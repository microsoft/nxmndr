# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The client's tile iterable and LoadModel result over real gRPC.

``stream_predict(tiles=...)`` takes one iterable of ``(tile_id, sample, options)``,
and ``load_model_result`` returns LoadModel's effective metadata so the host can
check the loaded model's capabilities before it streams.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2
from nxmndr.server import server
from nxmndr.tensor_bundle import unpack_tensor_bundle
from tst.support.grpc_harness import running_server
from tst.support.stream_doubles import CountingLoader, SegLogitsModel
from tst.support.ultralytics_double import FakeYOLO, make_module, write_checkpoint

pytestmark = pytest.mark.integration

VOLATILE = {"corr_id", "latency_infer_ms", "device_id", "device_type", "bundle_size_bytes"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


def _payload(resp):
    if resp.metadata.get("payload_format") == "npz":
        return {k: v.tolist() for k, v in unpack_tensor_bundle(resp.output).items()}
    return np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(tuple(resp.shape)).tolist()


def test_one_tile_iterable_gives_the_same_responses_as_aligned_iterables(tmp_path):
    loader = CountingLoader({"double://seg": lambda: (SegLogitsModel(), "onnx")})
    chips = [np.full((4, 4, 1), v, np.float32) for v in (1.0, 0.0, 1.0)]
    ids = ["r0_c0", "r0_c1", "r0_c2"]
    opts = [{"return_confidence": "true"}, None, {"return_embeddings": "true"}]
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = c.load_model("ignored", {"format": "onnx", "source": "double://seg", "task": "segmentation"})
        results = []
        for sid, kwargs in (
            ("aligned", {"samples": chips, "tile_ids": ids, "tile_options": opts}),
            ("one-iterable", {"tiles": iter(list(zip(ids, chips, opts)))}),
        ):
            spec = inference_pb2.ModelSpec(model_id=mid)
            assert c.open_session(session_id=sid, spec=spec, options={"task_type": "segmentation"}).status == "ok"
            resps = list(c.stream_predict(session_id=sid, **kwargs))
            assert c.close_session(sid).summary.ok_tiles == 3
            results.append(
                [
                    (r.metadata["tile_id"], _payload(r),
                     {k: v for k, v in r.metadata.items() if k not in VOLATILE | {"session_id"}})
                    for r in resps
                ]
            )
    assert results[0] == results[1]
    assert [t[2]["payload_format"] for t in results[1]] == ["npz", "raw", "npz"]


def test_load_model_result_reports_capabilities_of_the_loaded_model(tmp_path, monkeypatch):
    FakeYOLO.reset()
    monkeypatch.setitem(sys.modules, "ultralytics", make_module())
    ckpt = write_checkpoint(tmp_path / "delineate.pt")
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.PYTORCH, source=str(ckpt), model_class="ultralytics_yolo",
        task=inference_pb2.SEGMENTATION,
    )
    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    try:
        with InferenceGrpcClient(f"127.0.0.1:{port}", timeout=30, max_attempts=1) as c:
            first = c.load_model_result("", spec)
            again = c.load_model_result("", spec)
            by_mapping = c.load_model_result(
                "ignored", {"format": "pytorch", "source": str(ckpt), "model_class": "ultralytics_yolo",
                            "task": "segmentation"},
            )
    finally:
        server.stop_server(srv, service, grace=2.0)
    assert first.effective_metadata == {
        "model_cache_hit": "false",
        "capability.instances": "ultralytics_yolo",
        "capability.window": "full_chip",
    }
    assert again.model_id == by_mapping.model_id == first.model_id
    assert again.effective_metadata["model_cache_hit"] == "true"
    assert by_mapping.effective_metadata["model_cache_hit"] == "true"
    assert FakeYOLO.constructed == [str(ckpt)]
