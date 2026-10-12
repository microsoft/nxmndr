# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Ultralytics YOLO (DelineateAnything) over real gRPC through the production load path.

The server is ``create_server`` with its own InferenceService and ModelManager (default
loader ``build_model_record``, capacity from ``NXMNDR_MODEL_CACHE_CAPACITY``). Only
``ultralytics.YOLO`` is replaced, by ``tst.support.ultralytics_double``, so the
adapter, the registry loader, the model cache and its leases are all under test.
"""

from __future__ import annotations

import sys

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2, inference_pb2_grpc
from nxmndr.server import server
from tst.support.ultralytics_double import FakeYOLO, make_module, write_checkpoint

pytestmark = pytest.mark.integration

H, W = 16, 16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NXMNDR_MODEL_CACHE_CAPACITY", "2")


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.reset()
    monkeypatch.setitem(sys.modules, "ultralytics", make_module())
    return FakeYOLO


@pytest.fixture
def served(tmp_path):
    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    try:
        yield f"127.0.0.1:{port}", service
    finally:
        server.stop_server(srv, service, grace=2.0)


def _spec(path, *, fmt=inference_pb2.PYTORCH, model_id=""):
    return inference_pb2.ModelSpec(
        model_id=model_id,
        format=fmt,
        source=str(path),
        name="delineate",
        model_class="ultralytics_yolo",
        task=inference_pb2.SEGMENTATION,
    )


def _blocks(values=(1, 2, 3)):
    """Band 0: instance values[0] top-left 8x8, values[1] top-right 8x8, values[2] rows 12-15."""
    chip = np.zeros((H, W, 3), dtype=np.uint8)
    chip[:8, :8, 0] = values[0]
    chip[:8, 8:, 0] = values[1]
    chip[12:, :, 0] = values[2]
    return chip


def _expected(chip):
    band = chip[:, :, 0]
    return np.stack([band == v for v in sorted(set(band.ravel()) - {0})]).astype(np.uint8)


def _array(resp):
    assert resp.metadata.get("payload_format") == "raw"
    return np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(tuple(resp.shape))


def _stub(endpoint):
    channel = grpc.insecure_channel(endpoint)
    return channel, inference_pb2_grpc.InferenceServiceStub(channel)


def test_session_tiles_return_instance_stacks_and_reuse_one_loaded_model(tmp_path, fake_ultralytics, served):
    endpoint, service = served
    ckpt = write_checkpoint(tmp_path / "delineate.pt", group_mask=True)
    channel, stub = _stub(endpoint)
    with channel, InferenceGrpcClient(endpoint, timeout=30, max_attempts=1) as client:
        caps = client.capabilities()
        assert caps["instance_model_classes"] == "ultralytics_yolo"

        loaded = stub.LoadModel(inference_pb2.LoadModelRequest(spec=_spec(ckpt)))
        assert loaded.success, loaded.message
        meta = {e.key: e.value for e in loaded.effective_metadata}
        assert meta == {
            "model_cache_hit": "false",
            "capability.instances": "ultralytics_yolo",
            "capability.window": "full_chip",
        }
        mid = loaded.model_id
        record = service.model_manager.get(mid)
        assert record.backend == "ultralytics" and record.device_models == {}

        opened = client.open_session(session_id="yolo-a", spec=_spec(ckpt),
                                     options={"task_type": "segmentation", "conf": "0.3"})
        assert opened.status == "ok" and opened.model_cache_hit
        assert service.model_manager.pin_count(mid) == 1

        chip = _blocks()
        tiles = [
            ("r0_c0", chip, None),  # group mask present, dropped by default
            ("r0_c1", np.zeros((H, W, 3), np.uint8), None),  # nothing found
            ("r1_c0", chip, {"drop_group_masks": "false", "iou": "0.5", "max_det": "7"}),
            ("r1_c1", _blocks((5, 6, 7)).astype(np.uint16), None),  # other dtype, other values
        ]
        resps = list(
            client.stream_predict(
                session_id="yolo-a",
                samples=[t[1] for t in tiles],
                tile_ids=[t[0] for t in tiles],
                tile_options=[t[2] for t in tiles],
            )
        )
        by_tile = {r.metadata["tile_id"]: r for r in resps}
        assert sorted(by_tile) == [t[0] for t in tiles]
        for resp in resps:
            assert "error" not in resp.metadata, resp.metadata
            assert resp.metadata["result_type"] == "segmentation_mask"
            assert resp.metadata["task_type"] == "segmentation"
            assert resp.dtype == "uint8"
            assert "sam_prompt" not in resp.metadata

        np.testing.assert_array_equal(_array(by_tile["r0_c0"]), _expected(chip))
        assert list(by_tile["r0_c1"].shape) == [0, H, W]
        kept = _array(by_tile["r1_c0"])
        assert kept.shape == (4, H, W)
        np.testing.assert_array_equal(kept[0], (chip[:, :, 0] > 0).astype(np.uint8))
        np.testing.assert_array_equal(_array(by_tile["r1_c1"]), _expected(_blocks((5, 6, 7))))

        # A second session on the same checkpoint reuses the resident record.
        second = client.open_session(session_id="yolo-b", spec=_spec(ckpt))
        assert second.model_cache_hit and service.model_manager.pin_count(mid) == 2
        assert fake_ultralytics.constructed == [str(ckpt)]
        yolo = record.model.yolo
        # conf from the session options, iou and max_det from r1_c0's own options, else the defaults
        session_kwargs = {"verbose": False, "retina_masks": True, "conf": 0.3, "iou": 0.7, "max_det": 300}
        assert [c["kwargs"] for c in yolo.calls] == [
            session_kwargs, session_kwargs, {**session_kwargs, "iou": 0.5, "max_det": 7}, session_kwargs]
        assert yolo.calls[3]["dtype"] == "uint8"  # the adapter's conversion, not the wire dtype

        for sid in ("yolo-a", "yolo-b"):
            assert client.close_session(sid).status == "closed"
        assert service.model_manager.pin_count(mid) == 0

        listed = {m.model_id: m for m in stub.ListModels(inference_pb2.ListModelsRequest()).models}
        assert listed[mid].format == inference_pb2.PYTORCH and listed[mid].name == "ultralytics"


def test_unary_predict_equals_the_streamed_tile_and_masks_are_resized(tmp_path, fake_ultralytics, served):
    endpoint, service = served
    ckpt = write_checkpoint(tmp_path / "half.onnx", mask_divisor=2, mask_dtype="float32")
    chip = _blocks()
    with InferenceGrpcClient(endpoint, timeout=30, max_attempts=1) as client:
        mid = client.load_model(
            "ignored",
            {"format": "onnx", "source": str(ckpt), "model_class": "ultralytics_yolo", "task": "segmentation"},
        )
        unary = client.predict(mid, chip)
        streamed = list(client.stream_predict(model_id=mid, samples=[chip], tile_ids=["t"]))[0]
    for result in (unary, streamed):
        assert result.metadata["result_type"] == "segmentation_mask"
        out = np.frombuffer(result.output, dtype=np.dtype(result.dtype)).reshape(tuple(result.shape))
        np.testing.assert_array_equal(out, _expected(chip))
    assert unary.output == streamed.output
    assert fake_ultralytics.constructed == [str(ckpt)]
    assert service.model_manager.get(mid).model.yolo.predict(chip)[0].masks.data.shape == (3, 8, 8)


def test_unavailable_options_and_bad_chips_fail_their_tile_only(tmp_path, fake_ultralytics, served):
    endpoint, _ = served
    ckpt = write_checkpoint(tmp_path / "delineate.pt")
    chip = _blocks()
    with InferenceGrpcClient(endpoint, timeout=30, max_attempts=1) as client:
        opened = client.open_session(session_id="yolo-bad", spec=_spec(ckpt))
        assert opened.status == "ok"
        tiles = [
            ("emb", chip, {"return_embeddings": "true"}),
            ("conf", chip, {"return_confidence": "true"}),
            ("task", chip, {"task_type": "embedding"}),
            ("flag", chip, {"drop_group_masks": "maybe"}),
            ("conf_range", chip, {"conf": "1.5"}),
            ("iou_text", chip, {"iou": "high"}),
            ("max_det_zero", chip, {"max_det": "0"}),
            ("bands", np.zeros((H, W, 2), np.uint8), None),
            ("ok", chip, None),
        ]
        resps = list(
            client.stream_predict(
                session_id="yolo-bad",
                samples=[t[1] for t in tiles],
                tile_ids=[t[0] for t in tiles],
                tile_options=[t[2] for t in tiles],
            )
        )
    codes = {r.metadata["tile_id"]: (r.metadata.get("error_code"), r.metadata.get("error_scope")) for r in resps}
    assert codes == {
        "emb": ("malformed_options", "tile"),
        "conf": ("malformed_options", "tile"),
        "task": ("malformed_options", "tile"),
        "flag": ("malformed_options", "tile"),
        "conf_range": ("malformed_options", "tile"),
        "iou_text": ("malformed_options", "tile"),
        "max_det_zero": ("malformed_options", "tile"),
        "bands": ("malformed_payload", "tile"),
        "ok": (None, None),
    }


def test_load_refusals(tmp_path, fake_ultralytics, served, monkeypatch):
    endpoint, _ = served
    channel, stub = _stub(endpoint)
    with channel:
        hf = _spec("org/yolo", fmt=inference_pb2.HUGGINGFACE)
        with pytest.raises(grpc.RpcError) as err:
            stub.LoadModel(inference_pb2.LoadModelRequest(spec=hf))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        detect = write_checkpoint(tmp_path / "detect.pt", task="detect")
        with pytest.raises(grpc.RpcError) as err:
            stub.LoadModel(inference_pb2.LoadModelRequest(spec=_spec(detect)))
        assert err.value.code() == grpc.StatusCode.INTERNAL
        assert "'detect' model" in err.value.details()

        monkeypatch.setitem(sys.modules, "ultralytics", None)
        ok = write_checkpoint(tmp_path / "delineate.pt")
        with pytest.raises(grpc.RpcError) as err:
            stub.LoadModel(inference_pb2.LoadModelRequest(spec=_spec(ok)))
        assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        caps = stub.Capabilities(inference_pb2.CapabilitiesRequest())
        assert {c.key: c.value for c in caps.capabilities}["instance_model_classes"] == ""
    assert fake_ultralytics.constructed == [str(detect)]
