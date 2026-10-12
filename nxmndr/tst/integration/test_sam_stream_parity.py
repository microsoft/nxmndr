# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SAM prompting: unary Predict and StreamPredict give identical results.

A real loopback gRPC server runs the production dispatcher and SAM handler
(``nxmndr.models.sam``); only the transformers SAM 3 model/processor are replaced
by deterministic doubles, so prompt parsing, routing, nesting and post-processing
are all exercised. No weights or network.
"""

from __future__ import annotations

import json
import threading

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from nxmndr.models import sam as sam_support
from nxmndr.tensor_bundle import unpack_tensor_bundle
from tst.support.grpc_harness import running_server, wait_until
from tst.support.stream_doubles import (
    CountingLoader,
    CountingVariantLoader,
    FakeHFSamModel,
    NamedLikeSamModel,
)

pytestmark = pytest.mark.integration

H, W = 16, 16
VOLATILE_META = {
    "corr_id",
    "latency_infer_ms",
    "latency_total_ms",
    "device_id",
    "device_type",
    "session_id",
    "tile_id",
    "progress",
    "bundle_size_bytes",
}

PTS = json.dumps([[3.0, 3.0], [12.0, 12.0]])
LABELS_POS_NEG = json.dumps([1, 0])
BOX = json.dumps([2.0, 2.0, 10.0, 10.0])

CASES = {
    "text": {"sam_text_prompt": "field"},
    "points_pos_neg": {"sam_input_points": PTS, "sam_input_labels": LABELS_POS_NEG},
    "box": {"sam_input_bbox": BOX},
    "points_and_box": {
        "sam_input_points": json.dumps([[4.0, 4.0]]),
        "sam_input_labels": json.dumps([0]),
        "sam_input_bbox": BOX,
    },
    "geometry_over_text": {"sam_text_prompt": "field", "sam_input_bbox": BOX},
    "no_prompt": {},
    "conf_threshold_empties": {"sam_text_prompt": "square", "sam_conf_threshold": "0.7"},
    "mask_threshold_empties": {"sam_input_bbox": BOX, "sam_mask_threshold": "1.0"},
    "empty_text_result": {"sam_text_prompt": "nothing"},
    "embeddings_requested": {"sam_text_prompt": "square", "return_embeddings": "true"},
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sam_log():
    return []


@pytest.fixture
def variants(monkeypatch, sam_log):
    loader = CountingVariantLoader(sam_log)
    monkeypatch.setattr(sam_support, "load_sam_variant", loader)
    return loader


@pytest.fixture
def loader(sam_log):
    return CountingLoader(
        {
            "double://sam3": lambda: (FakeHFSamModel(sam_log), "huggingface"),
            "double://samples/sam-lookalike": lambda: (NamedLikeSamModel(), "onnx"),
        }
    )


def _image():
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def _load_sam(client):
    return client.load_model(
        "ignored", {"format": "huggingface", "source": "double://sam3", "task": "segmentation"}
    )


def _meta(metadata):
    return {k: v for k, v in metadata.items() if k not in VOLATILE_META}


def _payload(output, shape, dtype, metadata):
    if metadata.get("payload_format") == "npz":
        return unpack_tensor_bundle(output)
    return {"": np.frombuffer(output, dtype=np.dtype(dtype)).reshape(tuple(shape))}


def _assert_same(unary, streamed):
    assert _meta(unary.metadata) == _meta(streamed.metadata)
    u = _payload(unary.output, unary.shape, unary.dtype, unary.metadata)
    s = _payload(streamed.output, streamed.shape, streamed.dtype, streamed.metadata)
    assert sorted(u) == sorted(s)
    for key in u:
        assert u[key].dtype == s[key].dtype and u[key].shape == s[key].shape
        np.testing.assert_array_equal(u[key], s[key])


def test_sam_capability_is_reported_at_load(tmp_path, loader, variants):
    with running_server(tmp_path, loader=loader) as h:
        with grpc.insecure_channel(h.endpoint) as channel:
            stub = h.stub(channel)
            sam = stub.LoadModel(
                inference_pb2.LoadModelRequest(
                    spec=inference_pb2.ModelSpec(format=inference_pb2.HUGGINGFACE, source="double://sam3")
                )
            )
            other = stub.LoadModel(
                inference_pb2.LoadModelRequest(
                    spec=inference_pb2.ModelSpec(
                        format=inference_pb2.ONNX, source="double://samples/sam-lookalike", name="sam-ish"
                    )
                )
            )
        assert {e.key: e.value for e in sam.effective_metadata}["capability.sam"] == "sam3"
        assert "capability.sam" not in {e.key for e in other.effective_metadata}


@pytest.mark.parametrize("case", sorted(CASES))
def test_unary_and_stream_sam_results_are_identical(tmp_path, loader, variants, sam_log, case):
    options = dict(CASES[case])
    image = _image()
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load_sam(c)
        session = {"task_type": "segmentation"}
        unary = c.predict(mid, image, options=dict(session, **options))
        c.open_session(session_id="sess-sam", spec=inference_pb2.ModelSpec(model_id=mid), options=session)
        [streamed] = list(
            c.stream_predict(session_id="sess-sam", samples=[image], tile_ids=["r0_c0"], tile_options=[options])
        )
        assert streamed.metadata.get("error") is None, streamed.metadata
        assert streamed.metadata["tile_id"] == "r0_c0"
        _assert_same(unary, streamed)

        meta = unary.metadata
        assert meta["result_type"] == "segmentation_mask"
        has_geometry = any(k in options for k in ("sam_input_points", "sam_input_bbox"))
        expected_mode = "geometry" if has_geometry else ("text" if "sam_text_prompt" in options else "none")
        assert meta["sam_prompt"] == expected_mode

        out = _payload(unary.output, unary.shape, unary.dtype, meta)
        mask = out["mask"] if "mask" in out else out[""]
        if case == "text":
            assert mask.shape == (2, H, W)  # group mask dropped, two parcels kept
        elif case in ("conf_threshold_empties", "empty_text_result"):
            assert mask.shape == (0, H, W) and mask.dtype == np.uint16  # valid empty result
        elif case == "mask_threshold_empties":
            assert mask.shape == (H, W) and not mask.any()
        elif case == "no_prompt":
            assert mask.shape == (3, H, W)  # the record's own unprompted inference
        elif case == "box":
            expected = np.zeros((H, W), np.uint16)
            expected[2:10, 2:10] = 1
            np.testing.assert_array_equal(mask, expected)
        elif case == "points_pos_neg":
            assert mask[3, 3] == 1 and mask[12, 12] == 0
        elif case == "points_and_box":
            assert mask[8, 8] == 1 and mask[4, 4] == 0  # negative point carves the box

        # one object per prompt, labels and box passed as such (never 4 corner points)
        tracker_calls = [e for e in sam_log if e.get("kind") == "tracker" and "image_shape" in e]
        for call in tracker_calls:
            if "input_points" in call:
                assert len(call["input_points"]) == 1 and len(call["input_points"][0]) == 1
                assert len(call["input_labels"][0][0]) == len(call["input_points"][0][0])
            if "input_boxes" in call:
                assert call["input_boxes"] == [[[2.0, 2.0, 10.0, 10.0]]]
        if case == "points_pos_neg":
            assert tracker_calls[-1]["input_points"] == [[[[3.0, 3.0], [12.0, 12.0]]]]
            assert tracker_calls[-1]["input_labels"] == [[[1, 0]]]
            assert "input_boxes" not in tracker_calls[-1]
        if case == "box":
            assert "input_points" not in tracker_calls[-1]
        if case == "geometry_over_text":
            assert not [e for e in sam_log if e.get("kind") == "text" and "image_shape" in e]


@pytest.mark.parametrize(
    "bad",
    [
        {"sam_input_points": "[[[[1, 2]]]]", "sam_input_labels": "[1]"},  # legacy nesting
        {"sam_input_points": "[[1, 2], [3, 4]]", "sam_input_labels": "[1]"},  # count mismatch
        {"sam_input_points": "[[1, 2]]"},  # points without labels
        {"sam_input_bbox": "[1, 2, 3"},  # malformed JSON
        {"sam_input_bbox": "[30, 40, 10, 20]"},  # inverted box
        {"sam_input_points": "[[1, 2]]", "sam_input_labels": "[2]"},  # label not 0/1
        {"sam_conf_threshold": "1.5", "sam_text_prompt": "field"},
    ],
)
def test_malformed_prompts_are_errors_in_both_paths(tmp_path, loader, variants, bad):
    image = _image()
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load_sam(c)
        with pytest.raises(InferenceGrpcError) as err:
            c.predict(mid, image, options=bad)
        assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT
        record = h.manager.get(mid)
        assert record.model.unprompted_calls == 0  # never fell back to unprompted inference

        resps = list(
            c.stream_predict(model_id=mid, samples=[image, image], tile_ids=["bad", "good"],
                             tile_options=[bad, {"sam_text_prompt": "field"}])
        )
        assert resps[0].metadata["error_code"] == "malformed_options"
        assert resps[0].metadata["error_scope"] == "tile"
        assert resps[1].metadata.get("error") is None and resps[1].metadata["sam_prompt"] == "text"
        assert record.model.unprompted_calls == 0


def test_successive_tiles_with_different_prompts_do_not_leak(tmp_path, loader, variants, sam_log):
    image = _image()
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load_sam(c)
        c.open_session(session_id="sess-leak", spec=inference_pb2.ModelSpec(model_id=mid),
                       options={"task_type": "segmentation"})
        resps = list(
            c.stream_predict(
                session_id="sess-leak",
                samples=[image] * 3,
                tile_ids=["points", "text", "none"],
                tile_options=[
                    {"sam_input_points": PTS, "sam_input_labels": LABELS_POS_NEG},
                    {"sam_text_prompt": "field"},
                    None,
                ],
            )
        )
        assert [r.metadata["sam_prompt"] for r in resps] == ["geometry", "text", "none"]
        text_calls = [e for e in sam_log if e.get("kind") == "text" and "image_shape" in e]
        assert len(text_calls) == 1 and "input_points" not in text_calls[0]
        assert h.manager.get(mid).model.unprompted_calls == 1


def test_non_sam_model_with_sam_in_its_name_is_not_routed_to_sam(tmp_path, loader, variants, sam_log):
    image = _image().astype(np.float32) / 255.0
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = c.load_model(
            "sam-model", {"format": "onnx", "source": "double://samples/sam-lookalike", "name": "sam3"}
        )
        result = c.predict(mid, image, options={"sam_text_prompt": "field", "task_type": "segmentation"})
        assert "sam_prompt" not in result.metadata
        assert sam_log == [] and sum(variants.calls.values()) == 0
        assert len(h.manager.get(mid).model.calls) == 1


def test_sam_resources_load_once_across_tiles_and_concurrent_sessions(tmp_path, loader, variants):
    image = _image()
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        spec = inference_pb2.ModelSpec(format=inference_pb2.HUGGINGFACE, source="double://sam3")
        results, errors = {}, []

        def run(session_id):
            try:
                with h.client() as own:
                    opened = own.open_session(session_id=session_id, spec=spec,
                                              options={"task_type": "segmentation"})
                    assert opened.status == "ok"
                    results[session_id] = list(
                        own.stream_predict(
                            session_id=session_id,
                            samples=[image] * 6,
                            tile_ids=[f"t{i}" for i in range(6)],
                            options={"sam_input_bbox": BOX},
                        )
                    )
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(sid,)) for sid in ("s1", "s2")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert errors == []
        assert all(r.metadata.get("error") is None for rs in results.values() for r in rs)
        assert sum(len(rs) for rs in results.values()) == 12

        assert loader.loads["double://sam3"] == 1  # one record for both sessions
        assert dict(variants.calls) == {("sam3_tracker", "cpu"): 1}  # tracker built once
        mid = results["s1"][0].metadata["model_id"]
        assert results["s2"][0].metadata["model_id"] == mid
        assert h.manager.pin_count(mid) == 2  # two open sessions, no running calls

        c.close_session("s1")
        c.close_session("s2")
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)
        assert sum(variants.disposals.values()) == 0  # still cached while the record lives
        resp = c._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert resp.success and h.manager.get(mid) is None
        assert dict(variants.disposals) == {("sam3_tracker", "cpu"): 1}  # disposed exactly once
