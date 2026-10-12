# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unary Predict payload handling, called in-process.

SAM routing comes from the loaded model's capability (never its name), prompts use
the flat v1 encoding (points ``[[x, y], ...]`` with matching 0/1 labels, one box),
and malformed prompts are INVALID_ARGUMENT.
"""

import json

import grpc
import numpy as np
import pytest

from nxmndr.inference import inference_pb2
from nxmndr.models import sam as sam_support
from nxmndr.server import managers, server
from tst.support.stream_doubles import (
    CountingLoader,
    CountingVariantLoader,
    FakeHFSamModel,
    NamedLikeSamModel,
)


class _DummyContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _service(tmp_path, factories, fmt=inference_pb2.ONNX, source="double://m"):
    manager = managers.ModelManager(None, capacity=4, loader=CountingLoader(factories))
    svc = server.InferenceService(model_cache_dir=tmp_path / "cache", model_manager=manager)
    load = svc.LoadModel(
        inference_pb2.LoadModelRequest(spec=inference_pb2.ModelSpec(format=fmt, source=source)),
        _DummyContext(),
    )
    assert load.success, load.message
    return svc, load.model_id


@pytest.fixture
def sam_log():
    return []


@pytest.fixture
def variants(monkeypatch, sam_log):
    loader = CountingVariantLoader(sam_log)
    monkeypatch.setattr(sam_support, "load_sam_variant", loader)
    return loader


def _sam_service(tmp_path, sam_log, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    return _service(
        tmp_path,
        {"double://sam3": lambda: (FakeHFSamModel(sam_log), "huggingface")},
        fmt=inference_pb2.HUGGINGFACE,
        source="double://sam3",
    )


def test_predict_routes_sam_prompts(tmp_path, sam_log, variants, monkeypatch):
    service, mid = _sam_service(tmp_path, sam_log, monkeypatch)
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    req = inference_pb2.PredictRequest(
        model_id=mid, input=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
    )
    req.options["sam_text_prompt"] = "tree"
    req.options["sam_input_points"] = json.dumps([[10, 12], [3, 4]])
    req.options["sam_input_labels"] = json.dumps([1, 0])
    req.options["sam_input_bbox"] = json.dumps([5, 5, 14, 14])
    req.options["task_type"] = "segmentation"

    ctx = _DummyContext()
    response = service.Predict(req, ctx)
    assert ctx.code is None, ctx.details

    # geometry takes precedence over text; one object prompt with labels and a real box
    [call] = [e for e in sam_log if e.get("kind") == "tracker" and "image_shape" in e]
    assert call["image_shape"] == (16, 16, 3)
    assert call["input_points"] == [[[[10.0, 12.0], [3.0, 4.0]]]]
    assert call["input_labels"] == [[[1, 0]]]
    assert call["input_boxes"] == [[[5.0, 5.0, 14.0, 14.0]]]
    assert not [e for e in sam_log if e.get("kind") == "text" and "image_shape" in e]

    output = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert output.shape == (16, 16)
    assert output.dtype == np.uint16
    assert output[12, 10] == 1 and output[4, 3] == 0
    assert response.metadata.get("result_type") == "segmentation_mask"
    assert response.metadata.get("sam_prompt") == "geometry"


def test_predict_malformed_sam_prompt_is_an_error(tmp_path, sam_log, variants, monkeypatch):
    service, mid = _sam_service(tmp_path, sam_log, monkeypatch)
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    req = inference_pb2.PredictRequest(
        model_id=mid, input=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
    )
    req.options["sam_input_points"] = "[[1, 2"  # malformed JSON
    req.options["sam_input_labels"] = "[1]"
    ctx = _DummyContext()
    response = service.Predict(req, ctx)
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
    assert response.metadata["error_code"] == "malformed_options"
    assert service.model_manager.get(mid).model.unprompted_calls == 0  # no silent fallback


def test_predict_does_not_route_by_model_name(tmp_path, sam_log, variants):
    service, mid = _service(
        tmp_path, {"double://samples/sam3": lambda: (NamedLikeSamModel(), "onnx")},
        source="double://samples/sam3",
    )
    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    req = inference_pb2.PredictRequest(
        model_id=mid, input=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
    )
    req.options["sam_text_prompt"] = "field"
    req.options["task_type"] = "segmentation"
    response = service.Predict(req, _DummyContext())
    assert "sam_prompt" not in response.metadata
    assert sam_log == [] and sum(variants.calls.values()) == 0


def test_predict_segmentation_mask_non_sam(tmp_path):
    class _Model:
        def predict(self, arr, return_embeddings=False):
            # Return a 2D mask directly; prepare_segmentation_mask will accept (H, W)
            return np.ones((2, 2), dtype=np.uint16)

    service, mid = _service(tmp_path, {"double://m": lambda: (_Model(), "onnx")})

    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    req = inference_pb2.PredictRequest(
        model_id=mid,
        input=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )
    req.options["task_type"] = "segmentation"

    response = service.Predict(req, _DummyContext())

    mask = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert mask.shape == (2, 2)
    assert response.metadata.get("result_type") == "segmentation_mask"
    # Since channel 1 > channel 0 everywhere, mask should be all ones
    assert np.all(mask == 1)


def test_predict_segmentation_mask_from_logits(tmp_path):
    class _Model:
        def predict(self, arr, return_embeddings=False):
            # Return logits shaped (N, C, H, W); channel 1 wins everywhere.
            return np.concatenate(
                [
                    np.zeros((1, 1, 2, 2), dtype=np.float32),
                    np.ones((1, 1, 2, 2), dtype=np.float32),
                ],
                axis=1,
            )

    service, mid = _service(tmp_path, {"double://m": lambda: (_Model(), "onnx")})

    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    req = inference_pb2.PredictRequest(
        model_id=mid,
        input=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )
    req.options["task_type"] = "segmentation"

    response = service.Predict(req, _DummyContext())

    mask = np.frombuffer(response.output, dtype=response.dtype).reshape(response.shape)
    assert mask.shape == (2, 2)
    assert response.metadata.get("result_type") == "segmentation_mask"
    assert np.all(mask == 1)
