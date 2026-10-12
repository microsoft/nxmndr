# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model input contract (the "Model input" section of the server.py module docstring).

The model receives each chip exactly as sent, and ModelSpec.preprocessing and
postprocessing are refused instead of silently ignored.
"""

from __future__ import annotations

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from tst.support.grpc_harness import running_server
from tst.support.stream_doubles import CountingLoader, EchoModel

pytestmark = pytest.mark.integration


@pytest.fixture
def echo():
    return EchoModel()


@pytest.fixture
def loader(echo):
    return CountingLoader({"double://echo": lambda: (echo, "onnx")})


def _normalize():
    return inference_pb2.TransformSpec(name="normalize", params={"mean": "0.485,0.456,0.406"})


def test_the_model_gets_the_chip_exactly_as_sent(tmp_path, loader, echo):
    chip = np.arange(5 * 4 * 3, dtype=np.uint16).reshape(5, 4, 3) * 257
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = c.load_model("ignored", {"format": "onnx", "source": "double://echo"})
        c.predict(mid, chip)
        list(c.stream_predict(model_id=mid, samples=[chip], tile_ids=["t"]))
    assert len(echo.seen) == 2
    for seen in echo.seen:
        assert seen.shape == (5, 4, 3) and seen.dtype == np.uint16
        np.testing.assert_array_equal(seen, chip)


def test_transform_specs_are_refused_by_load_model_and_open_session(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = c.load_model("ignored", {"format": "onnx", "source": "double://echo"})
        stub = c._get_stub()

        pre = inference_pb2.ModelSpec(format=inference_pb2.ONNX, source="double://echo")
        pre.preprocessing.append(_normalize())
        with pytest.raises(grpc.RpcError) as err:
            stub.LoadModel(inference_pb2.LoadModelRequest(spec=pre))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "preprocessing" in err.value.details()

        by_id = inference_pb2.ModelSpec(model_id=mid)
        by_id.postprocessing.append(inference_pb2.TransformSpec(name="argmax"))
        with pytest.raises(InferenceGrpcError) as err:
            c.open_session(session_id="s-post", spec=by_id)
        assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT

        with pytest.raises(InferenceGrpcError) as err:
            c.open_session(session_id="s-pre", spec=pre)
        assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT
        assert h.manager.pin_count(mid) == 0 and len(h.manager.list_models()) == 1
