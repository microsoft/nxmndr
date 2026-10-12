# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import grpc
import pytest

from nxmndr.client import InferenceGrpcClient, InferenceGrpcError


class _FakeRpcError(grpc.RpcError):
    def __init__(self, message):
        super().__init__()
        self._message = message

    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return self._message


def test_with_retry_raises_after_attempts(monkeypatch):
    attempts = {"count": 0}

    def boom():
        attempts["count"] += 1
        raise _FakeRpcError("fail")

    client = InferenceGrpcClient("localhost:0", max_attempts=2, backoff_seconds=0)
    with pytest.raises(InferenceGrpcError):
        client._with_retry(boom)
    assert attempts["count"] == 2


def test_stream_predict_cancel_fn(monkeypatch):
    sent = []

    class _StubResp:
        def __init__(self, idx):
            self.idx = idx
            self.metadata = {}
            self.end_of_sequence = True
            self.output = b""
            self.shape = [1, 1, 1, 1]
            self.dtype = "float32"

    class _StubIter:
        def __iter__(self):
            yield _StubResp(0)
            yield _StubResp(1)

    class _StubChannel:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _StubStub:
        def __init__(self, channel):
            self.channel = channel

        def StreamPredict(self, req_iter, timeout=None):
            for req in req_iter:
                sent.append(req)
            return _StubIter()

    def _fake_create_channel(self):
        return _StubChannel()

    def _fake_stub(channel):
        return _StubStub(channel)

    monkeypatch.setattr("nxmndr.client.InferenceGrpcClient._create_channel", _fake_create_channel)
    monkeypatch.setattr("nxmndr.client.inference_pb2_grpc.InferenceServiceStub", _fake_stub)

    client = InferenceGrpcClient("localhost:0", max_attempts=1, backoff_seconds=0)
    cancel_flag = {"stop": False}

    resps = list(
        client.stream_predict(
            model_id="m1",
            samples=[(0,)],
            session_id="s1",
            tile_ids=["t1"],
            cancel_fn=lambda: cancel_flag["stop"],
        )
    )
    cancel_flag["stop"] = True
    resps = list(
        client.stream_predict(
            model_id="m1",
            samples=[(0,), (1,)],
            session_id="s1",
            tile_ids=["t1", "t2"],
            cancel_fn=lambda: cancel_flag["stop"],
        )
    )
    assert len(resps) == 0  # cancelled before consumption
    assert sent  # requests were produced
