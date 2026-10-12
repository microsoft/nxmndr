# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""StreamPredict servicer called in-process (no transport).

Models enter through the ModelManager lease API, a session must come from
OpenSession (no fabricated sessions), chunk counts are not limited, and limit
failures are tile-scope.
"""

import logging

import grpc
import numpy as np

from nxmndr.inference import inference_pb2
from nxmndr.server import managers, server
from tst.support.stream_doubles import CountingLoader


class _DummyContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class _SegModel:
    def predict(self, arr, return_embeddings=False):
        batch, _, h, w = arr.shape
        logits = np.zeros((batch, 2, h, w), dtype=np.float32)
        logits[:, 1, ...] = 1.0
        return logits


class _ZeroModel:
    def predict(self, arr, return_embeddings=False):
        return np.zeros((1, 1, 1, 1), dtype=np.float32)


def _service(tmp_path, model, task=inference_pb2.SEGMENTATION):
    manager = managers.ModelManager(
        None, capacity=4, loader=CountingLoader({"double://m": lambda: (model, "onnx")})
    )
    svc = server.InferenceService(model_cache_dir=tmp_path / "cache", model_manager=manager)
    load = svc.LoadModel(
        inference_pb2.LoadModelRequest(
            spec=inference_pb2.ModelSpec(format=inference_pb2.ONNX, source="double://m", task=task)
        ),
        _DummyContext(),
    )
    assert load.success, load.message
    return svc, load.model_id


def _open(svc, session_id, model_id, **kwargs):
    resp = svc.OpenSession(
        inference_pb2.OpenSessionRequest(
            session_id=session_id, spec=inference_pb2.ModelSpec(model_id=model_id), **kwargs
        ),
        _DummyContext(),
    )
    assert resp.status == "ok", resp.error
    return resp


def _tile(arr, tile_id, session_id="", chunk_bytes=0):
    raw = arr.tobytes()
    step = chunk_bytes or len(raw)
    reqs = []
    for pos in range(0, len(raw), step):
        ctx = {"tile_id": tile_id}
        if session_id:
            ctx["session_id"] = session_id
        reqs.append(
            inference_pb2.StreamPredictRequest(
                chunk=raw[pos : pos + step], shape=list(arr.shape), dtype=str(arr.dtype), context=ctx
            )
        )
    reqs.append(inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": tile_id}))
    return reqs


def test_stream_predict_two_samples(tmp_path):
    svc, mid = _service(tmp_path, _SegModel())
    _open(svc, "sessA", mid)
    arr1 = np.zeros((1, 3, 2, 2), dtype=np.float32)
    arr2 = np.ones((1, 3, 2, 2), dtype=np.float32)
    reqs = _tile(arr1, "tile1", "sessA") + _tile(arr2, "tile2", "sessA")

    responses = list(svc.StreamPredict(iter(reqs), _DummyContext()))
    assert len(responses) == 2
    assert [r.metadata["tile_id"] for r in responses] == ["tile1", "tile2"]
    for resp in responses:
        assert resp.metadata.get("error") is None
        assert resp.metadata.get("result_type") == "segmentation_mask"
        assert resp.metadata.get("payload_format") == "raw"
        mask = np.frombuffer(resp.output, dtype=resp.dtype).reshape(resp.shape)
        assert np.all(mask == 1)
        assert resp.metadata.get("device_id")
        assert resp.metadata.get("session_id") == "sessA"
        assert resp.end_of_sequence is True


def test_stream_predict_missing_model(tmp_path):
    svc, _ = _service(tmp_path, _SegModel())
    req = inference_pb2.StreamPredictRequest(model_id="missing", end_of_sequence=True)
    ctx = _DummyContext()
    responses = list(svc.StreamPredict(iter([req]), ctx))
    assert responses == []
    assert ctx.code == grpc.StatusCode.NOT_FOUND


def test_stream_predict_unknown_session_is_not_fabricated(tmp_path):
    svc, mid = _service(tmp_path, _SegModel())
    ctx = _DummyContext()
    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    responses = list(svc.StreamPredict(iter(_tile(arr, "t", "never-opened")), ctx))
    assert ctx.code == grpc.StatusCode.NOT_FOUND
    assert [r.metadata["error_code"] for r in responses] == ["unknown_session"]
    assert "never-opened" not in svc._sessions
    assert svc._session_state("never-opened") == "unknown"
    assert svc.model_manager.pin_count(mid) == 0


def test_stream_predict_many_chunks_are_not_backpressure(tmp_path):
    """20 data chunks plus EOS for one tile used to be rejected as backpressure."""
    svc, mid = _service(tmp_path, _ZeroModel(), task=inference_pb2.TASK_TYPE_UNSPECIFIED)
    svc._default_max_inflight = 16
    arr = np.zeros((1, 1, 1, 20), dtype=np.float32)
    reqs = _tile(arr, "frag", chunk_bytes=4)
    for req in reqs:
        req.model_id = mid
    assert len(reqs) == 21
    responses = list(svc.StreamPredict(iter(reqs), _DummyContext()))
    assert len(responses) == 1
    assert responses[0].metadata.get("error") is None


def test_stream_predict_chunk_over_limit(tmp_path):
    svc, mid = _service(tmp_path, _ZeroModel(), task=inference_pb2.TASK_TYPE_UNSPECIFIED)
    svc._default_chunk_bytes = 4
    _open(svc, "sessX", mid)
    wide = np.zeros((1, 1, 1, 2), dtype=np.float32)  # one 8-byte chunk > 4
    narrow = np.zeros((1, 1, 1, 1), dtype=np.float32)
    reqs = _tile(wide, "tile-ovr", "sessX") + _tile(narrow, "tile-ok", "sessX")

    resps = list(svc.StreamPredict(iter(reqs), _DummyContext()))
    assert len(resps) == 2
    assert "chunk exceeds" in resps[0].metadata.get("error")
    assert resps[0].metadata.get("error_code") == "chunk_too_large"
    assert resps[0].metadata.get("error_scope") == "tile"
    assert resps[0].metadata.get("session_id") == "sessX"
    assert resps[0].metadata.get("tile_id") == "tile-ovr"
    assert resps[1].metadata.get("tile_id") == "tile-ok"
    assert resps[1].metadata.get("error") is None


def test_stream_predict_client_disconnect_mid_stream(tmp_path):
    svc, mid = _service(tmp_path, _ZeroModel(), task=inference_pb2.TASK_TYPE_UNSPECIFIED)
    _open(svc, "sessC", mid)
    arr = np.zeros((1, 1, 1, 1), dtype=np.float32)

    def _requests():
        yield from _tile(arr, "t1", "sessC")
        raise grpc.RpcError()  # what the server's request iterator raises on client cancel

    resps = list(svc.StreamPredict(_requests(), _DummyContext()))
    assert [r.metadata.get("tile_id") for r in resps] == ["t1"]
    assert svc._session_state("sessC") == "disconnected"
    assert svc.model_manager.pin_count(mid) == 0


def test_session_lifecycle_open_close_cancel(tmp_path):
    svc, mid = _service(tmp_path, _ZeroModel(), task=inference_pb2.TASK_TYPE_UNSPECIFIED)
    open_resp = _open(svc, "sessL", mid, options={"max_tile_bytes": "4"})
    assert open_resp.session_id == "sessL"
    assert svc._sessions["sessL"].model_id == mid
    assert svc.model_manager.pin_count(mid) == 1

    ok = np.zeros((1, 1, 1, 1), dtype=np.float32)
    too_big = np.zeros((1, 1, 1, 2), dtype=np.float32)
    reqs = (
        _tile(ok, "a", "sessL") + _tile(ok, "b", "sessL") + _tile(ok, "c", "sessL")
        + _tile(too_big, "d", "sessL")
    )
    list(svc.StreamPredict(iter(reqs), _DummyContext()))

    close_resp = svc.CloseSession(
        inference_pb2.CloseSessionRequest(session_id="sessL"), _DummyContext()
    )
    assert close_resp.status == "closed"
    assert close_resp.summary.ok_tiles == 3
    assert close_resp.summary.failed_tiles == 1
    assert any("tile_too_large" in e for e in close_resp.summary.errors)
    assert "sessL" not in svc._sessions
    assert svc.model_manager.pin_count(mid) == 0

    # Cancel on a closed session reports its state; on an unknown one: not_found
    cancel_resp = svc.CancelSession(
        inference_pb2.CancelSessionRequest(session_id="sessL"), _DummyContext()
    )
    assert cancel_resp.status == "closed"
    cancel_ctx = _DummyContext()
    cancel_resp = svc.CancelSession(
        inference_pb2.CancelSessionRequest(session_id="never-opened"), cancel_ctx
    )
    assert cancel_resp.status == "not_found"
    assert cancel_ctx.code == grpc.StatusCode.NOT_FOUND


def test_unprefixed_context_keys_are_logged_once_per_stream(tmp_path, caplog):
    """Context keys without the opt. prefix are not options; the server says so once."""
    svc, mid = _service(tmp_path, _SegModel())
    _open(svc, "sessU", mid)
    arr = np.zeros((1, 3, 2, 2), dtype=np.float32)
    reqs = []
    for tile_id in ("u1", "u2"):
        tile = _tile(arr, tile_id, "sessU")
        tile[0].context["return_confidence"] = "true"  # unprefixed: not an option
        tile[0].context["batch_size"] = "8"  # the legacy key: not an option either
        reqs += tile
    with caplog.at_level(logging.WARNING, logger=server.logger.name):
        responses = list(svc.StreamPredict(iter(reqs), _DummyContext()))
    assert [r.metadata.get("error") for r in responses] == [None, None]
    assert all(r.metadata["payload_format"] == "raw" for r in responses)  # no confidence
    warnings = [r.getMessage() for r in caplog.records if "opt." in r.getMessage()]
    assert warnings == [
        "StreamPredict ignores context keys without the 'opt.' prefix: batch_size, "
        "return_confidence (per-tile options are context['opt.<name>'])"
    ]
