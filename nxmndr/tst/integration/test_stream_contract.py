# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""StreamPredict v1 contract over a real loopback gRPC transport.

Model doubles enter through the ModelManager loader hook; everything above them
(server, shared dispatcher, client) is production code. Covers per-tile options,
cancellation of a blocked stream, fragmented tiles and the client window, dtype
overrides, unknown sessions, per-tile vs per-stream failures, session binding, and
lease/pinning behavior.
"""

from __future__ import annotations

import threading
import time

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from nxmndr.server import server
from nxmndr.tensor_bundle import unpack_tensor_bundle
from tst.support import grpc_harness
from tst.support.grpc_harness import running_server, wait_until
from tst.support.stream_doubles import (
    CountingLoader,
    EchoModel,
    GatedModel,
    SegLogitsModel,
)

pytestmark = pytest.mark.integration

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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def gated():
    return GatedModel()


@pytest.fixture
def loader(gated):
    return CountingLoader(
        {
            "double://seg": lambda: (SegLogitsModel(), "onnx"),
            "double://seg-b": lambda: (SegLogitsModel(), "onnx"),
            "double://seg-c": lambda: (SegLogitsModel(), "onnx"),
            "double://echo": lambda: (EchoModel(), "onnx"),
            "double://gated": lambda: (gated, "onnx"),
        }
    )


def _load(client, source, task="segmentation"):
    return client.load_model("ignored", {"format": "onnx", "source": source, "task": task})


def _open(client, session_id, model_id, *, options=None, max_inflight=0, chunk_bytes=0):
    resp = client.open_session(
        session_id=session_id,
        spec=inference_pb2.ModelSpec(model_id=model_id),
        options=options or {},
        transport=inference_pb2.TransportCaps(max_inflight=max_inflight, chunk_bytes=chunk_bytes),
    )
    assert resp.status == "ok", resp.error
    return resp


def _decode(resp):
    if resp.metadata.get("payload_format") == "npz":
        return unpack_tensor_bundle(resp.output)
    return np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(tuple(resp.shape))


def _meta(resp):
    return {k: v for k, v in resp.metadata.items() if k not in VOLATILE_META}


def _consume(call, sink, errors):
    try:
        for resp in call:
            sink.append(resp)
    except Exception as exc:  # pragma: no cover - surfaced by assertions
        errors.append(exc)


# --------------------------------------------------------------- options


def test_mixed_per_tile_options_apply_to_their_tile_only(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        _open(c, "sess-mixed", mid, options={"task_type": "segmentation"})
        tiles = [np.full((8, 8, 3), v, np.float32) for v in (1.0, 0.0, 1.0, 1.0)]
        resps = list(
            c.stream_predict(
                session_id="sess-mixed",
                samples=tiles,
                tile_ids=["a", "b", "c", "d"],
                tile_options=[
                    {"return_confidence": "true"},
                    None,
                    {"task_type": "embedding"},
                    {"return_embeddings": "true"},
                ],
            )
        )
        by_tile = {r.metadata["tile_id"]: r for r in resps}
        assert sorted(by_tile) == ["a", "b", "c", "d"] and len(resps) == 4

        a = by_tile["a"]
        assert a.metadata["payload_format"] == "npz"
        assert a.metadata["bundle_keys"] == "confidence,mask"
        bundle = _decode(a)
        assert np.all(bundle["mask"] == 1) and bundle["confidence"].shape == (8, 8)

        # b follows a but carries no options: session options only, nothing leaks from a
        b = by_tile["b"]
        assert b.metadata["payload_format"] == "raw"
        assert b.metadata["result_type"] == "segmentation_mask"
        assert "has_confidence" not in b.metadata
        assert np.all(_decode(b) == 0)

        c_resp = by_tile["c"]
        assert c_resp.metadata["result_type"] == "embeddings"
        assert c_resp.metadata["bundle_keys"] == "embeddings"

        # d: embeddings requested per tile reach the model (streaming used to hardcode False)
        d = by_tile["d"]
        assert d.metadata["bundle_keys"] == "embeddings,mask"
        np.testing.assert_allclose(_decode(d)["embeddings"], [[1.0, 1.0, 1.0]])


def test_unary_and_stream_results_match_for_identical_options(tmp_path, loader):
    arr = np.random.default_rng(7).random((8, 8, 3), dtype=np.float32)
    option_sets = [
        {"task_type": "segmentation"},
        {"task_type": "segmentation", "return_confidence": "true"},
        {"task_type": "segmentation", "return_embeddings": "true"},
        {"task_type": "object_detection"},
        {"task_type": "embedding"},
    ]
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        for index, options in enumerate(option_sets):
            unary = c.predict(mid, arr, options=options)
            [streamed] = list(
                c.stream_predict(
                    model_id=mid, samples=[arr], tile_ids=[f"t{index}"], tile_options=[options]
                )
            )
            assert streamed.metadata.get("error") is None, streamed.metadata
            assert _meta(streamed) == {
                k: v for k, v in unary.metadata.items() if k not in VOLATILE_META
            }, options
            u, s = unary.output, streamed.output
            if unary.metadata.get("payload_format") == "npz":
                ub, sb = unpack_tensor_bundle(u), unpack_tensor_bundle(s)
                assert sorted(ub) == sorted(sb)
                for key in ub:
                    np.testing.assert_array_equal(ub[key], sb[key])
                    assert ub[key].dtype == sb[key].dtype
            else:
                assert (unary.dtype, tuple(unary.shape)) == (streamed.dtype, tuple(streamed.shape))
                assert u == s


def test_reserved_and_unprefixed_context_keys_are_not_options(tmp_path, loader):
    """A v1 server never treats unprefixed context keys as inference options."""
    with running_server(tmp_path, loader=loader) as h:
        with grpc.insecure_channel(h.endpoint) as channel:
            stub = h.stub(channel)
            with h.client() as c:
                mid = _load(c, "double://seg", task="")
            arr = np.ones((4, 4, 3), np.float32)
            reqs = [
                inference_pb2.StreamPredictRequest(
                    model_id=mid,
                    chunk=arr.tobytes(),
                    shape=list(arr.shape),
                    dtype="float32",
                    context={"tile_id": "t0", "task_type": "segmentation", "batch_size": "8"},
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": "t0"}),
            ]
            [resp] = list(stub.StreamPredict(iter(reqs)))
            assert resp.metadata["result_type"] == "raw"  # unprefixed task_type ignored
            assert resp.metadata.get("error") is None


# ---------------------------------------------------------- cancellation


def test_cancel_before_first_response_releases_blocked_caller(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://gated", task="")
        _open(c, "sess-cancel", mid)
        call = c.stream_predict(
            session_id="sess-cancel", samples=[np.ones((4, 4, 3), np.float32)], tile_ids=["t0"]
        )
        got, errors = [], []
        consumer = threading.Thread(target=_consume, args=(call, got, errors))
        consumer.start()
        assert gated.entered.wait(10), "server never started the tile"

        started = time.monotonic()
        call.cancel()  # the server is still inside predict: no response exists yet
        consumer.join(timeout=5)
        assert not consumer.is_alive(), "cancel did not release the blocked caller"
        assert time.monotonic() - started < 5
        assert got == [] and errors == [] and call.cancelled

        gated.release.set()  # let the running prediction finish
        assert wait_until(lambda: h.service._session_state("sess-cancel") == "disconnected")
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)


def test_cancel_fn_is_polled_while_no_response_arrives(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://gated", task="")
        flag = threading.Event()
        call = c.stream_predict(
            model_id=mid,
            samples=[np.ones((4, 4, 3), np.float32)],
            tile_ids=["t0"],
            cancel_fn=flag.is_set,
        )
        got, errors = [], []
        consumer = threading.Thread(target=_consume, args=(call, got, errors))
        consumer.start()
        assert gated.entered.wait(10)
        flag.set()
        consumer.join(timeout=5)
        assert not consumer.is_alive() and got == [] and errors == []
        gated.release.set()
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)


def test_cancel_session_stops_stream_before_next_dispatch(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client() as c, h.client() as control:
        mid = _load(c, "double://gated", task="")
        _open(c, "sess-cs", mid)
        call = c.stream_predict(
            session_id="sess-cs",
            samples=[np.ones((4, 4, 3), np.float32)] * 3,
            tile_ids=["t0", "t1", "t2"],
        )
        got, errors = [], []
        consumer = threading.Thread(target=_consume, args=(call, got, errors))
        consumer.start()
        assert gated.entered.wait(10)
        assert control.cancel_session("sess-cs", reason="user").status == "cancelled"
        assert h.manager.pin_count(mid) == 1  # the running tile keeps its execution lease
        gated.release.set()
        consumer.join(timeout=10)
        assert not consumer.is_alive() and errors == []
        assert [r.metadata.get("tile_id") for r in got] == ["t0", None]
        assert got[0].metadata.get("error") is None
        assert got[1].metadata["error_code"] == "cancelled"
        assert got[1].metadata["error_scope"] == "stream"
        assert gated.calls == 1  # t1 and t2 were never dispatched
        close = control.close_session("sess-cs")
        assert close.status == "cancelled" and close.summary.ok_tiles == 1
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)


# ----------------------------------------------------- limits


def test_fragmented_tiles_are_not_rejected_by_chunk_count(tmp_path, loader):
    with running_server(tmp_path, loader=loader, limits={"max_inflight": 16}) as h, h.client() as c:
        mid = _load(c, "double://echo", task="")
        _open(c, "sess-frag", mid, chunk_bytes=64)
        exactly_16 = np.arange(16 * 64, dtype=np.uint8).reshape(32, 32, 1)  # 16 chunks + EOS
        many = np.arange(100 * 8, dtype=np.uint8).reshape(20, 40, 1)
        [r16] = list(c.stream_predict(session_id="sess-frag", samples=[exactly_16], tile_ids=["t16"], chunk_bytes=64))
        [r100] = list(c.stream_predict(session_id="sess-frag", samples=[many], tile_ids=["t100"], chunk_bytes=8))
        assert r16.metadata.get("error") is None and r100.metadata.get("error") is None
        np.testing.assert_array_equal(_decode(r16), exactly_16)
        np.testing.assert_array_equal(_decode(r100), many)


def test_client_window_bounds_outstanding_tiles_without_deadlock(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        opened = _open(c, "sess-window", mid, max_inflight=2)
        assert opened.max_inflight == 2
        lock = threading.Lock()
        state = {"outstanding": 0, "peak": 0}

        def started(tile_id):
            with lock:
                state["outstanding"] += 1
                state["peak"] = max(state["peak"], state["outstanding"])

        def answered(resp):
            with lock:
                state["outstanding"] -= 1

        tiles = [np.full((8, 8, 3), i % 2, np.float32) for i in range(10)]
        resps = list(
            c.stream_predict(
                session_id="sess-window",
                samples=tiles,
                tile_ids=[f"t{i}" for i in range(10)],
                max_inflight=opened.max_inflight,
                on_tile_start=started,
                on_response=answered,
            )
        )
        assert sorted(r.metadata["tile_id"] for r in resps) == sorted(f"t{i}" for i in range(10))
        assert 1 <= state["peak"] <= 2


def test_tile_limits_fail_one_tile_and_the_stream_continues(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://echo", task="")
        _open(c, "sess-lim", mid, options={"max_tile_bytes": "64"}, chunk_bytes=32)
        small = np.zeros(16, np.uint8)  # one 16-byte chunk
        big = np.zeros(200, np.uint8)  # 7 chunks of 32 bytes, 200 > 64
        resps = list(
            c.stream_predict(
                session_id="sess-lim",
                samples=[small, big, small],
                tile_ids=["ok1", "big", "ok2"],
                chunk_bytes=32,
            )
        )
        assert [r.metadata["tile_id"] for r in resps] == ["ok1", "big", "ok2"]
        assert resps[1].metadata["error_code"] == "tile_too_large"
        assert resps[1].metadata["error_scope"] == "tile"
        assert resps[0].metadata.get("error") is None and resps[2].metadata.get("error") is None

        # a chunk over the negotiated chunk size fails only its tile
        resps = list(
            c.stream_predict(
                session_id="sess-lim",
                samples=[np.zeros(40, np.uint8), np.zeros(8, np.uint8)],
                tile_ids=["wide", "ok3"],
                chunk_bytes=40,
            )
        )
        assert resps[0].metadata["error_code"] == "chunk_too_large"
        assert resps[0].metadata["error_scope"] == "tile"
        assert resps[1].metadata.get("error") is None
        summary = c.close_session("sess-lim").summary
        assert (summary.ok_tiles, summary.failed_tiles) == (3, 2)


def test_stream_scope_error_ends_stream_and_fails_session(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://echo", task="")
        _open(c, "sess-scope", mid)
        arr = np.zeros(4, np.uint8)
        with grpc.insecure_channel(h.endpoint) as channel:
            reqs = [
                inference_pb2.StreamPredictRequest(
                    chunk=arr.tobytes(), shape=[4], dtype="uint8",
                    context={"session_id": "sess-scope", "tile_id": "t0"},
                ),
                inference_pb2.StreamPredictRequest(
                    end_of_sequence=True, context={"session_id": "sess-scope", "tile_id": "t0"}
                ),
                inference_pb2.StreamPredictRequest(
                    chunk=arr.tobytes(), shape=[4], dtype="uint8",
                    context={"session_id": "sess-scope"},  # no tile_id
                ),
                inference_pb2.StreamPredictRequest(
                    chunk=arr.tobytes(), shape=[4], dtype="uint8",
                    context={"session_id": "sess-scope", "tile_id": "t2"},
                ),
            ]
            resps = list(h.stub(channel).StreamPredict(iter(reqs)))
        assert len(resps) == 2
        assert resps[0].metadata["tile_id"] == "t0" and resps[0].metadata.get("error") is None
        assert resps[1].metadata["error_code"] == "missing_tile_id"
        assert resps[1].metadata["error_scope"] == "stream"
        assert "tile_id" not in resps[1].metadata
        assert h.service._session_state("sess-scope") == "failed"
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)


def test_malformed_payload_and_duplicate_tile_are_tile_scope(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://echo", task="")
        _open(c, "sess-bad", mid)
        with grpc.insecure_channel(h.endpoint) as channel:
            ctx = {"session_id": "sess-bad"}
            reqs = [
                # 3 bytes cannot be a float32 array of shape [2]
                inference_pb2.StreamPredictRequest(
                    chunk=b"abc", shape=[2], dtype="float32", context=dict(ctx, tile_id="bad")
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context=dict(ctx, tile_id="bad")),
                inference_pb2.StreamPredictRequest(
                    chunk=b"\x01\x02", shape=[2], dtype="uint8", context=dict(ctx, tile_id="dup")
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context=dict(ctx, tile_id="dup")),
                inference_pb2.StreamPredictRequest(
                    chunk=b"\x01\x02", shape=[2], dtype="uint8", context=dict(ctx, tile_id="dup")
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context=dict(ctx, tile_id="dup")),
                inference_pb2.StreamPredictRequest(
                    chunk=b"\x03", shape=[1], dtype="uint8",
                    context=dict(ctx, tile_id="opt", **{"opt.Bad-Name": "1"}),
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context=dict(ctx, tile_id="opt")),
            ]
            resps = list(h.stub(channel).StreamPredict(iter(reqs)))
        codes = [(r.metadata["tile_id"], r.metadata.get("error_code")) for r in resps]
        assert codes == [
            ("bad", "malformed_payload"),
            ("dup", None),
            ("dup", "duplicate_tile_id"),
            ("opt", "malformed_options"),
        ]
        assert all(r.metadata.get("error_scope") in (None, "tile") for r in resps)


# ------------------------------------------------------------------ dtype


def test_dtype_override_casts_bytes_or_is_rejected(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://echo", task="")
        values = np.array([[0, 1, 200, 255]], dtype=np.uint8)
        [resp] = list(c.stream_predict(model_id=mid, samples=[values], tile_ids=["u8"], dtype="float32"))
        assert resp.dtype == "float32"
        np.testing.assert_array_equal(_decode(resp), values.astype(np.float32))

        big_endian = np.array([1.5, -2.0], dtype=">f4")
        [resp] = list(c.stream_predict(model_id=mid, samples=[big_endian], tile_ids=["be"]))
        np.testing.assert_array_equal(_decode(resp), [1.5, -2.0])

        with pytest.raises(ValueError, match="not lossless"):
            list(
                c.stream_predict(
                    model_id=mid, samples=[np.array([0.5], np.float32)], tile_ids=["f"], dtype="uint8"
                )
            )


# ------------------------------------------------ sessions and leases


def test_stream_with_unknown_session_is_rejected_not_fabricated(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        seen = []
        with pytest.raises(InferenceGrpcError) as err:
            list(
                c.stream_predict(
                    model_id=mid,
                    session_id="never-opened",
                    samples=[np.ones((4, 4, 3), np.float32)],
                    tile_ids=["t0"],
                    on_response=seen.append,
                )
            )
        assert err.value.code == grpc.StatusCode.NOT_FOUND
        assert [r.metadata["error_code"] for r in seen] == ["unknown_session"]
        assert seen[0].metadata["error_scope"] == "stream"
        assert h.service._session_state("never-opened") == "unknown"
        assert "never-opened" not in h.service._sessions
        assert h.manager.pin_count(mid) == 0


def test_open_session_is_idempotent_and_bound_to_its_model(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        a = _load(c, "double://seg")
        b = _load(c, "double://seg-b")
        first = _open(c, "sess-idem", a, max_inflight=4)
        again = _open(c, "sess-idem", a, max_inflight=9)
        assert (again.session_id, again.max_inflight) == ("sess-idem", first.max_inflight)
        assert h.manager.pin_count(a) == 1
        with pytest.raises(InferenceGrpcError) as err:
            _open(c, "sess-idem", b)
        assert err.value.code == grpc.StatusCode.ALREADY_EXISTS
        assert c.close_session("sess-idem").status == "closed"
        assert h.manager.pin_count(a) == 0
        with pytest.raises(InferenceGrpcError) as err:
            _open(c, "sess-idem", a)  # closed IDs are tombstoned
        assert err.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_open_session_retry_reuses_client_generated_id(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        resp = c.open_session(spec=inference_pb2.ModelSpec(model_id=mid))
        assert resp.status == "ok" and len(resp.session_id) == 32


def test_pinned_session_model_stays_resident_under_a_tiny_cache(tmp_path, loader):
    with running_server(tmp_path, loader=loader, capacity=2) as h, h.client() as c:
        a = _load(c, "double://seg")
        _open(c, "sess-a", a)
        b = _load(c, "double://seg-b")
        _open(c, "sess-b", b)
        with pytest.raises(InferenceGrpcError) as err:
            _load(c, "double://seg-c")  # every slot is pinned
        assert err.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED

        [resp] = list(
            c.stream_predict(session_id="sess-a", samples=[np.ones((4, 4, 3), np.float32)], tile_ids=["t"])
        )
        assert resp.metadata.get("error") is None and resp.metadata["model_id"] == a

        c.close_session("sess-a")  # a's last lease ends: it becomes evictable
        cid = _load(c, "double://seg-c")
        assert h.manager.get(a) is None
        assert h.manager.get(b) is not None and h.manager.get(cid) is not None
        [resp] = list(
            c.stream_predict(session_id="sess-b", samples=[np.ones((4, 4, 3), np.float32)], tile_ids=["t"])
        )
        assert resp.metadata.get("error") is None
        assert loader.loads["double://seg"] == 1


def test_prediction_running_when_its_session_closes(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client() as c, h.client() as control:
        mid = _load(c, "double://gated", task="")
        _open(c, "sess-run", mid)
        call = c.stream_predict(
            session_id="sess-run", samples=[np.ones((4, 4, 3), np.float32)], tile_ids=["t0"]
        )
        got, errors = [], []
        consumer = threading.Thread(target=_consume, args=(call, got, errors))
        consumer.start()
        assert gated.entered.wait(10)

        closed = control.close_session("sess-run")
        assert closed.status == "closed"
        assert h.manager.pin_count(mid) == 1  # execution lease outlives the session
        resp = control._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert resp.success and h.manager.get(mid) is not None  # freed when the call ends

        gated.release.set()
        consumer.join(timeout=10)
        assert errors == [] and [r.metadata.get("error") for r in got] == [None]
        assert wait_until(lambda: h.manager.get(mid) is None)


def test_capabilities_advertise_stream_context_and_limits(tmp_path, loader):
    with running_server(tmp_path, loader=loader, limits={"chunk_bytes": 4096}) as h, h.client() as c:
        caps = c.capabilities()
        assert caps["stream_context_version"] == "1"
        assert caps["stream_default_chunk_bytes"] == "4096"
        ceiling = int(caps["stream_max_chunk_bytes"])
        assert int(caps["stream_max_tile_bytes"]) > 0 and int(caps["stream_max_inflight"]) > 0
        mid = _load(c, "double://seg")
        _open(c, "sess-wide", mid, chunk_bytes=8192)  # above the default: honored as requested
        with pytest.raises(InferenceGrpcError) as err:
            _open(c, "sess-too-wide", mid, chunk_bytes=ceiling + 1)
        assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_a_chunk_of_the_advertised_maximum_reaches_the_server(tmp_path, loader, monkeypatch):
    # A message carries its context (with the tile's options), shape and dtype besides
    # the chunk, and gRPC ends the whole stream for a message over its limit.
    limit = 2 * 1024 * 1024  # a small gRPC message limit, so the test sends megabytes, not 128 MiB
    monkeypatch.setattr(server, "GRPC_MAX_MESSAGE_LENGTH", limit)
    monkeypatch.setattr(
        grpc_harness,
        "GRPC_OPTIONS",
        [("grpc.max_send_message_length", limit), ("grpc.max_receive_message_length", limit)],
    )
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        ceiling = int(c.capabilities()["stream_max_chunk_bytes"])
        mid = _load(c, "double://echo", task="")
        _open(c, "sess-ceiling", mid, chunk_bytes=ceiling)
        tile = np.zeros(ceiling, dtype=np.uint8)  # the whole tile in one message of the maximum chunk
        [resp] = list(
            c.stream_predict(
                session_id="sess-ceiling", samples=[tile], tile_ids=["t0"], chunk_bytes=ceiling,
                options={"note": "x" * 1024},
            )
        )
        assert resp.metadata.get("error") is None, resp.metadata


def test_capabilities_survive_a_catalog_failure(tmp_path, loader, monkeypatch):
    from nxmndr.models import catalog

    def broken():
        raise RuntimeError("catalog probe failed")

    monkeypatch.setattr(catalog, "list_input_sizes", broken)
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        caps = c.capabilities()
    assert caps["stream_context_version"] == "1"
    assert "pytorch_model_classes" not in caps and "pytorch_model_input_sizes" not in caps


class _PeakModel:
    """Counts the predictions running at once; each blocks until ``release`` is set."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, arr, return_embeddings=False):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.entered.set()
        self.release.wait(10)
        with self.lock:
            self.running -= 1
        return np.array(arr, copy=True)


def test_inference_runs_at_most_max_cores_tiles_at_once(tmp_path):
    # The harness has 8 handler threads, so only the server's own inference bound
    # keeps the second session's tile out of the model while the first one runs.
    peak = _PeakModel()
    loader = CountingLoader({"double://peak": lambda: (peak, "onnx")})
    with running_server(tmp_path, loader=loader, max_cores=1) as h, h.client() as c:
        mid = _load(c, "double://peak", task="")
        results = {}

        def run(sid):
            _open(c, sid, mid)
            results[sid] = list(
                c.stream_predict(session_id=sid, samples=[np.zeros(4, np.uint8)], tile_ids=["t"])
            )

        threads = [threading.Thread(target=run, args=(sid,)) for sid in ("s1", "s2")]
        for t in threads:
            t.start()
        assert peak.entered.wait(10)
        time.sleep(0.5)  # time for the second tile to reach the model if nothing bounds it
        assert peak.peak == 1
        peak.release.set()
        for t in threads:
            t.join(10)
        assert peak.peak == 1
        assert sorted(results) == ["s1", "s2"]
        assert all(r[0].metadata.get("error") is None for r in results.values())


class _SlotSpy:
    """Wraps the service's inference semaphore; counts the calls that reached it."""

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()
        self.entered = 0

    def __enter__(self):
        with self.lock:
            self.entered += 1
        return self.inner.__enter__()

    def __exit__(self, *exc):
        return self.inner.__exit__(*exc)


@pytest.mark.parametrize("ending", ["cancel_session", "client_cancel"])
def test_a_tile_waiting_for_the_inference_slot_does_not_run_after_its_stream_ends(
    tmp_path, loader, gated, ending
):
    chip = np.ones((4, 4, 3), np.float32)
    with (
        running_server(tmp_path, loader=loader, max_cores=1) as h,
        h.client() as c,
        h.client() as control,
    ):
        slots = h.service._inference_slots = _SlotSpy(h.service._inference_slots)
        contexts = {}
        real_check = h.service._check_tile_may_run

        def recording_check(context, book, session_id, corr_id):
            contexts[session_id] = context
            return real_check(context, book, session_id, corr_id)

        h.service._check_tile_may_run = recording_check
        mid = _load(c, "double://gated", task="")
        _open(c, "s-running", mid)
        _open(c, "s-queued", mid)
        got_running, got_queued, errors = [], [], []
        running = c.stream_predict(session_id="s-running", samples=[chip], tile_ids=["t0"])
        threads = [threading.Thread(target=_consume, args=(running, got_running, errors))]
        threads[0].start()
        assert gated.entered.wait(10)  # s-running's tile holds the only inference slot
        queued = c.stream_predict(session_id="s-queued", samples=[chip], tile_ids=["t0"])
        threads.append(threading.Thread(target=_consume, args=(queued, got_queued, errors)))
        threads[1].start()
        assert wait_until(lambda: slots.entered == 2)  # s-queued's tile waits for the slot
        if ending == "cancel_session":
            assert control.cancel_session("s-queued", reason="user").status == "cancelled"
        else:
            queued.cancel()
            assert wait_until(lambda: not contexts["s-queued"].is_active())
        gated.release.set()
        for t in threads:
            t.join(10)
        assert gated.calls == 1  # the queued tile never reached the model
        assert errors == [] and [r.metadata.get("error") for r in got_running] == [None]
        if ending == "cancel_session":
            assert [r.metadata.get("error_code") for r in got_queued] == ["cancelled"]
            assert got_queued[0].metadata["error_scope"] == "stream"
            assert control.close_session("s-queued").summary.ok_tiles == 0
        else:
            assert got_queued == []
            assert wait_until(lambda: h.service._session_state("s-queued") == "disconnected")
        assert control.close_session("s-running").summary.ok_tiles == 1
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)


def test_open_session_refuses_a_negative_max_inflight(tmp_path, loader):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = _load(c, "double://seg")
        with pytest.raises(InferenceGrpcError) as err:
            _open(c, "sess-negative", mid, max_inflight=-1)
        assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT
        assert h.service._session_state("sess-negative") == "unknown"
        assert h.manager.pin_count(mid) == 0
        assert _open(c, "sess-default", mid, max_inflight=0).max_inflight == 16  # 0: the default


def test_serve_stops_grpc_then_shuts_down_the_model_cache(tmp_path, loader, monkeypatch):
    from nxmndr.server import managers, server

    created = []

    class _Capturing(server.InferenceService):
        def __init__(self, **kwargs):
            kwargs["model_manager"] = managers.ModelManager(None, capacity=2, loader=loader)
            super().__init__(**kwargs)
            created.append(self)

    class _Ctx:
        def set_code(self, code):
            raise AssertionError(f"unexpected status {code}")

        def set_details(self, details):
            pass

    monkeypatch.setattr(server, "InferenceService", _Capturing)
    started, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=server.serve,
        kwargs=dict(port=0, model_cache_dir=tmp_path, startup_event=started, stop_event=stop,
                    max_cores=1, stop_grace=1.0),
    )
    thread.start()
    assert started.wait(30)
    svc = created[0]
    load = svc.LoadModel(
        inference_pb2.LoadModelRequest(
            spec=inference_pb2.ModelSpec(format=inference_pb2.ONNX, source="double://seg")
        ),
        _Ctx(),
    )
    opened = svc.OpenSession(
        inference_pb2.OpenSessionRequest(
            session_id="sess-serve", spec=inference_pb2.ModelSpec(model_id=load.model_id)
        ),
        _Ctx(),
    )
    assert opened.status == "ok"

    stop.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert svc._session_state("sess-serve") == "shutdown"
    with pytest.raises(managers.CacheShutdownError):
        svc.model_manager.acquire_execution(model_id=load.model_id)
