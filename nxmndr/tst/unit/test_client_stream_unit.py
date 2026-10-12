# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InferenceGrpcClient request encoding, without a server (stub transport)."""

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcClient, InferenceGrpcError, prepare_tensor
from nxmndr.inference import inference_pb2


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code, message="boom"):
        super().__init__()
        self._code = code
        self._message = message

    def code(self):
        return self._code

    def details(self):
        return self._message


class _RecordingStub:
    """Consumes the request stream and answers one response per tile."""

    def __init__(self):
        self.requests = []
        self.open_requests = []
        self.open_failures = 0

    def StreamPredict(self, request_iterator, timeout=None):
        tiles = []
        for req in request_iterator:
            self.requests.append(req)
            if req.end_of_sequence:
                tiles.append(req.context["tile_id"])
        return iter(
            inference_pb2.StreamPredictResponse(end_of_sequence=True, metadata={"tile_id": t})
            for t in tiles
        )

    def OpenSession(self, request, timeout=None):
        self.open_requests.append(request)
        if self.open_failures:
            self.open_failures -= 1
            raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "connection reset")
        return inference_pb2.OpenSessionResponse(session_id=request.session_id, status="ok")


@pytest.fixture
def stubbed(monkeypatch):
    stub = _RecordingStub()
    client = InferenceGrpcClient("localhost:0", max_attempts=3, backoff_seconds=0)
    monkeypatch.setattr(client, "_get_stub", lambda: stub)
    return client, stub


def test_options_go_on_the_first_message_of_each_tile_only(stubbed):
    client, stub = stubbed
    tiles = [np.arange(8, dtype=np.uint8), np.arange(8, dtype=np.uint8)]
    resps = list(
        client.stream_predict(
            session_id="s1",
            samples=tiles,
            tile_ids=["r0_c0", "r0_c1"],
            tile_options=[{"sam_text_prompt": "field", "return_embeddings": True}, None],
            options={"task_type": "segmentation"},
            chunk_bytes=3,
        )
    )
    assert [r.metadata["tile_id"] for r in resps] == ["r0_c0", "r0_c1"]
    by_tile = {}
    for req in stub.requests:
        by_tile.setdefault(req.context["tile_id"], []).append(req)
    first, *rest = by_tile["r0_c0"]
    assert dict(first.context) == {
        "tile_id": "r0_c0",
        "session_id": "s1",
        "opt.task_type": "segmentation",
        "opt.sam_text_prompt": "field",
        "opt.return_embeddings": "True",
    }
    # 8 bytes in 3-byte chunks: 3 data messages, then EOS; later messages carry IDs only
    assert [len(r.chunk) for r in by_tile["r0_c0"]] == [3, 3, 2, 0]
    assert rest[-1].end_of_sequence
    for req in rest:
        assert dict(req.context) == {"tile_id": "r0_c0", "session_id": "s1"}
    # the second tile gets only the request-wide option: nothing leaks from r0_c0
    assert dict(by_tile["r0_c1"][0].context) == {
        "tile_id": "r0_c1",
        "session_id": "s1",
        "opt.task_type": "segmentation",
    }


def test_default_tile_ids_are_always_sent(stubbed):
    client, stub = stubbed
    list(client.stream_predict(model_id="m", samples=[np.zeros(2, np.uint8)] * 2))
    assert {req.context["tile_id"] for req in stub.requests} == {"tile-0", "tile-1"}
    assert all("session_id" not in req.context for req in stub.requests)


def test_reserved_option_name_is_an_error_not_a_silent_drop(stubbed):
    client, _ = stubbed
    with pytest.raises(ValueError, match="reserved"):
        list(client.stream_predict(model_id="m", samples=[np.zeros(2)], options={"session_id": "x"}))


def test_dtype_override_casts_the_bytes(stubbed):
    client, stub = stubbed
    values = np.array([0, 1, 200, 255], dtype=np.uint8)
    list(client.stream_predict(model_id="m", samples=[values], dtype="float32"))
    data = stub.requests[0]
    assert data.dtype == "float32"
    np.testing.assert_array_equal(np.frombuffer(data.chunk, dtype="<f4"), values.astype(np.float32))


def test_lossy_dtype_override_is_rejected(stubbed):
    client, stub = stubbed
    with pytest.raises(ValueError, match="not lossless"):
        list(client.stream_predict(model_id="m", samples=[np.array([0.5], np.float32)], dtype="uint8"))
    with pytest.raises(ValueError):
        prepare_tensor(np.array([1, 2], np.int16), "uint8")


def test_big_endian_input_is_sent_little_endian():
    arr = prepare_tensor(np.array([1.5, -2.0], dtype=">f4"))
    assert arr.dtype.byteorder in ("<", "=") and arr.dtype.name == "float32"
    np.testing.assert_array_equal(np.frombuffer(arr.tobytes(), "<f4"), [1.5, -2.0])


def test_retried_open_session_reuses_one_generated_id(stubbed):
    client, stub = stubbed
    stub.open_failures = 1
    resp = client.open_session(spec=inference_pb2.ModelSpec(model_id="m"))
    ids = [req.session_id for req in stub.open_requests]
    assert len(ids) == 2 and ids[0] == ids[1] and len(ids[0]) == 32
    assert resp.session_id == ids[0]


def test_deterministic_rejections_are_not_retried():
    client = InferenceGrpcClient("localhost:0", max_attempts=3, backoff_seconds=0)
    attempts = []

    def call():
        attempts.append(1)
        raise _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "malformed_options")

    with pytest.raises(InferenceGrpcError) as err:
        client._with_retry(call)
    assert len(attempts) == 1
    assert err.value.code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    ],
)
def test_server_answers_and_deadlines_are_not_retried(code):
    client = InferenceGrpcClient("localhost:0", max_attempts=3, backoff_seconds=0)
    attempts = []

    def call():
        attempts.append(1)
        raise _FakeRpcError(code, "inference_failed")

    with pytest.raises(InferenceGrpcError) as err:
        client._with_retry(call)
    assert len(attempts) == 1
    assert err.value.code == code


def test_unavailable_is_retried_on_the_same_channel(monkeypatch):
    made = []

    class _Channel:
        closed = False

        def close(self):
            self.closed = True

    def fake_create(self):
        made.append(_Channel())
        return made[-1]

    monkeypatch.setattr(InferenceGrpcClient, "_create_channel", fake_create)
    client = InferenceGrpcClient("localhost:0", max_attempts=3, backoff_seconds=0)
    attempts = []

    def call():
        client._get_channel()
        attempts.append(1)
        raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused")

    with pytest.raises(InferenceGrpcError) as err:
        client._with_retry(call)
    assert len(attempts) == 3 and err.value.code == grpc.StatusCode.UNAVAILABLE
    assert len(made) == 1 and not made[0].closed


def test_cancel_before_iteration_sends_nothing(stubbed):
    client, stub = stubbed
    call = client.stream_predict(model_id="m", samples=[np.zeros(2)])
    call.cancel()
    assert list(call) == [] and stub.requests == []


def test_misaligned_tile_ids_are_an_explicit_error(stubbed):
    client, _ = stubbed
    with pytest.raises(ValueError, match="fewer entries"):
        list(client.stream_predict(model_id="m", samples=[np.zeros(2)] * 2, tile_ids=["only-one"]))
    with pytest.raises(ValueError, match="fewer entries"):
        list(client.stream_predict(model_id="m", samples=[np.zeros(2)] * 2, tile_options=[{}]))


# ------------------------------------------ one tile iterable


def _stream_both_ways(client, stub, chips, ids, opts):
    list(client.stream_predict(session_id="s1", samples=chips, tile_ids=ids, tile_options=opts,
                               options={"task_type": "segmentation"}, chunk_bytes=3))
    aligned = [(dict(r.context), r.chunk, r.end_of_sequence) for r in stub.requests]
    stub.requests.clear()
    pulled = []

    def tiles():
        for item in zip(ids, chips, opts):
            pulled.append(item[0])
            yield item

    list(client.stream_predict(session_id="s1", tiles=tiles(), options={"task_type": "segmentation"},
                               chunk_bytes=3))
    one_iterable = [(dict(r.context), r.chunk, r.end_of_sequence) for r in stub.requests]
    return aligned, one_iterable, pulled


def test_one_tile_iterable_sends_exactly_what_three_aligned_iterables_send(stubbed):
    client, stub = stubbed
    chips = [np.arange(8, dtype=np.uint8), np.arange(8, 16, dtype=np.uint8), np.zeros(4, np.uint8)]
    ids = ["r0_c0", "r0_c1", "r1_c0"]
    opts = [{"sam_text_prompt": "field"}, None, {"return_confidence": "true"}]
    aligned, one_iterable, pulled = _stream_both_ways(client, stub, chips, ids, opts)
    assert one_iterable == aligned
    assert pulled == ids  # one pull per tile, in order
    firsts = [ctx for ctx, _, _ in one_iterable if any(k.startswith("opt.") for k in ctx)]
    assert [c["tile_id"] for c in firsts] == ids
    assert firsts[0]["opt.sam_text_prompt"] == "field" and "opt.sam_text_prompt" not in firsts[1]


def test_tiles_cannot_be_mixed_with_the_aligned_iterables(stubbed):
    client, _ = stubbed
    item = [("t", np.zeros(2), None)]
    for extra in ({"samples": [np.zeros(2)]}, {"tile_ids": ["t"]}, {"tile_options": [None]}):
        with pytest.raises(ValueError, match="not both"):
            client.stream_predict(model_id="m", tiles=item, **extra)
    with pytest.raises(ValueError, match="samples or tiles"):
        client.stream_predict(model_id="m")
    with pytest.raises(ValueError, match=r"tiles\[1\] must be"):
        list(client.stream_predict(model_id="m", tiles=[("a", np.zeros(2), None), ("b", np.zeros(2))]))


# ------------------------------------------ LoadModel effective metadata


class _LoadStub:
    def __init__(self):
        self.requests = []

    def LoadModel(self, request, timeout=None):
        self.requests.append(request)
        return inference_pb2.LoadModelResponse(
            success=True,
            model_id="srv-1",
            message="loaded",
            effective_metadata=[
                inference_pb2.MetadataEntry(key="model_cache_hit", value="false"),
                inference_pb2.MetadataEntry(key="capability.sam", value="sam3"),
            ],
        )


def test_load_model_result_returns_the_effective_metadata(monkeypatch):
    client = InferenceGrpcClient("localhost:0", max_attempts=1)
    stub = _LoadStub()
    monkeypatch.setattr(client, "_get_stub", lambda: stub)
    result = client.load_model_result("m", {"format": "huggingface", "source": "org/sam"})
    assert result.model_id == "srv-1" and result.message == "loaded"
    assert dict(result.effective_metadata) == {"model_cache_hit": "false", "capability.sam": "sam3"}
    assert client.load_model("m", {"format": "huggingface", "source": "org/sam"}) == "srv-1"

    spec = inference_pb2.ModelSpec(format=inference_pb2.HUGGINGFACE, source="org/sam", version="rev-2",
                                   model_id="from-spec")
    spec.metadata.add(key="k", value="v")
    client.load_model_result("", spec)
    client.load_model_result("override", spec)
    sent = [r.spec for r in stub.requests[2:]]
    assert [s.model_id for s in sent] == ["from-spec", "override"]
    assert all(s.version == "rev-2" and s.metadata[0].key == "k" for s in sent)
    assert spec.model_id == "from-spec"  # the caller's message is not modified
