# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model cache leases through the real server, client and gRPC transport.

Two cases over the real gRPC transport: with a tiny capacity, a model pinned by an
open session stays resident and usable while other models load and evict each
other, and becomes evictable after its last lease ends; and a prediction still
running when its session closes keeps its record alive until the call returns,
after which the record is disposed exactly once.

Also over gRPC: two clients that load an equal spec share one model ID, and one
client's UnloadModel or overwrite does not take the model from the other; an
UnloadModel while a session uses the model succeeds and the model is freed when the
session closes; a LoadModel whose client is gone when the load finishes keeps no
reference and frees no model; and a local model file rewritten in place is loaded again.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from nxmndr.server import server
from tst.support.grpc_harness import running_server, wait_until
from tst.support.stream_doubles import GatedModel, SegLogitsModel
from tst.unit.model_cache_test_utils import CountingRecord

pytestmark = pytest.mark.integration

CHIP = np.ones((4, 4, 3), np.float32)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


class _Loader:
    """Builds one dispose-counting record per load from a per-source model factory."""

    def __init__(self, factories):
        self.factories = factories
        self.records = defaultdict(list)
        self.lock = threading.Lock()

    def __call__(self, model_spec, key, metadata, device_plan):
        record = CountingRecord(
            model=self.factories[model_spec.model_path](),
            backend="onnx",
            metadata=dict(metadata),
            spec=model_spec,
        )
        with self.lock:
            self.records[model_spec.model_path].append(record)
        return record


def _load(client, source):
    return client.load_model("ignored", {"format": "onnx", "source": source, "task": "segmentation"})


def _open(client, session_id, model_id):
    resp = client.open_session(session_id=session_id, spec=inference_pb2.ModelSpec(model_id=model_id))
    assert resp.status == "ok", resp.error
    return resp


def _record_load_calls(monkeypatch):
    """Wrap LoadModel to record each call's gRPC context and the reply it returns."""

    contexts, replies = [], []
    real = server.InferenceService.LoadModel

    def recording(self, request, context):
        contexts.append(context)
        reply = real(self, request, context)
        replies.append(reply)
        return reply

    monkeypatch.setattr(server.InferenceService, "LoadModel", recording)
    return contexts, replies


def _predict(client, session_id):
    [resp] = list(client.stream_predict(session_id=session_id, samples=[CHIP], tile_ids=["t0"]))
    assert resp.metadata.get("error") is None, resp.metadata
    return resp


def test_session_pinned_model_survives_while_other_models_load_and_evict(tmp_path):
    sources = ("double://pinned", "double://m1", "double://m2", "double://m3")
    loader = _Loader({source: SegLogitsModel for source in sources})
    with running_server(tmp_path, loader=loader, capacity=2) as h, h.client() as c:
        pinned = _load(c, "double://pinned")
        _open(c, "s-pin", pinned)

        m1 = _load(c, "double://m1")  # fills the second slot
        m2 = _load(c, "double://m2")  # evicts m1, the only unpinned record
        assert h.manager.get(m1) is None and h.manager.get(m2) is not None
        m3 = _load(c, "double://m3")  # evicts m2
        assert h.manager.get(m2) is None and h.manager.get(m3) is not None
        assert h.manager.get(pinned) is not None and h.manager.pin_count(pinned) == 1

        assert _predict(c, "s-pin").metadata["model_id"] == pinned  # resident and usable
        _load(c, "double://m1")  # a reload evicts m3, never the pinned record
        assert h.manager.get(m3) is None and h.manager.get(pinned) is not None
        assert loader.records["double://pinned"][0].dispose_calls == 0
        assert [r.dispose_calls for r in loader.records["double://m1"]] == [1, 0]
        assert [r.dispose_calls for r in loader.records["double://m2"]] == [1]
        assert [r.dispose_calls for r in loader.records["double://m3"]] == [1]

        c.close_session("s-pin")  # the last lease on the pinned model ends
        assert wait_until(lambda: h.manager.pin_count(pinned) == 0)
        _load(c, "double://m2")  # the pinned model is now the least recently used
        assert h.manager.get(pinned) is None
        assert loader.records["double://pinned"][0].dispose_calls == 1
        assert len(loader.records["double://pinned"]) == 1  # loaded once throughout


@pytest.mark.parametrize("ending", ["close", "cancel"])
def test_running_prediction_keeps_its_record_until_the_call_returns(tmp_path, ending):
    gated = GatedModel()
    loader = _Loader({"double://gated": lambda: gated, "double://other": SegLogitsModel})
    with (
        running_server(tmp_path, loader=loader, capacity=1) as h,
        h.client() as c,
        h.client() as control,
    ):
        mid = _load(c, "double://gated")
        _open(c, "s-run", mid)
        [record] = loader.records["double://gated"]
        call = c.stream_predict(session_id="s-run", samples=[CHIP], tile_ids=["t0"])
        got, errors = [], []

        def consume():
            try:
                got.extend(call)
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert gated.entered.wait(10)  # the prediction is running

        if ending == "close":
            assert control.close_session("s-run").status == "closed"
        else:
            assert control.cancel_session("s-run").status == "cancelled"
        assert h.manager.pin_count(mid) == 1  # the execution lease outlives the session
        with pytest.raises(InferenceGrpcError) as err:
            _load(control, "double://other")  # the only slot is held by the running call
        assert err.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert h.manager.get(mid) is record and record.dispose_calls == 0

        gated.release.set()
        consumer.join(timeout=10)
        assert errors == [] and [r.metadata.get("error") for r in got] == [None]
        assert wait_until(lambda: h.manager.pin_count(mid) == 0)
        assert record.dispose_calls == 0  # released, not yet evicted

        _load(control, "double://other")  # now the record can be evicted
        assert h.manager.get(mid) is None
        assert record.dispose_calls == 1
    assert record.dispose_calls == 1  # server shutdown does not dispose it again


def test_unload_by_one_client_keeps_the_model_for_another(tmp_path):
    loader = _Loader({"double://shared": SegLogitsModel})
    with running_server(tmp_path, loader=loader) as h, h.client() as a, h.client() as b:
        mid = _load(a, "double://shared")
        assert _load(b, "double://shared") == mid  # a cache hit shares the model ID
        [record] = loader.records["double://shared"]
        released = a._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert released.success and "still loaded" in released.message
        result = b.predict(mid, CHIP, options={"task_type": "segmentation"})
        assert result.metadata.get("error") is None and record.dispose_calls == 0
        unloaded = b._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert unloaded.success and h.manager.get(mid) is None
        assert record.dispose_calls == 1


def test_unload_while_a_session_uses_the_model_frees_it_when_the_session_closes(tmp_path):
    loader = _Loader({"double://shared": SegLogitsModel})
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.ONNX, source="double://shared", task=inference_pb2.SEGMENTATION
    )
    with running_server(tmp_path, loader=loader) as h, h.client() as a, h.client() as b:
        mid = _load(a, "double://shared")
        _open(b, "s-b", mid)
        [record] = loader.records["double://shared"]
        reply = a._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert reply.success and "freed when its sessions and calls end" in reply.message
        _predict(b, "s-b")  # the session still runs on the model
        assert record.dispose_calls == 0
        assert b.close_session("s-b").status == "closed"
        assert h.manager.get(mid) is None and record.dispose_calls == 1
        fresh = a._get_stub().LoadModel(inference_pb2.LoadModelRequest(spec=spec, overwrite=True))
        assert fresh.success and fresh.model_id != mid


def test_a_load_whose_client_is_gone_keeps_no_reference(tmp_path, monkeypatch):
    gate = threading.Event()
    loader = _Loader({"double://slow": SegLogitsModel})

    def gated_loader(*args):
        gate.wait(30)
        return loader(*args)

    contexts, replies = _record_load_calls(monkeypatch)
    with running_server(tmp_path, loader=gated_loader) as h, h.client() as c:
        with h.client(timeout=0.5) as short, pytest.raises(InferenceGrpcError) as err:
            _load(short, "double://slow")
        assert err.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
        assert wait_until(lambda: contexts and not contexts[0].is_active())
        gate.set()  # the server finishes the load its client gave up on
        assert wait_until(lambda: replies)
        assert replies[0].message == "the client is gone"
        [record] = loader.records["double://slow"]
        [gone] = h.manager.list_models()  # cached with no reference, like a closed session's
        assert record.dispose_calls == 0
        mid = _load(c, "double://slow")  # the client's retry holds the only reference
        assert mid == gone
        reply = c._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert reply.success and reply.message == f"unloaded {mid}"
        assert record.dispose_calls == 1


@pytest.mark.parametrize("session_open", [False, True], ids=["warm", "pinned"])
def test_a_gone_load_that_hits_the_cache_does_not_free_the_model(
    tmp_path, monkeypatch, session_open
):
    loader = _Loader({"double://warm": SegLogitsModel})
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.ONNX, source="double://warm", task=inference_pb2.SEGMENTATION
    )
    contexts, replies = _record_load_calls(monkeypatch)
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        assert c.open_session(session_id="s1", spec=spec).status == "ok"
        if not session_open:  # warm: resident with no session and no LoadModel reference
            assert c.close_session("s1").status == "closed"
        [record] = loader.records["double://warm"]
        gate = threading.Event()
        prepare = h.service._prepare_load

        def gated_prepare(*args, **kwargs):
            gate.wait(30)
            return prepare(*args, **kwargs)

        monkeypatch.setattr(h.service, "_prepare_load", gated_prepare)
        with h.client(timeout=0.5) as short, pytest.raises(InferenceGrpcError) as err:
            _load(short, "double://warm")
        assert err.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
        assert wait_until(lambda: contexts and not contexts[0].is_active())
        gate.set()  # the load hits the cache after its client is gone
        assert wait_until(lambda: replies)
        assert replies[0].message == "the client is gone"
        if session_open:
            assert c.close_session("s1").status == "closed"
        [mid] = h.manager.list_models()
        assert record.dispose_calls == 0
        assert _load(c, "double://warm") == mid  # one reference: the gone load left none
        reply = c._get_stub().UnloadModel(inference_pb2.UnloadModelRequest(model_id=mid))
        assert reply.success and reply.message == f"unloaded {mid}"
        assert record.dispose_calls == 1


def test_overwrite_of_a_model_another_client_holds_is_refused(tmp_path):
    loader = _Loader({"double://shared": SegLogitsModel})
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.ONNX, source="double://shared", task=inference_pb2.SEGMENTATION
    )
    with running_server(tmp_path, loader=loader) as h, h.client() as a, h.client() as b:
        mid = _load(b, "double://shared")
        with pytest.raises(grpc.RpcError) as err:
            a._get_stub().LoadModel(inference_pb2.LoadModelRequest(spec=spec, overwrite=True))
        assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        result = b.predict(mid, CHIP, options={"task_type": "segmentation"})
        assert result.metadata.get("error") is None and h.manager.get(mid) is not None
        assert [r.dispose_calls for r in loader.records["double://shared"]] == [0]


def test_rewritten_local_file_is_loaded_again(tmp_path):
    import os

    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"v1")
    os.utime(weights, ns=(10**18, 10**18))
    source = str(weights)
    loader = _Loader({source: SegLogitsModel})
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        first = _load(c, source)
        assert _load(c, source) == first  # unchanged file: cache hit
        weights.write_bytes(b"v2")  # retrained into the same path
        os.utime(weights, ns=(10**18, 10**18 + 1))
        second = _load(c, source)
        assert second != first and len(loader.records[source]) == 2
        weights.write_bytes(b"v3")  # the plugin's path: OpenSession with the spec
        os.utime(weights, ns=(10**18, 10**18 + 2))
        spec = inference_pb2.ModelSpec(
            format=inference_pb2.ONNX, source=source, task=inference_pb2.SEGMENTATION
        )
        opened = c.open_session(session_id="s-retrained", spec=spec)
        assert opened.status == "ok" and opened.model_cache_hit is False
        assert len(loader.records[source]) == 3
        assert h.manager.get(second) is not None  # its holder keeps the earlier copy
