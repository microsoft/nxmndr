# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model cache keys, LRU, capacity, sessions, execution leases, TTL and shutdown.

Deterministic: an injected clock and loader, no timing sleeps. Concurrency cases
are in test_model_cache_concurrency.py.
"""

from __future__ import annotations

import gc
import logging
import threading
import weakref

import pytest

from nxmndr.inference import inference_pb2
from nxmndr.server.managers import (
    SESSION_CLOSE_REASONS,
    CacheExhaustedError,
    CacheShutdownError,
    ModelCacheError,
    ModelCacheKey,
    ModelInUseError,
    ModelLoadError,
    ModelManager,
    SessionClosedError,
    SessionConflictError,
    UnknownModelError,
    UnknownSessionError,
    cache_key_from_spec,
    local_source_stamp,
)
from tst.unit.model_cache_test_utils import WAIT_S, key, make_manager, spec, wait_until

TOKEN = "hf_Sup3rSecretTokenValue"


def _proto(**fields):
    base = dict(format=inference_pb2.ONNX, source="models/a.onnx", task=inference_pb2.SEGMENTATION)
    base.update(fields)
    return inference_pb2.ModelSpec(**base)


def _no_violations(loader):
    found = list(loader.violations)
    for record in loader.records:
        found.extend(record.violations)
    return found


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def test_equivalent_specs_have_equal_keys():
    a = _proto(name="first", model_id="client-a", checksum="ABCDEF")
    b = _proto(
        name="second",
        model_id="client-b",
        source="  models/a.onnx ",
        checksum="abcdef",
        preprocessing=[inference_pb2.TransformSpec(name="resize", params={"size": "224"})],
        metadata=[inference_pb2.MetadataEntry(key="note", value="x")],
        artifact_mime_type="application/onnx",
    )
    assert cache_key_from_spec(a) == cache_key_from_spec(b)
    assert cache_key_from_spec(a) == ModelCacheKey(
        format="onnx", source="models/a.onnx", checksum="abcdef", task="segmentation"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"format": inference_pb2.HUGGINGFACE},
        {"source": "models/b.onnx"},
        {"version": "v2"},
        {"checksum": "0123"},
        {"model_class": "UNet"},
        {"task": inference_pb2.CLASSIFICATION},
        {"lazy_load": True},
        {"token": TOKEN},
    ],
)
def test_load_affecting_fields_change_the_key(change):
    assert cache_key_from_spec(_proto()) != cache_key_from_spec(_proto(**change))


def test_torchhub_entry_point_name_is_part_of_the_key():
    hub_a = _proto(format=inference_pb2.TORCHHUB, source="org/hub", name="resnet18")
    hub_b = _proto(format=inference_pb2.TORCHHUB, source="org/hub", name="resnet50")
    assert cache_key_from_spec(hub_a) != cache_key_from_spec(hub_b)
    assert ("name", "resnet18") in cache_key_from_spec(hub_a).load_options


def test_key_holds_no_token_but_scopes_by_token():
    with_token = cache_key_from_spec(_proto(token=TOKEN))
    same_token = cache_key_from_spec(_proto(token=TOKEN))
    other_token = cache_key_from_spec(_proto(token=TOKEN + "x"))
    public = cache_key_from_spec(_proto())
    assert TOKEN not in repr(with_token)
    assert all(TOKEN not in str(value) for value in vars(with_token).values())
    assert with_token.auth_scope.startswith("hmac-sha256:")
    assert with_token == same_token
    assert with_token.auth_scope != other_token.auth_scope
    assert public.auth_scope == ""


def test_uploaded_artifact_is_keyed_by_content():
    import hashlib

    data = b"onnx-bytes"
    digest = hashlib.sha256(data).hexdigest()
    by_bytes = cache_key_from_spec(_proto(artifact=data, source="ignored/path.onnx"))
    assert by_bytes.source == f"sha256:{digest}"
    assert cache_key_from_spec(_proto(), artifact_sha256=digest.upper()) == by_bytes
    with pytest.raises(ValueError):
        cache_key_from_spec(_proto(), artifact_sha256="not-a-digest")


def test_source_stamp_is_part_of_the_key_except_for_uploads():
    assert cache_key_from_spec(_proto(), source_stamp="file:1:2") != cache_key_from_spec(_proto())
    upload = _proto(artifact=b"onnx-bytes")
    assert cache_key_from_spec(upload, source_stamp="file:1:2") == cache_key_from_spec(upload)


def test_local_source_stamp_follows_file_and_directory_contents(tmp_path):
    import os

    assert local_source_stamp("") == local_source_stamp(str(tmp_path / "missing")) == ""
    assert local_source_stamp("org/repo") == ""  # a hub repository ID, not a local path
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"aa")
    os.utime(weights, ns=(10**18, 10**18))
    first = local_source_stamp(str(weights))
    assert first and local_source_stamp(f" {weights} ") == first
    weights.write_bytes(b"bb")  # retrained: same size, new modification time
    os.utime(weights, ns=(10**18, 10**18 + 1))
    assert local_source_stamp(str(weights)) not in ("", first)
    snapshot = tmp_path / "snapshot"  # a Hugging Face snapshot directory
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    safetensors = snapshot / "model.safetensors"
    safetensors.write_bytes(b"x")
    os.utime(safetensors, ns=(10**18, 10**18))
    before = local_source_stamp(str(snapshot))
    assert before and local_source_stamp(str(snapshot)) == before
    os.utime(safetensors, ns=(10**18, 10**18 + 1))
    assert local_source_stamp(str(snapshot)) not in ("", before)


def test_key_load_options_are_sorted_and_unique():
    assert ModelCacheKey("onnx", "a", load_options=(("b", "2"), ("a", "1"))).load_options == (
        ("a", "1"),
        ("b", "2"),
    )
    assert ModelCacheKey("onnx", "a", load_options={"b": "2", "a": "1"}) == ModelCacheKey(
        "onnx", "a", load_options=(("a", "1"), ("b", "2"))
    )
    with pytest.raises(ValueError):
        ModelCacheKey("onnx", "a", load_options=(("a", "1"), ("a", "2")))


# ---------------------------------------------------------------------------
# LRU, reuse and capacity
# ---------------------------------------------------------------------------


def test_lru_evicts_least_recently_used_unpinned_record():
    manager, loader, _ = make_manager(capacity=3)
    ids = {name: manager.load(spec(), key=key(name)).model_id for name in "ABC"}
    assert manager.load(spec(), key=key("A")).cache_hit  # A becomes most recent
    d = manager.load(spec(), key=key("D"))
    assert not d.cache_hit
    assert manager.get(ids["B"]) is None
    assert list(manager.list_models()) == [ids["C"], ids["A"], d.model_id]
    evicted = [r for r in loader.records if r.model_id == ids["B"]][0]
    assert evicted.dispose_calls == 1 and evicted.disposed
    assert len(loader.calls) == 4
    assert _no_violations(loader) == []


def test_recency_updates_on_open_session_and_execution_but_not_on_get():
    manager, loader, _ = make_manager(capacity=3)
    a, b, c = (manager.load(spec(), key=key(n)).model_id for n in "ABC")
    manager.get(a)  # peek: no recency change
    manager.acquire_execution(model_id=b).release()  # B most recent
    manager.open_session("s-c", model_id=c)
    manager.close_session("s-c", reason="closed")  # C most recent
    manager.load(spec(), key=key("D"))
    assert manager.get(a) is None
    assert list(manager.list_models())[:2] == [b, c]


def test_equivalent_spec_reuses_the_loaded_record():
    manager, loader, _ = make_manager()
    first = manager.load(spec(), key=key("A"))
    again = manager.load(spec(), key=key("A"))
    lease = manager.open_session("s1", model_spec=spec(), key=key("A"))
    assert (again.model_id, again.cache_hit) == (first.model_id, True)
    assert (lease.model_id, lease.cache_hit, lease.reused) == (first.model_id, True, False)
    assert len(loader.calls) == 1
    other = manager.load(spec(), key=key("A", revision="v2"))
    assert other.model_id != first.model_id and len(loader.calls) == 2


def test_capacity_is_configurable_and_defaults_to_ten():
    loader_calls = []

    def loader(model_spec, cache_key, metadata, device_plan):
        loader_calls.append(cache_key)
        from nxmndr.server.managers import ModelRecord

        return ModelRecord(object(), "fake", metadata, model_spec)

    default = ModelManager(None, loader=loader)
    assert default.stats().capacity == 10
    ids = [default.load(spec(), key=key(str(i))).model_id for i in range(10)]
    assert default.stats().resident == 10
    default.load(spec(), key=key("10"))
    assert default.stats().resident == 10 and default.get(ids[0]) is None

    for capacity in (1, 4):
        manager, _, _ = make_manager(capacity=capacity)
        for i in range(capacity + 2):
            manager.load(spec(), key=key(str(i)))
            stats = manager.stats()
            assert stats.resident + stats.reserved <= capacity
        assert manager.stats().resident == capacity
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ModelManager(None, capacity=bad, loader=loader)


def test_all_pinned_capacity_raises_retryable_error_without_side_effects():
    manager, loader, _ = make_manager(capacity=2)
    a = manager.open_session("s-a", model_spec=spec(), key=key("A")).model_id
    b = manager.load(spec(), key=key("B")).model_id
    lease = manager.acquire_execution(model_id=b)
    with pytest.raises(CacheExhaustedError) as info:
        manager.load(spec(), key=key("C"))
    assert info.value.retryable is True and isinstance(info.value, ModelCacheError)
    with pytest.raises(CacheExhaustedError):
        manager.open_session("s-c", model_spec=spec(), key=key("C"))
    assert len(loader.calls) == 2
    assert all(r.dispose_calls == 0 for r in loader.records)
    assert manager.stats().resident == 2 and manager.stats().reserved == 0
    assert manager.session_state("s-c") == "unknown"
    lease.release()
    c = manager.load(spec(), key=key("C")).model_id  # B is now the only unpinned record
    assert manager.get(b) is None and manager.get(a) is not None and manager.get(c) is not None


# ---------------------------------------------------------------------------
# Unload and overwrite
# ---------------------------------------------------------------------------


def test_unload_of_a_pinned_record_disposes_it_when_the_last_pin_is_released():
    manager, loader, _ = make_manager()
    assert manager.unload("no-such-model") == "unknown"
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    lease = manager.acquire_execution(model_id=a)
    assert manager.unload(a) == "pending"  # a session-only record: no reference to release
    manager.close_session("s1", reason="closed")
    assert manager.get(a) is not None and loader.records[0].dispose_calls == 0  # the call
    lease.release()
    assert manager.get(a) is None and loader.records[0].dispose_calls == 1
    assert manager.unload(a) == "unknown" and loader.records[0].dispose_calls == 1
    b = manager.load(spec(), key=key("B")).model_id
    assert manager.unload(b) == "unloaded" and loader.records[1].dispose_calls == 1
    assert _no_violations(loader) == []


def test_unload_releases_one_load_reference_of_a_shared_model():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id  # client A
    assert manager.load(spec(), key=key("A")).model_id == a  # client B: cache hit
    assert manager.unload(a) == "released"  # A is done
    assert manager.get(a) is not None and loader.records[0].dispose_calls == 0
    manager.acquire_execution(model_id=a).release()  # B still predicts
    assert manager.unload(a) == "unloaded"  # B is done: the last reference
    assert manager.get(a) is None and loader.records[0].dispose_calls == 1
    assert manager.unload(a) == "unknown"
    assert _no_violations(loader) == []


def test_last_load_reference_of_a_pinned_record_is_released_and_the_record_freed_later():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    manager.load(spec(), key=key("A"))
    manager.open_session("s1", model_id=a)
    assert manager.unload(a) == "released"  # another load reference remains
    assert manager.unload(a) == "pending"  # the last one, while the session pins the record
    with pytest.raises(ModelInUseError):
        manager.load(spec(), key=key("A"), overwrite=True)  # the session still pins it
    assert manager.get(a) is not None and loader.records[0].dispose_calls == 0
    manager.close_session("s1", reason="closed")
    assert manager.get(a) is None and loader.records[0].dispose_calls == 1


def test_overwrite_works_once_the_session_of_an_unloaded_record_closes():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id  # client A: LoadModel
    manager.open_session("s-b", model_spec=spec(), key=key("A"))  # client B: by spec
    assert manager.unload(a) == "pending"  # A is done while B's session runs
    manager.close_session("s-b", reason="closed")  # B is done
    assert manager.get(a) is None and loader.records[0].dispose_calls == 1
    fresh = manager.load(spec(), key=key("A"), overwrite=True)
    assert fresh.model_id != a and fresh.cache_hit is False and len(loader.calls) == 2
    assert _no_violations(loader) == []


def test_a_load_reference_added_after_the_last_unload_keeps_the_record():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    lease = manager.acquire_execution(model_id=a)
    assert manager.unload(a) == "pending"
    assert manager.load(spec(), key=key("A")).model_id == a  # a new caller: cache hit
    lease.release()
    assert manager.get(a) is not None and loader.records[0].dispose_calls == 0
    assert manager.unload(a) == "unloaded" and loader.records[0].dispose_calls == 1


def test_session_expiry_frees_an_unloaded_record():
    manager, loader, clock = make_manager(ttl=10)
    a = manager.load(spec(), key=key("A")).model_id
    manager.open_session("s1", model_id=a)
    assert manager.unload(a) == "pending"
    clock.advance(11)
    assert manager.expire_sessions() == ["s1"]
    assert manager.get(a) is None and loader.records[0].dispose_calls == 1


def test_overwrite_refuses_a_record_that_a_load_reference_holds():
    manager, loader, _ = make_manager()
    held = manager.load(spec(), key=key("A")).model_id  # client B holds it
    with pytest.raises(ModelInUseError):
        manager.load(spec(), key=key("A"), overwrite=True)  # client A
    assert manager.get(held) is not None and loader.records[0].dispose_calls == 0
    manager.acquire_execution(model_id=held).release()  # B still predicts
    assert manager.unload(held) == "unloaded" and loader.records[0].dispose_calls == 1
    fresh = manager.load(spec(), key=key("A"), overwrite=True)  # not resident: plain load
    assert fresh.model_id != held and fresh.cache_hit is False and len(loader.calls) == 2
    assert _no_violations(loader) == []


def test_overwrite_refuses_a_pinned_record():
    manager, loader, _ = make_manager()
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    with pytest.raises(ModelInUseError):
        manager.load(spec(), key=key("A"), overwrite=True)
    assert len(loader.calls) == 1 and manager.get(a) is not None


def test_overwrite_reloads_a_record_that_nothing_holds():
    manager, loader, _ = make_manager()
    old = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.close_session("s1", reason="closed")  # no load reference, no pin
    new = manager.load(spec(), key=key("A"), overwrite=True)
    assert new.model_id != old and new.cache_hit is False
    assert manager.get(old) is None and loader.records[0].dispose_calls == 1
    fresh = manager.load(spec(), key=key("B"), overwrite=True)  # not resident: plain load
    assert fresh.cache_hit is False and len(loader.calls) == 3
    assert _no_violations(loader) == []


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_two_sessions_pin_one_model_independently():
    manager, _, _ = make_manager(capacity=1)
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.open_session("s2", model_id=a)
    assert manager.pin_count(a) == 2 and manager.stats().pinned == 1
    assert manager.close_session("s1", reason="closed") is True
    assert manager.pin_count(a) == 1
    with pytest.raises(CacheExhaustedError):
        manager.load(spec(), key=key("B"))
    assert manager.close_session("s2", reason="closed") is True
    assert manager.pin_count(a) == 0
    manager.load(spec(), key=key("B"))
    assert manager.get(a) is None


def test_retried_open_session_is_idempotent():
    manager, loader, _ = make_manager()
    first = manager.open_session("s1", model_spec=spec(), key=key("A"))
    by_key = manager.open_session("s1", model_spec=spec(), key=key("A"))
    by_id = manager.open_session("s1", model_id=first.model_id)
    assert (first.reused, by_key.reused, by_id.reused) == (False, True, True)
    assert by_key.model_id == by_id.model_id == first.model_id
    assert manager.pin_count(first.model_id) == 1 and len(loader.calls) == 1
    assert manager.close_session("s1", reason="closed") is True
    assert manager.pin_count(first.model_id) == 0


def test_session_id_bound_to_another_model_conflicts():
    manager, loader, _ = make_manager()
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    b = manager.load(spec(), key=key("B")).model_id
    with pytest.raises(SessionConflictError):
        manager.open_session("s1", model_id=b)
    with pytest.raises(SessionConflictError):
        manager.open_session("s1", model_spec=spec(), key=key("C"))
    assert len(loader.calls) == 2  # the conflicting key was never loaded
    assert manager.pin_count(a) == 1 and manager.pin_count(b) == 0


def test_open_session_argument_checks():
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    with pytest.raises(ValueError):
        manager.open_session("s1")
    with pytest.raises(ValueError):
        manager.open_session("s1", model_id=a, model_spec=spec(), key=key("A"))
    with pytest.raises(ValueError):
        manager.open_session("s1", key=key("A"))
    with pytest.raises(ValueError):
        manager.open_session("", model_id=a)
    with pytest.raises(TypeError):
        manager.load(spec(), key=("onnx", "A"))


def test_failed_open_leaves_no_session_and_no_tombstone():
    manager, loader, _ = make_manager()
    with pytest.raises(UnknownModelError):
        manager.open_session("s1", model_id="never-loaded")
    assert manager.session_state("s1") == "unknown"
    loader.fail_with = OSError("disk unavailable")
    with pytest.raises(ModelLoadError):
        manager.open_session("s1", model_spec=spec(), key=key("A"))
    assert manager.session_state("s1") == "unknown"
    assert manager.stats().resident == 0 and manager.stats().reserved == 0
    loader.fail_with = None
    lease = manager.open_session("s1", model_spec=spec(), key=key("A"))
    assert lease.reused is False and manager.pin_count(lease.model_id) == 1


@pytest.mark.parametrize("reason", SESSION_CLOSE_REASONS)
def test_close_releases_the_session_pin_exactly_once(reason):
    manager, _, _ = make_manager()
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.open_session("s2", model_id=a)
    assert manager.close_session("s1", reason=reason) is True
    assert manager.pin_count(a) == 1 and manager.session_state("s1") == reason
    for again in SESSION_CLOSE_REASONS:
        assert manager.close_session("s1", reason=again) is False
    assert manager.pin_count(a) == 1  # s2's pin untouched by the repeats
    with pytest.raises(SessionClosedError):
        manager.open_session("s1", model_id=a)
    with pytest.raises(SessionClosedError):
        manager.acquire_execution(session_id="s1")
    with pytest.raises(SessionClosedError):
        manager.touch_session("s1")


def test_close_unknown_session_returns_false_and_checks_reason():
    manager, _, _ = make_manager()
    assert manager.close_session("nobody", reason="closed") is False
    assert manager.session_state("nobody") == "unknown"
    with pytest.raises(ValueError):
        manager.close_session("nobody", reason="because")


def test_stream_without_open_session_is_rejected_not_fabricated():
    """StreamPredict naming a never-opened session gets no lease; none is fabricated."""

    manager, _, _ = make_manager()
    manager.load(spec(), key=key("A"))
    with pytest.raises(UnknownSessionError) as info:
        manager.acquire_execution(session_id="fabricated")
    assert isinstance(info.value, KeyError)
    with pytest.raises(UnknownSessionError):
        manager.touch_session("fabricated")
    assert manager.session_state("fabricated") == "unknown"
    assert manager.stats().pinned == 0


def test_acquire_execution_argument_and_model_checks():
    manager, _, _ = make_manager(capacity=1)
    a = manager.load(spec(), key=key("A")).model_id
    for kwargs in ({}, {"model_id": a, "session_id": "s1"}):
        with pytest.raises(ValueError):
            manager.acquire_execution(**kwargs)
    with pytest.raises(UnknownModelError):
        manager.acquire_execution(model_id="never-loaded")
    manager.load(spec(), key=key("B"))  # evicts A
    with pytest.raises(UnknownModelError):
        manager.acquire_execution(model_id=a)


def test_running_call_keeps_its_lease_after_the_session_closes():
    manager, loader, _ = make_manager(capacity=1)
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    lease = manager.acquire_execution(session_id="s1")
    record = loader.records[0]
    assert manager.close_session("s1", reason="cancelled") is True
    assert manager.pin_count(a) == 1
    assert lease.model_for_device("cpu:0") == "model:A"
    with pytest.raises(CacheExhaustedError):
        manager.load(spec(), key=key("B"))
    with pytest.raises(SessionClosedError):
        manager.acquire_execution(session_id="s1")
    assert record.dispose_calls == 0
    lease.release()
    lease.release()
    assert manager.pin_count(a) == 0
    manager.load(spec(), key=key("B"))
    assert manager.get(a) is None and record.dispose_calls == 1


def test_execution_lease_interface():
    manager, _, _ = make_manager()
    a = manager.load(
        spec(), key=key("A"), device_plan=[{"id": "cuda:0"}, {"id": "cuda:1"}]
    ).model_id
    with manager.acquire_execution(model_id=a) as lease:
        assert (lease.model_id, lease.session_id) == (a, "")
        assert lease.record is manager.get(a)
        assert lease.model_for_device("cuda:1") == "A@cuda:1"
        assert lease.model_for_device("cpu:0") == "model:A"
        assert manager.pin_count(a) == 1
    assert lease.released and manager.pin_count(a) == 0
    with pytest.raises(ModelCacheError):
        lease.model_for_device("cuda:0")
    with pytest.raises(ModelCacheError):
        lease.resource("x", object)


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def test_ttl_expires_idle_sessions_measured_from_last_activity():
    manager, _, clock = make_manager(ttl=100)
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.open_session("s2", model_id=a)
    clock.advance(60)
    manager.touch_session("s2")
    clock.advance(40)
    assert manager.expire_sessions() == []  # s1 idle exactly the TTL: not expired
    clock.advance(1)
    assert manager.expire_sessions() == ["s1"]
    assert manager.session_state("s1") == "expired" and manager.session_state("s2") == "open"
    assert manager.pin_count(a) == 1
    assert manager.expire_sessions() == []  # already expired: released once


def test_ttl_never_expires_a_session_with_a_running_execution():
    manager, _, clock = make_manager(ttl=100)
    manager.open_session("s1", model_spec=spec(), key=key("A"))
    lease = manager.acquire_execution(session_id="s1")
    clock.advance(10_000)
    assert manager.expire_sessions() == []
    lease.release()  # release counts as activity
    clock.advance(100)
    assert manager.expire_sessions() == []
    clock.advance(1)
    assert manager.expire_sessions() == ["s1"]


def test_tombstones_live_for_the_ttl():
    manager, _, clock = make_manager(ttl=100)
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.close_session("s1", reason="closed")
    clock.advance(100)
    assert manager.session_state("s1") == "closed"
    clock.advance(1)
    manager.expire_sessions()
    assert manager.session_state("s1") == "unknown"
    assert manager.open_session("s1", model_id=a).reused is False


# ---------------------------------------------------------------------------
# Aux resources
# ---------------------------------------------------------------------------


class _Resource:
    pass


def test_aux_resource_is_created_once_per_record_and_disposed_with_it():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    calls = []

    def factory():
        calls.append(1)
        return _Resource()

    with (
        manager.acquire_execution(model_id=a) as first,
        manager.acquire_execution(model_id=a) as second,
    ):
        tracker = first.resource("sam3_tracker@cuda:0", factory)
        assert second.resource("sam3_tracker@cuda:0", factory) is tracker
        assert first.resource("sam3@cuda:0", factory) is not tracker
    assert len(calls) == 2
    assert manager.stats().resident == 1  # aux resources use no cache slot
    ref = weakref.ref(tracker)
    del tracker
    assert manager.unload(a) == "unloaded"
    gc.collect()
    assert ref() is None


def test_aux_resource_failure_is_not_cached():
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("download failed")
        return _Resource()

    with manager.acquire_execution(model_id=a) as lease:
        with pytest.raises(RuntimeError):
            lease.resource("sam3@cpu", flaky)
        assert isinstance(lease.resource("sam3@cpu", flaky), _Resource)
    assert len(attempts) == 2


class _Disposable:
    """Records each dispose()/close() call and the thread it ran in."""

    def __init__(self, calls, name, *, has_dispose=True, has_close=True, fail=False):
        self._calls, self._name, self._fail = calls, name, fail
        if has_dispose:
            self.dispose = lambda: self._hook("dispose")
        if has_close:
            self.close = lambda: self._hook("close")

    def _hook(self, hook):
        self._calls.append((self._name, hook, threading.get_ident()))
        if self._fail:
            raise RuntimeError(f"{self._name} {hook} failed")


def test_aux_resources_are_disposed_once_in_the_disposing_thread(caplog):
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    calls = []
    made = {
        "both@cpu": _Disposable(calls, "both"),  # dispose() wins over close()
        "close-only@cpu": _Disposable(calls, "close-only", has_dispose=False),
        "failing@cpu": _Disposable(calls, "failing", fail=True),
        "plain@cpu": _Resource(),  # neither hook: only the reference is dropped
    }
    with manager.acquire_execution(model_id=a) as lease:
        for name, resource in made.items():
            assert lease.resource(name, lambda r=resource: r) is resource
    assert calls == []  # nothing is disposed while the record lives

    caplog.set_level(logging.WARNING)
    assert manager.unload(a) == "unloaded"  # the failing resource does not raise here
    me = threading.get_ident()
    assert sorted(calls) == sorted(
        [("both", "dispose", me), ("close-only", "close", me), ("failing", "dispose", me)]
    )
    assert "'failing@cpu' failed" in caplog.text  # logged instead
    manager.shutdown()
    assert len(calls) == 3  # never disposed a second time


def test_aux_resource_finished_after_disposal_is_disposed_by_its_creator():
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    calls = []
    with manager.acquire_execution(model_id=a) as lease:
        record = lease.record

        def factory():
            record.dispose()  # the record is disposed while the factory runs
            return _Disposable(calls, "late")

        with pytest.raises(ModelCacheError):
            lease.resource("late@cpu", factory)
    assert [(name, hook) for name, hook, _ in calls] == [("late", "dispose")]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_token_stays_out_of_metadata_repr_and_logs(caplog):
    caplog.set_level(logging.DEBUG)
    manager, loader, _ = make_manager()
    metadata = {"token": TOKEN, "hf_token": TOKEN, "name": "sam", "tokenizer": "keep"}
    a = manager.load(spec(token=TOKEN), key=key("A"), metadata=metadata).model_id
    record = manager.get(a)
    assert record.auth_token == TOKEN
    assert record.metadata == {"name": "sam", "tokenizer": "keep", "model_id": a}
    assert all(TOKEN not in str(v) for v in loader.metadata_seen[0].values())
    assert TOKEN not in repr(record)
    assert f"loading model {a}" in caplog.text  # the manager did log
    assert TOKEN not in caplog.text
    manager.unload(a)
    assert record.auth_token == ""


def test_load_error_message_redacts_the_token():
    manager, loader, _ = make_manager()
    original = RuntimeError(f"401 Unauthorized for token {TOKEN}")
    loader.fail_with = original
    with pytest.raises(ModelLoadError) as info:
        manager.load(spec(token=TOKEN), key=key("A"))
    assert TOKEN not in str(info.value)
    assert info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_closes_sessions_and_disposes_every_record_once():
    manager, loader, _ = make_manager()
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    manager.open_session("s2", model_spec=spec(), key=key("B"))
    manager.load(spec(), key=key("C"))
    manager.shutdown()
    assert [r.dispose_calls for r in loader.records] == [1, 1, 1]
    assert manager.session_state("s1") == manager.session_state("s2") == "shutdown"
    assert manager.stats().resident == 0 and manager.get(a) is None
    with pytest.raises(CacheShutdownError):
        manager.load(spec(), key=key("D"))
    with pytest.raises(CacheShutdownError):
        manager.open_session("s3", model_spec=spec(), key=key("A"))
    with pytest.raises(CacheShutdownError):
        manager.acquire_execution(model_id=a)
    assert manager.close_session("s1", reason="closed") is False
    assert manager.unload(a) == "unknown"
    manager.shutdown()  # repeated cleanup is safe
    assert [r.dispose_calls for r in loader.records] == [1, 1, 1]
    assert _no_violations(loader) == []


def test_shutdown_drains_execution_leases_before_disposal():
    manager, loader, _ = make_manager()
    manager.open_session("s1", model_spec=spec(), key=key("A"))
    lease = manager.acquire_execution(session_id="s1")
    record = loader.records[0]
    stopper = threading.Thread(target=manager.shutdown)
    stopper.start()
    wait_until(lambda: manager.session_state("s1") == "shutdown", "shutdown to start draining")
    assert stopper.is_alive() and record.dispose_calls == 0
    with pytest.raises(CacheShutdownError):
        manager.acquire_execution(model_id=lease.model_id)
    assert lease.model_for_device("cpu:0") == "model:A"  # still usable while draining
    lease.release()
    stopper.join(WAIT_S)
    assert not stopper.is_alive() and record.dispose_calls == 1


def test_shutdown_timeout_disposes_leased_records_with_a_warning(caplog):
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    lease = manager.acquire_execution(model_id=a)
    with caplog.at_level(logging.WARNING):
        manager.shutdown(drain_timeout_s=0)
    assert loader.records[0].dispose_calls == 1
    assert "live execution lease" in caplog.text
    lease.release()  # a late release after shutdown is harmless
    assert loader.records[0].dispose_calls == 1
