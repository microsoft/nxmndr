# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Concurrency of the model cache: coalescing, reservations, lock discipline.

Threads are ordered with events, barriers and waits on observable cache state.
No assertion depends on elapsed time.
"""

from __future__ import annotations

import random
import threading

import pytest

from nxmndr.server.managers import (
    SESSION_CLOSE_REASONS,
    CacheExhaustedError,
    CacheShutdownError,
    ModelLoadError,
    SessionClosedError,
    SessionConflictError,
)
from tst.unit.model_cache_test_utils import (
    WAIT_S,
    key,
    lock_is_free,
    make_manager,
    pending_waiters,
    spec,
    wait_until,
)


class _Runner:
    """Run callables on threads and collect results or exceptions."""

    def __init__(self):
        self.results = {}
        self.threads = []

    def start(self, name, fn):
        def body():
            try:
                self.results[name] = ("ok", fn())
            except BaseException as exc:  # recorded for the assertions
                self.results[name] = ("error", exc)

        thread = threading.Thread(target=body, name=name)
        self.threads.append(thread)
        thread.start()
        return thread

    def join(self):
        for thread in self.threads:
            thread.join(WAIT_S)
            assert not thread.is_alive(), f"{thread.name} did not finish"


def _violations(loader):
    found = list(loader.violations)
    for record in loader.records:
        found.extend(record.violations)
    return found


def test_concurrent_loads_of_one_key_coalesce_into_one_loader_call():
    manager, loader, _ = make_manager()
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("first", lambda: manager.load(spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    for i in range(7):
        runner.start(f"waiter{i}", lambda: manager.load(spec(), key=key("A")))
    wait_until(lambda: pending_waiters(manager, key("A")) == 7, "7 coalesced waiters")
    assert manager.stats().reserved == 1 and manager.stats().resident == 0
    loader.gate.set()
    runner.join()
    outcomes = list(runner.results.values())
    assert all(kind == "ok" for kind, _ in outcomes)
    assert len({result.model_id for _, result in outcomes}) == 1
    assert all(result.cache_hit is False for _, result in outcomes)
    assert len(loader.calls) == 1
    assert manager.stats().reserved == 0 and manager.stats().resident == 1
    assert _violations(loader) == []


def test_coalesced_loads_each_hold_a_load_reference():
    manager, loader, _ = make_manager()
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("first", lambda: manager.load(spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    for i in range(2):
        runner.start(f"waiter{i}", lambda: manager.load(spec(), key=key("A")))
    wait_until(lambda: pending_waiters(manager, key("A")) == 2, "2 coalesced waiters")
    loader.gate.set()
    runner.join()
    [model_id] = {result.model_id for _, result in runner.results.values()}
    assert manager.unload(model_id) == manager.unload(model_id) == "released"
    assert manager.get(model_id) is not None and loader.records[0].dispose_calls == 0
    assert manager.unload(model_id) == "unloaded"  # the third and last reference
    assert manager.get(model_id) is None and loader.records[0].dispose_calls == 1


def test_concurrent_open_sessions_coalesce_and_each_pins():
    manager, loader, _ = make_manager()
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("s0", lambda: manager.open_session("s0", model_spec=spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    for i in range(1, 4):
        runner.start(
            f"s{i}", lambda i=i: manager.open_session(f"s{i}", model_spec=spec(), key=key("A"))
        )
    wait_until(lambda: pending_waiters(manager, key("A")) == 3, "3 coalesced session opens")
    loader.gate.set()
    runner.join()
    leases = [result for _, result in runner.results.values()]
    model_ids = {lease.model_id for lease in leases}
    assert len(model_ids) == 1 and len(loader.calls) == 1
    assert manager.pin_count(model_ids.pop()) == 4


def test_concurrent_retry_of_one_session_id_pins_once():
    manager, loader, _ = make_manager()
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("open", lambda: manager.open_session("s1", model_spec=spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    runner.start("retry", lambda: manager.open_session("s1", model_spec=spec(), key=key("A")))
    wait_until(lambda: pending_waiters(manager, key("A")) == 1, "the retry to wait")
    loader.gate.set()
    runner.join()
    open_lease = runner.results["open"][1]
    retry_lease = runner.results["retry"][1]
    assert (open_lease.reused, retry_lease.reused) == (False, True)
    assert open_lease.model_id == retry_lease.model_id
    assert manager.pin_count(open_lease.model_id) == 1 and len(loader.calls) == 1


class _HoldOneWaiter(threading.Condition):
    """A cache condition variable that holds one named thread once, after it wakes.

    The held thread has released the lock, so other threads change the cache before
    it resumes: the order that a waiter slow to get the lock back meets.
    """

    def __init__(self, thread_name):
        super().__init__(threading.Lock())
        self.thread_name = thread_name
        self.held = threading.Event()
        self.resume = threading.Event()

    def wait(self, timeout=None):
        woke = super().wait(timeout)
        if threading.current_thread().name == self.thread_name and not self.held.is_set():
            self.held.set()
            self.release()
            try:
                self.resume.wait(WAIT_S)
            finally:
                self.acquire()
        return woke


@pytest.mark.parametrize("ending", ["closed", "conflict"])
def test_a_session_waiter_that_does_not_bind_frees_a_record_unloaded_during_its_handoff(ending):
    manager, loader, _ = make_manager()
    manager._cond = hold = _HoldOneWaiter("waiter")
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("first", lambda: manager.load(spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    runner.start("waiter", lambda: manager.open_session("s1", model_spec=spec(), key=key("A")))
    wait_until(lambda: pending_waiters(manager, key("A")) == 1, "the session open to wait")
    loader.gate.set()
    wait_until(lambda: "first" in runner.results and hold.held.is_set(), "the load to finish")
    model_id = runner.results["first"][1].model_id
    # The handoff the waiter has not picked up keeps the record from being freed.
    assert manager.unload(model_id) == "pending"
    if ending == "closed":  # another open of s1 binds the record and closes again
        manager.open_session("s1", model_id=model_id)
        assert manager.close_session("s1", reason="closed")
    else:  # another open of s1 binds another model
        manager.open_session("s1", model_spec=spec(), key=key("B"))
    hold.resume.set()
    runner.join()
    kind, exc = runner.results["waiter"]
    expected = SessionClosedError if ending == "closed" else SessionConflictError
    assert kind == "error" and isinstance(exc, expected)
    [record] = [r for r in loader.records if r.model_id == model_id]
    assert manager.get(model_id) is None and record.dispose_calls == 1
    assert _violations(loader) == []


def test_failed_load_rolls_back_for_the_loader_and_every_waiter():
    manager, loader, _ = make_manager(capacity=1)
    loader.gate = threading.Event()
    original = RuntimeError("corrupt weights")
    loader.fail_with = original
    runner = _Runner()
    runner.start("first", lambda: manager.load(spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    runner.start("waiter0", lambda: manager.load(spec(), key=key("A")))
    runner.start("waiter1", lambda: manager.open_session("s1", model_spec=spec(), key=key("A")))
    wait_until(lambda: pending_waiters(manager, key("A")) == 2, "2 waiters")
    loader.gate.set()
    runner.join()
    for name in ("first", "waiter0", "waiter1"):
        kind, exc = runner.results[name]
        assert kind == "error" and isinstance(exc, ModelLoadError), name
        assert exc.__cause__ is original, name
    stats = manager.stats()
    assert (stats.resident, stats.reserved) == (0, 0)
    assert manager.session_state("s1") == "unknown"
    loader.gate = None
    loader.fail_with = None
    assert manager.load(spec(), key=key("A")).cache_hit is False  # failure not cached
    assert len(loader.calls) == 2


def test_a_reservation_counts_against_capacity():
    manager, loader, _ = make_manager(capacity=2)
    manager.open_session("s-a", model_spec=spec(), key=key("A"))
    loader.gate = threading.Event()
    loader.entered.clear()
    runner = _Runner()
    runner.start("load-b", lambda: manager.load(spec(), key=key("B")))
    assert loader.entered.wait(WAIT_S)
    stats = manager.stats()
    assert (stats.resident, stats.reserved) == (1, 1)
    with pytest.raises(CacheExhaustedError):
        manager.load(spec(), key=key("C"))  # A pinned, B reserved: nothing to evict
    loader.gate.set()
    runner.join()
    assert runner.results["load-b"][0] == "ok"
    assert len(loader.calls) == 2


def test_loader_runs_without_the_global_lock():
    manager, loader, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    loader.gate = threading.Event()
    loader.entered.clear()
    runner = _Runner()
    runner.start("slow-load", lambda: manager.load(spec(), key=key("B")))
    assert loader.entered.wait(WAIT_S)
    # While B's loader is blocked, the cache keeps serving other work.
    assert manager.load(spec(), key=key("A")).cache_hit
    with manager.acquire_execution(model_id=a) as lease:
        assert lease.model_for_device("cpu:0") == "model:A"
    manager.open_session("s1", model_id=a)
    assert manager.close_session("s1", reason="closed")
    assert lock_is_free(manager)
    loader.gate.set()
    runner.join()
    assert _violations(loader) == []


def test_eviction_disposes_the_victim_without_the_global_lock():
    manager, loader, _ = make_manager(capacity=2)
    manager.load(spec(), key=key("A"))
    b = manager.load(spec(), key=key("B")).model_id
    victim = loader.records[0]
    victim.dispose_gate = threading.Event()
    runner = _Runner()
    runner.start("load-c", lambda: manager.load(spec(), key=key("C")))
    assert victim.dispose_entered.wait(WAIT_S)
    # The victim's disposal is blocked; other calls still go through.
    assert manager.stats().reserved == 1
    with manager.acquire_execution(model_id=b):
        pass
    victim.dispose_gate.set()
    runner.join()
    assert victim.dispose_calls == 1 and _violations(loader) == []


def test_concurrent_aux_resource_requests_share_one_factory_call():
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    record = manager.get(a)
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    def factory():
        calls.append(1)
        entered.set()
        gate.wait(WAIT_S)
        if not lock_is_free(manager):
            raise AssertionError("factory ran under the global lock")
        return object()

    def attempts_waiting():
        with record._cond:
            attempt = record._attempts.get("sam3_tracker@cuda:0")
            return attempt.waiters if attempt is not None else -1

    leases = [manager.acquire_execution(model_id=a) for _ in range(5)]
    runner = _Runner()
    runner.start("r0", lambda: leases[0].resource("sam3_tracker@cuda:0", factory))
    assert entered.wait(WAIT_S)
    for i in range(1, 5):
        runner.start(f"r{i}", lambda i=i: leases[i].resource("sam3_tracker@cuda:0", factory))
    wait_until(lambda: attempts_waiting() == 4, "4 resource waiters")
    gate.set()
    runner.join()
    values = {id(result) for kind, result in runner.results.values() if kind == "ok"}
    assert len(values) == 1 and len(runner.results) == 5 and len(calls) == 1
    for lease in leases:
        lease.release()


def test_concurrent_aux_resource_failure_reaches_waiters_but_is_not_cached():
    manager, _, _ = make_manager()
    a = manager.load(spec(), key=key("A")).model_id
    record = manager.get(a)
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    def failing():
        calls.append(1)
        entered.set()
        gate.wait(WAIT_S)
        raise OSError("gated repo")

    def attempts_waiting():
        with record._cond:
            attempt = record._attempts.get("sam3@cpu")
            return attempt.waiters if attempt is not None else -1

    with manager.acquire_execution(model_id=a) as lease:
        runner = _Runner()
        runner.start("creator", lambda: lease.resource("sam3@cpu", failing))
        assert entered.wait(WAIT_S)
        runner.start("waiter", lambda: lease.resource("sam3@cpu", failing))
        wait_until(lambda: attempts_waiting() == 1, "1 resource waiter")
        gate.set()
        runner.join()
        assert isinstance(runner.results["creator"][1], OSError)
        waiter_error = runner.results["waiter"][1]
        assert isinstance(waiter_error, ModelLoadError)
        assert isinstance(waiter_error.__cause__, OSError)
        assert lease.resource("sam3@cpu", object) is not None  # next call retries
    assert len(calls) == 1


def test_shutdown_waits_for_an_in_progress_load_and_disposes_it_once():
    manager, loader, _ = make_manager()
    loader.gate = threading.Event()
    runner = _Runner()
    runner.start("load", lambda: manager.load(spec(), key=key("A")))
    assert loader.entered.wait(WAIT_S)
    stopper = runner.start("shutdown", manager.shutdown)
    wait_until(lambda: manager._state == "draining", "shutdown to start")
    assert stopper.is_alive()
    loader.gate.set()
    runner.join()
    kind, exc = runner.results["load"]
    assert kind == "error" and isinstance(exc, CacheShutdownError)
    assert runner.results["shutdown"][0] == "ok"
    assert [r.dispose_calls for r in loader.records] == [1]
    assert manager.stats().resident == 0 and manager.stats().reserved == 0


def test_close_racing_a_running_call_releases_each_pin_once():
    """Close, cancel, fail and disconnect racing on one session with a live call."""

    manager, loader, _ = make_manager(capacity=1)
    a = manager.open_session("s1", model_spec=spec(), key=key("A")).model_id
    lease = manager.acquire_execution(session_id="s1")
    barrier = threading.Barrier(len(SESSION_CLOSE_REASONS))
    runner = _Runner()
    for reason in SESSION_CLOSE_REASONS:
        runner.start(
            reason,
            lambda r=reason: (barrier.wait(WAIT_S), manager.close_session("s1", reason=r))[1],
        )
    runner.join()
    assert sorted(result for _, result in runner.results.values()) == [False] * 5 + [True]
    assert manager.pin_count(a) == 1  # only the running call's lease
    with pytest.raises(SessionClosedError):
        manager.acquire_execution(session_id="s1")
    lease.release()
    assert manager.pin_count(a) == 0
    assert manager.unload(a) == "unloaded" and loader.records[0].dispose_calls == 1


def test_randomised_workload_keeps_every_invariant():
    capacity = 3
    manager, loader, clock = make_manager(capacity=capacity, ttl=50)
    keys = [key(f"m{i}") for i in range(6)]
    invariant_breaks = []
    users_lock = threading.Lock()

    def check_capacity():
        stats = manager.stats()
        if stats.resident + stats.reserved > capacity:
            invariant_breaks.append(stats)

    def use(record, delta):
        with users_lock:
            record.users += delta

    def worker(seed):
        rng = random.Random(seed)
        for step in range(150):
            check_capacity()
            op = rng.random()
            cache_key = rng.choice(keys)
            try:
                if op < 0.45:
                    sid = f"w{seed}-{step}"
                    manager.open_session(sid, model_spec=spec(), key=cache_key)
                    try:
                        lease = manager.acquire_execution(session_id=sid)
                        use(lease.record, +1)
                        lease.resource("aux@cpu", object)
                        use(lease.record, -1)
                        lease.release()
                    finally:
                        manager.close_session(sid, reason=rng.choice(SESSION_CLOSE_REASONS))
                elif op < 0.75:
                    model_id = manager.load(spec(), key=cache_key).model_id
                    lease = manager.acquire_execution(model_id=model_id)
                    use(lease.record, +1)
                    use(lease.record, -1)
                    lease.release()
                elif op < 0.9:
                    model_id = manager.load(spec(), key=cache_key).model_id
                    manager.unload(model_id)
                else:
                    clock.advance(20)
                    manager.expire_sessions()
            except (CacheExhaustedError, SessionClosedError):
                pass  # SessionClosedError: expired between open and acquire
            except KeyError:
                pass  # UnknownModelError: evicted between load and acquire

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S * 6)
        assert not thread.is_alive()
    check_capacity()
    assert invariant_breaks == []
    assert manager.stats().pinned == 0
    manager.shutdown()
    assert all(record.dispose_calls == 1 for record in loader.records)
    assert _violations(loader) == []
