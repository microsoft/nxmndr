# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic doubles for the model cache and lease tests.

No function here is named ``test_*`` so importing it never makes pytest collect it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, List, Optional

from nxmndr.server.managers import ModelCacheKey, ModelManager, ModelRecord

WAIT_S = 10.0  # upper bound for a synchronisation wait; never asserted on


class FakeClock:
    """Manually advanced clock for TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def key(name: str, **fields) -> ModelCacheKey:
    return ModelCacheKey(format="onnx", source=name, **fields)


def spec(name: str = "", token: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, token=token)


def lock_is_free(manager: ModelManager) -> bool:
    """True unless the calling thread already holds the cache's global lock.

    The lock is not reentrant, so a thread that holds it times out here. Other
    threads hold it only briefly, so the bound is never reached otherwise.
    """

    acquired = manager._cond.acquire(timeout=WAIT_S)
    if acquired:
        manager._cond.release()
    return acquired


def wait_until(predicate: Callable[[], bool], what: str) -> None:
    """Block until ``predicate()`` holds. Polls state; no test asserts on elapsed time."""

    deadline = time.monotonic() + WAIT_S
    pause = threading.Event()
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        pause.wait(0.0005)


def pending_waiters(manager: ModelManager, cache_key: ModelCacheKey) -> int:
    with manager._cond:
        pending = manager._loading.get(cache_key)
        return pending.waiters if pending is not None else -1


@dataclass(eq=False)
class CountingRecord(ModelRecord):
    """ModelRecord that counts dispose() calls and checks the lock is free."""

    dispose_calls: int = field(default=0, init=False)
    users: int = field(default=0, init=False)  # test-held leases and sessions
    manager: Optional[ModelManager] = field(default=None, init=False, repr=False)
    violations: List[str] = field(default_factory=list, init=False, repr=False)
    dispose_gate: Optional[threading.Event] = field(default=None, init=False, repr=False)
    dispose_entered: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def dispose(self) -> None:
        self.dispose_calls += 1
        self.dispose_entered.set()
        if self.manager is not None and not lock_is_free(self.manager):
            self.violations.append("dispose ran under the global lock")
            return  # report instead of disposing under the lock
        if self.users:
            self.violations.append(f"disposed with {self.users} test-held user(s)")
        if self.dispose_gate is not None:
            self.dispose_gate.wait(WAIT_S)
        super().dispose()


class FakeLoader:
    """Loader double: counts calls, can block on a gate and can fail."""

    def __init__(self) -> None:
        self.manager: Optional[ModelManager] = None
        self.calls: List[ModelCacheKey] = []
        self.metadata_seen: List[dict] = []
        self.records: List[CountingRecord] = []
        self.gate: Optional[threading.Event] = None
        self.entered = threading.Event()
        self.fail_with: Optional[BaseException] = None
        self.violations: List[str] = []
        self._lock = threading.Lock()

    def __call__(self, model_spec, cache_key, metadata, device_plan) -> CountingRecord:
        with self._lock:
            self.calls.append(cache_key)
            self.metadata_seen.append(dict(metadata))
        if self.manager is not None and not lock_is_free(self.manager):
            self.violations.append("loader ran under the global lock")
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(WAIT_S)
        if self.fail_with is not None:
            raise self.fail_with
        replicas = {d["id"]: f"{cache_key.source}@{d['id']}" for d in (device_plan or [])}
        record = CountingRecord(
            model=f"model:{cache_key.source}",
            backend="fake",
            metadata=dict(metadata),
            spec=model_spec,
            device_models=replicas,
        )
        record.manager = self.manager
        with self._lock:
            self.records.append(record)
        return record


def make_manager(capacity: int = 10, ttl: float = 100.0, clock: Optional[FakeClock] = None):
    loader = FakeLoader()
    clock = clock or FakeClock()
    manager = ModelManager(
        provider=None, capacity=capacity, session_ttl_s=ttl, clock=clock, loader=loader
    )
    loader.manager = manager
    return manager, loader, clock
