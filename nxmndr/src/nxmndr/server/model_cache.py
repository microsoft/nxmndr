# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bounded LRU cache of loaded models with session and execution leases.

The server imports these names from ``nxmndr.server.managers``; ``ModelManager``
there adds the default model loader. This module imports only the standard library.

Rules the implementation keeps:

* ``resident + reserved <= capacity`` at all times. Device replicas and aux
  resources are part of their record and do not use capacity.
* A record is pinned while an open session is bound to it or an execution lease
  on it is unreleased. Pinned records are never evicted or overwritten, and only
  shutdown disposes them.
  Holding a Python reference to a record is not a pin.
* Equal keys share one record, so several ``load`` callers can hold one model ID.
  Each ``load`` adds a load reference to the record it returns and each ``unload``
  releases one. An ``unload`` that leaves no load reference disposes the record at
  once if it is not pinned, else when its last pin is released, unless a ``load``
  adds a reference first. ``release_load`` takes back a reference whose caller never
  got the model ID and never disposes the record or marks it to be freed. A record
  with a load reference is never overwritten. Load references are not pins: LRU
  eviction ignores them, because a caller that never unloads would otherwise hold
  its cache slot until the server stops.
* A distinct load at full capacity evicts the least recently used unpinned
  record. If every slot is pinned or reserved it raises ``CacheExhaustedError``
  before it loads or evicts anything.
* The loader, ``ModelRecord.dispose`` and aux resource factories never run while
  the global lock is held.
* Each record is disposed exactly once.

Sessions come only from ``open_session``. ``acquire_execution(session_id=...)``
with an ID that was never opened raises ``UnknownSessionError`` and creates
nothing, so the server rejects a ``StreamPredict`` that names an unknown session
instead of fabricating a session.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..logging import get_logger

logger = get_logger(__name__)

SESSION_CLOSE_REASONS = ("closed", "cancelled", "expired", "failed", "disconnected", "shutdown")

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# ModelSpec.metadata keys that build a PyTorch catalog model; the same names as
# nxmndr.models.catalog.CONSTRUCTOR_KEYS (this module imports only the standard library).
_PYTORCH_CONSTRUCTOR_KEYS = ("num_classes", "in_channels")
# Per-process key for auth scopes: equal tokens map to equal scopes inside this
# process, and the scope cannot be reversed or correlated across processes.
_AUTH_SCOPE_KEY = secrets.token_bytes(32)
# A metadata key is treated as a credential if one of its words (split on
# non-alphanumerics) is in this set, or if it contains one of the fragments.
_SECRET_WORDS = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "apikey",
        "credential",
        "credentials",
        "authorization",
        "bearer",
    }
)
_SECRET_FRAGMENTS = ("api_key", "private_key", "access_key")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelCacheError(RuntimeError):
    """Base class of every model cache and lease error."""

    # KeyError subclasses below would otherwise quote their message.
    __str__ = RuntimeError.__str__


class CacheExhaustedError(ModelCacheError):
    """Every cache slot is pinned or reserved. Retry after a session or call ends."""

    retryable = True


class ModelInUseError(ModelCacheError):
    """Overwrite of a record that is pinned or has a load reference."""


class UnknownModelError(ModelCacheError, KeyError):
    """The model ID is not resident (never loaded, unloaded or evicted)."""


class UnknownSessionError(ModelCacheError, KeyError):
    """The session ID was never opened here, or its tombstone has expired."""


class SessionConflictError(ModelCacheError):
    """The session ID is already bound to another model."""


class SessionClosedError(ModelCacheError):
    """The session was closed, cancelled, expired, failed, disconnected or shut down."""


class ModelLoadError(ModelCacheError):
    """The loader raised. ``__cause__`` is the original exception."""


class CacheShutdownError(ModelCacheError):
    """The cache is shutting down or has shut down."""


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCacheKey:
    """Canonical identity of a loaded model. Holds no credential value."""

    format: str
    source: str
    revision: str = ""
    checksum: str = ""
    model_class: str = ""
    task: str = ""
    load_options: Tuple[Tuple[str, str], ...] = ()
    auth_scope: str = ""
    source_stamp: str = ""

    def __post_init__(self) -> None:
        for name in (
            "format",
            "source",
            "revision",
            "checksum",
            "model_class",
            "task",
            "auth_scope",
            "source_stamp",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"ModelCacheKey.{name} must be a str")
        raw = self.load_options
        items = raw.items() if isinstance(raw, Mapping) else raw
        options = tuple(sorted((str(k), str(v)) for k, v in items))
        names = [k for k, _ in options]
        if len(names) != len(set(names)):
            raise ValueError("ModelCacheKey.load_options has a duplicate option name")
        object.__setattr__(self, "load_options", options)


def _enum_name(spec: Any, field_name: str) -> str:
    number = int(getattr(spec, field_name))
    enum_type = spec.DESCRIPTOR.fields_by_name[field_name].enum_type
    value = enum_type.values_by_number.get(number)
    if value is None:
        return str(number)
    if value.name.endswith("_UNSPECIFIED"):
        return ""
    return value.name.lower()


def _auth_scope(token: str) -> str:
    if not token:
        return ""
    digest = hmac.new(_AUTH_SCOPE_KEY, token.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest[:32]}"


def local_source_stamp(source: str) -> str:
    """The size and modification time of a local file, or of each entry directly in a
    local directory, as one string; "" when ``source`` names neither.

    Part of the cache key (``source_stamp``), so a file rewritten in place, such as a
    retrained checkpoint, is loaded again instead of served from the earlier copy.
    Only metadata is read: hashing multi-gigabyte weights on every load would cost
    about as much as the load. A directory is not walked recursively, which bounds the
    cost for any path a client sends; a Hugging Face snapshot keeps its weights and
    config at the top level. A rewrite that keeps both the size and the modification
    time (a coarse filesystem clock, a copy that preserves the time) is not seen.
    """

    path = (source or "").strip()
    if not path:
        return ""
    try:
        info = os.stat(path)
        names = sorted(os.listdir(path)) if stat.S_ISDIR(info.st_mode) else None
    except (OSError, ValueError):  # ValueError: a NUL byte in the path
        return ""
    if names is None:
        return f"file:{info.st_size}:{info.st_mtime_ns}" if stat.S_ISREG(info.st_mode) else ""
    digest = hashlib.sha256()
    for name in names:
        try:
            entry = os.stat(os.path.join(path, name))  # follows snapshot symlinks to blobs
        except OSError:
            continue
        line = f"{name}\0{entry.st_size}\0{entry.st_mtime_ns}\n"
        digest.update(line.encode("utf-8", "surrogateescape"))
    return f"dir:{digest.hexdigest()[:32]}"


def cache_key_from_spec(
    spec: Any, *, artifact_sha256: str = "", source_stamp: str = ""
) -> ModelCacheKey:
    """Derive the cache key of an ``inference_pb2.ModelSpec``.

    Fields read: ``format``, ``source``, ``artifact``, ``version`` (the revision),
    ``checksum``, ``model_class``, ``task``, ``lazy_load``, ``name`` (TorchHub only,
    where it selects the hub entry point), the ``metadata`` entries ``num_classes`` and
    ``in_channels`` (PyTorch only, where they build a catalog model) and ``token``
    (only as an opaque scope). Uploaded artifact bytes are keyed by content:
    ``source = "sha256:<hex>"``. Otherwise ``source_stamp`` (the caller's
    ``local_source_stamp(source)``) is part of the key, so a rewritten local file gets
    a new key. Not part of the key, because the server does not use
    them to load: ``model_id``, the friendly ``name`` of other formats,
    ``preprocessing``, ``postprocessing``, other ``metadata`` entries and
    ``artifact_mime_type``.
    """

    fmt = _enum_name(spec, "format")
    artifact = bytes(getattr(spec, "artifact", b"") or b"")
    if artifact_sha256:
        digest = artifact_sha256.strip().lower()
        if not _SHA256_HEX.fullmatch(digest):
            raise ValueError("artifact_sha256 must be 64 hexadecimal characters")
        source = f"sha256:{digest}"
    elif artifact:
        source = f"sha256:{hashlib.sha256(artifact).hexdigest()}"
    else:
        source = (spec.source or "").strip()
    stamp = "" if (artifact_sha256 or artifact) else source_stamp
    options: List[Tuple[str, str]] = []
    if spec.lazy_load:
        options.append(("lazy_load", "true"))
    if fmt == "torchhub" and spec.name.strip():
        options.append(("name", spec.name.strip()))
    if fmt == "pytorch":
        entries = {e.key: e.value.strip() for e in getattr(spec, "metadata", ())}
        for name in _PYTORCH_CONSTRUCTOR_KEYS:
            value = entries.get(name, "")
            if value:
                options.append((name, str(int(value)) if value.isdigit() else value))
    return ModelCacheKey(
        format=fmt,
        source=source,
        revision=(spec.version or "").strip(),
        checksum=(spec.checksum or "").strip().lower(),
        model_class=(spec.model_class or "").strip(),
        task=_enum_name(spec, "task"),
        load_options=tuple(options),
        auth_scope=_auth_scope(spec.token or ""),
        source_stamp=stamp,
    )


def _is_secret_key(key: object) -> bool:
    lowered = str(key).lower()
    if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
        return True
    return any(word in _SECRET_WORDS for word in re.split(r"[^0-9a-z]+", lowered))


def _split_secrets(metadata: Optional[Mapping[str, object]]) -> Tuple[Dict[str, object], str]:
    """Return metadata without credential-like keys, and the value of ``token`` if any."""

    clean: Dict[str, object] = {}
    token = ""
    for key, value in dict(metadata or {}).items():
        if _is_secret_key(key):
            if str(key).lower() == "token" and isinstance(value, str):
                token = value
            continue
        clean[key] = value
    return clean, token


def _redact(text: str, *values: str) -> str:
    for value in values:
        if value:
            text = text.replace(value, "***")
    return text


def _dispose_resource(model_id: str, name: str, resource: object) -> None:
    """Call the resource's ``dispose()``, else its ``close()``, if it has one.

    Runs in the calling thread. An exception is logged, never raised, so one
    failing resource does not stop the disposal of the others.
    """

    for hook in ("dispose", "close"):
        method = getattr(resource, hook, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                logger.warning(
                    "model %s: %s() of resource %r failed: %s",
                    model_id,
                    hook,
                    name,
                    type(exc).__name__,
                )
            return


# ---------------------------------------------------------------------------
# Records, results and leases
# ---------------------------------------------------------------------------


class _ResourceAttempt:
    __slots__ = ("done", "error", "waiters")

    def __init__(self) -> None:
        self.done = False
        self.error: Optional[BaseException] = None
        self.waiters = 0


@dataclass(eq=False)
class ModelRecord:
    """A loaded model with its device replicas and aux resources."""

    model: Optional[object]
    backend: str
    metadata: Dict[str, object]
    # Out of repr: a proto ModelSpec carries its token field.
    spec: Optional[object] = field(default=None, repr=False)
    # Multi-device support: device_id -> model instance
    device_models: Dict[str, object] = field(default_factory=dict)
    model_id: str = ""
    key: Optional[ModelCacheKey] = None
    # A credential the model needs after loading (for example a Hugging Face
    # token for SAM variants). Never in metadata, keys, repr or logs.
    auth_token: str = field(default="", repr=False)
    _resources: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _attempts: Dict[str, _ResourceAttempt] = field(default_factory=dict, init=False, repr=False)
    _disposed: bool = field(default=False, init=False, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)

    @property
    def disposed(self) -> bool:
        with self._cond:
            return self._disposed

    def _resource(self, name: str, factory: Callable[[], object]) -> object:
        """Return the record-scoped resource ``name``, creating it once.

        Concurrent callers share one factory call. The factory runs without any
        lock held. A failure is raised to the caller and to concurrent waiters
        (as ``ModelLoadError``) and is not cached: the next call tries again.
        """

        with self._cond:
            while True:
                if self._disposed:
                    raise ModelCacheError(f"model {self.model_id} is disposed")
                if name in self._resources:
                    return self._resources[name]
                attempt = self._attempts.get(name)
                if attempt is None:
                    attempt = _ResourceAttempt()
                    self._attempts[name] = attempt
                    break
                attempt.waiters += 1
                while not attempt.done:
                    self._cond.wait()
                if attempt.error is not None:
                    raise ModelLoadError(
                        f"resource {name!r} of model {self.model_id} failed to load"
                    ) from attempt.error
        try:
            value = factory()
        except BaseException as exc:
            with self._cond:
                attempt.error = exc
                attempt.done = True
                self._attempts.pop(name, None)
                self._cond.notify_all()
            raise
        with self._cond:
            attempt.done = True
            self._attempts.pop(name, None)
            late = self._disposed
            if not late:
                self._resources[name] = value
            self._cond.notify_all()
        if late:
            # Never part of the record, so the record's disposal did not see it.
            _dispose_resource(self.model_id, name, value)
            del value
            raise ModelCacheError(f"model {self.model_id} was disposed while {name!r} loaded")
        return value

    def dispose(self) -> None:
        """Release replicas and aux resources. Idempotent.

        Each aux resource gets one ``dispose()`` call, or ``close()`` if it has no
        ``dispose``, in the calling thread; failures are logged, not raised.
        """

        with self._cond:
            if self._disposed:
                return
            self._disposed = True
            resources = list(self._resources.items())
            self._resources.clear()
            self._cond.notify_all()
        self.device_models.clear()
        self.model = None
        self.auth_token = ""
        for name, resource in resources:
            _dispose_resource(self.model_id, name, resource)
        del resources


@dataclass(frozen=True)
class LoadResult:
    model_id: str
    cache_hit: bool


@dataclass(frozen=True)
class SessionLease:
    session_id: str
    model_id: str
    cache_hit: bool
    reused: bool


@dataclass(frozen=True)
class CacheStats:
    capacity: int
    resident: int
    reserved: int
    pinned: int


class _Session:
    __slots__ = ("session_id", "model_id", "key", "last_activity", "active")

    def __init__(self, session_id: str, model_id: str, key: ModelCacheKey, now: float) -> None:
        self.session_id = session_id
        self.model_id = model_id
        self.key = key
        self.last_activity = now
        self.active = 0  # unreleased execution leases acquired through this session


class ExecutionLease:
    """Pins one record for the duration of one call or stream. Release exactly once."""

    def __init__(
        self,
        cache: "ModelCache",
        lease_id: str,
        record: ModelRecord,
        session: Optional[_Session],
    ) -> None:
        self._cache = cache
        self._session = session
        self._released = False
        self.lease_id = lease_id
        self.model_id = record.model_id
        self.session_id = session.session_id if session is not None else ""
        self.record = record

    @property
    def released(self) -> bool:
        return self._released

    def _ensure_live(self) -> None:
        if self._released:
            raise ModelCacheError(f"execution lease {self.lease_id} was released")

    def model_for_device(self, device_id: str) -> object:
        """The replica for ``device_id``, else the primary model."""

        self._ensure_live()
        replicas = self.record.device_models
        if device_id in replicas:
            return replicas[device_id]
        return self.record.model

    def resource(self, name: str, factory: Callable[[], object]) -> object:
        """A record-scoped aux resource, created once and disposed with the record."""

        self._ensure_live()
        return self.record._resource(name, factory)

    def release(self) -> None:
        self._cache._release_execution(self)

    def __enter__(self) -> "ExecutionLease":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"ExecutionLease(lease_id={self.lease_id!r}, model_id={self.model_id!r}, "
            f"session_id={self.session_id!r}, released={self._released})"
        )


class _Entry:
    __slots__ = ("record", "key", "sessions", "executions", "handoffs", "loads", "unloaded")

    def __init__(self, record: ModelRecord, key: ModelCacheKey) -> None:
        self.record = record
        self.key = key
        self.sessions: set[str] = set()
        self.executions: set[str] = set()
        # Coalesced waiters that have not yet picked up this freshly loaded
        # record. They keep it from eviction until they resume.
        self.handoffs = 0
        # load() calls that returned this record minus unload() calls that released
        # one. Not a pin: eviction ignores it.
        self.loads = 0
        # unload() left no load reference while the record was pinned: it is disposed
        # when its last pin is released. A new load reference clears it.
        self.unloaded = False

    def pins(self) -> int:
        return len(self.sessions) + len(self.executions)

    def evictable(self) -> bool:
        return self.pins() == 0 and self.handoffs == 0


class _PendingLoad:
    __slots__ = ("key", "model_id", "waiters", "done", "error")

    def __init__(self, key: ModelCacheKey, model_id: str) -> None:
        self.key = key
        self.model_id = model_id
        self.waiters = 0
        self.done = False
        self.error: Optional[BaseException] = None


Loader = Callable[[object, ModelCacheKey, dict, Optional[List[dict]]], ModelRecord]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ModelCache:
    """Thread-safe bounded LRU of model records with session and execution leases.

    ``nxmndr.server.managers.ModelManager`` is this class plus the default
    loader; see the module docstring for the rules.
    """

    def __init__(
        self,
        *,
        loader: Loader,
        capacity: int = 10,
        session_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be an integer >= 1")
        if session_ttl_s < 0:
            raise ValueError("session_ttl_s must be >= 0")
        self._loader = loader
        self._capacity = capacity
        self._ttl = float(session_ttl_s)
        self._clock = clock
        self._cond = threading.Condition(threading.Lock())
        self._records: "OrderedDict[str, _Entry]" = OrderedDict()  # least recent first
        self._by_key: Dict[ModelCacheKey, str] = {}
        self._loading: Dict[ModelCacheKey, _PendingLoad] = {}
        self._sessions: Dict[str, _Session] = {}
        self._tombstones: Dict[str, Tuple[str, float]] = {}  # session_id -> (reason, closed_at)
        self._live_executions = 0
        self._state = "running"  # running -> draining -> stopped

    # ---- loading --------------------------------------------------------

    def load(
        self,
        model_spec: object,
        *,
        key: ModelCacheKey,
        metadata: Optional[Mapping[str, object]] = None,
        device_plan: Optional[List[dict]] = None,
        overwrite: bool = False,
    ) -> LoadResult:
        """Make the model resident and add a load reference to it, not a pin (LoadModel).

        With ``overwrite=True`` a resident record for ``key`` that nothing holds is
        disposed and loaded again under a new model ID. A record that is pinned or
        has a load reference raises ``ModelInUseError``: the cache cannot tell the
        caller's own reference from another caller's, and disposing it would end the
        model ID for every holder. If ``key`` is being loaded by another call, this
        call waits for that load and returns its model ID, also with ``overwrite``.
        """

        _check_key(key)
        model_id, cache_hit, _ = self._acquire(
            key, model_spec, metadata, device_plan, overwrite=overwrite, session_id=None
        )
        return LoadResult(model_id=model_id, cache_hit=cache_hit)

    def _acquire(
        self,
        key: ModelCacheKey,
        model_spec: object,
        metadata: Optional[Mapping[str, object]],
        device_plan: Optional[List[dict]],
        *,
        overwrite: bool,
        session_id: Optional[str],
    ) -> Tuple[str, bool, bool]:
        """Resolve ``key`` to a resident record, loading it if needed.

        With ``session_id``, the session is bound to the record in the same lock
        hold that finds or inserts it. Returns (model_id, cache_hit, reused).
        """

        victims: List[ModelRecord] = []
        freed: List[ModelRecord] = []
        handoff: Optional[Tuple[str, bool, bool]] = None
        refusal: Optional[ModelCacheError] = None
        with self._cond:
            while True:
                self._check_running()
                if session_id is not None:
                    existing = self._session_precheck(session_id, key=key)
                    if existing is not None:
                        return existing.model_id, True, True
                model_id = self._by_key.get(key)
                if model_id is not None:
                    entry = self._records[model_id]
                    if not overwrite:
                        self._records.move_to_end(model_id)
                        if session_id is not None:
                            self._bind_session(session_id, entry)
                        else:
                            entry.loads += 1
                            entry.unloaded = False
                        return model_id, True, False
                    if entry.loads or not entry.evictable():
                        raise ModelInUseError(
                            f"model {model_id} is in use ({entry.pins()} pins, "
                            f"{entry.loads} load references); cannot overwrite"
                        )
                    self._remove_entry(model_id)
                    victims.append(entry.record)
                    overwrite = False
                pending = self._loading.get(key)
                if pending is not None:
                    pending.waiters += 1
                    while not pending.done:
                        self._cond.wait()
                    if pending.error is not None:
                        if isinstance(pending.error, CacheShutdownError):
                            raise CacheShutdownError("model cache shut down during load")
                        raise ModelLoadError(
                            f"loading model {pending.model_id} ({key.format}) failed"
                        ) from pending.error
                    self._check_running()
                    entry = self._records[pending.model_id]
                    entry.handoffs -= 1
                    self._records.move_to_end(pending.model_id)
                    if session_id is None:
                        entry.loads += 1
                        entry.unloaded = False
                        return pending.model_id, False, False
                    try:
                        existing = self._session_precheck(session_id, key=key)
                    except ModelCacheError as exc:
                        refusal = exc
                    else:
                        if existing is None:
                            self._bind_session(session_id, entry)
                            return pending.model_id, False, False
                        handoff = (existing.model_id, True, True)
                    # This waiter leaves without pinning the record. Its handoff kept an
                    # unload that came in the meantime from freeing the record, so free
                    # it now (outside the lock), as the release of a last pin would.
                    self._take_unloaded(entry, freed)
                    break
                if len(self._records) + len(self._loading) >= self._capacity:
                    victim = self._lru_evictable()
                    if victim is None:
                        raise CacheExhaustedError(
                            f"all {self._capacity} model cache slots are pinned or reserved"
                        )
                    self._remove_entry(victim.record.model_id)
                    victims.append(victim.record)
                    logger.info(
                        "evicting model %s (%s) to make room",
                        victim.record.model_id,
                        victim.key.format,
                    )
                model_id = uuid.uuid4().hex
                pending = _PendingLoad(key, model_id)
                self._loading[key] = pending
                break

        self._dispose_unloaded(freed)
        for victim_record in victims:
            self._dispose_record(victim_record)
        if refusal is not None:
            raise refusal
        if handoff is not None:
            return handoff

        clean, token = _split_secrets(metadata)
        clean["model_id"] = model_id
        spec_token = getattr(model_spec, "token", None)
        spec_token = spec_token if isinstance(spec_token, str) else ""
        logger.info("loading model %s (%s)", model_id, key.format)
        try:
            record = self._loader(model_spec, key, dict(clean), device_plan)
            if not isinstance(record, ModelRecord):
                raise TypeError(f"loader returned {type(record).__name__}, not ModelRecord")
        except BaseException as exc:
            with self._cond:
                self._loading.pop(key, None)
                pending.error = exc
                pending.done = True
                self._cond.notify_all()
            if not isinstance(exc, Exception):
                raise
            logger.warning("loading model %s failed: %s", model_id, type(exc).__name__)
            message = _redact(f"{type(exc).__name__}: {exc}", token, spec_token)
            raise ModelLoadError(
                f"loading model {model_id} ({key.format}) failed: {message}"
            ) from exc

        record.model_id = model_id
        record.key = key
        record.metadata, _ = _split_secrets(record.metadata)
        record.metadata["model_id"] = model_id
        if not record.auth_token:
            record.auth_token = token or spec_token

        reused: Optional[SessionLease] = None
        with self._cond:
            self._loading.pop(key, None)
            try:
                if self._state != "running":
                    pending.error = CacheShutdownError("model cache shut down during load")
                else:
                    entry = _Entry(record, key)
                    entry.handoffs = pending.waiters
                    self._records[model_id] = entry
                    self._by_key[key] = model_id
                    if session_id is not None:
                        reused = self._session_precheck(session_id, key=key)
                        if reused is None:
                            self._bind_session(session_id, entry)
                    else:
                        entry.loads = 1
            finally:
                pending.done = True
                self._cond.notify_all()
        if pending.error is not None:
            self._dispose_record(record)
            raise CacheShutdownError("model cache shut down during load")
        logger.info("loaded model %s (%s)", model_id, key.format)
        if reused is not None:
            return reused.model_id, True, True
        return model_id, False, False

    # ---- sessions -------------------------------------------------------

    def open_session(
        self,
        session_id: str,
        *,
        model_id: str = "",
        model_spec: object = None,
        key: Optional[ModelCacheKey] = None,
        metadata: Optional[Mapping[str, object]] = None,
        device_plan: Optional[List[dict]] = None,
    ) -> SessionLease:
        """Bind ``session_id`` to a model and pin it (OpenSession). Idempotent.

        Pass exactly one of ``model_id`` (a resident model) or ``key`` together
        with ``model_spec`` (load if needed). A retry with the same session ID and
        the same model returns ``reused=True`` and does not add a pin.
        """

        _check_session_id(session_id)
        if bool(model_id) == (key is not None):
            raise ValueError("open_session needs exactly one of model_id or key")
        if key is not None:
            _check_key(key)
            if model_spec is None:
                raise ValueError("open_session with key needs model_spec")
            mid, cache_hit, reused = self._acquire(
                key, model_spec, metadata, device_plan, overwrite=False, session_id=session_id
            )
            return SessionLease(
                session_id=session_id, model_id=mid, cache_hit=cache_hit, reused=reused
            )
        with self._cond:
            self._check_running()
            existing = self._session_precheck(session_id, model_id=model_id)
            if existing is not None:
                return existing
            entry = self._records.get(model_id)
            if entry is None:
                raise UnknownModelError(f"model {model_id} is not resident")
            self._bind_session(session_id, entry)
        return SessionLease(session_id=session_id, model_id=model_id, cache_hit=True, reused=False)

    def close_session(self, session_id: str, *, reason: str) -> bool:
        """Release the session pin once and tombstone the ID. False if not open."""

        if reason not in SESSION_CLOSE_REASONS:
            raise ValueError(f"unknown close reason {reason!r}")
        freed: List[ModelRecord] = []
        with self._cond:
            closed = self._close_locked(session_id, reason, self._clock(), freed)
        if closed:
            logger.info("session %s %s", session_id, reason)
        self._dispose_unloaded(freed)
        return closed

    def touch_session(self, session_id: str) -> None:
        """Record activity on an open session (for TTL expiry)."""

        with self._cond:
            now = self._clock()
            self._open_session_or_raise(session_id, now).last_activity = now

    def session_state(self, session_id: str) -> str:
        """``"open"``, the close reason while tombstoned, else ``"unknown"``."""

        with self._cond:
            if session_id in self._sessions:
                return "open"
            tombstone = self._live_tombstone(session_id, self._clock())
            return tombstone[0] if tombstone is not None else "unknown"

    def expire_sessions(self) -> List[str]:
        """Close (reason ``expired``) open sessions idle longer than the TTL.

        A session with a live execution lease is never expired. Also forgets
        tombstones older than the TTL. Returns the expired session IDs.
        """

        freed: List[ModelRecord] = []
        with self._cond:
            now = self._clock()
            expired = [
                sid
                for sid, session in self._sessions.items()
                if session.active == 0 and now - session.last_activity > self._ttl
            ]
            for sid in expired:
                self._close_locked(sid, "expired", now, freed)
            for sid, (_, closed_at) in list(self._tombstones.items()):
                if now - closed_at > self._ttl:
                    del self._tombstones[sid]
        for sid in expired:
            logger.info("session %s expired", sid)
        self._dispose_unloaded(freed)
        return expired

    # ---- execution ------------------------------------------------------

    def acquire_execution(self, *, model_id: str = "", session_id: str = "") -> ExecutionLease:
        """Pin a model for one call or stream. Pass exactly one of the two IDs."""

        if bool(model_id) == bool(session_id):
            raise ValueError("acquire_execution needs exactly one of model_id or session_id")
        with self._cond:
            self._check_running()
            now = self._clock()
            session: Optional[_Session] = None
            if session_id:
                session = self._open_session_or_raise(session_id, now)
                entry = self._records[session.model_id]
                session.active += 1
                session.last_activity = now
            else:
                entry = self._records.get(model_id)
                if entry is None:
                    raise UnknownModelError(f"model {model_id} is not resident")
            lease = ExecutionLease(self, uuid.uuid4().hex, entry.record, session)
            entry.executions.add(lease.lease_id)
            self._live_executions += 1
            self._records.move_to_end(entry.record.model_id)
        logger.debug("execution lease %s on model %s", lease.lease_id, lease.model_id)
        return lease

    def _release_execution(self, lease: ExecutionLease) -> None:
        freed: List[ModelRecord] = []
        with self._cond:
            if lease._released:
                return
            lease._released = True
            entry = self._records.get(lease.model_id)
            if entry is not None:
                entry.executions.discard(lease.lease_id)
                self._take_unloaded(entry, freed)
            self._live_executions -= 1
            if lease._session is not None:
                lease._session.active -= 1
                lease._session.last_activity = self._clock()
            self._cond.notify_all()
        logger.debug("released execution lease %s", lease.lease_id)
        self._dispose_unloaded(freed)

    # ---- unload and introspection ---------------------------------------

    def unload(self, model_id: str) -> str:
        """Release one load reference of a resident record (UnloadModel).

        Returns ``"unknown"`` if ``model_id`` is not resident, ``"released"`` while
        other load references remain, ``"unloaded"`` if the record was disposed now,
        and ``"pending"`` if no load reference remains but the record is in use (pinned,
        or not yet picked up by a load that waited for it): it is disposed when its
        last pin is released, unless a ``load`` adds a reference first. A record that
        only sessions loaded has no load reference, so unloading it releases none and
        leaves none.
        """

        with self._cond:
            entry = self._records.get(model_id)
            if entry is None:
                return "unknown"
            if entry.loads:
                entry.loads -= 1
            if entry.loads:
                outcome = "released"
            elif entry.evictable():
                self._remove_entry(model_id)
                outcome = "unloaded"
            else:
                entry.unloaded = True
                outcome = "pending"
            remaining = entry.loads
        if outcome == "released":
            logger.info("released a load reference of model %s (%d remain)", model_id, remaining)
        elif outcome == "pending":
            logger.info("model %s is unloaded when its last pin is released", model_id)
        else:
            self._dispose_record(entry.record)
            logger.info("unloaded model %s", model_id)
        return outcome

    def release_load(self, model_id: str) -> bool:
        """Take back one load reference whose ``load`` result never reached its caller.

        Unlike ``unload``, this never disposes the record or marks it to be freed: with
        no load reference left, the record stays cached, as a record that only sessions
        loaded does, until LRU eviction frees it. A pending free that the load cancelled
        stays cancelled. False if ``model_id`` is not resident or holds no load reference.
        """

        with self._cond:
            entry = self._records.get(model_id)
            if entry is None or not entry.loads:
                return False
            entry.loads -= 1
            return True

    def get(self, model_id: str) -> Optional[ModelRecord]:
        """Peek at a resident record. Does not pin it or change recency."""

        with self._cond:
            entry = self._records.get(model_id)
            return entry.record if entry is not None else None

    def list_models(self) -> Dict[str, ModelRecord]:
        """Resident records, least recently used first."""

        with self._cond:
            return {mid: entry.record for mid, entry in self._records.items()}

    def pin_count(self, model_id: str) -> int:
        """Open sessions bound to the model plus its unreleased execution leases."""

        with self._cond:
            entry = self._records.get(model_id)
            return entry.pins() if entry is not None else 0

    def stats(self) -> CacheStats:
        with self._cond:
            return CacheStats(
                capacity=self._capacity,
                resident=len(self._records),
                reserved=len(self._loading),
                pinned=sum(1 for entry in self._records.values() if entry.pins() > 0),
            )

    # ---- shutdown -------------------------------------------------------

    def shutdown(self, *, drain_timeout_s: Optional[float] = None) -> None:
        """Stop new work, close sessions, drain leases and loads, dispose every record.

        Waits for execution leases and in-progress loads without limit when
        ``drain_timeout_s`` is None. Records still leased after the timeout are
        disposed with a warning. Safe to call more than once.
        """

        with self._cond:
            if self._state != "running":
                while self._state != "stopped":
                    self._cond.wait()
                return
            self._state = "draining"
            now = self._clock()
            for sid in list(self._sessions):
                self._close_locked(sid, "shutdown", now, [])  # every record is disposed below
            self._cond.notify_all()
            deadline = (
                None if drain_timeout_s is None else time.monotonic() + max(0.0, drain_timeout_s)
            )
            while self._live_executions > 0 or self._loading:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            entries = list(self._records.values())
            self._records.clear()
            self._by_key.clear()
            live = self._live_executions
        if live:
            logger.warning(
                "model cache shutdown: drain timed out with %d live execution lease(s)", live
            )
        for entry in entries:
            if entry.executions:
                logger.warning(
                    "disposing model %s with %d live execution lease(s)",
                    entry.record.model_id,
                    len(entry.executions),
                )
            self._dispose_record(entry.record)
        with self._cond:
            self._state = "stopped"
            self._cond.notify_all()
        logger.info("model cache shut down (%d model(s) disposed)", len(entries))

    # ---- internals (lock held unless noted) ------------------------------

    def _check_running(self) -> None:
        if self._state != "running":
            raise CacheShutdownError("model cache is shut down")

    def _remove_entry(self, model_id: str) -> _Entry:
        entry = self._records.pop(model_id)
        if self._by_key.get(entry.key) == model_id:
            del self._by_key[entry.key]
        return entry

    def _lru_evictable(self) -> Optional[_Entry]:
        for entry in self._records.values():
            if entry.evictable():
                return entry
        return None

    def _live_tombstone(self, session_id: str, now: float) -> Optional[Tuple[str, float]]:
        tombstone = self._tombstones.get(session_id)
        if tombstone is not None and now - tombstone[1] > self._ttl:
            del self._tombstones[session_id]
            return None
        return tombstone

    def _open_session_or_raise(self, session_id: str, now: float) -> _Session:
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        tombstone = self._live_tombstone(session_id, now)
        if tombstone is not None:
            raise SessionClosedError(f"session {session_id} is {tombstone[0]}")
        raise UnknownSessionError(f"session {session_id} was not opened")

    def _session_precheck(
        self, session_id: str, *, key: Optional[ModelCacheKey] = None, model_id: str = ""
    ) -> Optional[SessionLease]:
        """Tombstone, conflict and idempotent-retry checks for open_session."""

        now = self._clock()
        tombstone = self._live_tombstone(session_id, now)
        if tombstone is not None:
            raise SessionClosedError(f"session {session_id} is {tombstone[0]}")
        session = self._sessions.get(session_id)
        if session is None:
            return None
        same = (key is not None and session.key == key) or (
            model_id and session.model_id == model_id
        )
        if not same:
            raise SessionConflictError(f"session {session_id} is bound to model {session.model_id}")
        session.last_activity = now
        self._records.move_to_end(session.model_id)
        return SessionLease(
            session_id=session_id, model_id=session.model_id, cache_hit=True, reused=True
        )

    def _bind_session(self, session_id: str, entry: _Entry) -> None:
        model_id = entry.record.model_id
        self._sessions[session_id] = _Session(session_id, model_id, entry.key, self._clock())
        entry.sessions.add(session_id)
        self._records.move_to_end(model_id)
        logger.info("session %s opened on model %s", session_id, model_id)

    def _close_locked(
        self, session_id: str, reason: str, now: float, freed: List[ModelRecord]
    ) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        entry = self._records.get(session.model_id)
        if entry is not None:
            entry.sessions.discard(session_id)
            self._take_unloaded(entry, freed)
        self._tombstones[session_id] = (reason, now)
        self._cond.notify_all()
        return True

    def _take_unloaded(self, entry: _Entry, freed: List[ModelRecord]) -> None:
        """Remove an unloaded entry whose last pin was just released; append its record
        to ``freed``, which the caller disposes outside the lock. Shutdown disposes every
        record itself, so nothing is taken once it has started."""

        if entry.unloaded and entry.evictable() and self._state == "running":
            self._remove_entry(entry.record.model_id)
            freed.append(entry.record)

    def _dispose_unloaded(self, records: List[ModelRecord]) -> None:
        """Dispose records taken by ``_take_unloaded`` (outside the global lock)."""

        for record in records:
            self._dispose_record(record)
            logger.info("unloaded model %s", record.model_id)

    def _dispose_record(self, record: ModelRecord) -> None:
        """Dispose outside the global lock."""

        try:
            record.dispose()
        except Exception as exc:
            logger.warning("disposing model %s failed: %s", record.model_id, type(exc).__name__)


def _check_key(key: object) -> None:
    if not isinstance(key, ModelCacheKey):
        raise TypeError("key must be a ModelCacheKey")


def _check_session_id(session_id: object) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")


__all__ = [
    "SESSION_CLOSE_REASONS",
    "ModelCacheError",
    "CacheExhaustedError",
    "ModelInUseError",
    "UnknownModelError",
    "UnknownSessionError",
    "SessionConflictError",
    "SessionClosedError",
    "ModelLoadError",
    "CacheShutdownError",
    "ModelCacheKey",
    "cache_key_from_spec",
    "local_source_stamp",
    "ModelRecord",
    "LoadResult",
    "SessionLease",
    "ExecutionLease",
    "CacheStats",
    "ModelCache",
]
