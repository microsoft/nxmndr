# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Server-side managers for model lifecycle and RPC worker orchestration.

This module extracts logic from server/server.py to improve separation of concerns.
The bounded model cache and its session/execution leases live in ``model_cache``;
this module re-exports the lease API names because ``server.py`` uses them as
``managers.<name>``.
"""

from __future__ import annotations

import ipaddress
import multiprocessing
import os
import threading
import time
from functools import partial
from multiprocessing.process import BaseProcess
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

from ..logging import get_logger
from .worker_ipc import WorkerClient, receive_message
from .model_cache import (
    SESSION_CLOSE_REASONS,
    CacheExhaustedError,
    CacheShutdownError,
    CacheStats,
    ExecutionLease,
    LoadResult,
    ModelCache,
    ModelCacheError,
    ModelCacheKey,
    ModelInUseError,
    ModelLoadError,
    ModelRecord,
    SessionClosedError,
    SessionConflictError,
    SessionLease,
    UnknownModelError,
    UnknownSessionError,
    cache_key_from_spec,
    local_source_stamp,
)

logger = get_logger(__name__)


def _load_model_object(model_spec, provider):
    """Construct the model for ``model_spec`` without the process-global model cache.

    ``InferenceSession`` consults ``nxmndr.memory.get_model_cache()``, a second
    LRU keyed only by spec class, path/repo and name. Going through it here would
    keep evicted models alive and hand back a model loaded for another revision
    or token, so the local path calls the registered loader directly.
    """

    from ..inference import InferenceSession  # registers the built-in loaders
    from ..models import get_loader
    from ..validation import validate_model_spec

    if hasattr(provider, "load_spec"):
        # Remote and RPC providers load through the provider, not in this process.
        return InferenceSession(model_spec, provider).model
    validate_model_spec(model_spec)
    loader = get_loader(model_spec)
    if loader is None:
        raise ValueError(f"No registered model loader for spec type: {type(model_spec).__name__}")
    return loader(model_spec, provider, SimpleNamespace(model_spec=model_spec))


def _build_pytorch_record(rpc_workers, model_spec, metadata: Dict[str, object]) -> ModelRecord:
    """A generic PyTorch record: its model lives in the RPC worker under the record's model ID.

    The record owns that worker-side model as the aux resource ``RPC_MODEL_RESOURCE``,
    so evicting, unloading or shutting down the record frees it in the worker.
    """

    if rpc_workers is None:
        raise ValueError(
            "generic PyTorch models run in the server's PyTorch RPC worker; this "
            "ModelManager has none (rpc_workers)"
        )
    model_id = str(metadata.get("model_id") or "")
    if not model_id:
        raise ValueError("a PyTorch record needs the model_id the cache assigned")
    info = rpc_workers.load(model_id, model_spec)
    try:
        record_metadata = dict(metadata)
        record_metadata["rpc_device"] = str(info.get("device", ""))
        record_metadata["rpc_fingerprint"] = str(info.get("fingerprint", ""))
        # A catalog model: model_family, model_class, num_classes, in_channels, input_size.
        record_metadata.update(info.get("catalog") or {})
        record = ModelRecord(None, "pytorch", record_metadata, model_spec)
        handle = _RpcModelHandle(rpc_workers, model_id, info)
        record._resource(RPC_MODEL_RESOURCE, lambda: handle)
    except BaseException:
        rpc_workers.unload(model_id)
        raise
    logger.info(
        "ModelManager loaded model_id=%s type=PytorchModelSpec in the RPC worker on %s (%s)",
        model_id,
        handle.device,
        handle.fingerprint,
    )
    return record


def build_model_record(
    provider,
    model_spec,
    key: ModelCacheKey,
    metadata: Dict[str, object],
    device_plan: Optional[List[Dict[str, str]]] = None,
    *,
    rpc_workers: Optional["RpcWorkerManager"] = None,
) -> ModelRecord:
    """Default loader: load ``model_spec``, optionally replicating to multiple devices.

    Args:
        provider: The inference provider models are loaded for
        model_spec: The model specification to load
        key: The cache key of the spec (unused here; part of the loader signature)
        metadata: Record metadata; ``metadata["model_id"]`` is the assigned model_id
        device_plan: Optional list of device dicts with 'id' keys (e.g., [{'id': 'cuda:0'}, {'id': 'cuda:1'}])
        rpc_workers: The server's ``RpcWorkerManager``; generic PyTorch specs load there.
    """
    model_type = model_spec.__class__.__name__
    model_id = metadata.get("model_id", "")

    if model_type == "PytorchModelSpec":
        return _build_pytorch_record(rpc_workers, model_spec, metadata)
    if model_type not in ("OnnxModelSpec", "HuggingFaceModelSpec", "UltralyticsModelSpec"):
        raise ValueError(f"Unsupported spec type {model_type}")
    model = _load_model_object(model_spec, provider)
    backend = model_type[:-9].lower()  # "onnx", "huggingface" or "ultralytics"
    record = ModelRecord(model, backend, metadata, model_spec)

    # Multi-device replication for ONNX/HuggingFace models
    if device_plan and len(device_plan) > 0:
        primary_device = device_plan[0].get("id", "cpu")
        record.device_models[primary_device] = model

        # Replicate to additional devices
        for device_info in device_plan[1:]:
            device_id = device_info.get("id", "cpu")
            try:
                if backend == "huggingface" and hasattr(model, "to"):
                    # For HuggingFace models, create a copy and move to device
                    import copy

                    device_model = copy.deepcopy(model)
                    device_model.to(device_id)
                    record.device_models[device_id] = device_model
                    logger.info(f"Replicated model {model_id} to {device_id}")
                else:
                    # For ONNX, we typically can't replicate easily
                    # Just reference the same model for now
                    record.device_models[device_id] = model
            except Exception as e:
                logger.warning(f"Failed to replicate model to {device_id}: {e}")
                record.device_models[device_id] = model
    logger.info(
        f"ModelManager loaded model_id={model_id} type={model_type} devices={list(record.device_models.keys())}"
    )
    return record


class ModelManager(ModelCache):
    """Bounded LRU of loaded models with session and execution leases.

    Supports loading models to multiple devices for parallel inference. ``loader``
    defaults to ``build_model_record`` bound to ``provider`` and ``rpc_workers`` (the
    server's ``RpcWorkerManager``, needed for generic PyTorch specs); tests inject
    their own.
    """

    def __init__(
        self,
        provider,
        *,
        capacity: int = 10,
        session_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        loader: Optional[Callable[..., ModelRecord]] = None,
        rpc_workers: Optional["RpcWorkerManager"] = None,
    ):
        self._provider = provider
        default_loader = partial(build_model_record, provider, rpc_workers=rpc_workers)
        super().__init__(
            loader=loader if loader is not None else default_loader,
            capacity=capacity,
            session_ttl_s=session_ttl_s,
            clock=clock,
        )


# Compatibility names for the private PyTorch worker.
RPC_DRIVER_NAME = "driver"
RPC_WORKER_NAME = "worker"
# Bound on the spawned child's readiness and on calls without an explicit timeout.
RPC_STARTUP_TIMEOUT_SECONDS = 60.0
# Bound on one model load in the worker (_rpc_load). It detects a hung worker; it does
# not bound normal load latency. The largest catalog weights file (regnet_y_128gf,
# 2.55 GB) built in 11 s on macOS arm64, and 300 s also covers reading it at 9 MB/s
# from slow network storage. It is above the QGIS plugin's 120 s deadline for LoadModel
# and OpenSession, so a slow load that is still progressing lands in the model cache for
# the client's next attempt instead of being abandoned.
RPC_LOAD_TIMEOUT_SECONDS = 300.0
# Bound on the stop call and each join of the child.
RPC_STOP_TIMEOUT_SECONDS = 2.0
# Aux resource name of a PyTorch record's model in the RPC worker.
RPC_MODEL_RESOURCE = "pytorch_rpc_model"
# Legacy configuration is retained for explicit refusal of external workers.
RPC_DEFAULT_ADDR = "127.0.0.1"
RPC_EXTERNAL_DEFAULT_PORT = 29500
EXTERNAL_RPC_ERROR = (
    "external PyTorch RPC workers are disabled: torch.distributed.rpc deserializes "
    "unauthenticated Python objects. Use nxmndr-server on the worker host and "
    "RemoteInferenceProvider over gRPC instead"
)


class RpcWorkerStartError(RuntimeError):
    """The PyTorch RPC worker could not be started, or can no longer be used."""


class RpcWorkerNotJoinedError(RpcWorkerStartError):
    """Legacy external worker configuration is not supported."""


_PROCESS_LOCK = threading.Lock()
_process_workers: set["RpcWorkerManager"] = set()


def _is_loopback(addr: str) -> bool:
    """True for a loopback IP or ``localhost``; False for any other address or host name."""

    addr = addr.strip()
    if addr.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False


def _terminate(proc) -> None:
    """End a child process: wait, terminate, then kill, each bounded."""

    if proc is None or proc.pid is None:
        return  # never started
    proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)


def _run_rpc_worker(
    conn,
    model_classes,
    device: str = "cpu",
) -> None:
    """Spawn target; only the trusted parent supplies classes and the private pipe."""

    from .rpc_worker import configure_cudnn, run_ipc_worker

    configure_cudnn()
    run_ipc_worker(conn, model_classes, device=device)


class _RpcModelHandle:
    """Aux resource of a PyTorch ModelRecord: its model in the RPC worker.

    ``dispose()`` (called once, when the record is evicted, unloaded or shut down)
    frees that model in the worker.
    """

    def __init__(self, workers: "RpcWorkerManager", model_id: str, info: Dict[str, object]):
        self.workers = workers
        self.model_id = model_id
        self.device = str(info.get("device", ""))
        self.fingerprint = str(info.get("fingerprint", ""))

    def infer(self, tensor, *, device_hint: str, timeout: float):
        return self.workers.infer(self.model_id, tensor, device_hint=device_hint, timeout=timeout)

    def dispose(self) -> None:
        self.workers.unload(self.model_id)


class RpcWorkerManager:
    """Local PyTorch worker, reached only through an inherited, data-only pipe.

    Register custom classes before the first load: the class allowlist is supplied
    at spawn, never imported from a worker message. External Torch RPC is refused.
    """

    # The child's process target; tests replace it with a module-level function.
    worker_target = staticmethod(_run_rpc_worker)

    def __init__(
        self,
        *,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        startup_timeout_s: float = RPC_STARTUP_TIMEOUT_SECONDS,
        external_worker: Optional[bool] = None,
        device: str = "cpu",
    ):
        self._lock = threading.Lock()
        self._calls_lock = threading.Lock()
        self._calls = 0
        self._client: Optional[WorkerClient] = None
        self._model_classes: Dict[str, type] = {}
        self._proc: Optional[BaseProcess] = None
        self._state = "idle"
        if external_worker is None:
            flag = os.environ.get("NXMNDR_RPC_EXTERNAL_WORKER", "").strip().lower()
            external_worker = flag not in {"", "0", "false", "no", "off"}
        self._external = bool(external_worker)
        # Retained only for the existing external-mode upload refusal, not for binding.
        self._addr = (master_addr or os.environ.get("MASTER_ADDR", "")).strip() or RPC_DEFAULT_ADDR
        self._port = int(master_port or os.environ.get("MASTER_PORT", "0") or 0)
        self.startup_timeout_s = float(startup_timeout_s)
        self.device = str(device or "cpu")

    @property
    def worker_may_be_remote(self) -> bool:
        """External mode on a non-loopback address: the worker may run on another host."""

        return self._external and not _is_loopback(self._addr)

    # ---- start ------------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the private worker once; raise ``RpcWorkerStartError`` on failure."""

        with self._lock:
            if self._state == "running":
                proc = self._proc
                if proc is not None and proc.is_alive():
                    return
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker exited (exit code {getattr(proc, 'exitcode', None)}); "
                    "restart the server to use PyTorch models"
                )
            if self._state == "stopped":
                raise RpcWorkerStartError(
                    "the PyTorch RPC worker was stopped; restart the server to use PyTorch models"
                )
            self._start_locked()

    def _start_locked(self) -> None:
        from ..models import get_registry

        if self._external:
            raise RpcWorkerNotJoinedError(EXTERNAL_RPC_ERROR)
        self._model_classes.update(get_registry().list_model_classes())
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(
            target=type(self).worker_target,
            args=(child, self._model_classes, self.device),
            daemon=True,
            name="nxmndr-pytorch-worker",
        )
        try:
            proc.start()
            child.close()
            self._await_ready(proc, parent, self.startup_timeout_s)
            self._client = WorkerClient(parent)
        except BaseException:
            parent.close()
            _terminate(proc)
            raise
        finally:
            child.close()
        self._proc = proc
        self._state = "running"
        with _PROCESS_LOCK:
            _process_workers.add(self)
        logger.info("PyTorch worker pid=%s ready on private IPC", proc.pid)

    @staticmethod
    def _await_ready(proc, conn, timeout: float) -> None:
        """Only the spawned child holds the other end of this anonymous pipe."""

        deadline = time.monotonic() + timeout
        while True:
            if not proc.is_alive():
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker exited with code {proc.exitcode} before it was ready"
                )
            if time.monotonic() >= deadline:
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker was not ready within {timeout:g} s"
                )
            if conn.poll(min(0.1, max(0, deadline - time.monotonic()))):
                try:
                    ready = receive_message(conn)
                except EOFError as exc:
                    proc.join(timeout=RPC_STOP_TIMEOUT_SECONDS)
                    raise RpcWorkerStartError(
                        f"the PyTorch RPC worker exited with code {proc.exitcode} before it was ready"
                    ) from exc
                if ready != {"ready": proc.pid}:
                    raise RpcWorkerStartError("invalid PyTorch worker readiness message")
                return

    # ---- models -------------------------------------------------------------

    def load(self, model_id: str, model_spec) -> Dict[str, object]:
        """Build ``model_spec`` in the worker under ``model_id`` (starting the worker first).

        Returns the worker's ``{"model_id", "device", "fingerprint"}``. A model the
        worker refused (a catalog model it cannot build, or a registered class whose
        weights do not load strictly) raises ``catalog.CatalogError``
        (``CatalogUnavailableError`` when a catalog family package is missing) with the
        worker's one-line message; the worker's traceback is logged here at DEBUG. A
        load that fails here, or takes longer than ``RPC_LOAD_TIMEOUT_SECONDS``, is
        abandoned in the worker.
        """

        from ..models import catalog

        model_class = getattr(model_spec, "model_class", None)
        payload = {
            "model_class": model_class,
            "model_path": model_spec.model_path,
            "name": getattr(model_spec, "name", None),
        }
        if isinstance(model_class, str):
            # A catalog name, built by the worker (nxmndr.models.catalog). An unknown or
            # unserved name fails here, before the worker starts.
            catalog.resolve(model_class)
            payload["catalog_args"] = dict(getattr(model_spec, "catalog_args", None) or {})
        elif not callable(model_class):
            raise ValueError(catalog.unknown_class_message(model_class))
        else:
            class_name = f"{model_class.__module__}.{model_class.__qualname__}"
            with self._lock:
                if self._state == "idle":
                    self._model_classes[class_name] = model_class
                elif self._model_classes.get(class_name) is not model_class:
                    class_name = next(
                        (name for name, cls in self._model_classes.items() if cls is model_class),
                        "",
                    )
                    if not class_name:
                        raise ValueError("register custom model classes before the worker starts")
            payload["model_class"] = class_name
        payload["model_path"] = str(payload["model_path"])
        self.ensure_started()
        from .rpc_worker import _rpc_load

        try:
            info = self._call(_rpc_load, (model_id, payload), timeout=RPC_LOAD_TIMEOUT_SECONDS)
        except Exception:
            # A timeout does not cancel the build: it may still finish in
            # the worker. Have the worker drop model_id, now or when its build ends; a
            # retry gets a new model ID, so nothing else names it.
            self.unload(model_id, abandon=True)
            raise
        refused = info.get("refused") if isinstance(info, dict) else None
        if refused:
            logger.debug(
                "the PyTorch RPC worker refused model %s:\n%s",
                model_id,
                refused.get("traceback", ""),
            )
            if refused.get("unavailable"):
                raise catalog.CatalogUnavailableError(refused.get("message", ""))
            raise catalog.CatalogError(refused.get("message", ""))
        return info

    def infer(self, model_id: str, tensor, *, device_hint: str = "cpu", timeout: float):
        """Run the worker's model ``model_id`` (``rpc_worker.UnknownModelError`` if absent)."""

        from .rpc_worker import _rpc_infer

        return self._call(_rpc_infer, (model_id, tensor, device_hint), timeout=timeout)

    def unload(self, model_id: str, *, abandon: bool = False) -> bool:
        """Free the worker's model ``model_id``. False if the worker or the model is gone.

        ``abandon``: the load of ``model_id`` failed in this process; the worker also
        drops the model if that load is still building there or has not arrived.
        """

        if not self.is_alive():
            logger.info("PyTorch RPC worker not running; nothing to free for model %s", model_id)
            return False
        from .rpc_worker import _rpc_unload

        args = (model_id, True) if abandon else (model_id,)
        try:
            return bool(self._call(_rpc_unload, args))
        except Exception:
            logger.warning(
                "freeing model %s in the PyTorch RPC worker failed", model_id, exc_info=True
            )
            return False

    def status(self) -> Dict[str, object]:
        """The worker's ``_rpc_status()``: state, loaded model IDs with device and fingerprint."""

        from .rpc_worker import _rpc_status

        return self._call(_rpc_status, ())

    def _call(self, func, args, **kwargs):
        """Call an allowlisted worker operation over the private pipe.

        ``RpcWorkerStartError`` if the worker is not running; ``kwargs``: ``timeout``.
        """

        with self._calls_lock:
            client = self._client
            if not self.is_alive() or client is None:
                raise RpcWorkerStartError("the PyTorch RPC worker is not running")
            self._calls += 1
        try:
            return client.call(func.__name__, args, **kwargs)
        finally:
            with self._calls_lock:
                self._calls -= 1

    # ---- stop ---------------------------------------------------------------

    def stop(self):
        with self._lock:
            proc, self._proc = self._proc, None
            if self._state == "running":
                self._state = "stopped"
                alive = proc is not None and proc.is_alive()
                with self._calls_lock:  # no _call starts from here on: the state is stopped
                    in_flight = self._calls
                graceful = alive and in_flight == 0
                if alive and self._client is not None:
                    from .rpc_worker import _rpc_stop

                    try:
                        self._client.call(
                            _rpc_stop.__name__,
                            (graceful,),
                            timeout=RPC_STOP_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        logger.warning(
                            "_rpc_stop failed; terminating the PyTorch RPC worker", exc_info=True
                        )
            if self._client is not None:
                self._client.close()
            self._state = "stopped"
        _terminate(proc)
        with _PROCESS_LOCK:
            _process_workers.discard(self)

    def shutdown(self):
        """Stop the worker (``InferenceService.shutdown`` and the server's atexit hook).

        A second call does nothing.
        """

        self.stop()

    def is_alive(self) -> bool:
        proc = self._proc
        return self._state == "running" and proc is not None and proc.is_alive()


def shutdown_process_rpc_worker() -> None:
    """Stop the PyTorch RPC worker this process started, if any (the server's atexit hook)."""

    with _PROCESS_LOCK:
        owners = list(_process_workers)
    for owner in owners:
        owner.shutdown()


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
    "ModelManager",
    "build_model_record",
    "RpcWorkerManager",
    "RpcWorkerStartError",
    "RpcWorkerNotJoinedError",
    "RPC_MODEL_RESOURCE",
    "shutdown_process_rpc_worker",
]
