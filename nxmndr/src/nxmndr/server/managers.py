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
import socket
import threading
import time
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import torch.distributed as dist
import torch.distributed.rpc as torch_rpc

from ..logging import get_logger
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


# The PyTorch RPC worker's names in its two-rank torch.distributed.rpc group.
RPC_DRIVER_NAME = "driver"
RPC_WORKER_NAME = "worker"
# Bound on starting the PyTorch RPC worker, applied twice: to the child's readiness
# handshake and to the torch rendezvous that follows it. It is also the RPC group's
# default timeout, which applies to the calls that set none (unload, status); a load,
# an inference call and the stop have their own bounds. 60 s is torch's default RPC
# timeout (torch.distributed.rpc.constants.DEFAULT_RPC_TIMEOUT_SEC); a warm start took
# 0.5 s on macOS arm64.
RPC_STARTUP_TIMEOUT_SECONDS = 60.0
# Bound on one model load in the worker (_rpc_load). It detects a hung worker; it does
# not bound normal load latency. The largest catalog weights file (regnet_y_128gf,
# 2.55 GB) built in 11 s on macOS arm64, and 300 s also covers reading it at 9 MB/s
# from slow network storage. It is above the QGIS plugin's 120 s deadline for LoadModel
# and OpenSession, so a slow load that is still progressing lands in the model cache for
# the client's next attempt instead of being abandoned.
RPC_LOAD_TIMEOUT_SECONDS = 300.0
# Stopping the worker: the _rpc_stop call and each join of the child get this long, and
# so does the wait for the worker in a graceful RPC shutdown (the values the earlier
# stop() used). A graceful shutdown also waits until no call of this process is in
# flight, whatever its timeout (torch 2.11's TensorPipe agent): up to
# RPC_LOAD_TIMEOUT_SECONDS for a load. So stop() leaves gracefully only when no call is
# in flight, and otherwise at once on both ranks.
RPC_STOP_TIMEOUT_SECONDS = 2.0
# Aux resource name of a PyTorch record's model in the RPC worker.
RPC_MODEL_RESOURCE = "pytorch_rpc_model"
# The address the RPC group listens on (MASTER_ADDR) unless configured: loopback, so
# only processes on this host can reach it. torch.distributed.rpc has no
# authentication and unpickles every call, so a worker on another host needs an
# explicit MASTER_ADDR (an address of this host that it can reach, or 0.0.0.0).
RPC_DEFAULT_ADDR = "127.0.0.1"
# External worker mode (NXMNDR_RPC_EXTERNAL_WORKER=1): the server does not spawn the
# worker; one started with ``python -m nxmndr.server.rpc_worker`` reports to
# MASTER_PORT + 1 and joins the group at MASTER_ADDR:MASTER_PORT. Default port: 29500,
# run_worker's and torch's usual MASTER_PORT.
RPC_EXTERNAL_DEFAULT_PORT = 29500


class RpcWorkerStartError(RuntimeError):
    """The PyTorch RPC worker could not be started, or can no longer be used."""


class RpcWorkerNotJoinedError(RpcWorkerStartError):
    """External worker mode: no worker reported within the startup timeout."""


# torch.distributed and torch.distributed.rpc are process-global, and a process
# cannot join a second group after its first (torch numbers its store prefixes per
# process, so a fresh child never meets a second attempt). So one manager per
# process may enter the torch rendezvous, once.
_PROCESS_LOCK = threading.Lock()
_process_owner: Optional["RpcWorkerManager"] = None


def _free_port(addr: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((addr, 0))
        return sock.getsockname()[1]


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

    if isinstance(proc, _ExternalWorker):
        proc.close()  # not this server's child: drop its connection; it exits on _rpc_stop
        return
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
    master_addr: str,
    master_port: int,
    ready_port: int,
    startup_timeout_s: float,
    device: str = "cpu",
) -> None:
    """Process target of the PyTorch RPC worker.

    Module level so the ``spawn`` start method can pickle it; a nested function
    cannot be pickled. The worker module is imported in the child only. The worker
    starts with no model: the server loads each model by its model ID, on ``device``.
    """

    from .rpc_worker import configure_cudnn, run_worker

    configure_cudnn()  # the server's cuDNN policy; a spawned child does not inherit it
    run_worker(
        RPC_WORKER_NAME,
        None,
        rank=1,
        world_size=2,
        master_addr=master_addr,
        master_port=master_port,
        ready_port=ready_port,
        startup_timeout_s=startup_timeout_s,
        device=device,
    )


class _ExternalWorker:
    """Stands in for the child process when the worker was started by hand (external mode).

    The worker keeps its handshake connection open for its whole life and never writes
    to it after ``ready``, so the connection closing means the worker process exited.
    A worker host that drops off the network closes nothing: RPC calls time out instead.
    This server cannot end that process; it exits on ``_rpc_stop``.
    """

    exitcode = None

    def __init__(self, pid: int, host: str, conn: socket.socket):
        self.pid = pid
        self.host = host
        self._conn = conn
        conn.setblocking(False)

    def is_alive(self) -> bool:
        try:
            # b"": the worker closed it. Stray bytes are dropped, so a later close is seen.
            return self._conn.recv(64) != b""
        except BlockingIOError:
            return True  # open, nothing to read
        except OSError:
            return False  # reset, or closed here

    def close(self) -> None:
        self._conn.close()


def _await_external(addr: str, port: int, timeout: float) -> _ExternalWorker:
    """Wait on ``addr:port + 1`` for an external worker's ``ready <pid>``; answer ``ok``.

    The accepted connection stays open in the returned ``_ExternalWorker``: it tracks
    the worker process.
    """

    try:
        listener = socket.create_server((addr, port + 1))
    except OSError as exc:
        raise RpcWorkerStartError(
            f"cannot listen for the external PyTorch RPC worker on {addr}:{port + 1}: {exc}"
        ) from exc
    deadline = time.monotonic() + timeout
    with listener:
        listener.settimeout(0.1)
        while time.monotonic() < deadline:
            try:
                conn, peer = listener.accept()
            except socket.timeout:
                continue
            conn.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                line = conn.recv(64).decode("ascii", "replace").split()
                if len(line) == 2 and line[0] == "ready" and line[1].isdigit():
                    conn.sendall(b"ok\n")
                    return _ExternalWorker(int(line[1]), peer[0], conn)
            except OSError:
                pass
            conn.close()
    raise RpcWorkerNotJoinedError(
        f"no external PyTorch RPC worker joined at {addr}:{port} within {timeout:g} s; start "
        f"one with: python -m nxmndr.server.rpc_worker --master <this host's address> --port {port}"
    )


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
    """The PyTorch RPC worker of one server: start, per-model load/infer/unload, stop.

    Start order: the child process starts first and reports over a loopback socket
    that it is about to rendezvous; only then does this process join the two-rank
    group as the driver, so both ranks are present. Each phase is bounded by
    ``startup_timeout_s``. A child that exits or never reports fails the load with
    ``RpcWorkerStartError`` and is terminated; a later load tries again. A failure
    inside the torch rendezvous also terminates the child, but torch cannot rejoin a
    group in this process, so later loads fail until the server restarts; so does a
    worker that exits after it started.

    External mode (``external_worker``, or ``NXMNDR_RPC_EXTERNAL_WORKER=1``): no child
    is spawned; the first phase waits for a worker started elsewhere to report to
    ``MASTER_PORT + 1`` (``_await_external``). A wait that times out raises
    ``RpcWorkerNotJoinedError`` and a later load waits again. The worker keeps that
    report connection open, so its closing tells this server that the worker exited.

    The spawned worker builds every model on ``device``. An external worker uses its
    own ``--device``: the server's devices describe the server's host, not the worker's.

    ``stop`` (from ``stop_server``) ends the worker for good: torch keeps per-process
    RPC state that a second group cannot reuse, so no later manager in this process
    can start one, and it is refused with that reason. It does not wait for a call in
    flight (a load still building, an inference call): that call fails.
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
        self._calls = 0  # rpc_sync calls in flight (_call); stop reads it
        self._proc = None
        self._state = "idle"  # idle -> running -> stopped; "broken" after a failed rendezvous
        self._error = ""
        if external_worker is None:
            flag = os.environ.get("NXMNDR_RPC_EXTERNAL_WORKER", "").strip().lower()
            external_worker = flag not in {"", "0", "false", "no", "off"}
        self._external = bool(external_worker)
        # An empty MASTER_ADDR would bind every interface: it means the default.
        self._addr = (master_addr or os.environ.get("MASTER_ADDR", "")).strip() or RPC_DEFAULT_ADDR
        self._port = int(master_port or os.environ.get("MASTER_PORT", "0") or 0)
        self._store = None  # the rendezvous TCPStore server, bound to self._addr
        self.startup_timeout_s = float(startup_timeout_s)
        self.device = str(device or "cpu")

    @property
    def worker_may_be_remote(self) -> bool:
        """External mode on a non-loopback address: the worker may run on another host."""

        return self._external and not _is_loopback(self._addr)

    # ---- start ------------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the worker and join its RPC group once; raise ``RpcWorkerStartError``."""

        with self._lock:
            if self._state == "running":
                proc = self._proc
                if proc is not None and proc.is_alive():
                    return
                if isinstance(proc, _ExternalWorker):
                    gone = (
                        f"the external PyTorch RPC worker (pid {proc.pid} on {proc.host}) "
                        "disconnected"
                    )
                else:
                    gone = (
                        "the PyTorch RPC worker exited (exit code "
                        f"{getattr(proc, 'exitcode', None)})"
                    )
                raise RpcWorkerStartError(
                    f"{gone}; torch cannot rejoin an RPC group in this process, so restart "
                    "the server to use PyTorch models"
                )
            if self._state == "stopped":
                raise RpcWorkerStartError(
                    "the PyTorch RPC worker was stopped; restart the server to use PyTorch models"
                )
            if self._state == "broken":
                raise RpcWorkerStartError(
                    f"the PyTorch RPC group of this server failed to start ({self._error}); "
                    "restart the server to use PyTorch models"
                )
            self._start_locked()

    def _start_locked(self) -> None:
        global _process_owner
        with _PROCESS_LOCK:
            if _process_owner is not None and _process_owner is not self:
                if _process_owner._state in ("stopped", "broken"):
                    raise RpcWorkerStartError(
                        "this process already ran its PyTorch RPC group; "
                        "torch.distributed.rpc keeps per-process state that a second group "
                        "cannot reuse, so generic PyTorch models need a server in a new process"
                    )
                raise RpcWorkerStartError(
                    "another server in this process already runs the PyTorch RPC worker; "
                    "torch allows one RPC group per process"
                )
            if dist.is_initialized() or torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
                raise RpcWorkerStartError(
                    "this process already belongs to a torch.distributed group; the PyTorch "
                    "RPC worker needs a process of its own"
                )
            timeout = self.startup_timeout_s
            addr = self._addr
            if self._external:
                port = self._port or RPC_EXTERNAL_DEFAULT_PORT
                proc = _await_external(addr, port, timeout)
            else:
                port = self._port or _free_port(addr)
                listener = socket.create_server((addr, 0))
                listener.settimeout(0.1)
                # spawn: the child must not inherit this process's gRPC and torch threads.
                proc = multiprocessing.get_context("spawn").Process(
                    target=type(self).worker_target,
                    args=(addr, port, listener.getsockname()[1], timeout, self.device),
                    daemon=True,
                    name="nxmndr-pytorch-rpc-worker",
                )
                try:
                    proc.start()
                    self._await_ready(proc, listener, timeout)
                except BaseException:
                    _terminate(proc)
                    raise
                finally:
                    listener.close()
            _process_owner = self  # from here on this process has used its one attempt
            init_method = f"tcp://{addr}:{port}"
            try:
                # torch's TCPStore server listens on every interface whatever init_method
                # names; bind the rendezvous port on addr here and hand the socket to a
                # multi-tenant TCPStore, which init_process_group and init_rpc then reuse.
                listen_fd = socket.create_server((addr, port)).detach()
                self._store = dist.TCPStore(
                    addr,
                    port,
                    2,
                    True,
                    timedelta(seconds=timeout),
                    wait_for_workers=False,
                    multi_tenant=True,
                    master_listen_fd=listen_fd,
                )
                dist.init_process_group(
                    backend="gloo",
                    rank=0,
                    world_size=2,
                    init_method=init_method,
                    timeout=timedelta(seconds=timeout),
                )
                opts = torch_rpc.TensorPipeRpcBackendOptions(
                    init_method=init_method, rpc_timeout=timeout
                )
                torch_rpc.init_rpc(RPC_DRIVER_NAME, rank=0, world_size=2, rpc_backend_options=opts)
            except BaseException as exc:
                self._state = "broken"
                self._error = f"{type(exc).__name__}: {exc}"
                _terminate(proc)
                self._leave_group(graceful=False)
                self._store = None
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker did not complete the rendezvous at {init_method} "
                    f"within {timeout:g} s: {self._error}"
                ) from exc
            self._proc = proc
            self._state = "running"
        logger.info("PyTorch RPC worker pid=%s joined the RPC group at %s", proc.pid, init_method)

    @staticmethod
    def _await_ready(proc, listener, timeout: float) -> None:
        """Wait for the child's ``ready <pid>`` line; fail fast if it exits."""

        deadline = time.monotonic() + timeout
        while True:
            # Checked on every pass, so a stray local connection cannot extend the wait.
            if not proc.is_alive():
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker exited with code {proc.exitcode} before it "
                    "was ready to join its RPC group"
                )
            if time.monotonic() >= deadline:
                raise RpcWorkerStartError(
                    f"the PyTorch RPC worker was not ready to join its RPC group within "
                    f"{timeout:g} s"
                )
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(max(0.1, deadline - time.monotonic()))
                try:
                    line = conn.recv(64).decode("ascii", "replace").split()
                except OSError:
                    line = []
            if line == ["ready", str(proc.pid)]:
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
        self.ensure_started()
        from .rpc_worker import _rpc_load

        try:
            info = self._call(_rpc_load, (model_id, payload), timeout=RPC_LOAD_TIMEOUT_SECONDS)
        except Exception:
            # torch does not cancel a call that timed out: the build may still finish in
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
            logger.warning("freeing model %s in the PyTorch RPC worker failed", model_id, exc_info=True)
            return False

    def status(self) -> Dict[str, object]:
        """The worker's ``_rpc_status()``: state, loaded model IDs with device and fingerprint."""

        from .rpc_worker import _rpc_status

        return self._call(_rpc_status, ())

    def _call(self, func, args, **kwargs):
        """``rpc_sync`` of the worker function ``func``, counted while it runs (``stop``).

        ``RpcWorkerStartError`` if the worker is not running; ``kwargs``: ``timeout``.
        """

        with self._calls_lock:
            if not self.is_alive():
                raise RpcWorkerStartError("the PyTorch RPC worker is not running")
            self._calls += 1
        try:
            return torch_rpc.rpc_sync(RPC_WORKER_NAME, func, args=args, **kwargs)
        finally:
            with self._calls_lock:
                self._calls -= 1

    # ---- stop ---------------------------------------------------------------

    @staticmethod
    def _leave_group(*, graceful: bool) -> None:
        try:
            if torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
                torch_rpc.shutdown(graceful=graceful, timeout=RPC_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("PyTorch RPC shutdown failed", exc_info=True)
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            logger.warning("destroying the torch.distributed process group failed", exc_info=True)

    def stop(self):
        with self._lock:
            proc, self._proc = self._proc, None
            if self._state == "running":
                self._state = "stopped"
                alive = proc is not None and proc.is_alive()
                with self._calls_lock:  # no _call starts from here on: the state is stopped
                    in_flight = self._calls
                # A graceful shutdown would wait for the calls in flight (see
                # RPC_STOP_TIMEOUT_SECONDS). The worker ends either way, so with a call in
                # flight both ranks leave at once and that call fails.
                graceful = alive and in_flight == 0
                if alive:
                    from .rpc_worker import _rpc_stop

                    try:
                        torch_rpc.rpc_sync(
                            RPC_WORKER_NAME,
                            _rpc_stop,
                            args=(graceful,),
                            timeout=RPC_STOP_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        graceful = False
                        logger.warning(
                            "_rpc_stop failed; terminating the PyTorch RPC worker", exc_info=True
                        )
                self._leave_group(graceful=graceful)
            self._store = None
        _terminate(proc)

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

    owner = _process_owner
    if owner is not None:
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
