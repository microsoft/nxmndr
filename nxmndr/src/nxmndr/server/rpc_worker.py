# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The private PyTorch worker: a second process that holds generic PyTorch models.

The server's ``RpcWorkerManager`` (``nxmndr.server.managers``) starts one worker per
server process and calls a fixed set of operations over an inherited anonymous
pipe. Messages contain JSON and tensor bytes, never Python objects or imports.

Models are routed by identity: the worker keeps a registry keyed by the model cache
record's model ID. ``_rpc_load(model_id, spec)`` builds one model, ``_rpc_infer``
runs exactly that model (``UnknownModelError`` if it is not loaded), ``_rpc_unload``
frees it, ``_rpc_stop`` ends the worker. One ID never names two models. A load the
server stopped waiting for is abandoned (``_rpc_unload(model_id, True)``): torch does
not cancel the call, so the worker drops that model when its build ends.

A catalog model that cannot be built as requested, or a registered class whose weights
do not load strictly, is returned as a refusal. ``RpcWorkerManager.load`` raises it
with the one-line message and logs the traceback at DEBUG.

External Torch RPC is disabled. Remote inference uses the nxmndr gRPC service.
"""

import argparse
import hashlib
import itertools
import os
import threading
import time
import traceback
from typing import Optional

import torch

from ..logging import get_logger

logger = get_logger(__name__)

# The single slot of ``TorchRpcInferenceProvider``. The server's model IDs
# are 32-character hex strings, so they never collide with it.
DEFAULT_MODEL_ID = "default"


class UnknownModelError(LookupError):
    """``_rpc_infer`` named a model ID that this worker has not loaded (or has unloaded)."""


class _LoadedModel:
    __slots__ = ("model", "device", "fingerprint")

    def __init__(self, model, device: str, fingerprint: str):
        self.model = model
        self.device = device
        self.fingerprint = fingerprint


# Worker operations run in different threads.
_worker_state = {
    "models": {},  # model_id -> _LoadedModel
    "loading": set(),  # model IDs whose _rpc_load is running
    "abandoned": set(),  # model IDs whose load the server gave up on (_rpc_unload)
    "load_calls": 0,  # _rpc_load calls served, refused ones included (_rpc_status)
    "spec": None,
    "initialized": False,
    "stop_requested": False,
    "rpc_initialized": False,
}
_models_lock = threading.Lock()
_worker_model_classes: dict[str, type] = {}


def _get_worker_state():
    """Get the global worker state."""
    return _worker_state


def _spec_value(spec, name, default=None):
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _load_model(model_spec, device: str = "cpu"):
    """Build ``model_spec``: a registered class, or a catalog name (``nxmndr.models.catalog``).

    Returns the model on ``device`` and its ``_fingerprint``, taken on the CPU before
    the move: on a GPU the hash would first copy every weight back to the host.
    """

    model_class = _spec_value(model_spec, "model_class")
    if isinstance(model_class, str):
        model_class = _worker_model_classes.get(model_class, model_class)
    target = "cpu"
    if device and device.startswith("cuda") and torch.cuda.is_available():
        target = device
    elif device == "mps" and torch.backends.mps.is_available():
        target = device
    model_path = _spec_value(model_spec, "model_path")
    if isinstance(model_class, str):
        from ..models import catalog

        model = catalog.build(model_class, model_path, _spec_value(model_spec, "catalog_args"))
    elif callable(model_class):
        model = model_class()
        # Loaded on the CPU, then moved once below with the model (as catalog models are),
        # rather than read onto the GPU, copied into CPU parameters and moved back.
        _load_state_strict(model, model_class, model_path)
    else:
        raise TypeError(
            f"model_class {model_class!r} is not a PyTorch model class or a catalog name; the "
            "server resolves ModelSpec.model_class by the names registered with "
            "register_pytorch_model, then by nxmndr.models.catalog"
        )
    fingerprint = _fingerprint(model)
    if target != "cpu":
        model.to(target)
    model.eval()
    return model, fingerprint


def _load_state_strict(model, model_class, model_path) -> None:
    """Load the weights file ``model_path`` (on the CPU) into ``model`` with strict=True,
    as catalog models load.

    A file that lacks entries would leave those layers randomly initialized and the
    model would serve their output. Every failure is a refusal: ``catalog.CatalogError``
    with one line, which the server answers with INVALID_ARGUMENT. It names the file
    that is missing or that ``torch.load`` refuses, the file that is not a mapping of
    entry names (with its type or first keys), the first entry that differs from the
    model with the counts (a checkpoint that nests the state dict lacks every entry), or
    the first error of ``load_state_dict``. Values that are not tensors are accepted
    when ``load_state_dict`` accepts them: the class's own ``state_dict()`` can hold
    ``get_extra_state()`` or a quantized layer's packed parameters.
    """

    from ..models import catalog

    name = getattr(model_class, "__qualname__", str(model_class))
    state = catalog._load_state(name, model_path, tensors_only=False)
    mismatch = catalog._mismatch(model.state_dict(), state)
    if mismatch:
        raise catalog.CatalogError(
            f"weights file {model_path} does not match {name} (strict load): {mismatch}"
        )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise catalog.CatalogError(
            f"weights file {model_path} does not load into {name}: {catalog._first_line(exc)}"
        ) from exc


def configure_cudnn() -> None:
    """Disable cuDNN when the bundled cuDNN cannot serve the attached GPUs.

    torch >= 2.9 ships cuDNN >= 9.11, which has no kernels for SM < 7.5 (Volta and older);
    convolutions then fail with "unable to find an engine". Falling back to the native CUDA
    kernels keeps inference on the GPU. Override with NXMNDR_CUDNN=0|1.

    ``nxmndr-server`` (``server.main``) calls it at startup, and so does each PyTorch
    RPC worker: the setting is per process, and a spawned worker starts with torch's
    default.
    """
    override = os.environ.get("NXMNDR_CUDNN")
    if override is not None:
        torch.backends.cudnn.enabled = override.strip() not in {"0", "false", "no", "off"}
        logger.info(
            "cuDNN %s by NXMNDR_CUDNN", "enabled" if torch.backends.cudnn.enabled else "disabled"
        )
        return
    try:
        if not torch.cuda.is_available():
            return
        compiled = torch._C._cudnn.getCompileVersion() if hasattr(torch._C, "_cudnn") else None
        min_cc = min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
        if compiled and (compiled[0], compiled[1]) >= (9, 11) and min_cc < (7, 5):
            torch.backends.cudnn.enabled = False
            logger.warning(
                "cuDNN %s does not support SM %d.%d GPUs; cuDNN disabled, using native CUDA kernels",
                ".".join(str(v) for v in compiled),
                min_cc[0],
                min_cc[1],
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cuDNN capability check skipped: %s", exc)


def _model_device(model) -> str:
    """The device of the model's first parameter, else of its first buffer, else cpu."""

    try:
        return str(next(itertools.chain(model.parameters(), model.buffers())).device)
    except Exception:  # neither, or not a torch module
        return "cpu"


def _to_cpu(value):
    """``value`` with every tensor in it on the CPU, also inside dicts, lists and tuples.

    The private channel carries CPU tensor bytes.
    """

    if torch.is_tensor(value):
        return value.detach().cpu() if value.device.type != "cpu" else value
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _fingerprint(model) -> str:
    """sha256 of the model class and every state-dict entry (name, dtype, shape, bytes).

    An entry that is not a tensor (``get_extra_state()``, a quantized layer's packed
    parameters) adds the tensors inside its lists, tuples and dicts and the repr of
    every other value.
    """

    digest = hashlib.sha256()
    cls = type(model)
    digest.update(f"{cls.__module__}.{cls.__qualname__}".encode("utf-8"))
    digest.update(str(getattr(model, "catalog_name", "")).encode("utf-8"))
    for name, value in model.state_dict().items():
        _digest_entry(digest, name, value)
    return "sha256:" + digest.hexdigest()


def _digest_entry(digest, name: str, value) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.is_quantized:  # a byte view of it does not hold its values
            tensor = tensor.dequantize()
        tensor = tensor.contiguous()
        digest.update(f"|{name}|{value.dtype}|{tuple(tensor.shape)}|".encode("utf-8"))
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (list, tuple, dict)):
        digest.update(f"|{name}|{type(value).__name__}|{len(value)}|".encode("utf-8"))
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            _digest_entry(digest, f"{name}[{key!r}]", item)
    else:
        digest.update(f"|{name}|{value!r}|".encode("utf-8"))


def _register(model_id: str, model, fingerprint: str) -> Optional[dict]:
    """Keep ``model`` under ``model_id``; None if the server abandoned that load."""

    entry = _LoadedModel(model, _model_device(model), fingerprint)
    with _models_lock:
        if model_id in _worker_state["abandoned"]:  # the server gave up on this load
            _worker_state["abandoned"].discard(model_id)
            return None
        _worker_state["models"][model_id] = entry
    info = {"model_id": model_id, "device": entry.device, "fingerprint": entry.fingerprint}
    if hasattr(model, "catalog_info"):  # a catalog model: its family, class and arguments
        info["catalog"] = model.catalog_info()
    return info


def _rpc_ping():
    """Ping function to check if worker is alive."""
    return {"status": "ok", "timestamp": time.time()}


def _catalog_refusal(exc: Exception):
    """The refusal ``_rpc_load`` returns for a catalog build error, or None for any other."""

    from ..models import catalog

    if not isinstance(exc, catalog.CatalogError):
        return None
    return {
        "message": str(exc),
        "unavailable": isinstance(exc, catalog.CatalogUnavailableError),
        "traceback": traceback.format_exc(),
    }


def _rpc_load(model_id, spec, device=None):
    """Build the model of ``spec`` under ``model_id``; return its device and fingerprint.

    ``device`` defaults to the worker's own (``run_worker(device=...)``, else cpu).

    ``spec`` has ``model_class`` (a locally allowlisted class name, or a catalog name with its
    ``catalog_args``), ``model_path`` and ``name``, as a mapping or as attributes. A
    model ID that is loaded or loading is refused, so one ID never names two models.
    A catalog model's result also has ``catalog`` (``CatalogModel.catalog_info``).
    A catalog model that cannot be built, or a registered class whose weights do not
    load strictly, is not loaded; the result is then
    ``{"model_id", "refused": {"message", "unavailable", "traceback"}}``, with the
    ``CatalogError`` message and whether its family package is missing.
    A load the server abandoned (``_rpc_unload(model_id, True)``) before it arrived or
    while it built keeps nothing; the result is then ``{"model_id", "abandoned": True}``.
    """

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    with _models_lock:
        _worker_state["load_calls"] += 1
        if model_id in _worker_state["abandoned"]:  # abandoned before this call arrived
            _worker_state["abandoned"].discard(model_id)
            return {"model_id": model_id, "abandoned": True}
        if model_id in _worker_state["models"] or model_id in _worker_state["loading"]:
            raise ValueError(f"model {model_id} is already loaded in this PyTorch RPC worker")
        _worker_state["loading"].add(model_id)
    device = device or _worker_state.get("device") or "cpu"
    try:
        try:
            model, fingerprint = _load_model(spec, device)
        except Exception as exc:
            refusal = _catalog_refusal(exc)
            if refusal is None:
                raise
            return {"model_id": model_id, "refused": refusal}
        info = _register(model_id, model, fingerprint)
    finally:
        with _models_lock:
            _worker_state["loading"].discard(model_id)
    if info is None:
        device = _model_device(model)
        del model
        _release(device)
        logger.info("Dropped model %s: the server gave up on its load", model_id)
        return {"model_id": model_id, "abandoned": True}
    logger.info(
        "Loaded model %s (%s) on %s, %s",
        model_id,
        _spec_value(spec, "name") or "unnamed",
        info["device"],
        info["fingerprint"],
    )
    return info


def _release(device: str) -> None:
    """Return cached CUDA memory after a model on ``device`` was dropped."""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rpc_unload(model_id, abandon=False):
    """Free the model of ``model_id``. True if it was loaded.

    ``abandon``: the server stopped waiting for the load of ``model_id``. If that load
    is still building or has not arrived yet, ``_rpc_load`` drops its model. An ID
    whose load had already failed stays in the abandoned set: one 32-character entry
    per such attempt, until the worker ends with its server.
    """

    with _models_lock:
        entry = _worker_state["models"].pop(model_id, None)
        if entry is None and abandon:
            _worker_state["abandoned"].add(model_id)
    if entry is None:
        return False
    device = entry.device
    del entry
    _release(device)
    logger.info("Unloaded model %s", model_id)
    return True


def _rpc_load_model(spec_dict):
    """Load a locally allowlisted model into the provider's default slot."""
    logger.info(f"Loading model on RPC worker: {spec_dict.get('name', 'unnamed')}")

    try:
        state = _get_worker_state()

        model_name = spec_dict.get("name", "unnamed")
        model, fingerprint = _load_model(spec_dict, state.get("device") or "cpu")

        # Store in the provider's slot
        _register(DEFAULT_MODEL_ID, model, fingerprint)
        state["spec"] = spec_dict
        state["initialized"] = True

        logger.info(f"Successfully loaded model {model_name} on RPC worker")
        return {"success": True, "message": f"Model {model_name} loaded successfully"}

    except Exception as e:
        error_msg = f"Failed to load model on RPC worker: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


def _rpc_infer(model_id, tensor, device_hint="cpu"):
    with _models_lock:
        entry = _worker_state["models"].get(model_id)
    if entry is None:
        raise UnknownModelError(f"model {model_id} is not loaded in the PyTorch RPC worker")
    model = entry.model
    model_device = torch.device(_model_device(model))
    if model_device != tensor.device:  # the model's device in this worker decides
        tensor = tensor.to(model_device)
    with torch.no_grad():
        out = model(tensor)
    return _to_cpu(out)


def _rpc_stop(graceful=True):
    """End the worker; ``graceful`` is retained for compatibility with local callers."""

    state = _get_worker_state()
    state["stop_graceful"] = bool(graceful)
    state["stop_requested"] = True
    return {"status": "stopping"}


def _rpc_status():
    state = _get_worker_state()
    with _models_lock:
        models = {
            mid: {"device": entry.device, "fingerprint": entry.fingerprint}
            for mid, entry in state["models"].items()
        }
        loading = sorted(state["loading"])
    return {
        "status": "ok" if state.get("rpc_initialized") else "initializing",
        "model_loaded": bool(models),
        "models": models,
        "loading": loading,
        "load_calls": state.get("load_calls", 0),
        "stop_requested": state.get("stop_requested"),
        "pid": os.getpid(),
    }


def run_ipc_worker(conn, model_classes, *, device="cpu"):
    """Serve only our parent's inherited pipe; never open a network endpoint."""

    from .worker_ipc import receive_message, send_message

    global _worker_model_classes
    _worker_model_classes = model_classes
    state = _get_worker_state()
    state["device"] = device
    state["rpc_initialized"] = True
    state["stop_requested"] = False
    operations = {
        fn.__name__: fn
        for fn in (
            _rpc_load,
            _rpc_load_model,
            _rpc_infer,
            _rpc_unload,
            _rpc_status,
            _rpc_ping,
            _rpc_stop,
        )
    }
    send_lock = threading.Lock()

    def respond(request):
        try:
            operation = operations.get(request["operation"])
            if operation is None:
                raise ValueError("unknown worker operation")
            reply = {"id": request["id"], "result": operation(*request["args"])}
            with send_lock:
                send_message(conn, reply)
        except Exception as exc:
            logger.exception("PyTorch worker operation failed")
            try:
                with send_lock:
                    send_message(
                        conn, {"id": request["id"], "error": f"{type(exc).__name__}: {exc}"}
                    )
            except (OSError, EOFError):
                logger.debug("Parent closed the worker pipe")

    send_message(conn, {"ready": os.getpid()})
    try:
        while not state["stop_requested"]:
            request = receive_message(conn)
            if not isinstance(request, dict) or set(request) != {"id", "operation", "args"}:
                raise ValueError("invalid worker request")
            if type(request["id"]) is not int or not isinstance(request["operation"], str):
                raise ValueError("invalid worker request identity")
            if not isinstance(request["args"], (tuple, list)):
                raise ValueError("invalid worker arguments")
            if request["operation"] == "_rpc_stop":
                respond(request)
                break
            # A hung model must not hold up stop, status, or abandonment. Daemon
            # threads exit with the worker when the parent's pipe closes.
            threading.Thread(target=respond, args=(request,), daemon=True).start()
    except EOFError:
        logger.info("Parent closed the worker pipe; exiting")
    finally:
        conn.close()
        state["rpc_initialized"] = False


def run_worker(*args, **kwargs):
    """Legacy network worker entry point: fail closed instead of starting Torch RPC."""

    from .managers import EXTERNAL_RPC_ERROR

    raise RuntimeError(EXTERNAL_RPC_ERROR)


def main(argv=None) -> int:
    """External Torch RPC is disabled; use nxmndr-server's gRPC API instead."""

    from .managers import EXTERNAL_RPC_ERROR, RPC_EXTERNAL_DEFAULT_PORT, RPC_STARTUP_TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(
        prog="python -m nxmndr.server.rpc_worker", description=main.__doc__
    )
    parser.add_argument("--master", required=True, help="the nxmndr server's address")
    parser.add_argument(
        "--port", type=int, default=RPC_EXTERNAL_DEFAULT_PORT, help="its MASTER_PORT"
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or mps")
    parser.add_argument(
        "--timeout", type=float, default=RPC_STARTUP_TIMEOUT_SECONDS, help="rendezvous bound, s"
    )
    parser.parse_args(argv)
    parser.error(EXTERNAL_RPC_ERROR)


if __name__ == "__main__":
    from nxmndr.server.rpc_worker import main as _main

    raise SystemExit(_main())
