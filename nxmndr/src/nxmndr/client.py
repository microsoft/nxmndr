# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""High-level gRPC client for the nxmndr inference service.

Host-side module: imports only the standard library, NumPy (1.x or 2.x), grpc and
protobuf, and the generated bindings. It never imports the ML runtime.

Streaming context v1: every message of a tile carries ``context["tile_id"]`` and,
for session streams, ``context["session_id"]``. Per-tile inference options travel
as ``context["opt.<name>"]`` on the tile's first message only; unprefixed keys are
reserved. The declared dtype always describes the bytes actually sent.

Transport follows the endpoint (``parse_endpoint``): ``https://host[:port]`` opens a
TLS channel (default port 443; the server certificate is checked against ``host``
with the default root certificates unless ``credentials`` are given);
``http://host[:port]`` (default port 80), ``grpc://host[:port]`` and a bare
``host:port`` open a plaintext channel. An https endpoint never falls back to
plaintext.
"""

from __future__ import annotations

import re
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import grpc
import numpy as np
import time

from .inference import inference_pb2, inference_pb2_grpc

# Match server's message size limits
GRPC_MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128MB
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]


# Stream context v1 (see module docstring).
STREAM_CONTEXT_CAPABILITY = "stream_context_version"
STREAM_CONTEXT_VERSION = "1"
CONTEXT_SESSION_ID = "session_id"
CONTEXT_TILE_ID = "tile_id"
RESERVED_CONTEXT_KEYS = ("session_id", "tile_id", "batch_size")
TILE_OPTION_PREFIX = "opt."
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# How often a ``cancel_fn`` callable is polled while a stream is blocked.
CANCEL_POLL_SECONDS = 0.05

# Unary calls are retried only on UNAVAILABLE, the code gRPC documents as transient
# and fixed by retrying the failing call with a backoff (the server could not be
# reached or stopped serving). With any other code the server ran the request and
# answered (INTERNAL is a failed inference), or the deadline passed while it may
# still be running, so a retry would run the request again.
_RETRYABLE_CODES = frozenset({grpc.StatusCode.UNAVAILABLE})


class InferenceGrpcError(RuntimeError):
    """Raised when a gRPC request to the inference service fails.

    ``code`` is the gRPC status code when the server returned one.
    """

    def __init__(self, message: str, *, code: Optional[grpc.StatusCode] = None) -> None:
        super().__init__(message)
        self.code = code


def _rpc_error_parts(err) -> tuple:
    details = getattr(err, "details", None)
    message = details() if callable(details) else str(details or err)
    code_fn = getattr(err, "code", None)
    code = code_fn() if callable(code_fn) else None
    return message, code


@dataclass(frozen=True)
class Endpoint:
    """How a client endpoint is dialed: gRPC ``target`` and whether TLS is used."""

    target: str
    tls: bool
    host: str


_PLAINTEXT_SCHEMES = {"http": 80, "grpc": None}


def _split_host_port(authority: str, endpoint: str):
    if authority.startswith("["):  # [IPv6]:port
        close = authority.find("]")
        if close < 0:
            raise ValueError(f"endpoint {endpoint!r}: unclosed '[' in the host")
        host, rest = authority[: close + 1], authority[close + 1 :]
        if rest and not rest.startswith(":"):
            raise ValueError(f"endpoint {endpoint!r}: unexpected {rest!r} after the host")
        port = rest[1:] if rest else ""
    else:
        host, _, port = authority.partition(":")
    if not host or host == "[]":
        raise ValueError(f"endpoint {endpoint!r} has no host")
    if port and not port.isdigit():
        raise ValueError(f"endpoint {endpoint!r}: port {port!r} is not a number")
    return host, (int(port) if port else None)


def parse_endpoint(endpoint: str) -> Endpoint:
    """The gRPC target and transport an endpoint names (see the module docstring).

    ``https://`` means TLS, ``http://`` and ``grpc://`` mean plaintext; a trailing
    ``/`` is dropped and any other path is an error. An endpoint without one of
    these schemes (``host:port``, or a gRPC target such as ``unix:///path``) is
    dialed as given, in plaintext.
    """

    text = str(endpoint or "").strip()
    if not text:
        raise ValueError("endpoint is required")
    scheme, sep, rest = text.partition("://")
    scheme = scheme.lower()
    if not sep or (scheme != "https" and scheme not in _PLAINTEXT_SCHEMES):
        return Endpoint(target=text, tls=False, host="")
    authority, _, path = rest.partition("/")
    if path.strip("/") or "?" in authority or "#" in authority or "@" in authority:
        raise ValueError(
            f"endpoint {endpoint!r}: a gRPC endpoint is {scheme}://host[:port], without "
            "a path, query or user"
        )
    host, port = _split_host_port(authority, endpoint)
    if port is None:
        port = 443 if scheme == "https" else _PLAINTEXT_SCHEMES[scheme]
    target = f"{host}:{port}" if port is not None else host
    return Endpoint(target=target, tls=scheme == "https", host=host.strip("[]"))


def validate_option_name(name: str) -> str:
    """Per-tile option names: ``^[a-z][a-z0-9_]*$`` and not a reserved context key."""

    if not isinstance(name, str) or not _OPTION_NAME.match(name):
        raise ValueError(f"option name {name!r} must match {_OPTION_NAME.pattern}")
    if name in RESERVED_CONTEXT_KEYS:
        raise ValueError(f"option name {name!r} is a reserved context key")
    return name


def encode_tile_context(
    *,
    tile_id: str,
    session_id: str = "",
    options: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """``StreamPredictRequest.context`` for one message of a tile.

    Pass ``options`` only for the tile's first message. ``None`` values are skipped;
    other values are sent as ``str(value)``.
    """

    if not tile_id:
        raise ValueError("tile_id is required")
    ctx: Dict[str, str] = {CONTEXT_TILE_ID: str(tile_id)}
    if session_id:
        ctx[CONTEXT_SESSION_ID] = str(session_id)
    for name, value in (options or {}).items():
        if value is None:
            continue
        ctx[TILE_OPTION_PREFIX + validate_option_name(name)] = str(value)
    return ctx


def prepare_tensor(sample, dtype: Optional[str] = None) -> np.ndarray:
    """C-contiguous little-endian array whose dtype is exactly what will be declared.

    A ``dtype`` override casts the data when NumPy can do so without loss
    (``np.can_cast(..., "safe")``); anything else is rejected, so bytes are never
    relabelled with a dtype they do not have.
    """

    arr = np.asarray(sample)
    if dtype is not None:
        target = np.dtype(dtype)
        if arr.dtype.newbyteorder("=") != target.newbyteorder("="):
            if not np.can_cast(arr.dtype, target, casting="safe"):
                raise ValueError(
                    f"cannot send {arr.dtype} data as {target}: the cast is not lossless; "
                    "convert the array explicitly before streaming"
                )
            arr = arr.astype(target)
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return np.ascontiguousarray(arr)


@dataclass
class PredictResult:
    """Container for Predict responses returned by the inference service."""

    output: bytes
    shape: Sequence[int]
    dtype: str
    metadata: Mapping[str, str]


@dataclass
class LoadModelResult:
    """A LoadModel answer: the model ID and the server's effective metadata.

    ``effective_metadata`` holds, for example, ``model_cache_hit``, ``capability.sam``
    (SAM 3 models) and ``capability.instances`` / ``capability.window`` (instance
    models such as ``ultralytics_yolo``), so a host can check prompts or tiling against
    the loaded model before it streams.
    """

    model_id: str
    effective_metadata: Mapping[str, str]
    message: str = ""


@dataclass
class ModelRegistryEntry:
    """Container for model registry entries returned by the inference service."""

    entry_id: str
    model_id: str
    display_name: str
    source: str
    task: str
    format: str
    metadata: Mapping[str, str]
    last_used_epoch_ms: int


class InferenceGrpcClient:
    """Simple wrapper around the nxmndr inference gRPC service.

    Maintains one persistent channel, created at first use; gRPC reconnects it after
    a failure and only ``close()`` closes it. A unary call is retried, up to
    ``max_attempts`` times, only when it fails with UNAVAILABLE. The endpoint
    selects the transport (``parse_endpoint``): ``https://`` is TLS, with
    ``credentials`` if given, else ``grpc.ssl_channel_credentials()`` (default
    roots); ``credentials`` given with a plaintext endpoint also select TLS.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: Optional[float] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        parsed = parse_endpoint(endpoint)
        self._endpoint = parsed.target
        self._timeout = timeout
        if credentials is None and parsed.tls:
            credentials = grpc.ssl_channel_credentials()
        self._credentials = credentials
        self._max_attempts = max(1, int(max_attempts or 1))
        self._backoff_seconds = max(0.0, float(backoff_seconds or 0.0))
        # Persistent channel (created lazily)
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[inference_pb2_grpc.InferenceServiceStub] = None

    def _get_channel(self) -> grpc.Channel:
        """Get or create a persistent channel."""
        if self._channel is None:
            self._channel = self._create_channel()
        return self._channel

    def _get_stub(self) -> inference_pb2_grpc.InferenceServiceStub:
        """Get or create the service stub."""
        if self._stub is None:
            self._stub = inference_pb2_grpc.InferenceServiceStub(self._get_channel())
        return self._stub

    def _reset_channel(self) -> None:
        """Close the channel; the next call opens a new one."""
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
        self._channel = None
        self._stub = None

    def _create_channel(self) -> grpc.Channel:
        if self._credentials is not None:
            return grpc.secure_channel(self._endpoint, self._credentials, options=GRPC_OPTIONS)
        return grpc.insecure_channel(self._endpoint, options=GRPC_OPTIONS)

    def _with_retry(self, func):
        last_err = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return func()
            except grpc.RpcError as err:  # pragma: no cover - network dependent
                last_err = err
                _, code = _rpc_error_parts(err)
                if code not in _RETRYABLE_CODES:
                    break
                # The channel reconnects by itself; closing it would also end every
                # other call (a running stream) on this client.
                if attempt >= self._max_attempts:
                    break
                time.sleep(self._backoff_seconds * attempt)
        if last_err:
            message, code = _rpc_error_parts(last_err)
            raise InferenceGrpcError(message, code=code) from last_err
        raise InferenceGrpcError("request failed")

    def close(self) -> None:
        """Close the persistent channel."""
        self._reset_channel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def load_model(
        self,
        model_id: str,
        spec: Mapping[str, str],
    ) -> str:
        """Load a model; return the server-assigned model_id (see :meth:`load_model_result`)."""

        return self.load_model_result(model_id, spec).model_id

    def load_model_result(
        self,
        model_id: str,
        spec,
    ) -> LoadModelResult:
        """Load a model on the remote inference service.

        Args:
            model_id: Unique identifier for the model
            spec: Model specification dict with keys like:
                - format: Model format (huggingface, pytorch, onnx, torchhub)
                - source: Model source (repo_id for huggingface, path for others)
                - task: Task type (segmentation, detection, classification, etc)
                - name: Display name for the model
                or an ``inference_pb2.ModelSpec``, sent as given except that a
                non-empty ``model_id`` replaces its ``model_id``.

        Returns:
            A :class:`LoadModelResult`: the server-assigned model_id for use in
            predict calls, and the server's ``effective_metadata``.
        """
        if isinstance(spec, inference_pb2.ModelSpec):
            given = inference_pb2.ModelSpec()
            given.CopyFrom(spec)
            if model_id:
                given.model_id = str(model_id)
            return self._load_model_request(inference_pb2.LoadModelRequest(spec=given))
        if not model_id:
            raise ValueError("model_id is required")
        if not spec:
            raise ValueError("spec is required")

        format_map = {
            "huggingface": inference_pb2.HUGGINGFACE,
            "pytorch": inference_pb2.PYTORCH,
            "onnx": inference_pb2.ONNX,
            "torchhub": inference_pb2.TORCHHUB,
        }

        task_map = {
            "segmentation": inference_pb2.SEGMENTATION,
            "detection": inference_pb2.OBJECT_DETECTION,
            "object_detection": inference_pb2.OBJECT_DETECTION,
            "classification": inference_pb2.CLASSIFICATION,
            "embedding": inference_pb2.EMBEDDING,
        }

        format_str = str(spec.get("format", "")).lower()
        task_str = str(spec.get("task", "")).lower()

        format_enum = format_map.get(format_str, inference_pb2.HUGGINGFACE)
        task_enum = task_map.get(task_str, inference_pb2.TASK_TYPE_UNSPECIFIED)

        source_value = str(spec.get("source", ""))
        name_value = str(spec.get("name", source_value))
        token_value = str(spec.get("token", ""))

        model_class_value = str(spec.get("model_class", ""))

        model_spec = inference_pb2.ModelSpec(
            model_id=str(model_id),
            format=format_enum,
            source=source_value,
            task=task_enum,
            name=name_value,
            model_class=model_class_value,
        )

        if token_value:
            model_spec.token = token_value

        return self._load_model_request(inference_pb2.LoadModelRequest(spec=model_spec))

    def _load_model_request(self, request: inference_pb2.LoadModelRequest) -> LoadModelResult:
        def _call():
            stub = self._get_stub()
            return stub.LoadModel(request, timeout=self._timeout)

        response = self._with_retry(_call)

        if not response.success:
            raise InferenceGrpcError(f"LoadModel failed: {response.message}")

        return LoadModelResult(
            model_id=response.model_id,
            effective_metadata={str(e.key): str(e.value) for e in response.effective_metadata},
            message=str(response.message or ""),
        )

    def predict(
        self,
        model_id: str,
        tensor: np.ndarray,
        *,
        options: Optional[Mapping[str, str]] = None,
    ) -> PredictResult:
        """Run a single Predict call against the remote inference service."""

        if not model_id:
            raise ValueError("model_id is required")
        if tensor is None:
            raise ValueError("tensor is required")

        array = prepare_tensor(tensor)
        request = inference_pb2.PredictRequest(
            model_id=str(model_id),
            input=array.tobytes(),
            shape=[int(dim) for dim in array.shape],
            dtype=array.dtype.name,
        )

        if options:
            for key, value in options.items():
                if value is None:
                    continue
                request.options[str(key)] = str(value)

        def _call():
            stub = self._get_stub()
            return stub.Predict(request, timeout=self._timeout)

        response = self._with_retry(_call)

        return PredictResult(
            output=bytes(response.output or b""),
            shape=tuple(int(dim) for dim in response.shape),
            dtype=response.dtype or str(array.dtype),
            metadata=dict(response.metadata),
        )

    # ---- Streaming & Sessions ----
    def capabilities(self) -> Dict[str, str]:
        """Server capabilities as a dict (e.g. ``stream_context_version``, stream limits)."""

        def _call():
            stub = self._get_stub()
            return stub.Capabilities(inference_pb2.CapabilitiesRequest(), timeout=self._timeout)

        response = self._with_retry(_call)
        return {str(cap.key): str(cap.value) for cap in response.capabilities}

    def open_session(
        self,
        *,
        session_id: Optional[str] = None,
        spec: Optional[inference_pb2.ModelSpec] = None,
        manifest: Optional[inference_pb2.TileManifest] = None,
        transport: Optional[inference_pb2.TransportCaps] = None,
        options: Optional[Mapping[str, str]] = None,
    ) -> inference_pb2.OpenSessionResponse:
        """Open (or idempotently re-open) a session.

        The session ID is generated here when not given, so a retried open reuses the
        same ID and the server returns the existing session instead of a second one.
        """
        session_id = session_id or uuid.uuid4().hex
        req = inference_pb2.OpenSessionRequest(
            session_id=session_id,
        )
        if spec:
            req.spec.CopyFrom(spec)
        if manifest:
            req.manifest.CopyFrom(manifest)
        if transport:
            req.transport.CopyFrom(transport)
        if options:
            for k, v in options.items():
                req.options[str(k)] = str(v)

        def _call():
            stub = self._get_stub()
            return stub.OpenSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def close_session(
        self, session_id: str, *, force: bool = False
    ) -> inference_pb2.CloseSessionResponse:
        req = inference_pb2.CloseSessionRequest(session_id=str(session_id or ""), force=force)

        def _call():
            stub = self._get_stub()
            return stub.CloseSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def cancel_session(
        self, session_id: str, *, reason: str = ""
    ) -> inference_pb2.CancelSessionResponse:
        req = inference_pb2.CancelSessionRequest(
            session_id=str(session_id or ""), reason=str(reason or "")
        )

        def _call():
            stub = self._get_stub()
            return stub.CancelSession(req, timeout=self._timeout)

        return self._with_retry(_call)

    def stream_predict(
        self,
        *,
        model_id: str = "",
        samples: Optional[Iterable[np.ndarray]] = None,
        tiles: Optional[Iterable[Tuple[str, np.ndarray, Optional[Mapping[str, object]]]]] = None,
        session_id: Optional[str] = None,
        tile_ids: Optional[Iterable[str]] = None,
        tile_options: Optional[Iterable[Optional[Mapping[str, object]]]] = None,
        options: Optional[Mapping[str, object]] = None,
        chunk_bytes: Optional[int] = None,
        max_inflight: Optional[int] = None,
        dtype: Optional[str] = None,
        cancel_fn=None,
        on_response=None,
        on_error=None,
        on_tile_start=None,
    ) -> "StreamPredictCall":
        """Bidirectional streaming inference, one response per tile.

        Args:
            model_id: model identifier; required without ``session_id`` (for a session
                stream the server uses the session's model).
            samples: iterable of numpy arrays, one per tile (consumed lazily).
            tiles: instead of ``samples``/``tile_ids``/``tile_options``, one iterable of
                ``(tile_id, sample, tile_options_or_None)``, consumed lazily, one item
                per tile, so ids and options cannot fall out of step with samples.
            session_id: session opened with :meth:`open_session`.
            tile_ids: tile ids aligned with samples (default ``tile-<index>``).
            tile_options: per-tile option mappings aligned with samples (``None`` for none).
            options: options applied to every tile; ``tile_options`` override them.
            chunk_bytes: split each tile into messages of at most this many bytes
                (use the negotiated value, at most Capabilities ``stream_max_chunk_bytes``).
            max_inflight: at most this many tiles sent but not yet answered
                (use ``OpenSessionResponse.max_inflight``); ``None`` means unbounded.
            dtype: declared dtype; the data is cast losslessly or rejected.
            cancel_fn: optional callable polled every ``CANCEL_POLL_SECONDS``; returning
                True cancels the RPC even while no response has arrived.
            on_response: optional callback(resp) invoked per response.
            on_error: optional callback(err_msg) invoked if the stream fails.
            on_tile_start: optional callback(tile_id) invoked when a tile starts sending.

        Returns:
            A :class:`StreamPredictCall`: iterate it for responses, call ``cancel()`` to
            stop. Iteration ends quietly after a cancel.
        """

        model_id = str(model_id or "")
        if not model_id and not session_id:
            raise ValueError("model_id or session_id is required")
        if tiles is not None:
            if samples is not None or tile_ids is not None or tile_options is not None:
                raise ValueError("pass tiles, or samples with tile_ids/tile_options, not both")
        elif samples is None:
            raise ValueError("samples or tiles is required")
        if max_inflight is not None and int(max_inflight) < 1:
            raise ValueError("max_inflight must be >= 1")
        if chunk_bytes is not None and int(chunk_bytes) < 0:
            raise ValueError("chunk_bytes must be >= 0")
        return StreamPredictCall(
            self,
            model_id=model_id,
            samples=samples,
            tiles=tiles,
            session_id=str(session_id or ""),
            tile_ids=tile_ids,
            tile_options=tile_options,
            options=options,
            chunk_bytes=int(chunk_bytes or 0),
            max_inflight=int(max_inflight) if max_inflight is not None else None,
            dtype=dtype,
            cancel_fn=cancel_fn,
            on_response=on_response,
            on_error=on_error,
            on_tile_start=on_tile_start,
        )

    def list_model_registry(
        self, provider_id: Optional[str] = None
    ) -> Sequence[ModelRegistryEntry]:
        """Fetch registry entries stored on the remote inference service."""

        request = inference_pb2.ListModelRegistryRequest()
        if provider_id:
            request.provider_id = str(provider_id)

        try:
            stub = self._get_stub()
            response = stub.ListModelRegistry(request, timeout=self._timeout)
        except grpc.RpcError as err:  # pragma: no cover - requires network
            details = getattr(err, "details", None)
            message = details() if callable(details) else str(details or err)
            raise InferenceGrpcError(message) from err

        return [
            ModelRegistryEntry(
                entry_id=str(entry.entry_id or ""),
                model_id=str(entry.model_id or ""),
                display_name=str(entry.display_name or ""),
                source=str(entry.source or ""),
                task=str(entry.task or ""),
                format=str(entry.format or ""),
                metadata=dict(entry.metadata),
                last_used_epoch_ms=int(entry.last_used_epoch_ms or 0),
            )
            for entry in response.entries
        ]

    def evict_model_registry_entry(self, entry_id: str, provider_id: Optional[str] = None) -> bool:
        """Remove an existing registry entry on the remote inference service."""

        entry_id = str(entry_id or "").strip()
        if not entry_id:
            raise ValueError("entry_id is required")

        request = inference_pb2.EvictModelRegistryRequest(entry_id=entry_id)
        if provider_id:
            request.provider_id = str(provider_id)

        try:
            stub = self._get_stub()
            response = stub.EvictModelRegistryEntry(request, timeout=self._timeout)
        except grpc.RpcError as err:  # pragma: no cover - requires network
            details = getattr(err, "details", None)
            message = details() if callable(details) else str(details or err)
            raise InferenceGrpcError(message) from err

        return bool(response.success)


class _TileWindow:
    """Bounds tiles sent but not yet answered; ``close()`` releases every waiter."""

    def __init__(self, limit: Optional[int]):
        self._limit = limit
        self._outstanding = 0
        self._closed = False
        self._cond = threading.Condition()

    def acquire(self) -> bool:
        with self._cond:
            while (
                not self._closed
                and self._limit is not None
                and self._outstanding >= self._limit
            ):
                self._cond.wait()
            if self._closed:
                return False
            self._outstanding += 1
            return True

    def release(self) -> None:
        with self._cond:
            if self._outstanding > 0:
                self._outstanding -= 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class StreamPredictCall:
    """One ``StreamPredict`` RPC: iterate for responses, ``cancel()`` to stop it.

    Cancellation cancels the underlying gRPC call immediately, so a caller blocked
    waiting for the first response is released even if the server never answers.
    A caller that stops reading before the stream ends calls ``cancel()`` (or closes
    the iterator); until then the RPC stays open. When the RPC ends by itself, the
    request thread and the ``cancel_fn`` watcher stop even if the caller keeps the
    iterator.
    """

    def __init__(
        self,
        client: "InferenceGrpcClient",
        *,
        model_id: str,
        samples: Optional[Iterable[np.ndarray]],
        session_id: str,
        tile_ids: Optional[Iterable[str]],
        tile_options: Optional[Iterable[Optional[Mapping[str, object]]]],
        options: Optional[Mapping[str, object]],
        chunk_bytes: int,
        max_inflight: Optional[int],
        dtype: Optional[str],
        cancel_fn: Optional[Callable[[], bool]],
        on_response,
        on_error,
        on_tile_start,
        tiles=None,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._samples = samples
        self._tiles = tiles
        self._session_id = session_id
        self._tile_ids = tile_ids
        self._tile_options = tile_options
        self._options = dict(options or {})
        self._chunk_bytes = chunk_bytes
        self._dtype = dtype
        self._cancel_fn = cancel_fn
        self._on_response = on_response
        self._on_error = on_error
        self._on_tile_start = on_tile_start
        self._window = _TileWindow(max_inflight)
        self._lock = threading.Lock()
        self._call = None
        self._cancelled = False
        self._request_error: Optional[BaseException] = None
        self._done = threading.Event()
        self._started = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Cancel the RPC now (idempotent, thread-safe)."""
        with self._lock:
            self._cancelled = True
            call = self._call
        self._window.close()
        if call is not None and callable(getattr(call, "cancel", None)):
            call.cancel()

    def _cancel_requested(self) -> bool:
        if self._cancelled:
            return True
        if self._cancel_fn is not None:
            try:
                if self._cancel_fn():
                    self.cancel()
                    return True
            except Exception:
                pass
        return False

    def _tile_items(self) -> Iterator[tuple]:
        """``(tile_id, sample, per_tile_options)`` per tile, pulled lazily."""
        if self._tiles is not None:
            for index, item in enumerate(self._tiles):
                try:
                    tile_id, sample, per_tile = item
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"tiles[{index}] must be (tile_id, sample, options)") from exc
                yield tile_id, sample, per_tile
            return
        id_iter = iter(self._tile_ids) if self._tile_ids is not None else None
        opt_iter = iter(self._tile_options) if self._tile_options is not None else None
        missing = object()
        for index, sample in enumerate(self._samples):
            tile_id = next(id_iter, missing) if id_iter is not None else f"tile-{index}"
            per_tile = next(opt_iter, missing) if opt_iter is not None else None
            if tile_id is missing or per_tile is missing:
                raise ValueError(f"tile_ids/tile_options have fewer entries than samples ({index + 1})")
            yield tile_id, sample, per_tile

    def _requests(self) -> Iterator[inference_pb2.StreamPredictRequest]:
        try:
            for tile_id, sample, per_tile in self._tile_items():
                tile_id = str(tile_id)
                tile_opts = dict(self._options)
                tile_opts.update(per_tile or {})
                arr = prepare_tensor(sample, self._dtype)
                first_ctx = encode_tile_context(
                    tile_id=tile_id, session_id=self._session_id, options=tile_opts
                )
                later_ctx = encode_tile_context(tile_id=tile_id, session_id=self._session_id)
                if self._cancel_requested() or not self._window.acquire():
                    return
                if self._on_tile_start:
                    try:
                        self._on_tile_start(tile_id)
                    except Exception:
                        pass
                raw = arr.tobytes()
                step = self._chunk_bytes if self._chunk_bytes > 0 else max(len(raw), 1)
                chunks = [raw[i : i + step] for i in range(0, len(raw), step)] or [b""]
                shape_list = [int(dim) for dim in arr.shape]
                for position, chunk in enumerate(chunks):
                    yield inference_pb2.StreamPredictRequest(
                        model_id=self._model_id,
                        chunk=chunk,
                        shape=shape_list,
                        dtype=arr.dtype.name,
                        context=first_ctx if position == 0 else later_ctx,
                    )
                # end-of-sequence marker for this tile
                yield inference_pb2.StreamPredictRequest(end_of_sequence=True, context=later_ctx)
        except Exception as exc:
            # Surface the caller's error (e.g. a rejected dtype cast) from iteration
            # instead of letting gRPC turn it into an opaque cancelled RPC.
            self._request_error = exc
            with self._lock:
                call = self._call
            if call is not None and callable(getattr(call, "cancel", None)):
                call.cancel()

    def _on_call_done(self, _call) -> None:
        # The RPC is over (server status, deadline, cancel or transport failure), so
        # nothing more can be sent: release a request thread waiting for a window slot
        # and stop the cancel_fn watcher, even if the caller keeps the response
        # iterator without finishing or closing it.
        self._done.set()
        self._window.close()

    def _watch_cancel_fn(self) -> None:
        while not self._done.wait(CANCEL_POLL_SECONDS):
            if self._cancel_requested():
                return

    def __iter__(self) -> Iterator[inference_pb2.StreamPredictResponse]:
        if self._started:
            raise RuntimeError("a StreamPredictCall can be iterated only once")
        self._started = True
        return self._run()

    def _run(self) -> Iterator[inference_pb2.StreamPredictResponse]:
        if self._cancel_requested():
            self._done.set()
            return
        finished = False
        try:
            stub = self._client._get_stub()
            call = stub.StreamPredict(self._requests(), timeout=self._client._timeout)
            with self._lock:
                self._call = call
                cancelled = self._cancelled
            if cancelled and callable(getattr(call, "cancel", None)):
                call.cancel()
            if callable(getattr(call, "add_done_callback", None)):
                call.add_done_callback(self._on_call_done)
            if self._cancel_fn is not None:
                threading.Thread(
                    target=self._watch_cancel_fn, name="nxmndr-stream-cancel", daemon=True
                ).start()
            for resp in call:
                if self._cancelled:
                    break
                if resp.metadata.get(CONTEXT_TILE_ID):
                    self._window.release()
                if self._on_response:
                    try:
                        self._on_response(resp)
                    except Exception:
                        pass
                yield resp
            if self._request_error is not None:
                raise self._request_error
            finished = True
        except grpc.RpcError as err:
            if self._request_error is not None:
                raise self._request_error from err
            if self._cancelled:
                return
            message, code = _rpc_error_parts(err)
            if self._on_error:
                try:
                    self._on_error(message)
                except Exception:
                    pass
            raise InferenceGrpcError(message, code=code) from err
        finally:
            self._done.set()
            self._window.close()
            if not finished:
                with self._lock:
                    call = self._call
                if call is not None and callable(getattr(call, "cancel", None)):
                    call.cancel()


__all__ = [
    "InferenceGrpcClient",
    "LoadModelResult",
    "PredictResult",
    "InferenceGrpcError",
    "ModelRegistryEntry",
    "StreamPredictCall",
    "encode_tile_context",
    "Endpoint",
    "parse_endpoint",
    "prepare_tensor",
    "validate_option_name",
    "STREAM_CONTEXT_CAPABILITY",
    "STREAM_CONTEXT_VERSION",
    "inference_pb2",
    "inference_pb2_grpc",
]
