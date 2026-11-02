# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Data-only messages over an anonymous pipe inherited by our spawned worker.

There is deliberately no listener, rendezvous, pickle decoder, or callable import.
Model classes are supplied by the trusted parent only at multiprocessing spawn.
"""

from __future__ import annotations

import base64
import itertools
import json
import queue
import threading
from concurrent.futures import Future, TimeoutError

import torch

MAX_MESSAGE_BYTES = 512 * 1024 * 1024
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def _encode(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if torch.is_tensor(value):
        tensor = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
        if tensor.layout != torch.strided or str(tensor.dtype) not in _DTYPES:
            raise TypeError(f"unsupported worker tensor: {tensor.layout}, {tensor.dtype}")
        return {
            "type": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": base64.b64encode(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).decode(
                "ascii"
            ),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(v) for v in value],
        }
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"type": "dict", "items": {k: _encode(v) for k, v in value.items()}}
    raise TypeError(f"unsupported worker value: {type(value).__name__}")


def _decode(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid worker value")
    kind = value.get("type")
    if kind == "tensor":
        dtype = _DTYPES.get(value.get("dtype"))
        shape = value.get("shape")
        if dtype is None or not isinstance(shape, list) or len(shape) > 64:
            raise ValueError("invalid worker tensor type or shape")
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 0:
                raise ValueError("invalid worker tensor dimension")
            count *= dim
        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) != count * torch.empty((), dtype=dtype).element_size():
            raise ValueError("worker tensor byte count does not match its shape")
        if not count:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    if kind in ("list", "tuple") and isinstance(value.get("items"), list):
        items = [_decode(v) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict" and isinstance(value.get("items"), dict):
        return {k: _decode(v) for k, v in value["items"].items()}
    raise ValueError("invalid worker value type")


def send_message(conn, value) -> None:
    data = json.dumps(_encode(value), separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("worker message exceeds size limit")
    conn.send_bytes(data)


def receive_message(conn):
    return _decode(json.loads(conn.recv_bytes(MAX_MESSAGE_BYTES)))


class WorkerClient:
    """Multiplex calls so a timed-out load can still be abandoned or stopped."""

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()
        self._pending: dict[int, Future[dict]] = {}
        self._outgoing: queue.Queue = queue.Queue()
        self._ids = itertools.count()
        self._closed = False
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._writer = threading.Thread(target=self._send, daemon=True)
        self._reader.start()
        self._writer.start()

    def call(self, operation: str, args=(), *, timeout: float = 60.0):
        request_id = next(self._ids)
        future: Future[dict] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("the PyTorch worker connection is closed")
            self._pending[request_id] = future
        try:
            self._outgoing.put({"id": request_id, "operation": operation, "args": args})
            try:
                reply = future.result(timeout=timeout)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"PyTorch worker call {operation} timeout after {timeout:g} s"
                ) from exc
            if "error" in reply:
                raise RuntimeError(f"PyTorch worker {operation} failed: {reply['error']}")
            return reply["result"]
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def _send(self):
        # Pipe writes can block if a worker hangs. Keep them off calling threads
        # so their deadlines (including stop's) also bound transmission.
        while True:
            request = self._outgoing.get()
            if request is None:
                return
            with self._lock:
                if self._closed or request["id"] not in self._pending:
                    continue
            try:
                send_message(self._conn, request)
            except (OSError, EOFError) as exc:
                self._fail(f"the PyTorch worker connection closed: {exc}")
                return
            except (ValueError, TypeError, RuntimeError) as exc:
                with self._lock:
                    future = self._pending.pop(request["id"], None)
                if future is not None:
                    future.set_exception(exc)

    def _receive(self):
        try:
            while True:
                reply = receive_message(self._conn)
                with self._lock:
                    future = self._pending.pop(reply["id"], None)
                if future is not None:
                    future.set_result(reply)
        except (
            EOFError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            OverflowError,
        ) as exc:
            self._fail(f"the PyTorch worker connection closed: {exc}")

    def _fail(self, message):
        with self._lock:
            self._closed = True
            pending, self._pending = self._pending, {}
        for future in pending.values():
            future.set_exception(RuntimeError(message))
        while True:
            try:
                self._outgoing.get_nowait()
            except queue.Empty:
                break
        self._outgoing.put(None)

    def close(self):
        self._fail("the PyTorch worker was stopped")
        self._conn.close()
