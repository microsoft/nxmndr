# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loopback gRPC server with an injected ModelManager for streaming tests.

Binds 127.0.0.1 on an ephemeral port, never reads NXMNDR_REMOTE_* and pins every
stream limit explicitly so NXMNDR_STREAM_* in the environment cannot change results.
"""

from __future__ import annotations

import contextlib
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import grpc

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2, inference_pb2_grpc
from nxmndr.server import managers
from nxmndr.server.server import GRPC_OPTIONS, InferenceService

DEFAULT_LIMITS = {
    "max_inflight": 16,
    "chunk_bytes": 8 * 1024 * 1024,
    "tile_bytes": 128 * 1024 * 1024,
}


@dataclass
class Harness:
    service: InferenceService
    manager: object
    endpoint: str

    def client(self, **kwargs) -> InferenceGrpcClient:
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("max_attempts", 1)
        return InferenceGrpcClient(self.endpoint, **kwargs)

    def stub(self, channel):
        return inference_pb2_grpc.InferenceServiceStub(channel)


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.01) -> bool:
    """Wait for a condition set by another thread (bounded poll, not a fixed sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@contextlib.contextmanager
def running_server(
    tmp_path: Path,
    *,
    loader,
    capacity: int = 4,
    session_ttl_s: float = 3600.0,
    max_cores: int = 1,
    limits: Optional[dict] = None,
    workers: int = 8,
):
    manager = managers.ModelManager(
        None, capacity=capacity, session_ttl_s=session_ttl_s, loader=loader
    )
    service = InferenceService(
        model_cache_dir=tmp_path / "model-cache", max_cores=max_cores, model_manager=manager
    )
    lim = dict(DEFAULT_LIMITS, **(limits or {}))
    service._default_max_inflight = int(lim["max_inflight"])
    service._default_chunk_bytes = int(lim["chunk_bytes"])
    service._default_tile_bytes = int(lim["tile_bytes"])
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers), options=GRPC_OPTIONS)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield Harness(service, manager, f"127.0.0.1:{port}")
    finally:
        server.stop(0).wait()
        service.shutdown(grace=5.0)


def onnx_spec(source: str, *, task=inference_pb2.SEGMENTATION, name: str = "") -> inference_pb2.ModelSpec:
    return inference_pb2.ModelSpec(format=inference_pb2.ONNX, source=source, task=task, name=name)


def hf_spec(source: str, *, task=inference_pb2.SEGMENTATION) -> inference_pb2.ModelSpec:
    return inference_pb2.ModelSpec(format=inference_pb2.HUGGINGFACE, source=source, task=task)
