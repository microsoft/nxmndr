# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""create_server / stop_server: the non-blocking server factory the worker uses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading

import grpc
import pytest
from aiohttp import web

from nxmndr.inference import inference_pb2, inference_pb2_grpc
from nxmndr.server import managers, server
from tst.support.grpc_harness import wait_until

pytestmark = pytest.mark.integration


def _accepts(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def _ipv6_loopback_available():
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


def _free_ipv6_port():
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
        sock.bind(("::", 0))
        return sock.getsockname()[1]


class _SpyServer:
    def __init__(self, inner, calls):
        self._inner, self._calls = inner, calls

    def stop(self, grace):
        self._calls.append(("grpc.stop", grace))
        return self._inner.stop(grace)


def test_create_server_binds_loopback_and_stop_server_stops_grpc_the_cache_then_the_rpc_worker(
    tmp_path, monkeypatch, caplog
):
    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    calls = []
    try:
        assert isinstance(service, server.InferenceService) and port > 0
        assert _accepts("127.0.0.1", port)
        if socket.has_ipv6:
            assert not _accepts("::1", port)  # loopback IPv4 only, not every interface

        caplog.set_level(logging.DEBUG, logger="nxmndr.server")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            secrets = {
                "authorization": "Bearer sekrit-bearer-1",
                "x-api-key": "sekrit-api-key-2",
                "hf-token": "sekrit-hf-token-3",
                "x-session-secret": "sekrit-4",
            }
            metadata = list(secrets.items()) + [("x-request-id", "visible-request-id")]
            assert stub.Health(inference_pb2.HealthRequest(), metadata=metadata).ready
            caps = stub.Capabilities(inference_pb2.CapabilitiesRequest())
        assert {c.key: c.value for c in caps.capabilities}["stream_context_version"] == "1"

        logged = [r.getMessage() for r in caplog.records if r.name == "nxmndr.server"]
        request_lines = [m for m in logged if "method=/nxmndr.inference" in m or "Health" in m]
        assert any("visible-request-id" in m for m in request_lines)  # the interceptor logged
        for value in secrets.values():
            assert all(value not in m for m in logged)
        assert all(name in "".join(request_lines) for name in secrets)  # keys stay visible

        real_shutdown = service.model_manager.shutdown

        def shutdown(*, drain_timeout_s=None):
            calls.append(("cache.shutdown", drain_timeout_s))
            real_shutdown(drain_timeout_s=drain_timeout_s)

        monkeypatch.setattr(service.model_manager, "shutdown", shutdown)
        real_rpc_shutdown = service.rpc_manager.shutdown

        def rpc_shutdown():
            calls.append(("rpc.shutdown",))
            real_rpc_shutdown()

        monkeypatch.setattr(service.rpc_manager, "shutdown", rpc_shutdown)
    finally:
        server.stop_server(_SpyServer(srv, calls), service, grace=1.5)

    assert calls == [("grpc.stop", 1.5), ("cache.shutdown", 1.5), ("rpc.shutdown",)]
    with pytest.raises(managers.CacheShutdownError):
        service.model_manager.acquire_execution(model_id="any")
    assert not _accepts("127.0.0.1", port)


def test_create_server_raises_when_the_port_is_taken(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(RuntimeError):
            server.create_server(port=taken.getsockname()[1], model_cache_dir=tmp_path / "c")


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback on this host")
@pytest.mark.parametrize("host", ["::", "[::]"])
def test_one_ipv6_host_binds_both_the_grpc_server_and_the_http_server(host, tmp_path, monkeypatch):
    # nxmndr-server passes one --host to both servers. grpcio needs an IPv6 address
    # in brackets and aiohttp refuses the brackets, so either spelling must reach both.
    srv, port, service = server.create_server(
        host, 0, model_cache_dir=tmp_path / "cache", max_cores=1
    )
    try:
        assert _accepts("::1", port)
    finally:
        server.stop_server(srv, service, grace=0)

    async def bare_app(**_kwargs):
        return web.Application()

    monkeypatch.setattr(server, "create_unified_web_app", bare_app)
    http_port = _free_ipv6_port()

    async def http_server_accepts():
        task = asyncio.create_task(server.run_unified_server(host, http_port))
        try:
            for _ in range(100):
                if task.done():
                    task.result()  # raises the bind error
                if _accepts("::1", http_port):
                    return True
                await asyncio.sleep(0.05)
            return False
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert asyncio.run(http_server_accepts())


def test_control_calls_are_served_while_streams_outnumber_max_cores(tmp_path, monkeypatch):
    entered = []
    real = server.InferenceService.StreamPredict

    def counting(self, request_iterator, context):
        entered.append(1)
        yield from real(self, request_iterator, context)

    monkeypatch.setattr(server.InferenceService, "StreamPredict", counting)
    release = threading.Event()

    def idle_requests():  # a client that sends nothing yet: its stream holds a handler thread
        release.wait(30)
        return
        yield

    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            calls = [stub.StreamPredict(idle_requests()) for _ in range(2)]
            assert wait_until(lambda: len(entered) == 2, timeout=5)  # both streams run at once
            assert stub.Health(inference_pb2.HealthRequest(), timeout=5).ready
            with pytest.raises(grpc.RpcError) as err:
                stub.CancelSession(inference_pb2.CancelSessionRequest(session_id="none"), timeout=5)
            assert err.value.code() == grpc.StatusCode.NOT_FOUND  # served, not DEADLINE_EXCEEDED
            release.set()
            assert [list(call) for call in calls] == [[], []]
    finally:
        release.set()
        server.stop_server(srv, service, grace=1.0)


def test_a_large_max_cores_opens_a_stream_per_slot_and_still_serves_control_calls(
    tmp_path, monkeypatch
):
    # 64 inference slots on a 64-CPU host: more than GRPC_HANDLER_THREADS, so a pool of
    # that fixed size would be full with 32 streams, and Health would wait.
    monkeypatch.setattr(server.os, "cpu_count", lambda: 64)
    entered = []
    real = server.InferenceService.StreamPredict

    def counting(self, request_iterator, context):
        entered.append(1)
        yield from real(self, request_iterator, context)

    monkeypatch.setattr(server.InferenceService, "StreamPredict", counting)
    release = threading.Event()

    def idle_requests():
        release.wait(30)
        return
        yield

    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=64)
    try:
        assert service._inference_slot_count == 64
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            calls = [stub.StreamPredict(idle_requests()) for _ in range(64)]
            assert wait_until(lambda: len(entered) == 64, timeout=10)  # all 64 run at once
            assert stub.Health(inference_pb2.HealthRequest(), timeout=5).ready
            release.set()
            assert all(list(call) == [] for call in calls)
    finally:
        release.set()
        server.stop_server(srv, service, grace=1.0)


def test_create_server_runs_the_session_expiry_sweep_until_stop_server(tmp_path):
    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    try:
        assert service._expiry_thread is not None and service._expiry_thread.is_alive()
    finally:
        server.stop_server(srv, service, grace=0)
    assert not service._expiry_thread.is_alive()
