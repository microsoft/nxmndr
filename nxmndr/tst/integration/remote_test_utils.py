# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import logging
import grpc
from typing import Optional, Tuple
from urllib.parse import urlparse

from concurrent import futures
from nxmndr.server.server import InferenceService, GRPC_OPTIONS
from nxmndr.inference import inference_pb2_grpc

DEFAULT_CONNECT_TIMEOUT = 30
# RemoteTestServer reuses the server named by NXMNDR_REMOTE_HOST/PORT only when this
# is "1"; otherwise it always starts its own loopback server, so variables left in the
# shell cannot point the tests at another server.
REMOTE_OPT_IN_ENV = "NXMNDR_TEST_USE_REMOTE_SERVER"

logger = logging.getLogger(__name__)


def get_remote_host() -> str:
    """Return remote host from env NXMNDR_REMOTE_HOST or default 'localhost'."""
    return os.getenv("NXMNDR_REMOTE_HOST", "localhost")


def get_remote_port() -> Optional[int]:
    """Return remote port from env NXMNDR_REMOTE_PORT if set."""
    port = os.getenv("NXMNDR_REMOTE_PORT")
    if port:
        try:
            return int(port)
        except ValueError:
            pass
    return None


def remote_server_opted_in() -> bool:
    """True when NXMNDR_TEST_USE_REMOTE_SERVER=1 allows reusing an external server."""
    return os.getenv(REMOTE_OPT_IN_ENV, "").strip() == "1"


def start_test_server(port: int = 0):
    """Start inference server (test mode) on 127.0.0.1 and return (server, bound_port).

    Not named ``test_*``, so importing it into a test module or conftest never
    makes pytest collect it as a test.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_OPTIONS)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    bound_port = server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    logger.info(f"[remote_test_utils] Started test gRPC server on port {bound_port}")
    return server, bound_port


__all__ = [
    "get_remote_host",
    "get_remote_port",
    "remote_server_opted_in",
    "start_test_server",
    "RemoteTestServer",
    "server_channel",
]


def _resolve_endpoint(
    raw_host: str,
    port_env: Optional[int],
) -> Tuple[str, Optional[int], Optional[str]]:
    """Normalize host/port/scheme from env values.

    Returns tuple of (host, port, scheme).
    Scheme is only populated when raw_host included an explicit scheme.
    """

    scheme = None
    host = raw_host.strip()
    port = port_env

    if host.startswith(("http://", "https://")):
        parsed = urlparse(host)
        scheme = parsed.scheme
        host = parsed.hostname or "localhost"
        if port is None:
            port = parsed.port
        # Do not infer port; require explicit value elsewhere
    else:
        # Allow host:port syntax
        if ":" in host and host.count(":") == 1:
            maybe_host, maybe_port = host.rsplit(":", 1)
            try:
                inferred_port = int(maybe_port)
            except ValueError:
                pass
            else:
                host = maybe_host
                if port is None:
                    port = inferred_port

        # Leave port as-is; caller will decide whether that's acceptable

    return host, port, scheme


class RemoteTestServer:
    """Context manager for a test gRPC inference server returning only a channel.

    Starts an ephemeral server on 127.0.0.1. Only with NXMNDR_TEST_USE_REMOTE_SERVER=1
    does it reuse the server named by NXMNDR_REMOTE_{HOST,PORT} instead.
    """

    def __init__(self, port: int = 0):
        self._requested_port = port
        self._server = None
        self._channel = None
        self._port = None
        self._ephemeral = False
        self._scheme: Optional[str] = None
        self._host: str = "localhost"

    def __enter__(self):
        if remote_server_opted_in():
            raw_host = get_remote_host()
            port_env = get_remote_port()
        else:
            raw_host, port_env = "localhost", None  # never an inherited endpoint
        host, port, scheme = _resolve_endpoint(raw_host, port_env)
        self._host = host
        self._scheme = scheme

        # Decide final host/port and whether to start a server
        if host != "localhost" or scheme:
            # External host reuse (port optional when scheme provided)
            self._port = port
            if self._port is None and not self._scheme:
                raise RuntimeError(
                    "Remote host requires an explicit port or HTTPS scheme for TLS auto-detection."
                )
            if self._port is None:
                logger.info(
                    f"[RemoteTestServer] Reusing external server at {raw_host} (resolved host={host}, scheme={self._scheme or 'insecure'})"
                )
            else:
                logger.info(f"[RemoteTestServer] Reusing external server at {host}:{self._port}")
        elif port_env is not None:
            # Local host reuse on specified port
            self._port = port_env
            logger.info(f"[RemoteTestServer] Reusing localhost server at {host}:{self._port}")
        else:
            # Start ephemeral server (bound to 127.0.0.1 only)
            self._server, self._port = start_test_server(self._requested_port)
            self._host = "127.0.0.1"
            self._ephemeral = True
            logger.info(f"[RemoteTestServer] Started ephemeral server at {self._host}:{self._port}")

        # Create channel (single location)
        target = self._host if self._port is None else f"{self._host}:{self._port}"
        if self._scheme == "https":
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(target, credentials, options=GRPC_OPTIONS)
        elif self._scheme == "http":
            if self._port is None:
                raise RuntimeError(
                    "HTTP scheme requires explicit NXMNDR_REMOTE_PORT or host:port in NXMNDR_REMOTE_HOST."
                )
            self._channel = grpc.insecure_channel(target, options=GRPC_OPTIONS)
        else:
            if self._port is None:
                raise RuntimeError(
                    "Port is required when connecting without explicit scheme. Set NXMNDR_REMOTE_PORT or use a URL."
                )
            self._channel = grpc.insecure_channel(target, options=GRPC_OPTIONS)
        grpc.channel_ready_future(self._channel).result(timeout=DEFAULT_CONNECT_TIMEOUT)
        return self._channel

    def __exit__(self, exc_type, exc, tb):
        # Only stop if we started an ephemeral server locally
        if self._ephemeral and self._server is not None:
            try:
                self._server.stop(0)
                logger.info(f"[RemoteTestServer] Stopped ephemeral server on port {self._port}")
            finally:
                self._server = None
        self._channel = None


def server_channel(port: int = 0) -> RemoteTestServer:
    """Generator-style context for use with 'with server_channel() as channel:'"""
    return RemoteTestServer(port)
