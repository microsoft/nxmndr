# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Every server entry point binds loopback unless --host names another address.

A gRPC client can make the server run code it supplies, and the HTTP proxy spends
the operator's credentials for every caller, so listening on the network must be
an explicit choice. The servers are replaced by recorders: no socket is bound.
"""

from __future__ import annotations

import logging
import sys

import pytest

import nxmndr.__main__ as cli
from nxmndr.server import azure_openai_proxy, server

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_log_levels():
    loggers = [logging.getLogger(), logging.getLogger("nxmndr.server")]  # server.main sets both
    levels = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, levels):
        lg.setLevel(level)


def test_nxmndr_server_binds_grpc_on_loopback_unless_host_is_given(monkeypatch):
    calls = []

    class _Server:
        def wait_for_termination(self):
            return None

    def create_server(host, port, **_kwargs):
        calls.append((host, port))
        return _Server(), port, None

    monkeypatch.setattr(server, "create_server", create_server)
    monkeypatch.setattr(server, "stop_server", lambda *_a, **_k: None)
    monkeypatch.setattr(server.rpc_worker, "configure_cudnn", lambda: None)
    for argv in (["nxmndr-server"], ["nxmndr-server", "--host", "0.0.0.0", "--grpc-port", "50999"]):
        monkeypatch.setattr(sys, "argv", argv)
        server.main()
    assert calls == [("127.0.0.1", 50051), ("0.0.0.0", 50999)]


def test_the_other_entry_points_bind_loopback_by_default(monkeypatch):
    seen = {}

    def serve(port=50051, *, host=None, **_kwargs):
        seen["python -m nxmndr grpc"] = host

    async def run_unified_server(host=None, port=None, **_kwargs):
        seen["python -m nxmndr server"] = host

    async def run_server(host=None, port=None):
        seen["azure_openai_proxy"] = host

    monkeypatch.setattr(server, "serve", serve)
    monkeypatch.setattr(server, "run_unified_server", run_unified_server)
    monkeypatch.setattr(azure_openai_proxy, "run_server", run_server)
    assert cli.main(["grpc"]) == 0
    assert cli.main(["server"]) == 0
    monkeypatch.setattr(sys, "argv", ["azure_openai_proxy"])
    azure_openai_proxy.main()
    assert seen == {
        name: "127.0.0.1"
        for name in ("python -m nxmndr grpc", "python -m nxmndr server", "azure_openai_proxy")
    }
