# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Test-server hygiene.

The helper is not collectable, binds loopback only, ignores inherited
NXMNDR_REMOTE_HOST/PORT unless NXMNDR_TEST_USE_REMOTE_SERVER=1, and keeps the
registry file and model artifacts out of the user's ~/.cache/nxmndr.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2, inference_pb2_grpc
from nxmndr.server.registry import CACHE_DIR_ENV, default_cache_root
from nxmndr.server.server import InferenceService
from tst.integration import remote_test_utils
from tst.integration.remote_test_utils import RemoteTestServer, start_test_server

pytestmark = pytest.mark.integration

EXAMPLE_ONNX = Path(__file__).resolve().parent / "example_model" / "example_model.onnx"


def _accepts(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def _healthy(channel):
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    return stub.Health(inference_pb2.HealthRequest(), timeout=10).ready


def test_no_helper_is_named_like_a_test():
    assert [n for n in vars(remote_test_utils) if n.lower().startswith("test")] == []


def test_start_test_server_binds_loopback_only():
    server, port = start_test_server()
    try:
        assert _accepts("127.0.0.1", port)
        if socket.has_ipv6:
            assert not _accepts("::1", port)  # not every interface ([::])
    finally:
        server.stop(0).wait()


def test_remote_endpoint_variables_are_ignored_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(remote_test_utils.REMOTE_OPT_IN_ENV, raising=False)
    monkeypatch.setenv("NXMNDR_REMOTE_HOST", "192.0.2.1")  # TEST-NET-1: must not be dialled
    monkeypatch.setenv("NXMNDR_REMOTE_PORT", "9")
    ctx = RemoteTestServer()
    with ctx as channel:
        assert ctx._ephemeral and ctx._host == "127.0.0.1"
        assert _healthy(channel)


def test_remote_endpoint_is_reused_only_with_the_opt_in(monkeypatch):
    external, port = start_test_server()
    try:
        monkeypatch.setenv(remote_test_utils.REMOTE_OPT_IN_ENV, "1")
        monkeypatch.setenv("NXMNDR_REMOTE_HOST", "127.0.0.1")
        monkeypatch.setenv("NXMNDR_REMOTE_PORT", str(port))
        ctx = RemoteTestServer()
        with ctx as channel:
            assert not ctx._ephemeral and ctx._server is None and ctx._port == port
            assert _healthy(channel)
    finally:
        external.stop(0).wait()


def test_registry_and_artifacts_stay_in_the_test_cache_dir(tmp_path, monkeypatch):
    root = tmp_path / "nxmndr-cache"  # set by tst/conftest.py for every test
    assert default_cache_root() == root
    service = InferenceService()
    assert service.registry_store._path == root / "model_registry.json"
    assert service.registry_store.artifact_root() == root / "models"

    server, port = start_test_server()
    try:
        client = InferenceGrpcClient(f"127.0.0.1:{port}", timeout=10, max_attempts=1)
        model_id = client.load_model(
            "registry-hygiene", {"format": "onnx", "source": str(EXAMPLE_ONNX)}
        )
        entries = client.list_model_registry()
        assert any(e.model_id == model_id for e in entries)
        assert model_id in (root / "model_registry.json").read_text(encoding="utf-8")
    finally:
        server.stop(0).wait()

    monkeypatch.delenv(CACHE_DIR_ENV)
    assert default_cache_root() == Path.home() / ".cache" / "nxmndr"  # the production default
