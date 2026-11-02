# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RpcWorkerManager starts its PyTorch RPC worker under the ``spawn`` start method.

``spawn`` pickles the process target, so a nested function fails in the parent with
"Can't pickle local object". The manager always uses the ``spawn`` context: the
child must not inherit the server's gRPC and torch threads, as ``fork`` would do.
The worker's arguments are plain values (no model spec: models are loaded later by
model ID; the device it builds them on). The full start, with the real rendezvous, runs in
tst/integration/test_rpc_pytorch_service.py.
"""

from __future__ import annotations

import os

import pytest

from nxmndr.server import managers
from tst.support.rpc_probe import MARKER_ENV, marker_worker

pytestmark = pytest.mark.integration


def test_worker_process_starts_and_runs_under_spawn(tmp_path, monkeypatch):
    marker = tmp_path / "rpc-worker-ran"
    monkeypatch.setenv(MARKER_ENV, str(marker))
    monkeypatch.setattr(managers.RpcWorkerManager, "worker_target", staticmethod(marker_worker))
    manager = managers.RpcWorkerManager(master_addr="127.0.0.1", startup_timeout_s=120.0)
    with pytest.raises(
        managers.RpcWorkerStartError, match="exited with code 0 before it was ready"
    ):
        manager.ensure_started()
    text = marker.read_text(encoding="utf-8")
    pid = int(text.split()[0].split("=")[1])
    assert pid != os.getpid() and "method=spawn" in text
    assert manager._state == "idle"
    assert manager._proc is None and not manager.is_alive()
