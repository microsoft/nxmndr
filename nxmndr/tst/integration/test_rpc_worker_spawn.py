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
import pickle
from multiprocessing.reduction import ForkingPickler

import pytest

from nxmndr.server import managers
from tst.support.rpc_probe import MARKER_ENV, marker_worker

pytestmark = pytest.mark.integration


class _RecordedProcess:
    """Stands in for a spawn-context Process and keeps what it was given."""

    made = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.target, self.args, self.kwargs = target, tuple(args), dict(kwargs or {})
        self.pid, self.exitcode = None, None
        _RecordedProcess.made.append(self)

    def start(self):
        pass

    def is_alive(self):
        return False


class _RecordedContext:
    methods = []

    def __init__(self, method):
        _RecordedContext.methods.append(method)
        self.Process = _RecordedProcess


@pytest.fixture
def no_torch_group(monkeypatch):
    """This pytest process may hold a group from another test; these starts never reach torch."""

    monkeypatch.setattr(managers.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(managers.torch_rpc, "_is_current_rpc_agent_set", lambda: False)


def test_worker_target_and_arguments_pickle_for_spawn(monkeypatch, no_torch_group):
    _RecordedProcess.made.clear()
    _RecordedContext.methods.clear()
    monkeypatch.setattr(managers.multiprocessing, "get_context", _RecordedContext)
    manager = managers.RpcWorkerManager(
        master_addr="127.0.0.1", master_port=29555, startup_timeout_s=7.0, device="cuda:0"
    )
    with pytest.raises(managers.RpcWorkerStartError, match="before it was ready"):
        manager.ensure_started()

    assert _RecordedContext.methods == ["spawn"]
    [proc] = _RecordedProcess.made
    payload = bytes(ForkingPickler.dumps((proc.target, proc.args, proc.kwargs)))
    target, args, kwargs = pickle.loads(payload)
    assert target is managers._run_rpc_worker
    addr, port, ready_port, timeout, device = args
    assert (addr, port, timeout) == ("127.0.0.1", 29555, 7.0) and isinstance(ready_port, int)
    assert device == "cuda:0"  # the worker builds its models on the server's device
    assert manager._state == "idle"  # the torch rendezvous was never entered: a retry may start again


def test_worker_process_starts_and_runs_under_spawn(tmp_path, monkeypatch, no_torch_group):
    marker = tmp_path / "rpc-worker-ran"
    monkeypatch.setenv(MARKER_ENV, str(marker))
    monkeypatch.setattr(managers.RpcWorkerManager, "worker_target", staticmethod(marker_worker))
    manager = managers.RpcWorkerManager(master_addr="127.0.0.1", startup_timeout_s=120.0)
    with pytest.raises(managers.RpcWorkerStartError, match="exited with code 0 before it was ready"):
        manager.ensure_started()
    text = marker.read_text(encoding="utf-8")
    pid = int(text.split()[0].split("=")[1])
    assert pid != os.getpid() and "ready_port=" in text
    assert manager._proc is None and not manager.is_alive()
