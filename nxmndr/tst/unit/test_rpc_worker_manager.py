# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RpcWorkerManager and the PyTorch record, without a real RPC group.

- Every ``rpc_sync`` call passes the worker function object, never its name
  (PyTorch rejects a string with "function should be callable").
- A failed ``_rpc_stop`` is logged, then the child is terminated.
- A stop does not wait for a call in flight: both ranks leave the RPC group at once.
- A generic PyTorch record owns its worker-side model as an aux resource: disposing
  the record frees that model exactly once.
- A catalog model the worker cannot build is returned as a refusal, raised in the
  server as the ``CatalogError`` with its one-line message, and answered with
  INVALID_ARGUMENT (FAILED_PRECONDITION for a missing family package), never INTERNAL.
  A registered class whose weights file does not load strictly, for any reason (also a
  missing file or a nested checkpoint), is refused the same way.
- The spawned worker gets the first GPU of the server's device plan and the cuDNN policy.
- A load that fails or times out in the server is abandoned in the worker, which drops
  the model when its build ends; the fingerprint is taken before the model moves.
- A model's device is that of its parameters, else of its buffers; the input moves there
  and every output tensor, also inside a dict, list or tuple, comes back on the CPU.
- After a stopped server, a later one in the process is refused with the reason.
- The RPC group listens on loopback unless MASTER_ADDR names another address.
- An external worker's report connection stays open: when it closes, the worker counts
  as gone and the next load names it.
- Uploaded PyTorch weights are refused when an external worker may be on another host.
The real worker, rendezvous and routing run in tst/integration/test_rpc_pytorch_service.py.
"""

from __future__ import annotations

import logging

import grpc
import pytest

from nxmndr.models import PytorchModelSpec, catalog
from nxmndr.server import managers, rpc_worker
from tst.support.rpc_models import AddConst


class _Proc:
    def __init__(self):
        self.pid = 4242
        self.exitcode = None
        self.alive = True
        self.calls = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.calls.append("join")

    def terminate(self):
        self.calls.append("terminate")
        self.alive = False

    def kill(self):
        self.calls.append("kill")
        self.alive = False


def _running_manager(monkeypatch):
    manager = managers.RpcWorkerManager()
    manager._state = "running"
    manager._proc = _Proc()
    left = []
    monkeypatch.setattr(managers.RpcWorkerManager, "_leave_group", staticmethod(lambda graceful: left.append(graceful)))
    return manager, left


def test_rpc_calls_pass_the_worker_functions_not_their_names(monkeypatch, tmp_path):
    calls = []

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        calls.append((to, func, args))
        if func is rpc_worker._rpc_load:
            return {"model_id": args[0], "device": "cpu", "fingerprint": "sha256:x"}
        return True

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, left = _running_manager(monkeypatch)
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"), name="w")

    assert manager.load("m1", spec)["model_id"] == "m1"
    manager.infer("m1", "tensor", device_hint="cpu", timeout=5.0)
    assert manager.unload("m1") is True
    manager.status()
    manager.stop()

    funcs = [func for _, func, _ in calls]
    assert funcs == [
        rpc_worker._rpc_load,
        rpc_worker._rpc_infer,
        rpc_worker._rpc_unload,
        rpc_worker._rpc_status,
        rpc_worker._rpc_stop,
    ]
    assert all(callable(f) and not isinstance(f, str) for f in funcs)
    assert all(to == managers.RPC_WORKER_NAME for to, _, _ in calls)
    assert calls[0][2] == ("m1", {"model_class": AddConst, "model_path": str(tmp_path / "w.pt"), "name": "w"})
    assert calls[1][2] == ("m1", "tensor", "cpu")
    assert calls[4][2] == (True,)
    assert left == [True]  # a live worker with no call in flight gets a graceful RPC shutdown


def test_a_failed_rpc_stop_is_logged_then_the_worker_is_terminated(monkeypatch, caplog):
    def failing_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        assert func is rpc_worker._rpc_stop
        raise RuntimeError("worker unreachable")

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", failing_rpc_sync)
    manager, left = _running_manager(monkeypatch)
    proc = manager._proc
    with caplog.at_level(logging.WARNING):
        manager.stop()
    assert any(
        "_rpc_stop failed" in r.getMessage() and r.exc_info and "worker unreachable" in str(r.exc_info[1])
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert "terminate" in proc.calls
    assert left == [False]  # no graceful shutdown with an unreachable peer
    assert manager._state == "stopped" and not manager.is_alive()


def test_a_stop_with_a_call_in_flight_leaves_the_rpc_group_without_waiting(monkeypatch):
    import threading

    calls = []
    started, release = threading.Event(), threading.Event()

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        calls.append((func, args))
        if func is rpc_worker._rpc_load:  # a load still building in the worker
            started.set()
            release.wait(timeout=30)
            raise RuntimeError("pipe closed")  # what the in-flight call gets on the leave
        return {"status": "stopping"} if func is rpc_worker._rpc_stop else False

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, left = _running_manager(monkeypatch)
    spec = PytorchModelSpec(model_class=AddConst, model_path="w.pt", name="w")
    failed = []

    def load():
        try:
            manager.load("m1", spec)
        except RuntimeError as exc:
            failed.append(str(exc))

    loader = threading.Thread(target=load)
    loader.start()
    try:
        assert started.wait(timeout=30)
        manager.stop()
    finally:
        release.set()
        loader.join(timeout=30)
    # The worker is told to leave at once too, and the graceful barrier is skipped.
    assert (rpc_worker._rpc_stop, (False,)) in calls
    assert left == [False]
    assert failed == ["pipe closed"]
    with pytest.raises(managers.RpcWorkerStartError, match="not running"):
        manager.status()  # no call starts after the stop


def test_an_unregistered_model_class_is_refused_before_any_worker_starts(monkeypatch, tmp_path):
    manager = managers.RpcWorkerManager()
    monkeypatch.setattr(manager, "_start_locked", lambda: pytest.fail("worker started"))
    spec = PytorchModelSpec(model_class="not_registered", model_path=str(tmp_path / "w.pt"))
    with pytest.raises(ValueError, match="not registered"):
        manager.load("m1", spec)


class _Workers:
    def __init__(self):
        self.loaded, self.unloaded = [], []

    def load(self, model_id, spec):
        self.loaded.append((model_id, spec))
        return {"model_id": model_id, "device": "cpu", "fingerprint": f"sha256:{model_id}"}

    def unload(self, model_id):
        self.unloaded.append(model_id)
        return True


def test_a_pytorch_record_owns_its_worker_model_and_frees_it_once(tmp_path):
    workers = _Workers()
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"))
    record = managers.build_model_record(
        None, spec, None, {"model_id": "abc", "task": 0}, None, rpc_workers=workers
    )
    assert workers.loaded == [("abc", spec)]
    assert record.backend == "pytorch" and record.model is None
    assert record.metadata["rpc_fingerprint"] == "sha256:abc"
    handle = record._resource(managers.RPC_MODEL_RESOURCE, lambda: pytest.fail("created twice"))
    assert handle.model_id == "abc"
    record.dispose()
    record.dispose()
    assert workers.unloaded == ["abc"]


def test_a_pytorch_spec_without_the_rpc_worker_fails_to_load(tmp_path):
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"))
    with pytest.raises(ValueError, match="RPC worker"):
        managers.build_model_record(None, spec, None, {"model_id": "abc"}, None)


def test_the_worker_registry_refuses_a_second_model_under_one_id(tmp_path):
    import torch

    path = tmp_path / "w.pt"
    torch.save({"bias": torch.tensor(1.0)}, str(path))
    payload = {"model_class": AddConst, "model_path": str(path), "name": "w"}
    try:
        info = rpc_worker._rpc_load("unit-m1", payload)
        assert info["device"] == "cpu" and info["fingerprint"].startswith("sha256:")
        with pytest.raises(ValueError, match="already loaded"):
            rpc_worker._rpc_load("unit-m1", payload)
        out = rpc_worker._rpc_infer("unit-m1", torch.zeros(2), "cpu")
        assert out.tolist() == [1.0, 1.0]
        assert rpc_worker._rpc_unload("unit-m1") is True
        with pytest.raises(rpc_worker.UnknownModelError):
            rpc_worker._rpc_infer("unit-m1", torch.zeros(2), "cpu")
        assert rpc_worker._rpc_unload("unit-m1") is False
    finally:
        rpc_worker._rpc_unload("unit-m1")


def test_the_worker_refuses_a_catalog_model_it_cannot_build(tmp_path):
    import torch
    import torchvision

    path = tmp_path / "r18.pt"
    torch.save(torchvision.models.resnet18(weights=None, num_classes=5).state_dict(), str(path))
    payload = {
        "model_class": "torchvision:resnet18",
        "model_path": str(path),
        "name": "r18",
        "catalog_args": {"num_classes": 7},
    }
    before = rpc_worker._rpc_status()["load_calls"]
    try:
        reply = rpc_worker._rpc_load("unit-refused", payload)
        assert set(reply) == {"model_id", "refused"}
        refused = reply["refused"]
        assert refused["unavailable"] is False
        assert refused["message"].startswith(f"weights file {path} does not match ")
        assert "\n" not in refused["message"]
        assert refused["traceback"].startswith("Traceback")
        assert "CatalogError" in refused["traceback"]
        status = rpc_worker._rpc_status()
        assert status["load_calls"] == before + 1
        assert "unit-refused" not in status["models"]
        # Nothing was kept under the ID: a model that fits loads under it.
        payload["catalog_args"] = {}
        assert "refused" not in rpc_worker._rpc_load("unit-refused", payload)
        assert rpc_worker._rpc_status()["load_calls"] == before + 2
    finally:
        rpc_worker._rpc_unload("unit-refused")


def test_a_refusal_is_raised_in_the_server_as_the_catalog_error(monkeypatch, caplog, tmp_path):
    replies = []

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        assert func is rpc_worker._rpc_load
        return {"model_id": args[0], "refused": replies.pop(0)}

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, _ = _running_manager(monkeypatch)
    spec = PytorchModelSpec(
        model_class="torchvision:resnet18", model_path=str(tmp_path / "w.pt"), name="w"
    )
    replies.append({"message": "does not match", "unavailable": False, "traceback": "Traceback X"})
    with caplog.at_level(logging.DEBUG, logger=managers.logger.name):
        with pytest.raises(catalog.CatalogError) as refused:
            manager.load("m1", spec)
    assert type(refused.value) is catalog.CatalogError and str(refused.value) == "does not match"
    assert any(
        r.levelno == logging.DEBUG and "Traceback X" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]

    replies.append({"message": "needs torchgeo", "unavailable": True, "traceback": ""})
    with pytest.raises(catalog.CatalogUnavailableError, match="needs torchgeo"):
        manager.load("m2", spec)


def test_a_catalog_refusal_is_a_request_error_on_the_wire():
    from nxmndr.server.server import InferenceService

    def failed_load(cause):
        try:
            try:
                raise cause
            except Exception as exc:
                raise managers.ModelLoadError(f"loading model m1 (pytorch) failed: {exc}") from exc
        except managers.ModelLoadError as exc:
            return exc

    mismatch = failed_load(catalog.CatalogError("weights file w.pt does not match x"))
    assert InferenceService._cache_error_status(mismatch) == grpc.StatusCode.INVALID_ARGUMENT
    assert InferenceService._cache_error_details(mismatch) == "weights file w.pt does not match x"
    missing = failed_load(catalog.CatalogUnavailableError("torchgeo:unet needs torchgeo"))
    assert InferenceService._cache_error_status(missing) == grpc.StatusCode.FAILED_PRECONDITION
    assert InferenceService._cache_error_details(missing) == "torchgeo:unet needs torchgeo"
    other = failed_load(RuntimeError("worker gone"))
    assert InferenceService._cache_error_status(other) == grpc.StatusCode.INTERNAL
    assert InferenceService._cache_error_details(other) == str(other)


def _listener():
    import socket

    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.1)
    return listener, listener.getsockname()[1]


def test_the_ready_handshake_accepts_only_its_child_and_keeps_its_deadline():
    import socket
    import threading
    import time

    listener, port = _listener()
    proc = _Proc()
    stop = threading.Event()

    def stray():  # a local process that keeps connecting with the wrong line
        while not stop.is_set():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1) as conn:
                    conn.sendall(b"ready 1\n")
            except OSError:
                pass
            time.sleep(0.01)

    thread = threading.Thread(target=stray, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(managers.RpcWorkerStartError, match="within 0.5 s"):
            managers.RpcWorkerManager._await_ready(proc, listener, 0.5)
        assert time.monotonic() - started < 2.0
    finally:
        stop.set()
        thread.join(timeout=5)

    with socket.create_connection(("127.0.0.1", port), timeout=1) as conn:
        conn.sendall(f"ready {proc.pid}\n".encode("ascii"))
        managers.RpcWorkerManager._await_ready(proc, listener, 5.0)  # returns: its own child
    proc.alive = False
    with pytest.raises(managers.RpcWorkerStartError, match="exited with code None"):
        managers.RpcWorkerManager._await_ready(proc, listener, 5.0)
    listener.close()


def test_the_spawned_worker_applies_the_cudnn_policy_then_runs_on_its_device(monkeypatch):
    calls = []
    monkeypatch.setattr(rpc_worker, "configure_cudnn", lambda: calls.append(("cudnn",)))
    monkeypatch.setattr(rpc_worker, "run_worker", lambda *a, **kw: calls.append(("run", a, kw)))
    managers._run_rpc_worker("127.0.0.1", 1, 2, 3.0, "cuda:0")
    assert [c[0] for c in calls] == ["cudnn", "run"]
    assert calls[1][2]["device"] == "cuda:0" and calls[1][2]["ready_port"] == 2


def test_the_server_gives_its_rpc_worker_the_first_gpu_of_its_device_plan(monkeypatch, tmp_path):
    from nxmndr.server import server

    assert server._rpc_worker_device([{"type": "gpu", "ordinal": 1, "id": "cuda:1"}]) == "cuda:1"
    assert server._rpc_worker_device([{"type": "cpu", "ordinal": 0, "id": "cpu:0"}]) == "cpu"
    assert server._rpc_worker_device([]) == "cpu"

    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 2)
    service = server.InferenceService(model_cache_dir=tmp_path / "gpu", max_cores=2)
    assert service.rpc_manager.device == "cuda:0"
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)
    service = server.InferenceService(model_cache_dir=tmp_path / "cpu", max_cores=2)
    assert service.rpc_manager.device == "cpu"


def test_a_load_that_fails_in_the_server_is_abandoned_in_the_worker(monkeypatch, tmp_path):
    calls = []

    def fake_rpc_sync(to, func, args=(), kwargs=None, timeout=None):
        calls.append((func, args, timeout))
        if func is rpc_worker._rpc_load:
            raise RuntimeError("RPC ran for more than set timeout (300000 ms)")
        return False

    monkeypatch.setattr(managers.torch_rpc, "rpc_sync", fake_rpc_sync)
    manager, _ = _running_manager(monkeypatch)
    spec = PytorchModelSpec(model_class=AddConst, model_path=str(tmp_path / "w.pt"), name="w")
    with pytest.raises(RuntimeError, match="set timeout"):
        manager.load("m1", spec)
    assert [(func, args[0], timeout) for func, args, timeout in calls[:1]] == [
        (rpc_worker._rpc_load, "m1", managers.RPC_LOAD_TIMEOUT_SECONDS)
    ]
    assert [(func, args) for func, args, _ in calls[1:]] == [(rpc_worker._rpc_unload, ("m1", True))]


def test_a_load_abandoned_while_it_builds_drops_its_model(tmp_path):
    import threading

    import torch

    from tst.support import rpc_models

    path = rpc_models.save_add_const(tmp_path / "w.pt", 1.0)
    payload = {"model_class": rpc_models.GatedAddConst, "model_path": path, "name": "w"}
    rpc_models.GATED_STARTED.clear()
    rpc_models.GATED_RELEASE.clear()
    replies = []
    thread = threading.Thread(target=lambda: replies.append(rpc_worker._rpc_load("unit-gated", payload)))
    thread.start()
    try:
        assert rpc_models.GATED_STARTED.wait(timeout=30)
        assert rpc_worker._rpc_status()["loading"] == ["unit-gated"]
        assert rpc_worker._rpc_unload("unit-gated", True) is False  # still building
    finally:
        rpc_models.GATED_RELEASE.set()
        thread.join(timeout=30)
    try:
        assert replies == [{"model_id": "unit-gated", "abandoned": True}]
        status = rpc_worker._rpc_status()
        assert "unit-gated" not in status["models"] and status["loading"] == []
        # Nothing is left under the ID: a later load of it is served.
        payload["model_class"] = AddConst
        assert rpc_worker._rpc_load("unit-gated", payload)["device"] == "cpu"
        assert rpc_worker._rpc_infer("unit-gated", torch.zeros(1), "cpu").tolist() == [1.0]
    finally:
        rpc_worker._rpc_unload("unit-gated")


def test_a_load_abandoned_before_it_arrives_is_not_built(tmp_path):
    from tst.support import rpc_models

    path = rpc_models.save_add_const(tmp_path / "w.pt", 1.0)
    payload = {"model_class": AddConst, "model_path": path, "name": "w"}
    try:
        assert rpc_worker._rpc_unload("unit-early", True) is False
        assert rpc_worker._rpc_load("unit-early", payload) == {
            "model_id": "unit-early",
            "abandoned": True,
        }
        assert "unit-early" not in rpc_worker._rpc_status()["models"]
    finally:
        rpc_worker._rpc_unload("unit-early")


def test_the_fingerprint_is_taken_on_the_cpu_before_the_model_moves(monkeypatch, tmp_path):
    from tst.support import rpc_models

    events, taken = [], []

    class MovingAddConst(AddConst):
        def to(self, *args, **kwargs):
            events.append("to")
            return self

    real_fingerprint = rpc_worker._fingerprint

    def fingerprint(model):
        events.append("fp")
        taken.append(real_fingerprint(model))
        return taken[-1]

    monkeypatch.setattr(rpc_worker, "_fingerprint", fingerprint)
    monkeypatch.setattr(rpc_worker.torch.cuda, "is_available", lambda: True)
    path = rpc_models.save_add_const(tmp_path / "w.pt", 1.0)
    payload = {"model_class": MovingAddConst, "model_path": path, "name": "w"}
    try:
        info = rpc_worker._rpc_load("unit-fp", payload, "cuda:0")
        assert events == ["fp", "to"]
        assert info["fingerprint"] == taken[0]
        assert rpc_worker._rpc_status()["models"]["unit-fp"]["fingerprint"] == taken[0]
    finally:
        rpc_worker._rpc_unload("unit-fp")


def test_a_model_device_comes_from_its_parameters_then_its_buffers():
    import torch

    class BufferOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.zeros((), device="meta"))  # stands for a GPU

    assert rpc_worker._model_device(BufferOnly()) == "meta"
    assert rpc_worker._model_device(AddConst().to("meta")) == "meta"
    assert rpc_worker._model_device(torch.nn.Identity()) == "cpu"  # neither: cpu


def _accelerator():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda:0" if torch.cuda.is_available() else None


@pytest.mark.skipif(_accelerator() is None, reason="needs an MPS or CUDA device")
def test_on_a_gpu_the_worker_moves_the_input_and_every_output_tensor(tmp_path):
    import torch

    from tst.support import rpc_models

    class BufferAddConst(torch.nn.Module):  # no parameters: the device is its buffer's
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.zeros(()))

        def forward(self, x):
            return x + self.bias

    class NestedAddConst(AddConst):  # a dict output, with a list and a tuple inside
        def forward(self, x):
            y = x + self.bias
            return {"output": y, "extra": [y * 2, (y * 3,)]}

    device = _accelerator()
    path = rpc_models.save_add_const(tmp_path / "w.pt", 1.0)
    try:
        for model_id, model_class in (("unit-gpu-buf", BufferAddConst), ("unit-gpu-nest", NestedAddConst)):
            payload = {"model_class": model_class, "model_path": path, "name": "w"}
            info = rpc_worker._rpc_load(model_id, payload, device)
            assert info["device"].split(":")[0] == device.split(":")[0], info
        out = rpc_worker._rpc_infer("unit-gpu-buf", torch.zeros(2), "cpu")
        assert out.device.type == "cpu" and out.tolist() == [1.0, 1.0]
        out = rpc_worker._rpc_infer("unit-gpu-nest", torch.zeros(2), "cpu")
        tensors = [out["output"], out["extra"][0], out["extra"][1][0]]
        assert [t.device.type for t in tensors] == ["cpu"] * 3  # TensorPipe sends CPU tensors only
        assert [t.tolist() for t in tensors] == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    finally:
        rpc_worker._rpc_unload("unit-gpu-buf")
        rpc_worker._rpc_unload("unit-gpu-nest")


@pytest.mark.parametrize(
    "state, first",
    [
        ({}, "missing bias"),
        ({"bias": 1.0, "scale": 2.0}, "unexpected scale"),
    ],
)
def test_a_registered_class_loads_strictly_and_a_mismatch_is_refused(tmp_path, state, first):
    import torch

    path = tmp_path / "w.pt"
    torch.save({key: torch.tensor(value) for key, value in state.items()}, str(path))
    payload = {"model_class": AddConst, "model_path": str(path), "name": "w"}
    try:
        reply = rpc_worker._rpc_load("unit-strict", payload)
        assert set(reply) == {"model_id", "refused"}, reply
        refused = reply["refused"]
        assert refused["message"].startswith(
            f"weights file {path} does not match AddConst (strict load): {first}"
        ), refused["message"]
        assert "\n" not in refused["message"] and refused["unavailable"] is False
        assert "unit-strict" not in rpc_worker._rpc_status()["models"]
    finally:
        rpc_worker._rpc_unload("unit-strict")


@pytest.mark.parametrize(
    "case",
    [
        "nested",
        "nested_checkpoint",
        "extra_value",
        "not_a_mapping",
        "value_not_a_tensor",
        "unreadable",
        "missing",
        "load_error",
    ],
)
def test_every_strict_load_failure_of_a_registered_class_is_a_one_line_refusal(tmp_path, case):
    import argparse

    import torch

    class FailingAddConst(AddConst):
        def load_state_dict(self, state, strict=True):
            raise RuntimeError("cannot copy\nsecond line")

    path, bias = tmp_path / "w.pt", torch.tensor(1.0)
    files = {
        "nested": {"state_dict": {"bias": bias}},
        "nested_checkpoint": {"state_dict": {"bias": bias}, "epoch": 3},
        "extra_value": {"bias": bias, "version": 2},
        "not_a_mapping": [bias],
        "value_not_a_tensor": {"bias": 1.0},  # names match; load_state_dict refuses the value
        "unreadable": {"bias": bias, "hparams": argparse.Namespace(lr=0.1)},  # weights_only refuses
        "load_error": {"bias": bias},
    }
    if case in files:
        torch.save(files[case], str(path))
    mismatch = f"weights file {path} does not match AddConst (strict load): "
    expected = {
        "nested": mismatch + "missing bias (first of 2 differing entries: 1 missing, 1 unexpected)",
        "nested_checkpoint": mismatch
        + "missing bias (first of 3 differing entries: 1 missing, 2 unexpected)",
        "extra_value": mismatch + "unexpected version (first of 1 differing entries: 1 unexpected)",
        "not_a_mapping": f"AddConst: weights file {path} is not a state dict (a mapping of entry "
        "names to values); it holds list",
        "value_not_a_tensor": f"weights file {path} does not load into AddConst: RuntimeError: "
        'While copying the parameter named "bias", expected torch.Tensor',
        "unreadable": f"AddConst: weights file {path} is not a PyTorch state dict (UnpicklingError",
        "missing": f"AddConst: weights file {path} not found",
        "load_error": f"weights file {path} does not load into {FailingAddConst.__qualname__}: "
        "RuntimeError: cannot copy",
    }[case]
    model_class = FailingAddConst if case == "load_error" else AddConst
    payload = {"model_class": model_class, "model_path": str(path), "name": "w"}
    try:
        reply = rpc_worker._rpc_load("unit-strict-file", payload)
        assert set(reply) == {"model_id", "refused"}, reply
        message = reply["refused"]["message"]
        assert message.startswith(expected), message
        assert "\n" not in message and reply["refused"]["unavailable"] is False
    finally:
        rpc_worker._rpc_unload("unit-strict-file")


@pytest.mark.parametrize("kind", ["extra_state", "quantized"])
def test_a_registered_class_loads_its_own_state_dict_with_values_that_are_not_tensors(
    tmp_path, kind
):
    import torch

    class ExtraStateAddConst(AddConst):  # get_extra_state() is a dict in its state_dict()
        def __init__(self):
            super().__init__()
            self.version = 0

        def get_extra_state(self):
            return {"version": self.version}

        def set_extra_state(self, state):
            self.version = state["version"]

    if kind == "quantized" and torch.backends.quantized.engine == "none":
        engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
        if not engines:
            pytest.skip("this torch build has no quantized engine")
        # torch cannot select "none" again, so the engine stays set; no other test uses one.
        torch.backends.quantized.engine = engines[0]

    class QuantizedLinear(torch.ao.nn.quantized.Linear):  # a dtype and a tuple of packed params
        def __init__(self):
            super().__init__(4, 3)

    saved = ExtraStateAddConst() if kind == "extra_state" else QuantizedLinear()
    if kind == "extra_state":
        saved.version = 7
        with torch.no_grad():
            saved.bias.fill_(2.0)
    else:
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        saved.set_weight_bias(
            torch.quantize_per_tensor(weight, 0.1, 0, torch.qint8), torch.ones(3)
        )
    path = tmp_path / "w.pt"
    torch.save(saved.state_dict(), str(path))
    assert any(not torch.is_tensor(value) for value in saved.state_dict().values())
    payload = {"model_class": type(saved), "model_path": str(path), "name": "w"}
    try:
        info = rpc_worker._rpc_load("unit-own-state", payload)
        assert "refused" not in info, info
        assert info["fingerprint"] == rpc_worker._fingerprint(saved)  # every entry loaded
        if kind == "extra_state":
            assert rpc_worker._worker_state["models"]["unit-own-state"].model.version == 7
    finally:
        rpc_worker._rpc_unload("unit-own-state")


@pytest.mark.parametrize("state", ["stopped", "broken"])
def test_a_server_after_a_stopped_one_in_the_process_is_refused_with_the_reason(monkeypatch, state):
    earlier = managers.RpcWorkerManager()
    earlier._state = state
    monkeypatch.setattr(managers, "_process_owner", earlier)
    monkeypatch.setattr(managers.dist, "is_initialized", lambda: pytest.fail("torch reached"))
    with pytest.raises(managers.RpcWorkerStartError, match="new process"):
        managers.RpcWorkerManager().ensure_started()


def test_the_service_shutdown_stops_the_rpc_worker_after_the_cache(tmp_path, monkeypatch):
    from nxmndr.server import server

    service = server.InferenceService(model_cache_dir=tmp_path / "cache", max_cores=1)
    calls = []
    monkeypatch.setattr(
        service.model_manager, "shutdown", lambda drain_timeout_s=None: calls.append("cache")
    )
    monkeypatch.setattr(service.rpc_manager, "shutdown", lambda: calls.append("rpc"))
    service.shutdown(grace=0)
    assert calls == ["cache", "rpc"]


def test_the_rpc_group_listens_on_loopback_unless_an_address_is_given(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    assert managers.RpcWorkerManager(external_worker=True)._addr == "127.0.0.1"
    assert managers.RpcWorkerManager(external_worker=False)._addr == "127.0.0.1"
    monkeypatch.setenv("MASTER_ADDR", "")  # set but empty: a bind on it takes every interface
    assert managers.RpcWorkerManager(external_worker=True)._addr == "127.0.0.1"
    assert managers.RpcWorkerManager(external_worker=False)._addr == "127.0.0.1"
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.10")
    assert managers.RpcWorkerManager(external_worker=True)._addr == "192.0.2.10"


def test_the_external_handshake_connection_tracks_the_worker_process():
    import os
    import socket
    import threading
    import time

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        ready_port = probe.getsockname()[1]
    joined = {}

    def wait_for_worker():
        joined["worker"] = managers._await_external("127.0.0.1", ready_port - 1, 10.0)

    waiter = threading.Thread(target=wait_for_worker)
    waiter.start()
    conn = rpc_worker._join_server("127.0.0.1", ready_port, 1.0)  # retries until the server listens
    waiter.join(timeout=10)
    worker = joined["worker"]
    try:
        assert worker.pid == os.getpid() and worker.host == "127.0.0.1"
        assert worker.is_alive()
        conn.close()  # what the OS does when the worker process exits
        deadline = time.monotonic() + 5
        while worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not worker.is_alive()
    finally:
        worker.close()


def test_a_disconnected_external_worker_fails_the_next_load_with_its_reason():
    import socket

    ours, theirs = socket.socketpair()
    manager = managers.RpcWorkerManager(external_worker=True)
    manager._state = "running"
    manager._proc = managers._ExternalWorker(77, "192.0.2.9", ours)
    try:
        assert manager.is_alive()
        theirs.close()
        assert not manager.is_alive()
        with pytest.raises(
            managers.RpcWorkerStartError, match=r"\(pid 77 on 192\.0\.2\.9\) disconnected"
        ):
            manager.ensure_started()
    finally:
        ours.close()


def test_an_external_worker_may_be_remote_only_on_a_non_loopback_address(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    assert not managers.RpcWorkerManager(external_worker=True).worker_may_be_remote
    for addr in ("127.0.0.1", "127.0.1.1", "localhost", "::1"):
        manager = managers.RpcWorkerManager(master_addr=addr, external_worker=True)
        assert not manager.worker_may_be_remote, addr
    for addr in ("192.0.2.10", "0.0.0.0", "gpu-host"):
        manager = managers.RpcWorkerManager(master_addr=addr, external_worker=True)
        assert manager.worker_may_be_remote, addr
    spawned = managers.RpcWorkerManager(master_addr="192.0.2.10", external_worker=False)
    assert not spawned.worker_may_be_remote


def test_uploaded_pytorch_weights_are_refused_when_the_worker_may_be_remote(monkeypatch, tmp_path):
    import hashlib

    from nxmndr.inference import inference_pb2
    from nxmndr.server.server import InferenceService, _LoadRequestError
    from tst.support import rpc_models

    rpc_models.register()
    spec = inference_pb2.ModelSpec(
        format=inference_pb2.PYTORCH, model_class=rpc_models.ADD_CONST_CLASS, artifact=b"weights"
    )
    sha = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setenv("NXMNDR_RPC_EXTERNAL_WORKER", "1")
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.10")
    remote = InferenceService(model_cache_dir=tmp_path / "remote", max_cores=1)
    with pytest.raises(_LoadRequestError) as refused:
        remote._prepare_load(spec, "c1")
    assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "send source" in str(refused.value)
    assert not [p for p in (tmp_path / "remote").rglob("*") if sha in p.name]  # nothing written
    onnx = inference_pb2.ModelSpec(format=inference_pb2.ONNX, artifact=b"weights")
    assert remote._prepare_load(onnx, "c2").model_spec.model_path  # loads in the server

    monkeypatch.delenv("MASTER_ADDR")  # loopback: the worker shares this host's files
    local = InferenceService(model_cache_dir=tmp_path / "local", max_cores=1)
    assert sha in local._prepare_load(spec, "c3").model_spec.model_path
