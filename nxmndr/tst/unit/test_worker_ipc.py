# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression coverage for the data-only worker boundary."""

import json
import multiprocessing
import pickle
import threading

import pytest
import torch

from nxmndr.server import rpc_worker, worker_ipc


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_tensor_and_nested_result_round_trip(dtype):
    parent, child = multiprocessing.Pipe()
    try:
        tensor = torch.arange(6).reshape(2, 3).to(dtype).t()
        value = {"output": (tensor, [torch.empty(0, 3, dtype=dtype), torch.tensor(2)])}
        worker_ipc.send_message(parent, value)
        result = worker_ipc.receive_message(child)
        assert isinstance(result["output"], tuple)
        torch.testing.assert_close(result["output"][0], tensor)
        torch.testing.assert_close(result["output"][1][0], value["output"][1][0])
        torch.testing.assert_close(result["output"][1][1], torch.tensor(2))
    finally:
        parent.close()
        child.close()


def test_pickled_payload_is_never_executed(capsys):
    class Payload:
        def __reduce__(self):
            return print, ("UNEXPECTED_PICKLE_EXECUTION",)

    parent, child = multiprocessing.Pipe()
    try:
        parent.send_bytes(pickle.dumps(Payload()))
        with pytest.raises((UnicodeDecodeError, ValueError)):
            worker_ipc.receive_message(child)
        assert "UNEXPECTED_PICKLE_EXECUTION" not in capsys.readouterr().out
        with pytest.raises(TypeError, match="unsupported worker value"):
            worker_ipc.send_message(parent, Payload())
    finally:
        parent.close()
        child.close()


@pytest.mark.parametrize(
    "value",
    [
        {"type": "python", "module": "os", "name": "system"},
        {"type": "tensor", "dtype": "object", "shape": [1], "data": ""},
        {"type": "tensor", "dtype": "torch.float32", "shape": [-1], "data": ""},
        {"type": "tensor", "dtype": "torch.float32", "shape": [1], "data": ""},
    ],
)
def test_decoder_refuses_objects_and_invalid_tensor_buffers(value):
    parent, child = multiprocessing.Pipe()
    try:
        parent.send_bytes(json.dumps(value).encode())
        with pytest.raises(ValueError):
            worker_ipc.receive_message(child)
    finally:
        parent.close()
        child.close()


def test_worker_dispatch_refuses_arbitrary_functions_and_imports(monkeypatch):
    monkeypatch.setattr(rpc_worker, "_worker_state", dict(rpc_worker._worker_state))
    monkeypatch.setattr(rpc_worker, "_worker_model_classes", {})
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(target=rpc_worker.run_ipc_worker, args=(child, {}))
    thread.start()
    assert worker_ipc.receive_message(parent)["ready"]
    client = worker_ipc.WorkerClient(parent)
    try:
        with pytest.raises(RuntimeError, match="unknown worker operation"):
            client.call("builtins.eval", ("print('unexpected')",), timeout=5)
        reply = client.call(
            "_rpc_load", ("untrusted", {"model_class": "os.system", "model_path": ""}), timeout=5
        )
        assert "not registered" in reply["refused"]["message"]
        assert client.call("_rpc_status", timeout=5)["status"] == "ok"
        client.call("_rpc_stop", timeout=5)
    finally:
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_call_deadline_includes_a_blocked_pipe_write(monkeypatch):
    parent, child = multiprocessing.Pipe()
    started, release = threading.Event(), threading.Event()

    def blocked_send(*args):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(worker_ipc, "send_message", blocked_send)
    client = worker_ipc.WorkerClient(parent)
    try:
        with pytest.raises(TimeoutError, match="timeout after"):
            client.call("_rpc_status", timeout=0.1)
        assert started.is_set()
        with pytest.raises(TimeoutError, match="timeout after"):
            client.call("_rpc_stop", timeout=0.1)
    finally:
        client.close()
        child.close()
        release.set()
        client._writer.join(timeout=5)
        client._reader.join(timeout=5)
    assert not client._writer.is_alive()
    assert not client._reader.is_alive()


def test_a_pickle_reply_closes_the_channel_without_executing(capsys):
    class Payload:
        def __reduce__(self):
            return print, ("UNEXPECTED_PICKLE_EXECUTION",)

    parent, child = multiprocessing.Pipe()
    client = worker_ipc.WorkerClient(parent)

    def reply():
        worker_ipc.receive_message(child)
        child.send_bytes(pickle.dumps(Payload()))

    responder = threading.Thread(target=reply)
    responder.start()
    try:
        with pytest.raises(RuntimeError, match="connection closed"):
            client.call("_rpc_status", timeout=5)
        assert "UNEXPECTED_PICKLE_EXECUTION" not in capsys.readouterr().out
    finally:
        client.close()
        child.close()
        responder.join(timeout=5)
