# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generic PyTorch models through the real gRPC service and the real PyTorch RPC worker.

Each test starts a fresh server process (``tst.support.rpc_service_process``;
torch joins one RPC group per process) and talks to it only over gRPC with the
canonical client. Nothing is mocked: the server's ``RpcWorkerManager`` starts a
spawned worker, joins its RPC group and calls ``_rpc_load``/``_rpc_infer``/
``_rpc_unload`` with the function objects. The model is ``AddConst`` (input plus
the scalar in its weights file), so each model's output names the model.

Covers worker startup (the first OpenSession starts the worker; startup failures
are bounded and clean up the child), unary Predict and StreamPredict through the real
RPC invocation, and two models loaded at once (they route by identity under
concurrent pinned sessions; eviction and unload free the model in the worker; a
reload gets a new identity). A stop does not wait for a load still building in the worker.

The external-worker tests start the server with ``NXMNDR_RPC_EXTERNAL_WORKER=1`` and
the worker as a separate ``python -m nxmndr.server.rpc_worker`` process. Two put both
on this host's non-loopback address, as a worker on a second machine would be; one
keeps the loopback default and kills the worker, whose models then become unavailable.
The rendezvous store of the default server listens on loopback only, not on that address.
"""

from __future__ import annotations

import json
import os
import queue
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np
import pytest
import torch

from nxmndr.client import InferenceGrpcClient, InferenceGrpcError
from nxmndr.inference import inference_pb2
from tst.support import rpc_models

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed not available"),
]

ROOT = Path(__file__).resolve().parents[2]  # the nxmndr directory
CHIP = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)


class ServiceProcess:
    """One ``rpc_service_process`` server; ``command`` sends a line and returns its reply."""

    def __init__(self, tmp_path: Path, *args: str, env=None):
        self.log_path = tmp_path / "rpc_service_process.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        extra, env = env or {}, _child_env()
        env.update(extra)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tst.support.rpc_service_process", *args],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            env=env,
        )
        self._replies: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        hello = self._reply(timeout=180)
        self.endpoint = f"127.0.0.1:{hello['port']}"

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith("@@ "):
                self._replies.put(json.loads(line[3:]))

    def _reply(self, timeout=120):
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"no reply from the server process; log:\n{self.log_tail()}")

    def command(self, text: str, timeout=120) -> dict:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return self._reply(timeout)

    def client(self) -> InferenceGrpcClient:
        return InferenceGrpcClient(self.endpoint, timeout=120, max_attempts=1)

    def log_tail(self, n=4000) -> str:
        self._log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:]

    def stop(self) -> dict:
        reply = self.command("stop", timeout=120)
        assert self.proc.wait(timeout=60) == 0, self.log_tail()
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self._log.close()


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")) if p
    )
    for var in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "NXMNDR_REMOTE_HOST",
        "NXMNDR_REMOTE_PORT",
        "NXMNDR_RPC_EXTERNAL_WORKER",
    ):
        env.pop(var, None)
    return env


@pytest.fixture
def service_process(tmp_path):
    made = []

    def start(*args, env=None):
        made.append(ServiceProcess(tmp_path, *args, env=env))
        return made[-1]

    yield start
    for proc in made:
        proc.close()


def _spec(path, model_class=rpc_models.ADD_CONST_CLASS) -> inference_pb2.ModelSpec:
    return inference_pb2.ModelSpec(
        format=inference_pb2.PYTORCH,
        source=str(path),
        model_class=model_class,
        name=Path(path).name,
    )


def _array(output, shape, dtype):
    return np.frombuffer(output, dtype=np.dtype(dtype)).reshape(tuple(int(d) for d in shape))


def _predict(client, model_id, chip=CHIP):
    result = client.predict(model_id, chip)
    assert result.metadata.get("model_type") == "pytorch", result.metadata
    return _array(result.output, result.shape, result.dtype)


def _stream(client, session_id, chips, prefix):
    out = {}
    ids = [f"{prefix}{i}" for i in range(len(chips))]
    for resp in client.stream_predict(session_id=session_id, samples=chips, tile_ids=ids):
        meta = dict(resp.metadata)
        assert "error" not in meta, meta
        out[meta["tile_id"]] = _array(resp.output, resp.shape, resp.dtype)
    assert sorted(out) == sorted(ids)
    return [out[i] for i in ids]


def _open(client, session_id, **spec_fields):
    resp = client.open_session(session_id=session_id, spec=inference_pb2.ModelSpec(**spec_fields))
    assert resp.status == "ok", resp.error
    return resp


def _pytorch_model_ids(client):
    listed = client._get_stub().ListModels(inference_pb2.ListModelsRequest(), timeout=30)
    return sorted(m.model_id for m in listed.models if m.format == inference_pb2.PYTORCH)


def _worker_models(proc) -> dict:
    status = proc.command("status")
    assert "worker" in status, status
    return status["worker"]["models"]


def test_first_generic_pytorch_open_session_starts_the_worker_and_serves_both_rpcs(
    tmp_path, service_process
):
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process()
    with proc.client() as client:
        started = time.monotonic()
        opened = client.open_session(session_id="s-first", spec=_spec(weights))
        elapsed = time.monotonic() - started
        assert opened.status == "ok", f"{opened.error}\n{proc.log_tail()}"
        assert opened.model_cache_hit is False
        assert elapsed < 60.0, elapsed  # RPC_STARTUP_TIMEOUT_SECONDS bounds each start phase

        [model_id] = _pytorch_model_ids(client)
        models = _worker_models(proc)
        assert list(models) == [model_id]
        # The spawned worker builds on the first GPU of the server's device plan.
        assert models[model_id]["device"] == ("cuda:0" if torch.cuda.is_available() else "cpu")

        tiles = _stream(client, "s-first", [CHIP, CHIP * 2], "t")
        np.testing.assert_array_equal(tiles[0], CHIP + 1)
        np.testing.assert_array_equal(tiles[1], CHIP * 2 + 1)
        result = client.predict(model_id, CHIP)
        unary = _array(result.output, result.shape, result.dtype)
        assert unary.dtype == np.float32
        np.testing.assert_array_equal(unary, CHIP + 1)
        # The metadata names the device that ran the model: the worker's.
        assert result.metadata["device_id"] == models[model_id]["device"], result.metadata
        assert result.metadata["device_type"] == (
            "gpu" if torch.cuda.is_available() else "cpu"
        ), result.metadata
        # Predict records the record's metadata: it names the worker's model.
        [entry] = [e for e in client.list_model_registry() if e.model_id == model_id]
        assert entry.metadata["rpc_fingerprint"] == models[model_id]["fingerprint"]
        assert client.close_session("s-first").status == "closed"

    stopped = proc.stop()
    assert stopped["state"] == "stopped" and stopped["children"] == [], stopped


def test_a_tile_that_runs_longer_than_five_seconds_is_served(tmp_path, service_process):
    weights = rpc_models.save_sleepy_add_const(tmp_path / "sleepy.pt", 1.0, delay=6.0)
    proc = service_process()
    with proc.client() as client:
        model_id = client.load_model_result(
            "", _spec(weights, rpc_models.SLEEPY_ADD_CONST_CLASS)
        ).model_id
        np.testing.assert_array_equal(_predict(client, model_id), CHIP + 1)

    stopped = proc.stop()
    assert stopped["children"] == [], stopped


def test_two_models_route_by_identity_across_sessions_eviction_and_reload(tmp_path, service_process):
    path_a = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    path_b = rpc_models.save_add_const(tmp_path / "b.pt", 2.0)
    path_c = rpc_models.save_add_const(tmp_path / "c.pt", 3.0)
    proc = service_process("--capacity", "2")
    with proc.client() as client:
        id_a = client.load_model_result("", _spec(path_a)).model_id
        id_b = client.load_model_result("", _spec(path_b)).model_id
        assert id_a != id_b
        models = _worker_models(proc)
        assert set(models) == {id_a, id_b}
        assert models[id_a]["fingerprint"] != models[id_b]["fingerprint"]

        _open(client, "s-a", model_id=id_a)
        _open(client, "s-b", model_id=id_b)

        # Concurrent pinned sessions, plus unary calls on both models meanwhile.
        chips = [CHIP + k for k in range(4)]
        results, errors = {}, []
        barrier = threading.Barrier(3)

        def run_session(session_id):
            try:
                with proc.client() as own:
                    barrier.wait(timeout=30)
                    results[session_id] = _stream(own, session_id, chips, f"{session_id}-")
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=run_session, args=(sid,)) for sid in ("s-a", "s-b")]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=30)
        for _ in range(3):
            np.testing.assert_array_equal(_predict(client, id_a), CHIP + 1)
            np.testing.assert_array_equal(_predict(client, id_b), CHIP + 2)
        for thread in threads:
            thread.join(timeout=120)
        assert not errors, errors
        for chip, got_a, got_b in zip(chips, results["s-a"], results["s-b"]):
            np.testing.assert_array_equal(got_a, chip + 1)
            np.testing.assert_array_equal(got_b, chip + 2)

        # Both slots pinned: a third model is refused; nothing changes beneath the sessions.
        with pytest.raises(InferenceGrpcError) as refused:
            client.load_model_result("", _spec(path_c))
        assert refused.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert set(_worker_models(proc)) == {id_a, id_b}

        # A unpinned and least recently used: loading C evicts it and frees it in the worker.
        assert client.close_session("s-a").status == "closed"
        id_c = client.load_model_result("", _spec(path_c)).model_id
        assert set(_worker_models(proc)) == {id_b, id_c}
        with pytest.raises(InferenceGrpcError) as gone:
            client.predict(id_a, CHIP)
        assert gone.value.code == grpc.StatusCode.NOT_FOUND
        np.testing.assert_array_equal(_stream(client, "s-b", [CHIP], "s-b-again-")[0], CHIP + 2)
        np.testing.assert_array_equal(_predict(client, id_c), CHIP + 3)

        # Reloading A gives a new identity (C, unpinned, is evicted; B stays pinned).
        id_a2 = client.load_model_result("", _spec(path_a)).model_id
        assert id_a2 not in (id_a, id_b, id_c)
        models = _worker_models(proc)
        assert set(models) == {id_b, id_a2}
        np.testing.assert_array_equal(_predict(client, id_a2), CHIP + 1)
        np.testing.assert_array_equal(_predict(client, id_b), CHIP + 2)

        # Unload frees the worker-side model too.
        assert client.close_session("s-b").status == "closed"
        unload = client._get_stub().UnloadModel(
            inference_pb2.UnloadModelRequest(model_id=id_b), timeout=30
        )
        assert unload.success, unload.message
        assert set(_worker_models(proc)) == {id_a2}

    stopped = proc.stop()  # the cache shutdown frees A2 before the worker stops
    assert stopped["children"] == [], stopped


def test_worker_start_failures_are_bounded_clean_up_and_allow_a_retry(tmp_path, service_process):
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process("--target", "crash", "--startup-timeout", "30")
    with proc.client() as client:
        # A child that exits before it is ready fails the first OpenSession at once.
        started = time.monotonic()
        with pytest.raises(InferenceGrpcError) as crashed:
            client.open_session(session_id="s-crash", spec=_spec(weights))
        assert time.monotonic() - started < 20.0
        assert crashed.value.code == grpc.StatusCode.INTERNAL
        assert "exited with code 3" in str(crashed.value), str(crashed.value)
        status = proc.command("status")
        assert status["state"] == "idle" and status["children"] == [], status

        # A child that never reports ready fails after the startup timeout and is terminated.
        proc.command("timeout 3")
        proc.command("target silent")
        started = time.monotonic()
        with pytest.raises(InferenceGrpcError) as silent:
            client.load_model_result("", _spec(weights))
        waited = time.monotonic() - started
        assert 3.0 <= waited < 3.0 + 3 * 2.0 + 10.0, waited  # timeout, then bounded terminate
        assert "not ready to join its RPC group within 3 s" in str(silent.value), str(silent.value)
        status = proc.command("status")
        assert status["state"] == "idle" and status["children"] == [], status

        # Neither failure entered the torch rendezvous, so the next load starts a worker.
        proc.command("timeout 60")
        proc.command("target real")
        retried = client.open_session(session_id="s-retry", spec=_spec(weights))
        assert retried.status == "ok", retried.error
        [model_id] = _pytorch_model_ids(client)
        np.testing.assert_array_equal(_predict(client, model_id), CHIP + 1)

    stopped = proc.stop()
    assert stopped["children"] == [], stopped


def test_a_load_that_times_out_leaves_no_model_in_the_worker(tmp_path, service_process):
    slow = rpc_models.save_add_const(tmp_path / "slow.pt", 1.0)
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 2.0)
    proc = service_process()
    with proc.client() as client:
        proc.command("load_timeout 1")
        with pytest.raises(InferenceGrpcError) as timed_out:
            client.load_model_result("", _spec(slow, rpc_models.SLOW_ADD_CONST_CLASS))
        assert "timeout" in str(timed_out.value), str(timed_out.value)
        # The worker finishes the build it was given, then drops the model.
        deadline = time.monotonic() + 30
        while True:
            worker = proc.command("status")["worker"]
            if not worker["loading"] or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        assert worker["loading"] == [] and worker["models"] == {}, worker

        proc.command("load_timeout 300")
        model_id = client.load_model_result("", _spec(weights)).model_id
        assert list(_worker_models(proc)) == [model_id]
        np.testing.assert_array_equal(_predict(client, model_id), CHIP + 2)

    stopped = proc.stop()
    assert stopped["children"] == [], stopped


def test_stop_does_not_wait_for_a_load_still_building_in_the_worker(tmp_path, service_process):
    weights = rpc_models.save_add_const(tmp_path / "hung.pt", 1.0)
    proc = service_process()
    failed = []

    def load():
        with proc.client() as client:
            try:
                client.load_model_result("", _spec(weights, rpc_models.HUNG_ADD_CONST_CLASS))
            except InferenceGrpcError as exc:
                failed.append(exc)

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while True:
        status = proc.command("status")
        if status.get("worker", {}).get("loading") or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert status["worker"]["loading"], status  # HungAddConst sleeps 60 s in __init__

    stopped = proc.stop()
    # The stop's own bounds: grace 5 s for the gRPC calls and 5 s for the cache, then
    # the worker's _rpc_stop and its bounded terminate. Far below the 60 s load.
    assert stopped["seconds"] < 30.0, stopped
    assert stopped["state"] == "stopped" and stopped["children"] == [], stopped
    thread.join(timeout=30)
    assert len(failed) == 1, failed  # the server ended the load's call


def test_the_rendezvous_store_listens_on_the_rpc_address_only(tmp_path, service_process):
    lan = _lan_address()
    port = _free_port_pair("0.0.0.0")  # free on every interface; P + 1 goes unused here
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process(env={"MASTER_PORT": str(port)})  # spawned worker, loopback default
    with proc.client() as client:
        client.load_model_result("", _spec(weights))
    # torch's own store would listen on every interface, whatever MASTER_ADDR names.
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((lan, port), timeout=5).close()
    proc.stop()


def _lan_address() -> str:
    """This host's non-loopback IPv4 address (the default route's source), or skip."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1; a UDP connect sends nothing
            addr = probe.getsockname()[0]
    except OSError:
        addr = ""
    if not addr or addr.startswith("127.") or addr == "0.0.0.0":
        pytest.skip("no non-loopback IPv4 address on this host")
    return addr


def _free_port_pair(addr: str) -> int:
    """A port P on ``addr`` with P + 1 also free (the rendezvous and the ready line).

    The candidates lie below the ephemeral port ranges (49152 and up on macOS, 32768 and up
    on Linux). A port taken from ``bind((addr, 0))`` is an ephemeral one, and the OS can hand
    it to one of the server process's own sockets before the server binds it as the
    rendezvous port, which failed these tests now and then with "Address already in use".
    """

    rng = random.Random()
    for _ in range(50):
        port = rng.randrange(20000, 32000)
        try:
            with socket.socket() as first, socket.socket() as second:
                first.bind((addr, port))
                second.bind((addr, port + 1))
        except OSError:
            continue
        return port
    pytest.skip("no free port pair")


@pytest.fixture
def external_worker(tmp_path):
    """Start ``python -m nxmndr.server.rpc_worker`` as its own process (not a child of the server).

    Its working directory is ``tmp_path``, the server's is ``ROOT``: a relative weights
    path names a file that only the worker can open.
    """

    made = []

    def start(addr: str, port: int, device: str = "cpu"):
        log = open(tmp_path / f"external_worker_{len(made)}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "nxmndr.server.rpc_worker", "--master", addr,
             "--port", str(port), "--device", device],
            cwd=str(tmp_path), stdout=log, stderr=subprocess.STDOUT, env=_child_env(),
        )
        made.append((proc, log))
        return proc

    yield start
    for proc, log in made:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        log.close()


def _external_env(addr: str, port: int) -> dict:
    return {"NXMNDR_RPC_EXTERNAL_WORKER": "1", "MASTER_ADDR": addr, "MASTER_PORT": str(port)}


def _direct(weights, chip) -> np.ndarray:
    model = rpc_models.AddConst()
    model.load_state_dict(torch.load(weights, weights_only=True))
    with torch.no_grad():
        return model(torch.from_numpy(chip)).numpy()


def test_external_worker_on_a_lan_address_serves_load_session_and_stream(
    tmp_path, service_process, external_worker
):
    addr = _lan_address()
    port = _free_port_pair(addr)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.5)
    assert not (ROOT / "a.pt").exists()
    proc = service_process(env=_external_env(addr, port))
    worker = external_worker(addr, port, device)
    with proc.client() as client:
        # "a.pt" resolves in the worker's directory only: the worker opens the weights.
        model_id = client.load_model_result("", _spec("a.pt")).model_id
        status = proc.command("status")
        assert status["children"] == [], status  # the server spawned nothing
        assert status["worker"]["pid"] == worker.pid, status
        assert status["worker"]["models"][model_id]["device"].split(":")[0] == device, status

        _open(client, "s-ext", model_id=model_id)
        chips = [CHIP, CHIP * 2]
        tiles = _stream(client, "s-ext", chips, "t")
        for chip, tile in zip(chips, tiles):
            np.testing.assert_array_equal(tile, _direct(weights, chip))
        assert client.close_session("s-ext").status == "closed"

    stopped = proc.stop()  # _rpc_stop ends the external worker too
    assert stopped["state"] == "stopped", stopped
    assert worker.wait(timeout=60) == 0


def test_external_worker_that_never_joins_fails_the_load_then_a_late_worker_serves(
    tmp_path, service_process, external_worker
):
    addr = _lan_address()
    port = _free_port_pair(addr)
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    proc = service_process("--startup-timeout", "3", env=_external_env(addr, port))
    with proc.client() as client:
        started = time.monotonic()
        with pytest.raises(InferenceGrpcError) as missing:
            client.load_model_result("", _spec(weights))
        assert 3.0 <= time.monotonic() - started < 3.0 + 10.0
        assert missing.value.code == grpc.StatusCode.FAILED_PRECONDITION, str(missing.value)
        assert f"no external PyTorch RPC worker joined at {addr}:{port} within 3 s" in str(
            missing.value
        ), str(missing.value)
        assert proc.command("status")["state"] == "idle"

        # The failed wait never entered the torch rendezvous: a worker started now serves.
        proc.command("timeout 60")
        worker = external_worker(addr, port)
        model_id = client.load_model_result("", _spec(weights)).model_id
        np.testing.assert_array_equal(_predict(client, model_id), _direct(weights, CHIP))

    proc.stop()
    assert worker.wait(timeout=60) == 0


def test_an_external_worker_that_exits_makes_its_models_unavailable(
    tmp_path, service_process, external_worker
):
    port = _free_port_pair("127.0.0.1")
    weights = rpc_models.save_add_const(tmp_path / "a.pt", 1.0)
    other = rpc_models.save_add_const(tmp_path / "b.pt", 2.0)
    # No MASTER_ADDR: the loopback default, with the worker on this host.
    proc = service_process(env={"NXMNDR_RPC_EXTERNAL_WORKER": "1", "MASTER_PORT": str(port)})
    worker = external_worker("127.0.0.1", port)
    with proc.client() as client:
        model_id = client.load_model_result("", _spec(weights)).model_id
        np.testing.assert_array_equal(_predict(client, model_id), CHIP + 1)
        worker.kill()
        worker.wait(timeout=30)
        deadline = time.monotonic() + 10
        while proc.command("status")["alive"] and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.command("status")["alive"] is False
        with pytest.raises(InferenceGrpcError) as gone:
            client.predict(model_id, CHIP)
        assert gone.value.code == grpc.StatusCode.UNAVAILABLE, str(gone.value)
        with pytest.raises(InferenceGrpcError) as reload:
            client.load_model_result("", _spec(other))
        assert f"(pid {worker.pid} on 127.0.0.1) disconnected" in str(reload.value), str(
            reload.value
        )
    proc.stop()
