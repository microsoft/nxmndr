# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A real nxmndr gRPC server in its own process, for the PyTorch RPC service tests.

Run as ``python -m tst.support.rpc_service_process`` from the ``nxmndr`` directory.
Torch joins one RPC group per process, so every such test gets a fresh server
process; the pytest process only speaks gRPC to it.

Protocol (stdout lines prefixed ``@@ ``, JSON): first ``{"port": ..., "pid": ...}``.
Then one reply per stdin command:

- ``status``: the RPC worker manager's state, the live worker children
  (``multiprocessing.active_children``) and the worker's ``_rpc_status`` (loaded
  model IDs with device and fingerprint), or the error that prevented it.
- ``target real|crash|silent``: the child process target of the next worker start.
- ``timeout <seconds>``: the startup timeout of the next worker start.
- ``load_timeout <seconds>``: the bound on each later model load in the worker
  (``managers.RPC_LOAD_TIMEOUT_SECONDS``).
- ``stop`` (or end of input): ``stop_server``, which also stops the RPC worker;
  replies with the manager's state, the live worker children afterwards and the
  seconds ``stop_server`` took, and exits.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time


def _say(payload) -> None:
    sys.stdout.write("@@ " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _children():
    return sorted(p.pid for p in multiprocessing.active_children())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--target", choices=("real", "crash", "silent"), default="real")
    parser.add_argument("--startup-timeout", type=float, default=0.0)
    args = parser.parse_args()

    os.environ["NXMNDR_MODEL_CACHE_CAPACITY"] = str(args.capacity)
    from nxmndr.server import managers
    from nxmndr.server.server import create_server, stop_server
    from tst.support import rpc_models

    rpc_models.register()
    targets = {
        "real": managers._run_rpc_worker,
        "crash": rpc_models.crashing_worker,
        "silent": rpc_models.silent_worker,
    }
    managers.RpcWorkerManager.worker_target = staticmethod(targets[args.target])
    server, port, service = create_server("127.0.0.1", 0, max_cores=4)
    rpc = service.rpc_manager
    if args.startup_timeout:
        rpc.startup_timeout_s = args.startup_timeout
    _say({"port": port, "pid": os.getpid()})

    for line in sys.stdin:
        words = line.split()
        if not words:
            continue
        if words[0] == "status":
            reply = {"state": rpc._state, "alive": rpc.is_alive(), "children": _children()}
            try:
                reply["worker"] = rpc.status()
            except Exception as exc:  # reported, not raised: the test asserts on it
                reply["worker_error"] = f"{type(exc).__name__}: {exc}"
            _say(reply)
        elif words[0] == "target":
            managers.RpcWorkerManager.worker_target = staticmethod(targets[words[1]])
            _say({"target": words[1]})
        elif words[0] == "timeout":
            rpc.startup_timeout_s = float(words[1])
            _say({"timeout": rpc.startup_timeout_s})
        elif words[0] == "load_timeout":
            managers.RPC_LOAD_TIMEOUT_SECONDS = float(words[1])
            _say({"load_timeout": managers.RPC_LOAD_TIMEOUT_SECONDS})
        elif words[0] == "stop":
            break
    started = time.monotonic()
    stop_server(server, service, grace=5.0)
    seconds = round(time.monotonic() - started, 2)
    _say({"stopped": True, "state": rpc._state, "children": _children(), "seconds": seconds})
    return 0


if __name__ == "__main__":
    sys.exit(main())
