# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tiny PyTorch models and RPC worker targets for the PyTorch RPC service tests.

They live in an importable module because the ``spawn`` worker unpickles model
classes and process targets by reference.

``AddConst`` returns its input plus a scalar ``bias`` read from the weights file, so
two weights files give two models with distinguishable outputs from one class.
``SleepyAddConst`` first sleeps for the ``delay`` seconds in its weights file.
``SlowAddConst`` takes 3 s to construct (a slow load); ``HungAddConst`` takes 60 s
(a load still building when the server stops); ``GatedAddConst`` sets
``GATED_STARTED`` when its construction starts and waits for ``GATED_RELEASE``, so a
test can act while a load is in progress.
The ``*_worker`` functions replace ``RpcWorkerManager.worker_target`` to make the
child fail before it is ready to join the RPC group.
"""

from __future__ import annotations

import os
import threading
import time

import torch

ADD_CONST_CLASS = "nxmndr_test_add_const"
SLEEPY_ADD_CONST_CLASS = "nxmndr_test_sleepy_add_const"
SLOW_ADD_CONST_CLASS = "nxmndr_test_slow_add_const"
HUNG_ADD_CONST_CLASS = "nxmndr_test_hung_add_const"
GATED_STARTED = threading.Event()
GATED_RELEASE = threading.Event()


class AddConst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x + self.bias


class SleepyAddConst(AddConst):
    def __init__(self):
        super().__init__()
        self.register_buffer("delay", torch.zeros(()))

    def forward(self, x):
        time.sleep(float(self.delay))
        return super().forward(x)


class SlowAddConst(AddConst):
    def __init__(self):
        time.sleep(3.0)
        super().__init__()


class HungAddConst(AddConst):
    def __init__(self):
        time.sleep(60.0)
        super().__init__()


class GatedAddConst(AddConst):
    def __init__(self):
        GATED_STARTED.set()
        GATED_RELEASE.wait(timeout=30)
        super().__init__()


def save_add_const(path, bias: float) -> str:
    torch.save({"bias": torch.tensor(float(bias))}, str(path))
    return str(path)


def save_sleepy_add_const(path, bias: float, delay: float) -> str:
    torch.save({"bias": torch.tensor(float(bias)), "delay": torch.tensor(float(delay))}, str(path))
    return str(path)


def register() -> None:
    from nxmndr.models import register_pytorch_model

    register_pytorch_model(ADD_CONST_CLASS, AddConst)
    register_pytorch_model(SLEEPY_ADD_CONST_CLASS, SleepyAddConst)
    register_pytorch_model(SLOW_ADD_CONST_CLASS, SlowAddConst)
    register_pytorch_model(HUNG_ADD_CONST_CLASS, HungAddConst)


def crashing_worker(master_addr, master_port, ready_port, startup_timeout_s, device="cpu"):
    """Exits with code 3 before it reports ready (a broken worker environment)."""

    os._exit(3)


def silent_worker(master_addr, master_port, ready_port, startup_timeout_s, device="cpu"):
    """Never reports ready (a hung import); exits only when its parent is gone."""

    parent = os.getppid()
    deadline = time.monotonic() + 600
    while os.getppid() == parent and time.monotonic() < deadline:
        time.sleep(0.1)
