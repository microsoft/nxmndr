# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The compatibility provider uses a private child, never a distributed group."""

import socket

import numpy as np
import pytest
import torch.distributed as dist
import torch.distributed.rpc as rpc

from nxmndr.inference import InferenceSession, TorchRpcInferenceProvider
from nxmndr.models import PytorchModelSpec
from tst.support import rpc_models

pytestmark = pytest.mark.integration


def test_provider_load_predict_reload_and_shutdown_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network Torch RPC must not be initialized")

    monkeypatch.setattr(dist, "init_process_group", forbidden)
    monkeypatch.setattr(rpc, "init_rpc", forbidden)
    original_socket = socket.socket

    def local_socket(family=socket.AF_INET, *args, **kwargs):
        if family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail("worker startup must not create an IP socket")
        return original_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", local_socket)
    rpc_models.register()
    path = rpc_models.save_add_const(tmp_path / "weights.pt", 2)
    spec = PytorchModelSpec(model_class=rpc_models.AddConst, model_path=path)
    for _ in range(2):
        with TorchRpcInferenceProvider() as provider:
            assert provider.ping()["status"] == "ok"
            session = InferenceSession(spec, provider)
            arr = np.zeros((1, 3, 4, 4), dtype=np.float32)
            np.testing.assert_array_equal(session.run(arr), arr + 2)
            provider.load_spec(spec)
            np.testing.assert_array_equal(provider.predict(arr), arr + 2)
            proc = provider._manager._proc
        assert not proc.is_alive()


def test_provider_refuses_a_network_worker():
    with pytest.raises(ValueError, match="RemoteInferenceProvider"):
        TorchRpcInferenceProvider(master_addr="192.0.2.1")
