# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from pathlib import Path

from nxmndr.client import InferenceGrpcClient
from tst.example_model.modeling_exampleconv import export
from tst.integration.remote_test_utils import start_test_server


@pytest.mark.integration
def test_model_registry_list_and_evict(tmp_path):
    repo_root = Path.cwd()
    model_dir = repo_root / "tst" / "example_model"
    model_path = model_dir / "example_model.onnx"
    created_model = False
    if not model_path.exists():
        export(model_dir)
        created_model = True

    server = None
    bound_port = None
    try:
        server, bound_port = start_test_server()
        client = InferenceGrpcClient(f"127.0.0.1:{bound_port}", timeout=10)

        model_id = client.load_model(
            model_id="registry-test",
            spec={"format": "onnx", "source": str(model_path), "name": "registry-test"},
        )
        assert model_id

        entries = client.list_model_registry()
        assert entries, "registry should contain the loaded model"
        entry = next((e for e in entries if e.model_id == model_id), None)
        assert entry is not None
        assert entry.source == str(model_path)

        evicted = client.evict_model_registry_entry(entry.entry_id)
        assert evicted is True

        remaining = [e for e in client.list_model_registry() if e.model_id == model_id]
        assert not remaining

    finally:
        if server:
            server.stop(0)
        if created_model and model_path.exists():
            model_path.unlink()
