# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Top-level test configuration.

Auto-generates synthetic test model weights (PyTorch, ONNX, HuggingFace)
before the test session starts so binary model files do not need to be
checked into git.
"""

from pathlib import Path

import pytest

# Seed of the exported example model, so its weights are the same on every run
# and every machine.
EXAMPLE_MODEL_SEED = 0


def pytest_configure(config):
    """Generate test model artifacts if they are missing."""
    tst_dir = Path(__file__).parent
    example_dir = tst_dir / "example_model"
    integration_dir = tst_dir / "integration" / "example_model"

    from example_model.modeling_exampleconv import export

    # Generate models for unit tests
    export(example_dir, seed=EXAMPLE_MODEL_SEED)
    # Generate models for integration tests
    export(integration_dir, seed=EXAMPLE_MODEL_SEED)


@pytest.fixture(autouse=True)
def _isolated_nxmndr_cache(tmp_path, monkeypatch):
    """Keep every server's registry file and model artifacts in the test's tmp_path.

    Without this, InferenceService writes ~/.cache/nxmndr/model_registry.json and
    downloads into ~/.cache/nxmndr/models, the developer's real cache.
    """
    monkeypatch.setenv("NXMNDR_CACHE_DIR", str(tmp_path / "nxmndr-cache"))
