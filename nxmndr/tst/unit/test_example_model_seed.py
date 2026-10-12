# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The example model export is deterministic under a seed."""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from tst.conftest import EXAMPLE_MODEL_SEED
from tst.example_model.modeling_exampleconv import export

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example_model"


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _initializers(path):
    model = onnx.load(str(path))
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_seeded_exports_are_identical_and_unseeded_ones_differ(tmp_path):
    first, second, unseeded = tmp_path / "first", tmp_path / "second", tmp_path / "unseeded"
    export(first, seed=EXAMPLE_MODEL_SEED)
    export(second, seed=EXAMPLE_MODEL_SEED)
    export(unseeded)

    for name in ("example_model.pth", "pytorch_model.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    onnx_a, onnx_b = _initializers(first / "example_model.onnx"), _initializers(
        second / "example_model.onnx"
    )
    assert onnx_a and onnx_a.keys() == onnx_b.keys()
    assert all((onnx_a[k] == onnx_b[k]).all() for k in onnx_a)
    assert not _same_state(_state(first / "example_model.pth"), _state(unseeded / "example_model.pth"))


def test_session_fixtures_were_exported_with_the_pinned_seed(tmp_path):
    export(tmp_path, seed=EXAMPLE_MODEL_SEED)
    assert _same_state(_state(EXAMPLE_DIR / "example_model.pth"), _state(tmp_path / "example_model.pth"))
