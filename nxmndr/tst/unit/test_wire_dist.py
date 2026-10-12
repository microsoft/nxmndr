# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The nxmndr-wire wheel: exact contents, metadata, one schema, ML-free import."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import nxmndr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

_spec = importlib.util.spec_from_file_location("build_wire_dist", ROOT / "wire" / "build_wire_dist.py")
build_wire_dist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wire_dist)

PROBE = """
import importlib, importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
blocked = {"torch", "torchvision", "onnxruntime", "transformers", "PIL", "cv2", "skimage",
           "huggingface_hub", "ultralytics", "rasterio"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in blocked:
            raise ImportError(name)
        return None

sys.meta_path.insert(0, Blocker())
import numpy as np
import nxmndr, nxmndr.client, nxmndr.session_protocol, nxmndr.tensor_bundle
import nxmndr.constants, nxmndr.huggingface.search
from nxmndr.inference import inference_pb2, inference_pb2_grpc
bundle = nxmndr.tensor_bundle.pack_tensor_bundle({"mask": np.arange(6, dtype=np.uint16)})
print(json.dumps({
    "file": nxmndr.__file__,
    "search_file": nxmndr.huggingface.search.__file__,
    "stub": hasattr(inference_pb2_grpc, "InferenceServiceStub"),
    "roundtrip": nxmndr.tensor_bundle.unpack_tensor_bundle(bundle)["mask"].tolist(),
    "has_runtime": importlib.util.find_spec("nxmndr.inference.inference") is not None,
    "has_models": importlib.util.find_spec("nxmndr.models") is not None,
    "ml_loaded": sorted(m for m in ("torch", "transformers", "huggingface_hub") if m in sys.modules),
}))
"""


def test_wheel_contains_exactly_the_wire_modules(tmp_path):
    wheel = build_wire_dist.build(tmp_path / "dist")
    assert wheel.name == f"nxmndr_wire-{nxmndr.__version__}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as zf:
        names = {n for n in zf.namelist() if ".dist-info/" not in n}
        assert names == set(build_wire_dist.WIRE_FILES)
        # one schema: the wheel ships the canonical files byte for byte
        for rel in build_wire_dist.WIRE_FILES:
            assert zf.read(rel) == (SRC / rel).read_bytes(), rel
        metadata = next(zf.read(n).decode() for n in zf.namelist() if n.endswith("/METADATA"))
    assert "Name: nxmndr-wire" in metadata
    assert f"Version: {nxmndr.__version__}" in metadata
    requires = sorted(line.split(": ", 1)[1] for line in metadata.splitlines() if line.startswith("Requires-Dist"))
    assert requires == ["grpcio>=1.76.0", "numpy", "protobuf<7,>=6.31.1", "requests"]

    unpacked = tmp_path / "site"
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(unpacked)
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(unpacked)],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["file"].startswith(str(unpacked))
    assert report["search_file"].startswith(str(unpacked))
    assert report["stub"] and report["roundtrip"] == [0, 1, 2, 3, 4, 5]
    assert report["has_runtime"] is False  # the wheel carries no ML runtime module
    assert report["has_models"] is False
    assert report["ml_loaded"] == []
