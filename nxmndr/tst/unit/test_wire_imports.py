# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The host-side wire modules import without the ML runtime.

Runs a fresh interpreter in which importing torch, transformers, onnxruntime,
Pillow and friends raises, then imports every module the QGIS host may use.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from nxmndr.inference import image_utils
from nxmndr import tensor_bundle

SRC = Path(__file__).resolve().parents[2] / "src"

HOST_WIRE_MODULES = [
    "nxmndr",
    "nxmndr.session_protocol",
    "nxmndr.client",
    "nxmndr.inference",
    "nxmndr.inference.inference_pb2",
    "nxmndr.inference.inference_pb2_grpc",
    "nxmndr.tensor_bundle",
    "nxmndr.constants",
    "nxmndr.huggingface",
    "nxmndr.huggingface.search",
]
BLOCKED = [
    "torch", "torchvision", "onnxruntime", "transformers", "ultralytics", "huggingface_hub",
    "PIL", "cv2", "skimage", "rasterio",
]
NEVER_LOADED = [
    "nxmndr.models", "nxmndr.server", "nxmndr.inference.inference", "nxmndr.inference.image_utils",
]

PROBE = textwrap.dedent(
    """
    import importlib, importlib.abc, json, sys
    blocked = set(json.loads(sys.argv[1]))

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked for the host import test: {name}")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in json.loads(sys.argv[2]):
        importlib.import_module(name)
    import nxmndr.client as c
    c.InferenceGrpcClient("127.0.0.1:1").close()
    import nxmndr
    from nxmndr.huggingface import search
    found = search.HFModelSearchResult.from_dict(
        {"modelId": "org/m", "siblings": [{"rfilename": "a.bin"}, {"rfilename": "model.safetensors"}]})
    lazy = {}
    for name, call in (("hub", lambda: search.hf_model_supports_inference("org/m")),
                       ("spec", lambda: found.to_spec())):
        try:
            call()
            lazy[name] = "ran"
        except ImportError as exc:
            lazy[name] = "ImportError"
    print(json.dumps({"file": nxmndr.__file__,
                      "loaded": sorted(m for m in sys.modules if m.startswith("nxmndr")),
                      "ml_loaded": sorted(m for m in ("torch", "transformers", "huggingface_hub")
                                          if m in sys.modules),
                      "weights": search._select_primary_weight_file(found.siblings),
                      "extensions": sorted(nxmndr.MODEL_EXTENSIONS),
                      "lazy": lazy}))
    """
)


def test_wire_modules_import_without_ml_runtime():
    import json

    out = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(BLOCKED), json.dumps(HOST_WIRE_MODULES)],
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["file"] == str(SRC / "nxmndr" / "__init__.py")
    loaded = set(report["loaded"])
    assert set(HOST_WIRE_MODULES) <= loaded
    assert not loaded & set(NEVER_LOADED)
    assert report["ml_loaded"] == []  # no torch, transformers or huggingface_hub
    assert report["weights"] == "model.safetensors"
    assert ".onnx" in report["extensions"]
    # the ML-dependent helpers import lazily and fail loudly without the ML runtime
    assert report["lazy"] == {"hub": "ImportError", "spec": "ImportError"}


def test_image_utils_reexports_the_pil_free_codec():
    assert image_utils.pack_tensor_bundle is tensor_bundle.pack_tensor_bundle
    assert image_utils.unpack_tensor_bundle is tensor_bundle.unpack_tensor_bundle


def test_tensor_bundle_round_trip_keeps_dtype_and_shape():
    arrays = {
        "mask": np.arange(12, dtype=np.uint16).reshape(3, 4),
        "confidence": np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4),
        "embeddings": np.ones((1, 8), dtype=np.float64),
    }
    back = tensor_bundle.unpack_tensor_bundle(tensor_bundle.pack_tensor_bundle(arrays))
    assert sorted(back) == sorted(arrays)
    for key, value in arrays.items():
        assert back[key].dtype == value.dtype
        np.testing.assert_array_equal(back[key], value)
    assert tensor_bundle.unpack_tensor_bundle(b"") == {}
