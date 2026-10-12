# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build the ``nxmndr-wire`` wheel from the nxmndr source tree.

Stages exactly the host-side modules (``WIRE_FILES``) next to ``pyproject.toml`` in
a temporary directory and calls the setuptools build backend directly, so it works
offline with the build environment's own setuptools. The version is the full
package's ``nxmndr.__version__``, so host and worker always match.

Usage: python wire/build_wire_dist.py [--out DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"

# The modules the QGIS plugin may import on its host side (HOST_WIRE_MODULES in
# tst/unit/test_wire_imports.py, which imports them with the ML runtime blocked), plus
# the generated stub types and the schema they were generated from. The host list
# includes the Hugging Face Hub search helpers and the model-file constants they use,
# so the plugin and the server can share one search implementation.
WIRE_FILES = (
    "nxmndr/__init__.py",
    "nxmndr/client.py",
    "nxmndr/constants.py",
    "nxmndr/huggingface/__init__.py",
    "nxmndr/huggingface/search.py",
    "nxmndr/session_protocol.py",
    "nxmndr/tensor_bundle.py",
    "nxmndr/inference/__init__.py",
    "nxmndr/inference/inference_pb2.py",
    "nxmndr/inference/inference_pb2.pyi",
    "nxmndr/inference/inference_pb2_grpc.py",
    "nxmndr/inference/inference.proto",
)


def build(out_dir: Path) -> Path:
    """Build the wheel into ``out_dir`` and return its path."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nxmndr-wire-") as tmp:
        stage = Path(tmp)
        shutil.copy2(HERE / "pyproject.toml", stage / "pyproject.toml")
        for rel in WIRE_FILES:
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, target)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, setuptools.build_meta as b; print(b.build_wheel(sys.argv[1]))",
                str(out_dir),
            ],
            cwd=stage,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"wheel build failed:\n{result.stdout}\n{result.stderr}")
        name = result.stdout.strip().splitlines()[-1]
    return out_dir / name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "dist"), help="output directory")
    args = parser.parse_args(argv)
    print(build(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
