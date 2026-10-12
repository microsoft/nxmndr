# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NPZ tensor bundle codec shared by the server and host-side clients.

Imports only the standard library and NumPy, and uses only APIs available in
NumPy 1.x and 2.x, so a QGIS host with NumPy 1.26 can decode bundles written by a
worker with NumPy 2 (and the reverse). ``allow_pickle`` is always off.
"""

from __future__ import annotations

import io
from typing import Dict, Mapping

import numpy as np


def pack_tensor_bundle(tensors: Mapping[str, np.ndarray]) -> bytes:
    """Serialize a mapping of numpy arrays into a compressed NPZ payload."""

    if not tensors:
        raise ValueError("Tensor bundle cannot be empty")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.ascontiguousarray(val) for key, val in tensors.items()})
    buffer.seek(0)
    return buffer.read()


def unpack_tensor_bundle(payload: bytes) -> Dict[str, np.ndarray]:
    """Deserialize a compressed NPZ payload into numpy arrays."""

    if not payload:
        return {}

    buffer = io.BytesIO(payload)
    with np.load(buffer, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


__all__ = ["pack_tensor_bundle", "unpack_tensor_bundle"]
