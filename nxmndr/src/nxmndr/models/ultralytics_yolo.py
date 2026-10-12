# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Ultralytics YOLO instance-segmentation adapter (server side).

A ``ModelSpec`` with ``model_class = "ultralytics_yolo"`` and format ``PYTORCH`` or
``ONNX`` (a local checkpoint that ``ultralytics.YOLO`` opens) loads through this
adapter instead of the PyTorch RPC worker or onnxruntime. The QGIS host treats that
model class as an instance model with full-chip windows, so every prediction returns
the instance mask stack ``(N, H, W)`` uint8 (0 or 1) at the size of the chip it was
given. The server never assigns instance IDs: the host gives each instance a
job-unique ID. Within one chip the order is the model's order after group-mask
removal.

Ported from the YOLO path that a development version of the QGIS plugin ran inside
QGIS, before inference moved to a separate worker process:

- Chip conversion (``preprocess``): a 2-D chip is one band; one band is repeated to
  three; bands after the third are dropped; two bands are refused. Data that is not
  ``uint8`` becomes float, NaN becomes 0, values are multiplied by 255 when the
  maximum lies in (0, 1], then clipped to [0, 255] and cast to ``uint8``. Bands are
  passed in the order received.
- ``YOLO.predict(chip, verbose=False, retina_masks=True, conf=..., iou=..., max_det=...)``.
  A YOLO build that rejects ``retina_masks`` (``TypeError``) is called again without it.
  ``conf``, ``iou`` and ``max_det`` are per-request options (the in-QGIS path read
  them from ``model_info``). Their defaults are those of the DelineateAnything run in
  the Anaximander paper's case study: a detection confidence threshold of 0.1, and
  Ultralytics' own predict defaults for the other two, which that run left unchanged
  (``ultralytics`` 8.4.142 ``cfg/default.yaml``: ``iou: 0.7``, ``max_det: 300``;
  Ultralytics' own ``conf`` default is 0.25).
- ``results[0].masks.data`` is the stack. No result, ``masks is None`` or zero masks
  is a valid empty result ``(0, H, W)``.
- Group masks, which contain two or more other instances, are dropped with
  :func:`nxmndr.models.sam.drop_group_masks` unless ``drop_group_masks`` is false.
- Masks that are not chip-sized are resized to the chip by nearest neighbour at pixel
  centres. A mask whose aspect ratio differs from the chip's by one pixel or more
  (letterboxed output) cannot be placed and is an error.
- Masks with no pixel left after resizing are dropped; they would get no instance ID.

``ultralytics`` is imported only when a model is loaded, so it may be absent from an
environment that never loads one. It is licensed AGPL-3.0.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..logging import get_logger
from .models import Model, ModelSpec

logger = get_logger("nxmndr.models.ultralytics_yolo")

# ``ModelSpec.model_class`` that selects this adapter (the plugin's
# model_selection.ULTRALYTICS_YOLO).
ULTRALYTICS_YOLO = "ultralytics_yolo"
# Wire formats whose local checkpoint ultralytics.YOLO opens.
ULTRALYTICS_FORMATS = ("pytorch", "onnx")
# ``ModelRecord.backend`` of records loaded by this adapter.
ULTRALYTICS_BACKEND = "ultralytics"
# The only YOLO task that yields instance masks.
SEGMENT_TASK = "segment"
# Per-request option (default true): drop group masks, which contain two or more other
# instances (such as a whole-region mask) and are not instances themselves; false keeps
# every mask the model returned.
OPTION_DROP_GROUP_MASKS = "drop_group_masks"
# Per-request detection options passed to ``YOLO.predict`` and their defaults (module
# docstring): confidence and NMS IoU thresholds (decimals in [0, 1]) and the maximum
# number of detections per chip (integer >= 1).
OPTION_CONF = "conf"
OPTION_IOU = "iou"
OPTION_MAX_DET = "max_det"
DEFAULT_CONF = 0.1
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300


class InstanceInputError(ValueError):
    """The chip cannot be given to the YOLO model (wire error ``malformed_payload``)."""


def ultralytics_available() -> bool:
    """True when ``import ultralytics`` can succeed, without importing it."""

    if "ultralytics" in sys.modules:
        return sys.modules["ultralytics"] is not None
    return importlib.util.find_spec("ultralytics") is not None


@dataclass
class UltralyticsModelSpec(ModelSpec):
    """A local YOLO checkpoint (``.pt``, ``.onnx`` or another format YOLO opens)."""

    model_path: str
    name: Optional[str] = None


def to_yolo_chip(chip: Any) -> np.ndarray:
    """The ``uint8`` ``[H, W, 3]`` array given to ``YOLO.predict`` (module docstring)."""

    arr = np.asarray(chip)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise InstanceInputError(f"expected an [H, W, C] chip, got shape {list(arr.shape)}")
    bands = arr.shape[2]
    if bands == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif bands == 2:
        raise InstanceInputError("a YOLO chip needs 1 or at least 3 bands, got 2")
    else:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        values = np.nan_to_num(arr.astype(np.float32), nan=0.0)
        peak = float(values.max()) if values.size else 0.0
        if 0.0 < peak <= 1.0:
            values = values * 255.0
        arr = np.clip(values, 0, 255).astype(np.uint8)
    return np.array(arr, dtype=np.uint8, order="C", copy=True)


def _stack_from_results(results: Any) -> Optional[np.ndarray]:
    """``results[0].masks.data`` as a NumPy ``(N, h, w)`` array, or None when empty."""

    if not results:
        return None
    masks = getattr(results[0], "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"YOLO masks.data has shape {list(arr.shape)}; expected (N, H, W)")
    return arr


def _resize_nearest(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize of a boolean ``(N, h, w)`` stack to ``(N, height, width)``."""

    mh, mw = masks.shape[1], masks.shape[2]
    if (mh, mw) == (height, width):
        return masks
    if abs(mw * height / width - mh) >= 1.0:
        raise ValueError(
            f"YOLO masks are {mh}x{mw} but the chip is {height}x{width}; masks with another "
            "aspect ratio (letterboxed) cannot be placed on the chip"
        )
    rows = np.minimum(((np.arange(height) + 0.5) * mh / height).astype(np.int64), mh - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * mw / width).astype(np.int64), mw - 1)
    return masks[:, rows][:, :, cols]


def instance_stack(
    results: Any, height: int, width: int, *, drop_group_masks: bool = True, logger=None
) -> np.ndarray:
    """The ``(N, height, width)`` uint8 instance stack of one YOLO prediction."""

    arr = _stack_from_results(results)
    if arr is None or arr.shape[0] == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    binary = arr.astype(np.float32) > 0.5
    if drop_group_masks and binary.shape[0] > 1:
        from .sam import drop_group_masks as _drop_group_masks

        kept = _drop_group_masks(binary.astype(np.uint8), logger=logger)
        if logger is not None and len(kept) < binary.shape[0]:
            logger.info("YOLO: dropped %d group mask(s)", binary.shape[0] - len(kept))
        binary = binary[kept]
    binary = _resize_nearest(binary, height, width)
    binary = binary[binary.reshape(binary.shape[0], -1).any(axis=1)]
    return np.ascontiguousarray(binary, dtype=np.uint8)


class UltralyticsYoloModel(Model):
    """A loaded ``ultralytics.YOLO`` model; predictions are serialized by a lock
    because a YOLO object is not safe to call from several threads at once."""

    def __init__(self, yolo, inference_provider=None):
        self.yolo = yolo
        self.inference_provider = inference_provider
        self._lock = threading.Lock()

    def preprocess(self, input_data, task=None):
        return to_yolo_chip(input_data)

    def postprocess(self, model_output, task=None):
        return model_output

    def predict(
        self,
        input_data,
        return_embeddings: bool = False,
        *,
        drop_group_masks: bool = True,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
    ):
        """``(N, H, W)`` uint8 instance masks at the chip's height and width."""

        if return_embeddings:
            raise ValueError("ultralytics_yolo models return instance masks only, no embeddings")
        chip = self.preprocess(input_data)
        height, width = chip.shape[0], chip.shape[1]
        kwargs = {"verbose": False, "retina_masks": True, "conf": conf, "iou": iou, "max_det": max_det}
        with self._lock:
            try:
                results = self.yolo.predict(chip, **kwargs)
            except TypeError:
                kwargs.pop("retina_masks")
                results = self.yolo.predict(chip, **kwargs)
        return instance_stack(results, height, width, drop_group_masks=drop_group_masks, logger=logger)


def load_ultralytics_yolo(spec: UltralyticsModelSpec, provider, session):
    """Registry loader: open ``spec.model_path`` with ``ultralytics.YOLO``."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "model_class ultralytics_yolo needs the ultralytics package in the server's "
            f"Python environment ({exc})"
        ) from exc
    # YOLO() unpickles a .pt checkpoint (weights_only=False), so it runs code from the file:
    # only trusted clients may load models (README, Security).
    yolo = YOLO(spec.model_path)
    task = getattr(yolo, "task", None)
    if task is not None and task != SEGMENT_TASK:
        raise ValueError(
            f"{spec.model_path} is a YOLO {task!r} model; model_class ultralytics_yolo "
            f"needs a {SEGMENT_TASK!r} model, which returns instance masks"
        )
    return UltralyticsYoloModel(yolo, provider)


__all__ = [
    "ULTRALYTICS_YOLO",
    "ULTRALYTICS_FORMATS",
    "ULTRALYTICS_BACKEND",
    "OPTION_DROP_GROUP_MASKS",
    "OPTION_CONF",
    "OPTION_IOU",
    "OPTION_MAX_DET",
    "DEFAULT_CONF",
    "DEFAULT_IOU",
    "DEFAULT_MAX_DET",
    "InstanceInputError",
    "UltralyticsModelSpec",
    "UltralyticsYoloModel",
    "instance_stack",
    "load_ultralytics_yolo",
    "to_yolo_chip",
    "ultralytics_available",
]
