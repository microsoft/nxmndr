# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A deterministic stand-in for the ``ultralytics`` package (no weights, no network).

It replaces only ``ultralytics.YOLO``. The production adapter
(``nxmndr.models.ultralytics_yolo``), the registry loader, ``build_model_record``, the
ModelManager cache and its leases all run unchanged, because the double enters where
the adapter does ``from ultralytics import YOLO``: tests put :func:`make_module` into
``sys.modules["ultralytics"]`` (explicit test bootstrap; see ``worker_doubles`` for a
separate worker process).

``YOLO(path)`` reads a JSON "checkpoint" written by :func:`write_checkpoint`, so a real
checkpoint is never mistaken for a double. ``predict`` returns what Ultralytics 8.4
returns (``ultralytics/models/yolo/segment/predict.py``, ``construct_result``): a list
with one result whose ``masks`` is None when nothing is found, else an object whose
``data`` is a torch tensor ``(N, h, w)`` of 0/1 values.

Instances come from the chip: every distinct non-zero value of band 0 is one instance
(ascending value), its mask is ``band0 == value``. Checkpoint settings:

- ``group_mask``: when there are at least two instances, put their union first, like
  the whole-region mask SAM 3 and DelineateAnything return (``drop_group_masks`` must
  remove it).
- ``mask_divisor``: return masks at ``(h // k, w // k)``, sampled at every k-th pixel
  (the resize case).
- ``empty``: ``"none"`` gives ``masks = None`` when nothing is found (Ultralytics'
  behaviour); ``"zeros"`` gives ``data`` of shape ``(0, h, w)``.
- ``mask_dtype``: ``"uint8"`` (Ultralytics 8.4) or ``"float32"`` (older releases).
- ``retina_masks_supported``: false makes ``predict`` reject ``retina_masks`` with a
  ``TypeError``, like a YOLO build without that argument.
- ``task``: ``"segment"``, or ``"detect"`` for a detection-only model.
"""

from __future__ import annotations

import json
import threading
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

CHECKPOINT_MARKER = "nxmndr-test-ultralytics-double"

DEFAULTS: Dict[str, Any] = {
    "task": "segment",
    "group_mask": False,
    "mask_divisor": 1,
    "empty": "none",
    "mask_dtype": "uint8",
    "retina_masks_supported": True,
}


def write_checkpoint(path, **settings) -> Path:
    """Write a double's checkpoint file; unknown settings are refused."""

    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown double settings {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"double": CHECKPOINT_MARKER, **DEFAULTS, **settings}))
    return path


class FakeMasks:
    def __init__(self, data: torch.Tensor):
        self.data = data


class FakeResult:
    def __init__(self, masks, orig_shape):
        self.masks = masks
        self.orig_shape = orig_shape


# The detection settings a caller passes to ``predict`` (recorded in ``predict_options``).
DETECTION_KWARGS = ("conf", "iou", "max_det")


class FakeYOLO:
    """``ultralytics.YOLO`` stand-in. Class-level counters are shared by every
    instance in the process: ``constructed`` lists the checkpoint paths opened,
    ``predict_options`` the ``conf``/``iou``/``max_det`` arguments of every ``predict``
    call (the double does not apply them: it has no scores)."""

    lock = threading.Lock()
    constructed: List[str] = []
    predict_options: List[Dict[str, Any]] = []

    def __init__(self, model: str = "", task=None, verbose: bool = False):
        settings = json.loads(Path(model).read_text())
        if settings.get("double") != CHECKPOINT_MARKER:
            raise ValueError(f"{model} is not an ultralytics double checkpoint")
        self.settings = settings
        self.task = settings["task"]
        self.calls: List[Dict[str, Any]] = []
        with FakeYOLO.lock:
            FakeYOLO.constructed.append(str(model))

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.constructed = []
            cls.predict_options = []

    def predict(self, source=None, **kwargs):
        if "retina_masks" in kwargs and not self.settings["retina_masks_supported"]:
            raise TypeError("predict() got an unexpected keyword argument 'retina_masks'")
        chip = np.asarray(source)
        self.calls.append({"kwargs": dict(kwargs), "shape": chip.shape, "dtype": str(chip.dtype),
                           "chip": chip.copy()})
        with FakeYOLO.lock:
            FakeYOLO.predict_options.append({k: kwargs[k] for k in DETECTION_KWARGS if k in kwargs})
        h, w = chip.shape[0], chip.shape[1]
        if self.task != "segment":
            return [FakeResult(None, (h, w))]
        band = chip[:, :, 0]
        values = [int(v) for v in np.unique(band) if v != 0]
        masks = [band == v for v in values]
        if self.settings["group_mask"] and len(masks) >= 2:
            masks.insert(0, np.logical_or.reduce(masks))
        k = int(self.settings["mask_divisor"])
        mh, mw = h // k, w // k
        if not masks:
            if self.settings["empty"] == "none":
                return [FakeResult(None, (h, w))]
            data = torch.zeros((0, mh, mw), dtype=torch.uint8)
        else:
            stack = np.stack(masks)[:, ::k, ::k][:, :mh, :mw]
            data = torch.from_numpy(stack.astype(self.settings["mask_dtype"]))
        return [FakeResult(FakeMasks(data), (h, w))]


def make_module() -> types.ModuleType:
    """A module object to install as ``sys.modules["ultralytics"]``."""

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    module.__version__ = "0.0.0+nxmndr-test-double"
    return module


__all__ = ["CHECKPOINT_MARKER", "DETECTION_KWARGS", "FakeYOLO", "make_module", "write_checkpoint"]
