# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..tensor_bundle import pack_tensor_bundle, unpack_tensor_bundle


def load_images_to_batch(
    items: Sequence[Union[str, Path, np.ndarray]], target_size: Tuple[int, int] = None
) -> np.ndarray:
    """Load image paths or numpy arrays into a single NCHW float32 batch.

    Args:
        items: Sequence of file paths (str/Path) or already-loaded HWC numpy arrays.
        target_size: Optional (width, height) to resize with Pillow.

    Returns:
        np.ndarray shaped (N, C, H, W) normalized to [0,1].
    """
    arrays = []
    for item in items:
        if isinstance(item, (str, Path)):
            img = Image.open(item).convert("RGB")
            if target_size is not None:
                img = img.resize(target_size)
            arr = np.array(img)
        elif isinstance(item, np.ndarray):
            arr = item
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")
        if arr.ndim == 2:  # grayscale -> expand to 3 channels
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 3:  # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        arrays.append(arr.astype(np.float32) / 255.0)
    return np.stack(arrays, axis=0)


UINT16_MAX = 65535


def _label_mask(arr: np.ndarray) -> np.ndarray:
    """``arr`` as a ``uint16`` class-label map, refusing values a cast would change.

    Labels must be whole numbers in ``[0, 65535]``. A negative or larger value, a
    fractional value, NaN or infinity raises ``ValueError`` instead of wrapping or
    truncating (an identity output of 65536 became label 0 before). The dispatcher then
    returns the prediction unchanged as a ``raw`` result with a ``task_warning``.
    """

    if arr.dtype == np.uint16:
        return arr
    if arr.dtype.kind == "b":
        return arr.astype(np.uint16)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"segmentation labels must be numeric, got dtype {arr.dtype}")
    if arr.size:
        if arr.dtype.kind == "f" and not (np.isfinite(arr).all() and np.array_equal(arr, np.floor(arr))):
            raise ValueError("segmentation output has fractional or non-finite values; it is not a label map")
        low, high = arr.min(), arr.max()
        if low < 0 or high > UINT16_MAX:
            raise ValueError(f"segmentation labels span [{low}, {high}], outside the uint16 range [0, {UINT16_MAX}]")
    return arr.astype(np.uint16, copy=False)


def prepare_segmentation_mask_with_confidence(
    prediction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw model predictions into masks with per-pixel confidence scores.

    Unlike prepare_segmentation_mask, this preserves confidence information
    for client-side threshold-based filtering.

    Args:
        prediction: Raw model output shaped (N, C, H, W), (C, H, W), (N, H, W) or (H, W).

    Returns:
        Tuple of (mask, confidence) where:
        - mask: uint16 array of shape (H, W) with class labels
        - confidence: float32 array of shape (H, W) with confidence scores (0-1)

    Raises ``ValueError`` for unsupported shapes and for labels that are not whole
    numbers in ``[0, 65535]`` (``_label_mask``).
    """
    arr = np.asarray(prediction)

    if arr.ndim == 4:
        # (N, C, H, W)
        if arr.shape[1] == 1:
            # Single channel - already a mask, confidence is 1.0 everywhere
            result = arr.squeeze()
            if result.ndim != 2:
                raise ValueError(f"Unexpected shape after squeeze: {result.shape} from {arr.shape}")
            mask = _label_mask(result)
            confidence = np.ones_like(mask, dtype=np.float32)
            return mask, confidence
        else:
            # Multi-channel logits - compute softmax and extract max confidence
            # Squeeze batch dim if single batch
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)  # Now (C, H, W)
            else:
                arr = arr[0]  # Take first batch
            # Softmax for confidence
            exp_arr = np.exp(arr - arr.max(axis=0, keepdims=True))  # Numerical stability
            softmax = exp_arr / exp_arr.sum(axis=0, keepdims=True)
            # Argmax for class labels
            mask = _label_mask(np.argmax(arr, axis=0))
            # Max softmax value as confidence
            confidence = softmax.max(axis=0).astype(np.float32)
            return mask, confidence

    if arr.ndim == 3:
        if arr.dtype.kind in ("i", "u"):
            # Already integer mask
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)
            mask = _label_mask(arr)
            confidence = np.ones_like(mask, dtype=np.float32)
            return mask, confidence
        # Multi-channel logits (C, H, W)
        exp_arr = np.exp(arr - arr.max(axis=0, keepdims=True))
        softmax = exp_arr / exp_arr.sum(axis=0, keepdims=True)
        mask = _label_mask(np.argmax(arr, axis=0))
        confidence = softmax.max(axis=0).astype(np.float32)
        return mask, confidence

    if arr.ndim == 2:
        # Already a mask
        mask = _label_mask(arr)
        confidence = np.ones_like(mask, dtype=np.float32)
        return mask, confidence

    raise ValueError(f"Unsupported segmentation prediction shape: {arr.shape}")


def prepare_segmentation_mask(prediction: np.ndarray) -> np.ndarray:
    """Convert raw model predictions into integer segmentation masks.

    Accepts tensors shaped (N, C, H, W), (C, H, W), (N, H, W) or (H, W) and returns
    an array shaped (H, W) with ``uint16`` dtype representing the winning class
    per pixel. Raises ``ValueError`` for unsupported shapes and for labels that are
    not whole numbers in ``[0, 65535]`` (``_label_mask``): nothing wraps or truncates.
    """

    arr = np.asarray(prediction)

    if arr.ndim == 4:
        # (N, C, H, W) - if C=1, squeeze it out; otherwise argmax along channel dim
        if arr.shape[1] == 1:
            # Single channel - already a mask, squeeze out batch and channel dims
            result = arr.squeeze()
            if result.ndim != 2:
                raise ValueError(f"Unexpected shape after squeeze: {result.shape} from {arr.shape}")
            return _label_mask(result)
        else:
            # Multi-channel logits - argmax and squeeze batch dim
            result = np.argmax(arr, axis=1)
            if result.shape[0] == 1:
                result = result.squeeze(0)
            return _label_mask(result)

    if arr.ndim == 3:
        if arr.dtype.kind in ("i", "u"):
            # Already integer mask, squeeze if batch dim is 1
            if arr.shape[0] == 1:
                return _label_mask(arr.squeeze(0))
            return _label_mask(arr)
        # Multi-channel logits (C, H, W) - argmax along channel dim
        mask = _label_mask(np.argmax(arr, axis=0))
        return mask

    if arr.ndim == 2:
        return _label_mask(arr)

    raise ValueError(f"Unsupported segmentation prediction shape: {arr.shape}")


def masks_to_bounding_boxes(mask_batch: np.ndarray) -> np.ndarray:
    """Derive bounding boxes (x_min, y_min, x_max, y_max) per class label.

    Output layout: ``[batch_index, class_id, x_min, y_min, x_max, y_max]`` with
    coordinates expressed in pixel units.
    """

    masks = np.asarray(mask_batch)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    masks = masks.astype(np.uint16, copy=False)
    boxes = []

    for batch_idx, mask in enumerate(masks):
        labels = np.unique(mask)
        labels = labels[labels > 0]
        for label in labels:
            ys, xs = np.where(mask == label)
            if xs.size == 0 or ys.size == 0:
                continue
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            boxes.append(
                [
                    float(batch_idx),
                    float(label),
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ]
            )

    if boxes:
        return np.array(boxes, dtype=np.float32)

    return np.zeros((0, 6), dtype=np.float32)


__all__ = [
    "load_images_to_batch",
    "prepare_segmentation_mask",
    "prepare_segmentation_mask_with_confidence",
    "masks_to_bounding_boxes",
    "pack_tensor_bundle",
    "unpack_tensor_bundle",
    "parse_geotransform",
    "boxes_to_polygons",
]


def parse_geotransform(
    value: Union[str, Sequence[float], None],
) -> Tuple[float, float, float, float, float, float] | None:
    """Parse a geotransform string/list into a 6-tuple."""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) != 6:
            return None
        return tuple(float(v) for v in value)

    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(data, (list, tuple)) and len(data) == 6:
        return tuple(float(v) for v in data)

    if isinstance(data, MutableMapping):
        ordered = [data.get(str(idx)) for idx in range(6)]
        if all(item is not None for item in ordered):
            return tuple(float(v) for v in ordered)
    return None


def boxes_to_polygons(
    boxes: np.ndarray,
    geotransform: Tuple[float, float, float, float, float, float] | None,
) -> Iterable[Tuple[np.ndarray, Dict[str, float]]]:
    """Convert detection boxes into polygon coordinates using the geotransform."""

    if boxes.size == 0:
        return []

    gt = geotransform or (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    a, b, c, d, e, f = gt
    polygons = []
    for det in boxes:
        if det.size < 6:
            continue
        _, cls, x_min, y_min, x_max, y_max = det.astype(float)

        corners = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ],
            dtype=float,
        )

        world = []
        for px, py in corners:
            world_x = a + px * b + py * c
            world_y = d + px * e + py * f
            world.append([world_x, world_y])

        polygons.append((np.asarray(world, dtype=float), {"class_id": float(cls)}))
    return polygons
