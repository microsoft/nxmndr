# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SAM (Segment Anything Model) family handling for server-side inference.

This module centralizes all SAM-specific logic including SAM3, SAM3Tracker, and other variants.

Routing is by an explicit capability resolved from the loaded model's architecture
(``config.model_type``), never from a model's name, path or ID. SAM prompt options
use the flat v1 wire encoding in chip pixel coordinates (x = column, y = row):

- ``sam_text_prompt``: plain text
- ``sam_input_points``: JSON ``[[x, y], ...]``
- ``sam_input_labels``: JSON ``[1, 0, ...]`` (1 positive, 0 negative), same length as points
- ``sam_input_bbox``: JSON ``[x_min, y_min, x_max, y_max]``
- ``sam_conf_threshold``, ``sam_mask_threshold``: decimals in ``[0, 1]``

All points and the box form ONE object prompt. Malformed values raise
:class:`SamPromptError`; they never fall back to unprompted inference.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

SAM_OPTION_TEXT = "sam_text_prompt"
SAM_OPTION_POINTS = "sam_input_points"
SAM_OPTION_LABELS = "sam_input_labels"
SAM_OPTION_BOX = "sam_input_bbox"
SAM_OPTION_CONF = "sam_conf_threshold"
SAM_OPTION_MASK = "sam_mask_threshold"

DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5

# config.model_type values of the SAM 3 family and what each loaded class can serve.
SAM3_MODEL_TYPES = ("sam3", "sam3_tracker")
SAM_VARIANT_TEXT = "sam3"  # Sam3Model + Sam3Processor: text prompts
SAM_VARIANT_GEOMETRY = "sam3_tracker"  # Sam3TrackerModel + Sam3TrackerProcessor: points/box


class SamPromptError(ValueError):
    """A SAM prompt option is malformed (wire error code ``malformed_options``)."""


class SamInputError(ValueError):
    """The input chip cannot be given to SAM as-is (wire error code ``malformed_payload``)."""


@dataclass(frozen=True)
class SamPrompt:
    """One object prompt decoded from request options."""

    text: str = ""
    points: Tuple[Tuple[float, float], ...] = ()
    labels: Tuple[int, ...] = ()
    box: Optional[Tuple[float, float, float, float]] = None
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    mask_threshold: float = DEFAULT_MASK_THRESHOLD

    @property
    def has_geometry(self) -> bool:
        return bool(self.points) or self.box is not None

    @property
    def has_prompt(self) -> bool:
        return self.has_geometry or bool(self.text)


@dataclass(frozen=True)
class SamCapability:
    """SAM support of one loaded model record, resolved once from the loaded model.

    ``record_variant`` is the SAM 3 class the record itself holds; the other variant
    is loaded from ``pretrained_path`` (the record's own snapshot or repo, never a
    hardcoded repository) as a record-scoped resource.
    """

    family: str
    record_variant: str
    pretrained_path: str
    revision: Optional[str] = None


def resolve_sam_capability(model_obj: Any, spec: Any = None) -> Optional[SamCapability]:
    """Return the SAM capability of a loaded model, or None if it is not a SAM 3 model.

    Reads ``config.model_type`` of the loaded transformers model. Names, paths and
    model IDs are never consulted.
    """

    inner = getattr(model_obj, "model", None)
    config = getattr(inner, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in SAM3_MODEL_TYPES:
        return None
    path = getattr(model_obj, "repo_path", None) or getattr(spec, "repo_id", None) or ""
    return SamCapability(
        family="sam3",
        record_variant=str(model_type),
        pretrained_path=str(path),
        revision=getattr(spec, "revision", None),
    )


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamPromptError(f"{where}: expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise SamPromptError(f"{where}: must be finite, got {out!r}")
    return out


def _json(raw: Any, where: str) -> Any:
    if not isinstance(raw, str):
        raise SamPromptError(f"{where}: expected a JSON string")
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise SamPromptError(f"{where}: not valid JSON ({exc})") from exc


def _threshold(options: Mapping[str, str], key: str, default: float) -> float:
    if key not in options:
        return default
    try:
        value = float(options[key])
    except (TypeError, ValueError) as exc:
        raise SamPromptError(f"{key}: must be a number") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SamPromptError(f"{key}: must be in [0, 1], got {options[key]!r}")
    return value


def parse_sam_prompt(options: Mapping[str, str]) -> SamPrompt:
    """Decode SAM prompt options (v1 wire encoding). Malformed values raise SamPromptError."""

    text = str(options.get(SAM_OPTION_TEXT, "") or "").strip()
    raw_points = options.get(SAM_OPTION_POINTS)
    raw_labels = options.get(SAM_OPTION_LABELS)
    if (raw_points is None) != (raw_labels is None):
        raise SamPromptError("sam_input_points and sam_input_labels must be sent together")
    points: Tuple[Tuple[float, float], ...] = ()
    labels: Tuple[int, ...] = ()
    if raw_points is not None:
        parsed_points = _json(raw_points, SAM_OPTION_POINTS)
        parsed_labels = _json(raw_labels, SAM_OPTION_LABELS)
        if not isinstance(parsed_points, list) or not isinstance(parsed_labels, list):
            raise SamPromptError("sam_input_points and sam_input_labels must be JSON lists")
        if len(parsed_points) != len(parsed_labels):
            raise SamPromptError(
                f"sam_input_points ({len(parsed_points)}) and sam_input_labels "
                f"({len(parsed_labels)}) must have the same length"
            )
        pts = []
        for idx, pt in enumerate(parsed_points):
            if not isinstance(pt, list) or len(pt) != 2:
                raise SamPromptError(f"sam_input_points[{idx}] must be [x, y]")
            pts.append((_finite(pt[0], f"point {idx}.x"), _finite(pt[1], f"point {idx}.y")))
        labs = []
        for idx, lab in enumerate(parsed_labels):
            if isinstance(lab, bool) or lab not in (0, 1):
                raise SamPromptError(f"sam_input_labels[{idx}] must be 0 or 1, got {lab!r}")
            labs.append(int(lab))
        points, labels = tuple(pts), tuple(labs)
    box = None
    if SAM_OPTION_BOX in options:
        parsed_box = _json(options[SAM_OPTION_BOX], SAM_OPTION_BOX)
        if not isinstance(parsed_box, list) or len(parsed_box) != 4:
            raise SamPromptError("sam_input_bbox must be [x_min, y_min, x_max, y_max]")
        vals = tuple(_finite(v, f"sam_input_bbox[{idx}]") for idx, v in enumerate(parsed_box))
        if vals[2] <= vals[0] or vals[3] <= vals[1]:
            raise SamPromptError("sam_input_bbox needs x_min < x_max and y_min < y_max")
        box = vals
    return SamPrompt(
        text=text,
        points=points,
        labels=labels,
        box=box,
        conf_threshold=_threshold(options, SAM_OPTION_CONF, DEFAULT_CONF_THRESHOLD),
        mask_threshold=_threshold(options, SAM_OPTION_MASK, DEFAULT_MASK_THRESHOLD),
    )


def sam_image_from_array(image_array: np.ndarray):
    """Convert a chip to the PIL image SAM expects, without changing pixel values.

    SAM prompting takes rendered 8-bit imagery. Other dtypes are rejected rather than
    truncated: the host decides how native samples become display values.
    """

    from PIL import Image as PILImage

    arr = np.asarray(image_array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        raise SamInputError(
            f"SAM prompting needs a uint8 [H, W, 3] chip, got dtype {arr.dtype}; "
            "convert native samples to display values on the host"
        )
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        return PILImage.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return PILImage.fromarray(arr)
    raise SamInputError(f"SAM prompting needs an [H, W, 3] chip, got shape {tuple(arr.shape)}")


def handle_sam_inference(
    image_array: np.ndarray,
    prompt: SamPrompt,
    get_variant,
    device: str = "cpu",
    logger=None,
) -> np.ndarray:
    """Run one SAM prompt and return ``(N, H, W)`` uint8 masks (N may be 0).

    Geometry (points/box) takes precedence over text. ``get_variant(name)`` returns
    ``(model, processor)`` for ``SAM_VARIANT_GEOMETRY`` or ``SAM_VARIANT_TEXT``; the
    caller supplies cached, record-scoped resources so nothing is loaded per call.
    Callers handle the no-prompt case themselves (unprompted inference).
    """

    if not prompt.has_prompt:
        raise ValueError("No SAM prompts provided (need text, points or a box)")
    image = sam_image_from_array(image_array)

    if prompt.has_geometry:
        model, processor = get_variant(SAM_VARIANT_GEOMETRY)
        if logger:
            logger.info(
                "SAM3 tracker prompt: %d points (%d negative), box=%s",
                len(prompt.points),
                sum(1 for lab in prompt.labels if lab == 0),
                prompt.box is not None,
            )
        results = run_sam3_tracker_inference(
            model,
            processor,
            image,
            points=prompt.points,
            labels=prompt.labels,
            box=prompt.box,
            device=device,
            mask_threshold=prompt.mask_threshold,
        )
    else:
        model, processor = get_variant(SAM_VARIANT_TEXT)
        if logger:
            logger.info("SAM3 text prompt: %r", prompt.text)
        results = run_sam3_text_inference(
            model,
            processor,
            image,
            prompt.text,
            device=device,
            threshold=prompt.conf_threshold,
            mask_threshold=prompt.mask_threshold,
            logger=logger,
        )
    if logger:
        logger.info("SAM3 inference completed: output shape=%s", results["masks"].shape)
    return results["masks"]


class SamVariantResource:
    """A SAM variant (model + processor) cached as a record-scoped resource."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def close(self) -> None:
        self.model = None
        self.processor = None


def load_sam_variant(
    variant: str, capability: SamCapability, device: str, token: Optional[str] = None
) -> SamVariantResource:
    """Load the SAM 3 variant a record does not already hold, from the record's own source.

    A local snapshot directory is read with ``local_files_only``; a hub repository uses
    the record's revision and token. Called once per record and device by the lease's
    resource cache, never per tile.
    """

    import os

    if variant == SAM_VARIANT_GEOMETRY:
        from transformers import Sam3TrackerModel as model_cls
        from transformers import Sam3TrackerProcessor as processor_cls
    elif variant == SAM_VARIANT_TEXT:
        from transformers import Sam3Model as model_cls
        from transformers import Sam3Processor as processor_cls
    else:
        raise ValueError(f"unknown SAM variant {variant!r}")
    if not capability.pretrained_path:
        raise RuntimeError("SAM record has no snapshot path or repository to load variants from")
    kwargs: Dict[str, Any] = {}
    if os.path.isdir(capability.pretrained_path):
        kwargs["local_files_only"] = True
    else:
        if capability.revision:
            kwargs["revision"] = capability.revision
        if token:
            kwargs["token"] = token
    model = model_cls.from_pretrained(capability.pretrained_path, **kwargs).to(device)
    model.eval()
    processor = processor_cls.from_pretrained(capability.pretrained_path, **kwargs)
    return SamVariantResource(model, processor)


def _model_device(model, default: str):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return default


def _image_hw(image) -> Tuple[int, int]:
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        width, height = image.size
        return int(height), int(width)
    arr = np.asarray(image)
    return int(arr.shape[0]), int(arr.shape[1])


def _as_uint8_masks(masks, height: int, width: int) -> np.ndarray:
    """Binary mask stack as ``(N, H, W)`` uint8; an empty result keeps the chip size."""

    if masks is None or len(masks) == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    arr = np.asarray(masks)
    if arr.dtype != np.bool_:
        arr = arr > 0.5
    return arr.astype(np.uint8)


def _probability_to_logit(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


def run_sam3_text_inference(
    model,
    processor,
    image,
    text_prompt: str,
    device: str = "cuda",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    logger=None,
) -> Dict[str, Any]:
    """
    Run SAM3 text-based inference.

    Args:
        model: SAM3Model instance
        processor: SAM3Processor instance
        image: PIL Image or numpy array
        text_prompt: Text prompt for segmentation
        device: Device to run on
        threshold: Detection threshold
        mask_threshold: Mask threshold

    Returns:
        Dict with 'masks' ((N, H, W) uint8), 'boxes', 'scores'
    """
    height, width = _image_hw(image)
    device = _model_device(model, device)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = _as_uint8_masks(results["masks"], height, width)
    boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.zeros((0, 4))
    scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.zeros((0,))
    if len(masks) > 1:
        # drop whole-region "group" masks that contain other instances (see drop_group_masks)
        kept = drop_group_masks(masks, logger=logger)
        masks, boxes, scores = masks[kept], boxes[kept], scores[kept]

    return {"masks": masks, "boxes": boxes, "scores": scores}


def run_sam3_tracker_inference(
    model,
    processor,
    image,
    *,
    points: Tuple[Tuple[float, float], ...] = (),
    labels: Tuple[int, ...] = (),
    box: Optional[Tuple[float, float, float, float]] = None,
    device: str = "cuda",
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run SAM3Tracker on ONE object prompt: every point (with its label) and the box
    describe the same object, so the result is one mask.

    Nesting passed to the processor (one image, one object):
    ``input_points [image][object][point][xy]``, ``input_labels [image][object][point]``,
    ``input_boxes [image][object][xyxy]``. The box is a real box prompt, not corner points.

    Args:
        model: SAM3TrackerModel instance
        processor: SAM3TrackerProcessor instance
        image: PIL Image or numpy array
        points: (x, y) chip pixel coordinates
        labels: 1 (positive) or 0 (negative) per point
        box: (x_min, y_min, x_max, y_max) chip pixel coordinates
        device: Device to run on
        mask_threshold: probability threshold for the binary mask

    Returns:
        Dict with 'masks': (1, H, W) uint8
    """
    if len(points) != len(labels):
        raise SamPromptError("points and labels must have the same length")
    if not points and box is None:
        raise SamPromptError("a tracker prompt needs points or a box")

    height, width = _image_hw(image)
    device = _model_device(model, device)
    prompt_kwargs: Dict[str, Any] = {}
    if points:
        prompt_kwargs["input_points"] = [[[[float(x), float(y)] for x, y in points]]]
        prompt_kwargs["input_labels"] = [[[int(lab) for lab in labels]]]
    if box is not None:
        prompt_kwargs["input_boxes"] = [[[float(v) for v in box]]]

    tracker_inputs = processor(images=image, return_tensors="pt", **prompt_kwargs).to(device)

    with torch.no_grad():
        tracker_outputs = model(**tracker_inputs, multimask_output=False)

    # post_process_masks thresholds logits; convert the probability threshold.
    batch_masks = processor.post_process_masks(
        tracker_outputs.pred_masks.cpu(),
        tracker_inputs["original_sizes"],
        mask_threshold=_probability_to_logit(mask_threshold),
    )
    object_masks = batch_masks[0]  # (objects, masks per object, H, W) for the one image
    mask = object_masks[0][0]  # the one object, single-mask output
    return {"masks": _as_uint8_masks([np.asarray(mask)], height, width)}


def drop_group_masks(
    masks: np.ndarray,
    min_contained: int = 2,
    overlap: float = 0.8,
    mask_logits: Optional[np.ndarray] = None,
    logger=None,
) -> np.ndarray:
    """Indices of masks to keep after dropping "group" masks that contain other instances.

    For a concept like "field", SAM 3 returns both per-parcel instances and a whole-region mask
    (the union of all parcels). The region mask has the HIGHEST concept score, so it cannot be
    removed by score; it is identified structurally: a mask is dropped when at least
    ``min_contained`` other masks lie inside it (``overlap`` fraction of their pixels). On the
    case-study chips the region mask contained 4 to 56 parcels and covered 100% of the chip,
    parcels contained none (mean in-mask logit 1.3-1.5 for the region vs 1.7-5.5 for parcels).

    Args:
        masks: (N, H, W) binary masks (bool or 0/1).
        min_contained: drop a mask that contains at least this many other masks.
        overlap: fraction of the contained mask's pixels that must lie inside the container.
        mask_logits: optional (N, H, W) mask logits, only used to log the in-mask mean logit.
    A contained mask that itself covers >= ``overlap`` of the container is treated as a duplicate
    rather than a group member, so the smallest non-empty mask is never dropped.
    Returns:
        1-D array of kept indices (in the original order).
    """
    n = len(masks)
    if n < 2:
        return np.arange(n)
    binm = [np.asarray(m) > 0.5 for m in masks]
    areas = np.array([int(b.sum()) for b in binm])
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if areas[i] == 0:
            continue
        contained = 0
        for j in range(n):
            if j == i or areas[j] == 0 or areas[j] > areas[i]:
                continue
            inter = np.logical_and(binm[i], binm[j]).sum()
            # j inside i, but i not inside j: mutual containment is a duplicate detection, not a group
            if inter >= overlap * areas[j] and inter < overlap * areas[i]:
                contained += 1
                if contained >= min_contained:
                    break
        if contained >= min_contained:
            keep[i] = False
            if logger:
                extra = ""
                if mask_logits is not None:
                    extra = f", mean in-mask logit {float(np.asarray(mask_logits[i])[binm[i]].mean()):.2f}"
                logger.info(
                    "SAM3: dropping group mask %d (area %.2f of chip, contains >=%d instances%s)",
                    i, areas[i] / binm[i].size, contained, extra,
                )
    return np.flatnonzero(keep)


def select_masks_by_area_and_iou(
    masks: np.ndarray,
    order: np.ndarray,
    min_area: float = 0.001,
    max_area: float = 0.5,
    iou: float = 0.5,
) -> List[int]:
    """Score-free instance selection: walk ``masks`` in ``order`` (best first), drop masks whose
    area fraction is outside [min_area, max_area] (specks and whole-chip masks) and masks with
    IoU > ``iou`` against an already kept mask. Returns the kept indices in ``order``.
    """
    binm = [np.asarray(m) > 0.5 for m in masks]
    total = float(binm[0].size) if len(binm) else 1.0
    kept: List[int] = []
    for i in order:
        a = binm[i].sum() / total
        if a < min_area or a > max_area:
            continue
        ok = True
        for j in kept:
            inter = np.logical_and(binm[i], binm[j]).sum()
            union = np.logical_or(binm[i], binm[j]).sum()
            if union and inter / union > iou:
                ok = False
                break
        if ok:
            kept.append(int(i))
    return kept


def filter_overlapping_masks(masks: np.ndarray, keep_smaller: bool = True) -> np.ndarray:
    """
    Remove overlapping masks, optionally keeping the smaller one in each overlapping pair.

    Args:
        masks: Array of masks with shape (N, H, W)
        keep_smaller: If True, keep smaller mask; if False, keep larger mask

    Returns:
        Filtered array of masks
    """
    if len(masks) == 0:
        return masks

    keep = [True] * len(masks)
    areas = [np.sum(mask > 0.5) for mask in masks]

    for i in range(len(masks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not keep[j]:
                continue
            intersection = np.logical_and(masks[i] > 0.5, masks[j] > 0.5)
            if np.any(intersection):
                # Drop based on size comparison
                if keep_smaller:
                    if areas[i] < areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # i is dropped, skip further checks
                else:
                    if areas[i] > areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

    return np.array([mask for mask, k in zip(masks, keep) if k])
