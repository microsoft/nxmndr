# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared prediction dispatch for unary ``Predict`` and every ``StreamPredict`` tile.

One code path turns (leased model record, decoded input, effective options) into a
response array plus metadata: SAM prompt routing, the unprompted fallback, the chip
size of a fixed-size catalog model (``check_catalog_chip``), the result family of a
catalog model's output (``catalog_output``), task shaping (segmentation
masks and confidence, detection boxes, embeddings) and NPZ tensor bundles. The gRPC
layer only decodes transport, holds leases and adds correlation metadata, so unary
and streaming results cannot drift apart.

Stream context v1 (``StreamPredictRequest.context``) is also decoded here:
``session_id`` and ``tile_id`` are reserved, per-tile options use the ``opt.``
prefix, and every other unprefixed key is reserved and never used as an option.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..inference import inference_pb2
from ..inference.image_utils import (
    masks_to_bounding_boxes,
    prepare_segmentation_mask,
    prepare_segmentation_mask_with_confidence,
)
from ..models import sam as sam_support
from ..models import ultralytics_yolo
from ..tensor_bundle import pack_tensor_bundle

# ---------------------------------------------------------------- stream context v1

STREAM_CONTEXT_VERSION = "1"
CONTEXT_SESSION_ID = "session_id"
CONTEXT_TILE_ID = "tile_id"
CONTEXT_BATCH_SIZE = "batch_size"  # legacy; reserved, never an option
RESERVED_CONTEXT_KEYS = (CONTEXT_SESSION_ID, CONTEXT_TILE_ID, CONTEXT_BATCH_SIZE)
TILE_OPTION_PREFIX = "opt."
SESSION_CONTROL_OPTION_KEYS = ("max_tile_bytes", "batch_size")
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# ------------------------------------------------------------------ error codes

ERROR_MISSING_TILE_ID = "missing_tile_id"
ERROR_DUPLICATE_TILE_ID = "duplicate_tile_id"
ERROR_UNKNOWN_SESSION = "unknown_session"
ERROR_SESSION_NOT_OPEN = "session_not_open"
ERROR_MALFORMED_PAYLOAD = "malformed_payload"
ERROR_MALFORMED_OPTIONS = "malformed_options"
ERROR_CHUNK_TOO_LARGE = "chunk_too_large"
ERROR_TILE_TOO_LARGE = "tile_too_large"
ERROR_BACKPRESSURE = "backpressure"
ERROR_RESOURCE_EXHAUSTED = "resource_exhausted"
ERROR_INFERENCE_FAILED = "inference_failed"
ERROR_CANCELLED = "cancelled"
ERROR_INTERNAL = "internal"
SCOPE_TILE = "tile"
SCOPE_STREAM = "stream"


class DispatchError(Exception):
    """A request-level failure with a wire ``error_code``."""

    code = ERROR_INTERNAL


class MalformedOptionsError(DispatchError):
    code = ERROR_MALFORMED_OPTIONS


class MalformedPayloadError(DispatchError):
    code = ERROR_MALFORMED_PAYLOAD


@dataclass
class DecodedContext:
    session_id: str
    tile_id: str
    options: Dict[str, str]
    unknown_keys: Tuple[str, ...]


def validate_option_name(name: str) -> str:
    if not isinstance(name, str) or not _OPTION_NAME.match(name):
        raise MalformedOptionsError(f"option name {name!r} must match {_OPTION_NAME.pattern}")
    if name in RESERVED_CONTEXT_KEYS:
        raise MalformedOptionsError(f"option name {name!r} is a reserved context key")
    return name


def decode_tile_context(context: Mapping[str, str]) -> DecodedContext:
    """Split one message's context into reserved IDs, ``opt.`` options and ignored keys."""

    options: Dict[str, str] = {}
    unknown: List[str] = []
    for key, value in context.items():
        if key.startswith(TILE_OPTION_PREFIX):
            options[validate_option_name(key[len(TILE_OPTION_PREFIX):])] = value
        elif key not in RESERVED_CONTEXT_KEYS:
            unknown.append(key)
    return DecodedContext(
        session_id=context.get(CONTEXT_SESSION_ID, ""),
        tile_id=context.get(CONTEXT_TILE_ID, ""),
        options=options,
        unknown_keys=tuple(sorted(unknown)),
    )


def session_inference_options(session_options: Mapping[str, str]) -> Dict[str, str]:
    """``OpenSessionRequest.options`` minus the session-control keys."""

    return {k: v for k, v in session_options.items() if k not in SESSION_CONTROL_OPTION_KEYS}


def effective_tile_options(
    session_options: Mapping[str, str], tile_options: Mapping[str, str]
) -> Dict[str, str]:
    """Session options updated by this tile's own options; nothing carries between tiles."""

    merged = session_inference_options(session_options)
    merged.update(tile_options)
    return merged


# ------------------------------------------------------------------ input decoding


def decode_input(data: bytes, shape: Optional[Sequence[int]], dtype: Optional[str]) -> np.ndarray:
    """Decode wire bytes; the declared shape and dtype must describe the bytes exactly."""

    if shape is None or not dtype:
        raise MalformedPayloadError("missing shape or dtype")
    try:
        dt = np.dtype(dtype)
    except TypeError as exc:
        raise MalformedPayloadError(f"unknown dtype {dtype!r}") from exc
    if dt.hasobject or dt.kind not in "biufc":
        raise MalformedPayloadError(f"unsupported dtype {dtype!r}")
    dims = tuple(int(d) for d in shape)
    if any(d < 0 for d in dims):
        raise MalformedPayloadError(f"negative dimension in shape {list(dims)}")
    expected = int(np.prod(dims, dtype=np.int64)) * dt.itemsize
    if len(data) != expected:
        raise MalformedPayloadError(
            f"payload has {len(data)} bytes but shape {list(dims)} dtype {dt.name} needs {expected}"
        )
    return np.frombuffer(data, dtype=dt.newbyteorder("<")).reshape(dims)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def option_bool(options: Mapping[str, str], name: str) -> bool:
    raw = options.get(name)
    if raw is None:
        return False
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise MalformedOptionsError(f"option {name}={raw!r} must be true or false")


def option_fraction(options: Mapping[str, str], name: str, default: float) -> float:
    """A decimal option in ``[0, 1]``; ``default`` when absent."""

    raw = options.get(name)
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        value = math.nan
    if not (0.0 <= value <= 1.0):
        raise MalformedOptionsError(f"option {name}={raw!r} must be a decimal in [0, 1]")
    return value


def option_positive_int(options: Mapping[str, str], name: str, default: int) -> int:
    """An integer option >= 1; ``default`` when absent."""

    raw = options.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = 0
    if value < 1:
        raise MalformedOptionsError(f"option {name}={raw!r} must be an integer >= 1")
    return value


# ------------------------------------------------------------------ dispatch


@dataclass
class DispatchResult:
    array: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def output(self) -> bytes:
        return self.array.tobytes()

    @property
    def shape(self) -> List[int]:
        return [int(d) for d in self.array.shape]

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)


def _task_type(options: Mapping[str, str], model_metadata: Mapping[str, object]) -> str:
    task_type = str(options.get("task_type") or "").strip().lower()
    if task_type:
        return task_type
    task_hint = model_metadata.get("task_name")
    if isinstance(task_hint, str) and task_hint:
        return task_hint.lower()
    raw_task = model_metadata.get("task")
    if isinstance(raw_task, int) and raw_task:
        try:
            return inference_pb2.TaskType.Name(raw_task).lower()
        except ValueError:
            return ""
    return ""


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    return np.ascontiguousarray(value)


def shape_result(
    output,
    *,
    options: Mapping[str, str],
    model_metadata: Mapping[str, object],
    logger=None,
    instance_masks: bool = False,
    label_confidence: Optional[np.ndarray] = None,
) -> DispatchResult:
    """Task shaping and serialization shared by every prediction path.

    ``instance_masks``: ``output`` is an ``(N, H, W)`` instance stack that stays as it
    is (no argmax, no squeeze) as a ``segmentation_mask`` result.
    ``label_confidence``: the confidence tile of a segmentation ``output`` that is
    already a label tile (a catalog class vector, ``catalog_output``), used when
    ``return_confidence`` is set.
    """

    embeddings_from_model = None
    if isinstance(output, dict):
        if "embeddings" in output:
            embeddings_from_model = output["embeddings"]
        if "output" in output:
            output = output["output"]
        elif embeddings_from_model is not None:
            output = embeddings_from_model
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
    base_output = _to_numpy(output)
    if embeddings_from_model is not None:
        embeddings_from_model = _to_numpy(embeddings_from_model)

    task_type = _task_type(options, model_metadata)
    return_embeddings = option_bool(options, "return_embeddings")
    return_confidence = option_bool(options, "return_confidence")
    embedding_only = task_type in {"embedding", "embeddings", "feature", "features"}

    meta: Dict[str, str] = {}
    if task_type:
        meta["task_type"] = task_type
    model_task_name = model_metadata.get("task_name")
    if model_task_name:
        meta["model_task"] = str(model_task_name)

    response_array = base_output
    confidence = None
    if embedding_only:
        meta["result_type"] = "embeddings"
    elif task_type == "segmentation":
        try:
            if instance_masks:
                masks = np.ascontiguousarray(base_output)
            elif return_confidence and label_confidence is not None:
                masks = prepare_segmentation_mask(base_output)
                confidence = np.asarray(label_confidence, dtype=np.float32)
            elif return_confidence:
                masks, confidence = prepare_segmentation_mask_with_confidence(base_output)
            else:
                masks = prepare_segmentation_mask(base_output)
            response_array = masks
            meta["result_type"] = "segmentation_mask"
            meta["mask_shape"] = str(list(masks.shape))
            meta["mask_dtype"] = str(masks.dtype)
        except Exception as exc:
            if logger:
                logger.warning("prepare_segmentation_mask failed: %s", exc)
            meta["result_type"] = "raw"
            meta["task_warning"] = f"segmentation_fallback:{exc}"
            confidence = None
    elif task_type in {"object_detection", "object-detection", "detection"}:
        try:
            boxes = masks_to_bounding_boxes(prepare_segmentation_mask(base_output))
            response_array = boxes
            meta["result_type"] = "bounding_boxes"
            meta["boxes_count"] = str(boxes.shape[0])
            meta["box_format"] = "batch_index,class_id,x_min,y_min,x_max,y_max"
        except Exception as exc:
            meta["result_type"] = "raw"
            meta["task_warning"] = f"detection_fallback:{exc}"

    geotransform_opt = options.get("geotransform")
    projection_opt = options.get("projection") or options.get("crs") or options.get("crs_wkt")
    if geotransform_opt:
        meta["geotransform"] = str(geotransform_opt)
    if projection_opt:
        meta["projection"] = str(projection_opt)

    bundle: Optional[Dict[str, np.ndarray]] = None
    if embedding_only:
        bundle = {"embeddings": base_output}
    elif return_embeddings or confidence is not None:
        bundle = {}
        if return_embeddings:
            bundle["embeddings"] = (
                embeddings_from_model if embeddings_from_model is not None else base_output
            )
        result_type = meta.get("result_type", "raw")
        if result_type == "segmentation_mask":
            bundle["mask"] = np.ascontiguousarray(response_array)
            if confidence is not None:
                bundle["confidence"] = np.ascontiguousarray(confidence)
                meta["has_confidence"] = "true"
                if confidence.size:
                    meta["confidence_min"] = f"{float(confidence.min()):.4f}"
                    meta["confidence_max"] = f"{float(confidence.max()):.4f}"
                    meta["confidence_mean"] = f"{float(confidence.mean()):.4f}"
        elif result_type == "bounding_boxes":
            bundle["detections"] = np.ascontiguousarray(response_array)

    if return_embeddings or embedding_only:
        meta["return_embeddings"] = "true"
        embeddings = bundle.get("embeddings") if bundle else None
        if embeddings is not None:
            meta["embeddings_available"] = "true"
            meta["embeddings_shape"] = json.dumps(list(embeddings.shape))
            meta["embeddings_dtype"] = str(embeddings.dtype)

    if bundle:
        bundle_bytes = pack_tensor_bundle(bundle)
        response_array = np.frombuffer(bundle_bytes, dtype=np.uint8)
        meta["payload_format"] = "npz"
        meta["bundle_size_bytes"] = str(len(bundle_bytes))
        meta["bundle_keys"] = ",".join(sorted(bundle.keys()))
    else:
        meta["payload_format"] = "raw"

    meta.setdefault("result_type", "raw")
    return DispatchResult(np.ascontiguousarray(response_array), meta)


def _sam_variant_getter(lease, capability, device_id: str, torch_device: str):
    def get_variant(variant: str):
        if variant == capability.record_variant:
            model_obj = lease.model_for_device(device_id)
            return model_obj.model, model_obj.processor
        # The record's credential (ModelRecord.auth_token, set by the model cache from
        # the load request); never taken from metadata, which holds no token.
        token = lease.record.auth_token or None

        def factory():
            return sam_support.load_sam_variant(variant, capability, torch_device, token=token)

        resource = lease.resource(f"{variant}@{torch_device}", factory)
        return resource.model, resource.processor

    return get_variant


# The embedding-only task names of shape_result.
_EMBEDDING_TASKS = {"embedding", "embeddings", "feature", "features"}

# ------------------------------------------------------------ catalog output families

OUTPUT_CLASS_VECTOR = "class_vector"
OUTPUT_LABEL_MAP = "label_map"
OUTPUT_EMBEDDINGS = "embeddings"
_UINT16_LABELS = 65536


class UnsupportedOutputError(DispatchError):
    """A catalog model's output fits no result family (unary Predict: INVALID_ARGUMENT)."""

    code = ERROR_INFERENCE_FAILED


def _class_count(model_metadata: Mapping[str, object]) -> int:
    try:
        return int(str(model_metadata.get("num_classes") or "0"))
    except ValueError:
        return 0


def check_catalog_chip(chip_shape: Sequence[int], model_metadata: Mapping[str, object]) -> None:
    """Refuse a chip that a fixed-size catalog model cannot take (``MalformedPayloadError``).

    A catalog record's ``input_size`` is the square chip size its model requires
    (``nxmndr.models.catalog.input_size``; empty when it takes any size). A chip of
    another height or width is refused here, before it reaches the PyTorch worker, so
    the error names the size: unary ``Predict`` answers INVALID_ARGUMENT and a stream
    answers a tile-scope ``malformed_payload``.
    """

    size = str(model_metadata.get("input_size") or "")
    if not size or len(chip_shape) < 2:
        return
    height, width = int(chip_shape[0]), int(chip_shape[1])
    if (height, width) != (int(size), int(size)):
        raise MalformedPayloadError(
            f"{model_metadata.get('model_class', 'the model')} takes only {size} x {size} "
            f"chips; this chip is {height} x {width}"
        )


def catalog_output(output, chip_shape: Sequence[int], options, model_metadata):
    """The result family of a catalog model's output (``nxmndr.models.catalog``).

    The chip is ``[H, W, C]`` and the model returns one tensor with batch 1. The model
    has a class head when its load metadata ``num_classes`` is K >= 1 (the class count
    it was built with). The family is decided by that and by the output's shape:

    - Task ``embedding`` (option ``task_type`` or ModelSpec task), or no class head:
      ``[1, D]`` or ``[1, D, h, w]`` is ``embeddings`` (``result_type=embeddings``).
    - Class head, ``[1, K]``: a class vector. The paper's rule, a bare class vector
      becomes a constant tile: a ``segmentation_mask`` whose ``[H, W]`` uint16 tile is
      the argmax class everywhere, so the host writes that class over the chip's
      written window. With ``return_confidence`` the confidence tile is that class's
      softmax probability everywhere.
    - Class head, ``[1, K, H, W]``: per-pixel logits, the label-map path of every
      segmentation model (per-pixel argmax; with ``return_confidence`` the per-pixel
      softmax maximum).
    - Anything else is rejected with ``UnsupportedOutputError`` naming the shape: other
      ranks, batch other than 1, K other than ``num_classes``, logits that are not
      chip-sized (the host places only chip-sized outputs), and K = 1 (one logit has no
      argmax).

    With ``return_embeddings`` the bundle's ``embeddings`` is the model output.
    Returns ``(output, options, confidence, family)``: the first three for
    ``shape_result`` (``options`` with the family's ``task_type``; ``confidence`` only
    for a class vector), and the family name (response metadata ``output_family``).
    """

    arr = _to_numpy(output)
    shape = list(arr.shape)
    height, width = int(chip_shape[0]), int(chip_shape[1])
    classes = _class_count(model_metadata)
    embedding_task = _task_type(options, model_metadata) in _EMBEDDING_TASKS

    def unsupported(why: str) -> UnsupportedOutputError:
        return UnsupportedOutputError(
            f"{model_metadata.get('model_class', 'the model')} returned shape {shape} "
            f"({arr.dtype}), which fits no result family: {why}"
        )

    if arr.ndim not in (2, 4) or shape[0] != 1:
        raise unsupported("the families are [1, K], [1, K, H, W], [1, D] and [1, D, h, w]")
    shaping = dict(options)
    if embedding_task or classes < 1:
        shaping["task_type"] = "embedding"
        return arr, shaping, None, OUTPUT_EMBEDDINGS
    if shape[1] != classes:
        raise unsupported(f"the model was built with num_classes={classes}")
    if classes == 1:
        raise unsupported("one class logit has no argmax")
    shaping["task_type"] = "segmentation"
    if arr.ndim == 4:
        if shape[2:] != [height, width]:
            raise unsupported(f"per-pixel logits must be chip-sized ({height}x{width})")
        return arr, shaping, None, OUTPUT_LABEL_MAP
    if classes > _UINT16_LABELS:
        raise unsupported(f"more than {_UINT16_LABELS} classes do not fit a uint16 label tile")
    scores = arr[0].astype(np.float64)
    label = int(np.argmax(scores))
    tile = np.full((height, width), label, dtype=np.uint16)
    confidence = None
    if option_bool(options, "return_confidence"):
        exp = np.exp(scores - scores.max())
        probability = float(exp[label] / exp.sum())
        confidence = np.full((height, width), probability, dtype=np.float32)
    return {"output": tile, "embeddings": arr}, shaping, confidence, OUTPUT_CLASS_VECTOR


def _run_instance_prediction(model_obj, image, options, model_metadata, logger) -> DispatchResult:
    """An instance-mask model (``ultralytics_yolo``): always a ``segmentation_mask``
    result holding the ``(N, H, W)`` uint8 instance stack at the chip's size.

    Embeddings and per-pixel confidence do not exist for these models, so
    ``return_embeddings``, ``return_confidence`` and an embedding task are refused
    (``malformed_options``) instead of being ignored. Option ``drop_group_masks``
    (boolean, default true) controls the group-mask filter; ``conf``, ``iou`` and
    ``max_det`` go to ``YOLO.predict`` (defaults in ``nxmndr.models.ultralytics_yolo``).
    """

    if (
        option_bool(options, "return_embeddings")
        or option_bool(options, "return_confidence")
        or _task_type(options, model_metadata) in _EMBEDDING_TASKS
    ):
        raise MalformedOptionsError(
            f"{ultralytics_yolo.ULTRALYTICS_YOLO} models return instance masks only; "
            "embeddings and confidence are not available"
        )
    drop = True
    if ultralytics_yolo.OPTION_DROP_GROUP_MASKS in options:
        drop = option_bool(options, ultralytics_yolo.OPTION_DROP_GROUP_MASKS)
    detection = {
        "conf": option_fraction(options, ultralytics_yolo.OPTION_CONF, ultralytics_yolo.DEFAULT_CONF),
        "iou": option_fraction(options, ultralytics_yolo.OPTION_IOU, ultralytics_yolo.DEFAULT_IOU),
        "max_det": option_positive_int(options, ultralytics_yolo.OPTION_MAX_DET, ultralytics_yolo.DEFAULT_MAX_DET),
    }
    start = time.monotonic()
    try:
        masks = model_obj.predict(image, drop_group_masks=drop, **detection)
    except ultralytics_yolo.InstanceInputError as exc:
        raise MalformedPayloadError(str(exc)) from exc
    infer_ms = (time.monotonic() - start) * 1000.0
    shaping = dict(options)
    shaping["task_type"] = "segmentation"
    result = shape_result(
        masks, options=shaping, model_metadata=model_metadata, logger=logger, instance_masks=True
    )
    result.metadata["latency_infer_ms"] = f"{infer_ms:.2f}"
    return result


def run_prediction(
    lease,
    *,
    image: np.ndarray,
    options: Mapping[str, str],
    device_id: str,
    torch_device: str,
    infer: Callable[[np.ndarray, bool], object],
    logger=None,
) -> DispatchResult:
    """Run one prediction on a leased model record.

    ``infer(array, return_embeddings)`` runs the record's own model (ONNX, Hugging
    Face or the PyTorch RPC worker). SAM 3 records with a prompt go to the shared SAM
    handler instead; SAM 3 records without a prompt use ``infer`` (unprompted mode).
    """

    record = lease.record
    model_metadata = record.metadata or {}
    if record.backend == ultralytics_yolo.ULTRALYTICS_BACKEND:
        return _run_instance_prediction(
            lease.model_for_device(device_id), image, options, model_metadata, logger
        )
    capability = sam_support.resolve_sam_capability(record.model, record.spec)

    if model_metadata.get("model_family"):  # a catalog model
        check_catalog_chip(image.shape, model_metadata)
    start = time.monotonic()
    output = None
    sam_mode = ""
    if capability is not None:
        try:
            prompt = sam_support.parse_sam_prompt(options)
        except sam_support.SamPromptError as exc:
            raise MalformedOptionsError(str(exc)) from exc
        if prompt.has_prompt:
            getter = _sam_variant_getter(lease, capability, device_id, torch_device)
            try:
                output = sam_support.handle_sam_inference(
                    image, prompt, getter, device=torch_device, logger=logger
                )
            except sam_support.SamPromptError as exc:
                raise MalformedOptionsError(str(exc)) from exc
            except sam_support.SamInputError as exc:
                raise MalformedPayloadError(str(exc)) from exc
            sam_mode = "geometry" if prompt.has_geometry else "text"
    if output is None:
        output = infer(image, option_bool(options, "return_embeddings"))
    infer_ms = (time.monotonic() - start) * 1000.0

    confidence, family = None, ""
    if model_metadata.get("model_family"):  # a catalog model: catalog_output's rule
        output, options, confidence, family = catalog_output(
            output, image.shape, options, model_metadata
        )
    result = shape_result(
        output,
        options=options,
        model_metadata=model_metadata,
        logger=logger,
        label_confidence=confidence,
    )
    if family:
        result.metadata["output_family"] = family
    result.metadata["latency_infer_ms"] = f"{infer_ms:.2f}"
    if capability is not None:
        result.metadata["sam_prompt"] = sam_mode or "none"
    return result


__all__ = [
    "STREAM_CONTEXT_VERSION",
    "RESERVED_CONTEXT_KEYS",
    "TILE_OPTION_PREFIX",
    "SESSION_CONTROL_OPTION_KEYS",
    "SCOPE_TILE",
    "SCOPE_STREAM",
    "DispatchError",
    "MalformedOptionsError",
    "MalformedPayloadError",
    "DecodedContext",
    "DispatchResult",
    "decode_tile_context",
    "session_inference_options",
    "effective_tile_options",
    "decode_input",
    "option_bool",
    "option_fraction",
    "option_positive_int",
    "shape_result",
    "run_prediction",
    "catalog_output",
    "check_catalog_chip",
    "UnsupportedOutputError",
    "OUTPUT_CLASS_VECTOR",
    "OUTPUT_LABEL_MAP",
    "OUTPUT_EMBEDDINGS",
]
