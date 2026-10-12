# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Centralized model specification for all providers
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..exceptions import handle_inference_error
from ..logging import PerformanceContext, get_logger
from ..memory import optimize_tensor_copy
from ..validation import validate_input
from ..tasks import Task

from .registry import get_registry

logger = get_logger("nxmndr.models")


class Model(ABC):
    """Abstract base class for all model types."""

    @abstractmethod
    def preprocess(self, input_data, task: Optional[object] = None):
        pass

    @abstractmethod
    def postprocess(self, model_output, task: Optional[object] = None):
        pass

    @abstractmethod
    def predict(self, input_data, return_embeddings: bool = False):
        pass

    @classmethod
    def model_spec_from_json(cls, json_str):
        """Create ModelSpec from JSON using unified registry."""
        d = json.loads(json_str)
        t = d.get("type")
        registry = get_registry()
        spec_cls = registry.get_spec_class(t)
        if spec_cls is None:
            raise ValueError(f"Unknown ModelSpec type: {t}")
        return spec_cls.from_json(json_str)


class ModelSpec(ABC):
    """Base class for model specifications with unified JSON serialization."""

    @classmethod
    def from_json(cls, json_str: str):
        """Create ModelSpec from JSON string."""
        data = json.loads(json_str)
        # Remove 'type' field used for registry dispatch
        data.pop("type", None)
        # Filter only fields that exist in this dataclass
        if hasattr(cls, "__dataclass_fields__"):
            field_names = {f.name for f in fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in field_names}
        else:
            filtered_data = data
        return cls(**filtered_data)

    def to_json(self) -> str:
        """Serialize ModelSpec to JSON string."""
        if hasattr(self, "__dataclass_fields__"):
            data = asdict(self)
        else:
            # Fallback for non-dataclass specs
            data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        # Add type information for registry dispatch
        class_name = self.__class__.__name__
        if class_name.endswith("ModelSpec"):
            type_name = class_name[:-9].lower()  # Remove 'ModelSpec' suffix
        else:
            type_name = class_name.lower()
        data["type"] = type_name
        return json.dumps(data)


@dataclass
class TorchHubModelSpec(ModelSpec):
    name: str
    repo: str
    hub_args: Optional[tuple] = None
    hub_kwargs: Optional[dict] = None


@dataclass
class PytorchModelSpec(ModelSpec):
    model_class: nn.Module
    model_path: str
    name: Optional[str] = None
    # Constructor arguments of a catalog model_class (nxmndr.models.catalog), from
    # ModelSpec.metadata: num_classes, in_channels.
    catalog_args: Optional[dict] = None

    @classmethod
    def from_json(cls, json_str):
        """Custom from_json to handle model_class resolution."""
        d = json.loads(json_str)
        model_class_field = d["model_class"]
        # A registered class name resolves to its class; any other string stays as given
        # (a catalog name, built by nxmndr.models.catalog).
        registry = get_registry()
        resolved = registry.get_model_class(model_class_field)
        return cls(
            model_class=resolved if resolved is not None else model_class_field,
            model_path=d["model_path"],
            name=d.get("name"),
            catalog_args=d.get("catalog_args"),
        )


@dataclass
class HuggingFaceModelSpec(ModelSpec):
    repo_id: str
    filename: Optional[str] = None
    revision: Optional[str] = None
    token: Optional[str] = field(default=None, repr=False)
    local_files_only: bool = False
    cache_dir: Optional[str] = None


@dataclass
class OnnxModelSpec(ModelSpec):
    model_path: str
    name: Optional[str] = None


# Registration will be done in inference/__init__.py to avoid circular imports


class PytorchModel(Model):
    def __init__(self, model: nn.Module, inference_provider=None):
        self.model = model
        self.inference_provider = inference_provider
        # Get device from first parameter, avoiding tensor boolean evaluation
        try:
            self.device = next(iter(model.parameters())).device
        except StopIteration:
            self.device = torch.device("cpu")
        logger.debug(f"PytorchModel initialized on device: {self.device}")

    def enable_embedding_capture(self, layer_name: Optional[str] = None):
        """Enable capturing intermediate embeddings from PyTorch model.

        Captures from the last layer before the final output, or a specified layer.
        """

        def hook_fn(module, input, output):
            # Store the output of this layer as embeddings
            if torch.is_tensor(output):
                self._embeddings = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # Take first output if multiple
                self._embeddings = output[0].detach() if torch.is_tensor(output[0]) else output[0]

        # Find the layer to hook
        if layer_name:
            # Hook specific named layer
            for name, module in self.model.named_modules():
                if name == layer_name:
                    self._embedding_hook = module.register_forward_hook(hook_fn)
                    logger.debug("Registered embedding capture hook on layer: %s", name)
                    return
            logger.warning("Layer %s not found, falling back to default", layer_name)

        # Default: hook the second-to-last module (before final classifier/head)
        modules = list(self.model.children())
        if len(modules) >= 2:
            target_module = modules[-2]
            self._embedding_hook = target_module.register_forward_hook(hook_fn)
            logger.debug("Registered embedding capture hook on second-to-last module")
        elif len(modules) == 1:
            self._embedding_hook = modules[0].register_forward_hook(hook_fn)
            logger.debug("Registered embedding capture hook on only module")

    @handle_inference_error
    def preprocess(self, input_data, task: Optional[Task] = None):
        """Preprocess input with validation and device optimization."""
        validated_input, _ = validate_input(input_data)

        if isinstance(validated_input, np.ndarray):
            tensor = torch.from_numpy(validated_input)
        elif torch.is_tensor(validated_input):
            tensor = validated_input
        else:
            tensor = torch.tensor(validated_input)

        # Optimize device transfer
        if str(tensor.device) != str(self.device):
            tensor = optimize_tensor_copy(tensor, str(self.device))

        return tensor

    def postprocess(self, model_output, task: Optional[Task] = None):
        """Postprocess model output."""
        if torch.is_tensor(model_output):
            # Move to CPU for consistency and memory efficiency
            if model_output.device.type != "cpu":
                model_output = model_output.detach().cpu()
            return model_output.numpy()
        return model_output

    @handle_inference_error
    def predict(self, input_data, return_embeddings: bool = False):
        """Run model prediction with performance monitoring."""
        with PerformanceContext(
            logger,
            "pytorch_inference",
            input_shape=tuple(input_data.shape) if hasattr(input_data, "shape") else None,
        ):
            input_tensor = self.preprocess(input_data)

            # Set up embedding capture if requested
            captured_embeddings = []
            hook_handle = None

            if return_embeddings:

                def hook_fn(module, input, output):
                    if torch.is_tensor(output):
                        captured_embeddings.append(output.detach())
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        if torch.is_tensor(output[0]):
                            captured_embeddings.append(output[0].detach())

                # Hook the second-to-last module (before final classifier/head)
                modules = list(self.model.children())
                if len(modules) >= 2:
                    hook_handle = modules[-2].register_forward_hook(hook_fn)
                elif len(modules) == 1:
                    hook_handle = modules[0].register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    output = self.model(input_tensor)

                result = self.postprocess(output)

                # If embeddings were captured, return both
                if return_embeddings and captured_embeddings:
                    embeddings_tensor = captured_embeddings[0]
                    embeddings_np = (
                        embeddings_tensor.cpu().numpy()
                        if embeddings_tensor.device.type != "cpu"
                        else embeddings_tensor.numpy()
                    )
                    if isinstance(result, np.ndarray):
                        return {"output": result, "embeddings": embeddings_np}
                    elif torch.is_tensor(result):
                        result_np = (
                            result.cpu().numpy() if result.device.type != "cpu" else result.numpy()
                        )
                        return {"output": result_np, "embeddings": embeddings_np}

                return result
            finally:
                # Clean up hook
                if hook_handle is not None:
                    hook_handle.remove()


class OnnxModel(Model):
    def __init__(self, session, inference_provider=None):
        self.session = session
        self.inference_provider = inference_provider
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.debug(
            f"OnnxModel initialized with input: {self.input_name}, shape: {self.input_shape}"
        )

    @handle_inference_error
    def preprocess(self, input_data, task: Optional[Task] = None):
        """Preprocess input with validation."""
        validated_input, _ = validate_input(input_data)

        if not isinstance(validated_input, np.ndarray):
            validated_input = np.array(validated_input)

        return validated_input

    def postprocess(self, model_output, task: Optional[Task] = None):
        return model_output

    @handle_inference_error
    def predict(self, input_data, return_embeddings: bool = False):
        """Run ONNX inference with performance monitoring."""
        with PerformanceContext(
            logger,
            "onnx_inference",
            input_shape=tuple(input_data.shape) if hasattr(input_data, "shape") else None,
        ):
            input_array = self.preprocess(input_data)

            output = self.session.run(None, {self.input_name: input_array})

            # If capturing embeddings and model has multiple outputs, treat earlier outputs as embeddings
            if return_embeddings and isinstance(output, list) and len(output) > 1:
                # Last output is typically the final prediction, earlier ones are features
                main_output = output[-1]
                embeddings = output[-2] if len(output) >= 2 else output[0]
                return {"output": self.postprocess(main_output), "embeddings": embeddings}

            # onnxruntime returns a list; normalize to a single ndarray if length 1
            if isinstance(output, list) and len(output) == 1:
                output = output[0]

            return self.postprocess(output)


class HuggingFaceModel(PytorchModel):
    def __init__(self, model, processor, inference_provider=None, repo_path: Optional[str] = None):
        super().__init__(model, inference_provider)
        self.processor = processor
        self.repo_path = repo_path

    def preprocess(self, input_data, task: Optional[Task] = None):
        # Ensure input data is writable to avoid PyTorch warnings
        if isinstance(input_data, np.ndarray) and not input_data.flags.writeable:
            input_data = np.copy(input_data)

        # Check if this is a SAM3 model by checking processor type
        processor_class_name = (
            self.processor.__class__.__name__ if hasattr(self.processor, "__class__") else ""
        )
        is_sam3_model = "Sam3" in processor_class_name and "Tracker" not in processor_class_name

        # For SAM3 (Promptable Concept Segmentation), use a default text prompt
        if is_sam3_model:
            # Use generic prompt for automatic segmentation
            # SAM3 requires text prompts to know what to segment
            processed = self.processor(images=input_data, text="object", return_tensors="pt")
        else:
            # For other models (including SAM3 Tracker), process normally
            processed = self.processor(input_data, return_tensors="pt")

        if hasattr(processed, "to_dict"):
            processed_dict = processed.to_dict()
        elif isinstance(processed, Mapping):
            processed_dict = dict(processed)
        elif torch.is_tensor(processed):
            processed_dict = {"pixel_values": processed}
        elif isinstance(processed, np.ndarray):
            processed_dict = {"pixel_values": torch.from_numpy(processed)}
        else:
            processed_dict = {"pixel_values": torch.tensor(processed)}

        normalized_inputs: dict[str, torch.Tensor] = {}
        for key, value in processed_dict.items():
            if torch.is_tensor(value):
                tensor_value = value
            elif isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value)
            else:
                tensor_value = torch.tensor(value)
            if tensor_value.device != self.device:
                tensor_value = tensor_value.to(self.device)
            normalized_inputs[key] = tensor_value

        # Return both normalized inputs and the original processed object for SAM3 post-processing
        return normalized_inputs, processed if is_sam3_model else None

    @handle_inference_error
    def predict(self, input_data, return_embeddings: bool = False):
        with PerformanceContext(
            logger,
            "huggingface_inference",
            input_shape=tuple(input_data.shape) if hasattr(input_data, "shape") else None,
        ):
            processed_inputs, preprocessed_context = self.preprocess(input_data)

            captured_embeddings = None
            with torch.no_grad():
                output = self.model(**processed_inputs)

                # For HuggingFace models, try to capture image_embeddings if available
                if return_embeddings:
                    if hasattr(output, "image_embeddings"):
                        captured_embeddings = output.image_embeddings
                        logger.debug(
                            "Captured HuggingFace image_embeddings with shape: %s",
                            output.image_embeddings.shape,
                        )
                    elif isinstance(output, dict) and "image_embeddings" in output:
                        captured_embeddings = output["image_embeddings"]
                        logger.debug(
                            "Captured HuggingFace image_embeddings from dict with shape: %s",
                            output["image_embeddings"].shape,
                        )

            result = self.postprocess(output, preprocessed_context=preprocessed_context)

            # If embeddings were captured or we have SAM3 context, return dict
            if captured_embeddings is not None or preprocessed_context is not None:
                result_dict = {}

                # Add output
                if isinstance(result, np.ndarray):
                    result_dict["output"] = result
                    logger.info(
                        "HuggingFaceModel returning dict with output shape: %s, dtype: %s",
                        result.shape,
                        result.dtype,
                    )
                elif torch.is_tensor(result):
                    result_np = (
                        result.cpu().numpy() if result.device.type != "cpu" else result.numpy()
                    )
                    result_dict["output"] = result_np
                    logger.info(
                        "HuggingFaceModel returning dict with tensor output shape: %s, dtype: %s",
                        result_np.shape,
                        result_np.dtype,
                    )
                else:
                    result_dict["output"] = result
                    logger.warning(
                        "HuggingFaceModel returning dict with non-array output type: %s, value: %s",
                        type(result),
                        result,
                    )

                # Add embeddings if captured
                if captured_embeddings is not None:
                    result_dict["embeddings"] = (
                        captured_embeddings.cpu().numpy()
                        if captured_embeddings.device.type != "cpu"
                        else captured_embeddings.numpy()
                    )

                # Add preprocessed context for SAM3
                if preprocessed_context is not None:
                    result_dict["preprocessed_context"] = preprocessed_context

                return result_dict

            return result

    def postprocess(self, model_output, task: Optional[Task] = None, preprocessed_context=None):
        """Postprocess HuggingFace model output.

        HuggingFace models often return dictionaries with keys like 'logits', 'loss', etc.
        Extract the primary prediction tensor (logits) for inference.
        For SAM3, use the processor's post-processing to handle resizing correctly.
        """
        # Handle SAM3 pred_masks directly
        if isinstance(model_output, dict):
            # SAM3: prefer the processor's own instance post-processing, which scores every query
            # (pred_logits x presence_logits), keeps only those above the score threshold, resizes the
            # kept masks to the input size and binarises them. The hand-rolled branch below thresholds
            # all 200 queries and produced up to 100 low-confidence masks per chip.
            if (
                "pred_masks" in model_output
                and "pred_logits" in model_output
                and preprocessed_context is not None
                and hasattr(self.processor, "post_process_instance_segmentation")
            ):
                try:
                    import os as _os

                    score_thr = float(_os.environ.get("NXMNDR_SAM3_SCORE_THRESHOLD", "0.3"))  # transformers default
                    mask_thr = float(_os.environ.get("NXMNDR_SAM3_MASK_THRESHOLD", "0.5"))  # transformers default
                    original_sizes = preprocessed_context.get("original_sizes")
                    target_sizes = None
                    if original_sizes is not None:
                        sizes = original_sizes.tolist() if torch.is_tensor(original_sizes) else original_sizes
                        target_sizes = [tuple(int(v) for v in sz) for sz in sizes]
                    if score_thr <= 0:
                        # Score-free selection (unprompted use): the concept score is uninformative when no
                        # concept is given (presence ~0.01), so keep every query mask and select by geometry:
                        # area window, IoU dedupe in raw-logit order, then the group-mask filter.
                        from .sam import drop_group_masks, select_masks_by_area_and_iou

                        min_area = float(_os.environ.get("NXMNDR_SAM3_MIN_AREA", "0.001"))
                        max_area = float(_os.environ.get("NXMNDR_SAM3_MAX_AREA", "0.5"))
                        iou_thr = float(_os.environ.get("NXMNDR_SAM3_NMS_IOU", "0.5"))
                        pm = model_output["pred_masks"]
                        if target_sizes:
                            pm = torch.nn.functional.interpolate(
                                pm.float(), size=target_sizes[0], mode="bilinear", align_corners=False
                            )
                        binm = (pm.sigmoid()[0] > mask_thr).to(torch.uint8).cpu().numpy()
                        raw = model_output["pred_logits"].sigmoid()[0].detach().cpu().numpy()
                        order = np.argsort(-raw)
                        kept = select_masks_by_area_and_iou(binm, order, min_area, max_area, iou_thr)
                        masks = binm[kept] if kept else np.zeros((0,) + binm.shape[1:], np.uint8)
                        if masks.shape[0] > 1:
                            masks = masks[drop_group_masks(masks, logger=logger)]
                        logger.info(
                            "SAM3: score-free selection kept %d of %d masks (area %.3f-%.2f, IoU %.2f)",
                            masks.shape[0], binm.shape[0], min_area, max_area, iou_thr,
                        )
                        return masks
                    res = self.processor.post_process_instance_segmentation(
                        model_output, threshold=score_thr, mask_threshold=mask_thr, target_sizes=target_sizes
                    )[0]
                    scores, masks = res["scores"], res["masks"]
                    if len(masks) == 0:
                        logger.info("SAM3: no instance above score threshold %.2f", score_thr)
                        h, w = target_sizes[0] if target_sizes else tuple(model_output["pred_masks"].shape[-2:])
                        return np.zeros((0, int(h), int(w)), dtype=np.uint8)
                    order = torch.argsort(scores, descending=True)  # highest score first
                    masks = masks[order].to(torch.uint8).cpu().numpy()
                    scores = scores[order].detach().cpu().numpy()
                    if masks.shape[0] > 1:
                        from .sam import drop_group_masks

                        kept = drop_group_masks(masks, logger=logger)
                        masks, scores = masks[kept], scores[kept]
                        if masks.shape[0] == 0:
                            logger.info("SAM3: every instance was dropped as a group mask")
                            return masks  # (0, H, W) uint8
                    max_masks = 100
                    if masks.shape[0] > max_masks:
                        logger.warning("SAM3: limiting from %d to %d masks", masks.shape[0], max_masks)
                        masks, scores = masks[:max_masks], scores[:max_masks]
                    logger.info(
                        "SAM3: %d instances with score > %.2f (scores %.2f..%.2f) shape %s",
                        masks.shape[0], score_thr, float(scores.min()), float(scores.max()), masks.shape,
                    )
                    return masks
                except Exception as exc:  # pragma: no cover - fall back to the legacy thresholding
                    logger.warning("SAM3: processor post-processing failed (%s); using legacy thresholding", exc)
            # Handle SAM3 mask outputs
            if "pred_masks" in model_output:
                from PIL import Image

                # SAM3 returns pred_masks that need sigmoid and thresholding
                masks = model_output["pred_masks"]
                if torch.is_tensor(masks):
                    # Apply sigmoid to get probabilities
                    masks = torch.sigmoid(masks)
                    # Move to CPU and convert to numpy
                    if masks.device.type != "cpu":
                        masks = masks.detach().cpu()
                    masks_np = masks.numpy()

                    logger.info(f"SAM3: raw pred_masks after sigmoid shape: {masks_np.shape}")

                    # masks_np shape is typically [batch, num_queries, H, W]
                    # Remove batch dimension
                    if masks_np.ndim == 4:
                        masks_np = masks_np[0]  # Now [num_queries, H, W]

                    # Apply threshold to get binary masks
                    threshold = 0.1
                    binary_masks = (masks_np > threshold).astype(np.uint8)

                    logger.info(
                        f"SAM3: after threshold, have {binary_masks.shape[0]} masks to process"
                    )

                    # Resize back to original input dimensions if they were stored in context
                    if preprocessed_context is not None and hasattr(preprocessed_context, "get"):
                        original_sizes = preprocessed_context.get("original_sizes")
                        if original_sizes is not None and len(original_sizes) > 0:
                            orig_h, orig_w = original_sizes[0]  # First batch item
                            logger.info(
                                f"SAM3: resizing {binary_masks.shape[0]} masks to original size ({orig_h}, {orig_w})"
                            )

                            resized_masks = []
                            for i in range(binary_masks.shape[0]):
                                mask_2d = binary_masks[i, :, :]
                                # Only keep masks that have some pixels
                                if mask_2d.sum() > 0:
                                    # Resize using PIL
                                    mask_uint8 = (mask_2d * 255).astype(np.uint8)
                                    mask_pil = Image.fromarray(mask_uint8, mode="L")
                                    mask_resized = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                                    mask_resized_np = (np.array(mask_resized) > 127).astype(
                                        np.uint8
                                    )
                                    resized_masks.append(mask_resized_np)

                            if len(resized_masks) > 0:
                                result = np.stack(resized_masks, axis=0)
                                # Limit number of masks to prevent huge responses
                                max_masks = 100
                                if result.shape[0] > max_masks:
                                    logger.warning(
                                        f"SAM3: limiting from {result.shape[0]} to {max_masks} masks to avoid huge response"
                                    )
                                    result = result[:max_masks]
                                logger.info(
                                    f"SAM3: returning {result.shape[0]} non-empty masks with shape {result.shape}"
                                )
                                return result
                            else:
                                logger.warning("SAM3: all masks were empty after thresholding")
                                return np.array([])

                    # No resize needed, just filter empty masks
                    non_empty_masks = []
                    for i in range(binary_masks.shape[0]):
                        if binary_masks[i].sum() > 0:
                            non_empty_masks.append(binary_masks[i])

                    if len(non_empty_masks) > 0:
                        result = np.stack(non_empty_masks, axis=0)
                        # Limit number of masks to prevent huge responses
                        max_masks = 100
                        if result.shape[0] > max_masks:
                            logger.warning(
                                f"SAM3: limiting from {result.shape[0]} to {max_masks} masks to avoid huge response"
                            )
                            result = result[:max_masks]
                        logger.info(
                            f"SAM3: returning {result.shape[0]} non-empty masks with shape {result.shape}"
                        )
                        return result
                    else:
                        logger.warning("SAM3: all masks were empty after thresholding")
                        return np.array([])
            elif "logits" in model_output:
                model_output = model_output["logits"]
            else:
                for v in model_output.values():
                    if v is not None and (torch.is_tensor(v) or isinstance(v, np.ndarray)):
                        model_output = v
                        break

        return super().postprocess(model_output, task)


# Generic remote model that delegates to the server
class RemoteModel(Model):
    def __init__(self, inference_provider):
        self.inference_provider = inference_provider

    def preprocess(self, input_data, task: Optional[Task] = None):
        return input_data

    def postprocess(self, model_output, task: Optional[Task] = None):
        return model_output

    def predict(self, input_data):
        return self.inference_provider.predict(input_data)
