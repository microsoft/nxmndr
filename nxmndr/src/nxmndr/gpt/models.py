# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ChatGPT Vision model specifications and model classes.
"""

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union, Literal

import numpy as np
from PIL import Image

from ..exceptions import PredictionError, ValidationError
from ..logging import get_logger
from ..models import Model, ModelSpec
from ..server.proxy_routes import PROVIDERS
from ..tasks import Task

logger = get_logger(__name__)

# Image preprocessing limits for ChatGPT Vision API
MAX_IMAGE_DIMENSION = 2048  # Maximum pixel dimension before resizing
JPEG_ENCODE_QUALITY = 95  # JPEG quality for base64 encoding


@dataclass
class ModelEndpointSpec(ModelSpec):
    """Specification for Azure OpenAI endpoint configuration.

    ``provider`` is the explicit identity, ``"openai"`` or ``"azure_openai"``; empty
    (the default) detects it from the endpoint's host
    (``nxmndr.server.proxy_routes.resolve_provider``).
    """

    endpoint_url: str
    deployment_name: str
    api_version: str
    type: Literal["chat", "vision"] = "chat"
    provider: str = ""

    @classmethod
    def from_json(cls, json_str: str):
        """Create ModelEndpointSpec from JSON."""
        d = json.loads(json_str)
        return cls(
            endpoint_url=d["endpoint_url"],
            deployment_name=d["deployment_name"],
            api_version=d.get("api_version", "2025-01-01-preview"),
            type=d.get("type", "chat"),
            provider=d.get("provider", ""),
        )

    def validate(self):
        """Validate the endpoint specification."""
        if not self.endpoint_url:
            raise ValidationError("endpoint_url is required")
        if not self.deployment_name:
            raise ValidationError("deployment_name is required")
        if not self.api_version:
            raise ValidationError("api_version is required")
        if self.type not in ["chat", "vision"]:
            raise ValidationError("type must be either 'chat' or 'vision'")
        if self.provider and self.provider not in PROVIDERS:
            raise ValidationError(f"provider must be one of {PROVIDERS} or empty")

        # Basic URL validation
        if not (
            self.endpoint_url.startswith("http://") or self.endpoint_url.startswith("https://")
        ):
            raise ValidationError("endpoint_url must be a valid HTTP/HTTPS URL")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "endpoint_url": self.endpoint_url,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "type": self.type,
            "provider": self.provider,
        }

    def __str__(self) -> str:
        """String representation of the endpoint spec."""
        return f"ModelEndpointSpec(type={self.type}, endpoint={self.endpoint_url}, deployment={self.deployment_name}, api_version={self.api_version})"


@dataclass
class ChatGptVisionModelSpec(ModelSpec):
    """Base specification for ChatGPT Vision API models."""

    model_name: str = "gpt-4o"  # Default to latest vision model
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.0
    name: Optional[str] = None

    @classmethod
    def from_json(cls, json_str: str):
        """Create ChatGptVisionModelSpec from JSON."""
        d = json.loads(json_str)
        return cls(
            model_name=d.get("model_name", "gpt-4o"),
            system_prompt=d.get("system_prompt"),
            max_tokens=d.get("max_tokens", 1000),
            temperature=d.get("temperature", 0.0),
            name=d.get("name"),
        )

    def validate(self):
        """Validate the model specification."""
        if self.max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        if self.model_name not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]:
            logger.warning(f"Using unrecognized model: {self.model_name}")


@dataclass
class Gpt4VisionModelSpec(ChatGptVisionModelSpec):
    """Specification for GPT-4 Vision API models."""

    model_name: str = "gpt-4o"  # Default GPT-4 vision model

    @classmethod
    def from_json(cls, json_str: str):
        """Create Gpt4VisionModelSpec from JSON."""
        d = json.loads(json_str)
        return cls(
            model_name=d.get("model_name", "gpt-4o"),
            system_prompt=d.get("system_prompt"),
            max_tokens=d.get("max_tokens", 1000),
            temperature=d.get("temperature", 0.0),
            name=d.get("name"),
        )

    def validate(self):
        """Validate GPT-4 model specification."""
        if self.max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        # GPT-4 specific models
        gpt4_models = ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo", "gpt-4", "gpt-4-32k"]
        if self.model_name not in gpt4_models:
            logger.warning(f"Using unrecognized GPT-4 model: {self.model_name}")


@dataclass
class Gpt5VisionModelSpec(ModelSpec):
    """Specification for GPT-5 Vision API models using max_completion_tokens."""

    model_name: str = "gpt-5"  # Default GPT-5 model
    system_prompt: Optional[str] = None
    max_completion_tokens: int = 1000  # GPT-5 uses max_completion_tokens instead of max_tokens
    temperature: float = 0.0
    name: Optional[str] = None

    @classmethod
    def from_json(cls, json_str: str):
        """Create Gpt5VisionModelSpec from JSON."""
        d = json.loads(json_str)
        return cls(
            model_name=d.get("model_name", "gpt-5"),
            system_prompt=d.get("system_prompt"),
            max_completion_tokens=d.get(
                "max_completion_tokens", d.get("max_tokens", 1000)
            ),  # Fallback to max_tokens for compatibility
            temperature=d.get("temperature", 0.0),
            name=d.get("name"),
        )

    def validate(self):
        """Validate GPT-5 model specification."""
        if self.max_completion_tokens <= 0:
            raise ValidationError("max_completion_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        # GPT-5 specific models
        gpt5_models = ["gpt-5", "gpt-5-turbo", "gpt-5-vision"]
        if self.model_name not in gpt5_models:
            logger.warning(f"Using unrecognized GPT-5 model: {self.model_name}")


class ChatGptVisionModel(Model):
    """Model wrapper for ChatGPT Vision API calls."""

    def __init__(
        self,
        model_spec: Union[ChatGptVisionModelSpec, Gpt4VisionModelSpec, Gpt5VisionModelSpec],
        inference_provider=None,
    ):
        self.model_spec = model_spec
        self.inference_provider = inference_provider

        # Determine if this is GPT-5 for API parameter handling
        self.is_gpt5 = isinstance(model_spec, Gpt5VisionModelSpec) or (
            hasattr(model_spec, "model_name") and model_spec.model_name.startswith("gpt-5")
        )

        logger.info(
            f"Initialized ChatGptVisionModel with {model_spec.model_name} (GPT-5: {self.is_gpt5})"
        )

    def preprocess(self, input_data, task: Optional[Task] = None):
        """Convert input data to base64 encoded image string.

        Args:
            input_data: Can be numpy array, PIL Image, or file path
            task: Optional task context for preprocessing hints

        Returns:
            dict: Processed data with base64 image and metadata
        """
        try:
            # Convert various input formats to PIL Image
            if isinstance(input_data, str):
                # File path
                image = Image.open(input_data).convert("RGB")
            elif isinstance(input_data, np.ndarray):
                # Numpy array - assume HWC format
                if input_data.ndim == 2:
                    # Grayscale - convert to RGB
                    input_data = np.stack([input_data] * 3, axis=-1)
                elif input_data.ndim == 3 and input_data.shape[0] == 3:
                    # CHW format - transpose to HWC
                    input_data = input_data.transpose(1, 2, 0)

                # Normalize if needed (assume 0-1 range if max <= 1)
                if input_data.max() <= 1.0:
                    input_data = (input_data * 255).astype(np.uint8)

                image = Image.fromarray(input_data.astype(np.uint8))
            elif isinstance(input_data, Image.Image):
                # Already PIL Image
                image = input_data.convert("RGB")
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            # Resize if image is too large (ChatGPT has size limits)
            if max(image.size) > MAX_IMAGE_DIMENSION:
                ratio = MAX_IMAGE_DIMENSION / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to {new_size}")

            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=JPEG_ENCODE_QUALITY)
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return {"image_b64": image_b64, "original_size": image.size, "format": "jpeg"}

        except Exception as e:
            raise PredictionError(f"Failed to preprocess image: {str(e)}")

    def predict(self, input_data):
        """Delegate prediction to the inference provider."""
        if self.inference_provider is None:
            raise PredictionError("No inference provider available")
        return self.inference_provider.predict(input_data)

    def postprocess(self, model_output, task: Optional[Task] = None):
        """Process ChatGPT response.

        Args:
            model_output: Raw response from ChatGPT API
            task: Optional task context for output formatting

        Returns:
            Processed output (typically string or structured data)
        """
        try:
            if isinstance(model_output, dict):
                # Extract text from ChatGPT response structure
                if "choices" in model_output and len(model_output["choices"]) > 0:
                    content = model_output["choices"][0].get("message", {}).get("content", "")
                elif "content" in model_output:
                    content = model_output["content"]
                else:
                    content = str(model_output)
            else:
                content = str(model_output)

            # Task-specific post-processing
            if task == Task.IMAGE_CLASSIFICATION:
                # Try to extract structured classification results
                return self._parse_classification_output(content)
            elif task == Task.OBJECT_DETECTION:
                # Try to extract object detection results
                return self._parse_detection_output(content)
            else:
                # Return raw text for other tasks
                return content.strip()

        except Exception as e:
            logger.warning(f"Postprocessing failed, returning raw output: {e}")
            return str(model_output)

    def _parse_classification_output(self, text: str) -> Union[str, dict]:
        """Try to parse classification results from text."""
        # Simple heuristic - look for common classification patterns
        lines = text.strip().split("\n")

        # Look for numbered or bulleted lists
        classes = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                # Remove numbering/bullets and extract class name
                clean_line = line.lstrip("0123456789.-• ").strip()
                if clean_line:
                    classes.append(clean_line)

        if classes:
            return {"classes": classes, "raw_text": text}
        else:
            return text

    def _parse_detection_output(self, text: str) -> Union[str, dict]:
        """Try to parse object detection results from text."""
        # For now, return structured format with raw text
        # Could be enhanced to parse coordinates if ChatGPT provides them
        return {
            "detected_objects": text,
            "raw_text": text,
            "note": "Coordinate extraction not yet implemented",
        }
