# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Input validation utilities for the inference system.

This module provides comprehensive validation for model inputs, shapes,
types, and other data to prevent runtime errors.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from .exceptions import ValidationError
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validation with details about what was validated."""

    is_valid: bool
    message: Optional[str] = None
    corrected_value: Optional[Any] = None
    warnings: Optional[List[str]] = None


class InputValidator:
    """Comprehensive input validator for model inference."""

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: If True, reject any inputs that require correction.
                        If False, attempt to correct common issues.
        """
        self.strict_mode = strict_mode

    def validate_input_array(
        self,
        input_data: Any,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[str] = None,
        min_dims: int = 1,
        max_dims: int = 5,
        allow_batch_dim: bool = True,
    ) -> ValidationResult:
        """Validate input array/tensor data.

        Args:
            input_data: Input data to validate
            expected_shape: Expected shape (None means any shape is OK)
            expected_dtype: Expected data type
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions
            allow_batch_dim: Whether batch dimension is allowed/expected

        Returns:
            ValidationResult with validation outcome
        """
        warnings = []

        # Convert to numpy array if needed
        if torch.is_tensor(input_data):
            array = input_data.detach().cpu().numpy()
            warnings.append("Converted torch tensor to numpy array")
        elif not isinstance(input_data, np.ndarray):
            try:
                array = np.array(input_data)
                warnings.append(f"Converted {type(input_data).__name__} to numpy array")
            except Exception as e:
                return ValidationResult(
                    is_valid=False, message=f"Cannot convert input to array: {e}"
                )
        else:
            array = input_data

        # Check dimensions
        if not min_dims <= array.ndim <= max_dims:
            return ValidationResult(
                is_valid=False,
                message=f"Array has {array.ndim} dimensions, expected {min_dims}-{max_dims}",
            )

        # Check shape if specified
        if expected_shape is not None:
            if allow_batch_dim and len(expected_shape) == array.ndim - 1:
                # Check if array has batch dimension
                shape_to_check = array.shape[1:]
                warnings.append("Detected batch dimension")
            else:
                shape_to_check = array.shape

            if shape_to_check != expected_shape:
                if self.strict_mode:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Shape mismatch: got {array.shape}, expected {expected_shape}",
                    )
                else:
                    # Try to reshape if possible
                    try:
                        reshaped = array.reshape(expected_shape)
                        warnings.append(f"Reshaped array from {array.shape} to {expected_shape}")
                        array = reshaped
                    except ValueError:
                        return ValidationResult(
                            is_valid=False,
                            message=f"Cannot reshape {array.shape} to {expected_shape}",
                        )

        # Check dtype if specified
        corrected_array = array
        if expected_dtype is not None:
            if str(array.dtype) != expected_dtype:
                if self.strict_mode:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Dtype mismatch: got {array.dtype}, expected {expected_dtype}",
                    )
                else:
                    try:
                        corrected_array = array.astype(expected_dtype)
                        warnings.append(f"Converted dtype from {array.dtype} to {expected_dtype}")
                    except ValueError as e:
                        return ValidationResult(
                            is_valid=False,
                            message=f"Cannot convert dtype {array.dtype} to {expected_dtype}: {e}",
                        )

        # Check for NaN or inf values
        if np.issubdtype(corrected_array.dtype, np.floating):
            if np.any(np.isnan(corrected_array)):
                return ValidationResult(is_valid=False, message="Input contains NaN values")
            if np.any(np.isinf(corrected_array)):
                return ValidationResult(is_valid=False, message="Input contains infinite values")

        # Check value ranges for common cases
        if (
            expected_dtype in ["float32", "float64"]
            and np.all(corrected_array >= 0)
            and np.all(corrected_array <= 1)
        ):
            # Likely normalized image data - good
            pass
        elif expected_dtype in ["uint8"] and (
            np.any(corrected_array < 0) or np.any(corrected_array > 255)
        ):
            warnings.append("Values outside typical uint8 range [0, 255]")

        return ValidationResult(
            is_valid=True, corrected_value=corrected_array, warnings=warnings if warnings else None
        )

    def validate_model_spec(self, model_spec: Any) -> ValidationResult:
        """Validate a model specification."""
        if not hasattr(model_spec, "__class__"):
            return ValidationResult(
                is_valid=False, message="Invalid model spec: not a class instance"
            )

        spec_type = model_spec.__class__.__name__

        # Check for required attributes based on spec type
        required_attrs = {
            "PytorchModelSpec": ["model_class", "model_path"],
            "OnnxModelSpec": ["model_path"],
            "HuggingFaceModelSpec": ["repo_id"],
            "TorchHubModelSpec": ["repo", "name"],
        }

        if spec_type in required_attrs:
            for attr in required_attrs[spec_type]:
                if not hasattr(model_spec, attr) or getattr(model_spec, attr) is None:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Missing required attribute '{attr}' in {spec_type}",
                    )

        # Validate file paths exist for local models
        if hasattr(model_spec, "model_path"):
            from pathlib import Path

            model_path = Path(model_spec.model_path)
            if not model_path.exists():
                return ValidationResult(
                    is_valid=False, message=f"Model file not found: {model_path}"
                )

        return ValidationResult(is_valid=True)

    def validate_batch_size(self, batch_size: int, max_batch_size: int) -> ValidationResult:
        """Validate batch size."""
        if not isinstance(batch_size, int):
            return ValidationResult(
                is_valid=False, message=f"Batch size must be integer, got {type(batch_size)}"
            )

        if batch_size <= 0:
            return ValidationResult(
                is_valid=False, message=f"Batch size must be positive, got {batch_size}"
            )

        if batch_size > max_batch_size:
            if self.strict_mode:
                return ValidationResult(
                    is_valid=False,
                    message=f"Batch size {batch_size} exceeds maximum {max_batch_size}",
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    corrected_value=max_batch_size,
                    warnings=[f"Clamped batch size from {batch_size} to {max_batch_size}"],
                )

        return ValidationResult(is_valid=True)


def validate_input(
    input_data: Any,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[str] = None,
    strict: bool = False,
) -> Tuple[Any, List[str]]:
    """Convenience function for input validation.

    Args:
        input_data: Data to validate
        expected_shape: Expected shape
        expected_dtype: Expected data type
        strict: Whether to use strict validation

    Returns:
        Tuple of (validated_data, warnings)

    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator(strict_mode=strict)
    result = validator.validate_input_array(input_data, expected_shape, expected_dtype)

    if not result.is_valid:
        raise ValidationError(result.message or "Validation failed", invalid_value=input_data)

    validated_data = result.corrected_value if result.corrected_value is not None else input_data
    warnings = result.warnings or []

    # Log warnings
    for warning in warnings:
        logger.warning(f"Input validation: {warning}")

    return validated_data, warnings


def validate_model_spec(model_spec: Any, strict: bool = False) -> None:
    """Convenience function for model spec validation.

    Args:
        model_spec: Model specification to validate
        strict: Whether to use strict validation

    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator(strict_mode=strict)
    result = validator.validate_model_spec(model_spec)

    if not result.is_valid:
        raise ValidationError(
            result.message or "Validation failed",
            invalid_value=model_spec,
            expected=f"Valid {model_spec.__class__.__name__}",
        )


# Common validation patterns
COMMON_IMAGE_SHAPES = {
    "imagenet": (224, 224, 3),
    "cifar10": (32, 32, 3),
    "mnist": (28, 28, 1),
    "coco": (640, 640, 3),
}

COMMON_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
}
