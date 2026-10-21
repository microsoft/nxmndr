# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""nxmndr package root.

Main inference and ML toolkit with proper submodule organization.
Access functionality through submodules:
- nxmndr.logging: Logging utilities
- nxmndr.config: Configuration management
- nxmndr.exceptions: Custom exceptions
- nxmndr.validation: Input validation
- nxmndr.models: Model specifications
- nxmndr.inference: Inference sessions and providers
- nxmndr.server: Server functions
- nxmndr.constants: Model file extension constants and utilities
"""

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "InferenceGrpcClient",
    "InferenceGrpcError",
    "PredictResult",
    # Constants
    "MODEL_EXTENSIONS",
    "PYTORCH_EXTENSIONS",
    "ONNX_EXTENSIONS",
    "SAFETENSORS_EXTENSIONS",
    "is_model_file",
    "is_pytorch_model",
    "is_onnx_model",
    "get_model_type",
]


def __getattr__(name):
    if name in {"InferenceGrpcClient", "InferenceGrpcError", "PredictResult"}:
        from . import client

        return getattr(client, name)
    if name in {
        "MODEL_EXTENSIONS",
        "PYTORCH_EXTENSIONS",
        "ONNX_EXTENSIONS",
        "SAFETENSORS_EXTENSIONS",
        "is_model_file",
        "is_pytorch_model",
        "is_onnx_model",
        "get_model_type",
    }:
        from . import constants

        return getattr(constants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
