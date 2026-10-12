# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inference package.

The generated gRPC bindings (``inference_pb2``, ``inference_pb2_grpc``) import only
protobuf and grpc, so host-side clients can load them without the ML runtime.
The provider and session classes live in ``nxmndr.inference.inference`` (torch,
transformers, onnxruntime) and are imported only when one of them is requested.
"""

import importlib

_LAZY_SUBMODULES = {"inference_pb2", "inference_pb2_grpc"}
_LAZY_RUNTIME_NAMES = {
    "InferenceProvider",
    "LocalInferenceProvider",
    "RemoteInferenceProvider",
    "TorchRpcInferenceProvider",
    "InferenceSession",
}

__all__ = [
    "InferenceProvider",
    "LocalInferenceProvider",
    "RemoteInferenceProvider",
    "TorchRpcInferenceProvider",
    "InferenceSession",
    "inference_pb2",
    "inference_pb2_grpc",
]


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _LAZY_RUNTIME_NAMES:
        from . import inference as _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
