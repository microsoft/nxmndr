# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as torch_rpc
from transformers import AutoModel, AutoProcessor

from ..config import get_config
from ..exceptions import (
    InferenceError,
    ModelLoadError,
    PredictionError,
    handle_inference_error,
)
from ..logging import PerformanceContext, get_logger
from ..memory import get_model_cache
from ..validation import validate_input, validate_model_spec
from ..models import (
    HuggingFaceModel,
    HuggingFaceModelSpec,
    ModelSpec,
    OnnxModel,
    OnnxModelSpec,
    PytorchModel,
    PytorchModelSpec,
    RemoteModel,
    TorchHubModelSpec,
)
from ..models import get_loader, get_registry
from ..models.ultralytics_yolo import UltralyticsModelSpec, load_ultralytics_yolo

from . import inference_pb2, inference_pb2_grpc

logger = get_logger("nxmndr.inference")

# Defaults for ONNX export from TorchHub models
DEFAULT_ONNX_OPSET_VERSION = 11
DEFAULT_INPUT_SHAPE = (3, 224, 224)  # CHW format


# Model class overrides for specific repos that need explicit model classes
# instead of relying on AutoModel detection. Values are (module_path, model_class_name, processor_class_name).
MODEL_CLASS_OVERRIDES = {
    "facebook/sam3": ("transformers", "Sam3Model", "Sam3Processor"),
}


def _get_override_classes(repo_id: str):
    """Lazy import model and processor classes from overrides dictionary."""
    override = MODEL_CLASS_OVERRIDES.get(repo_id.lower())
    if not override:
        return None

    module_path, model_class_name, processor_class_name = override
    try:
        import importlib

        module = importlib.import_module(module_path)
        model_class = getattr(module, model_class_name)
        processor_class = getattr(module, processor_class_name)
        return model_class, processor_class
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Failed to load override classes for %s: %s. Falling back to AutoModel.", repo_id, e
        )
        return None


# Unified InferenceSession interface
class InferenceSession:
    """
    Unifies model loading and inference for all providers and specs.
    Mirrors the ort.InferenceSession interface.
    """

    @handle_inference_error
    def __init__(self, model_spec, provider):
        self.config = get_config()

        # Validate model spec
        validate_model_spec(model_spec)

        self.model_spec = model_spec
        self.provider = provider

        # For providers that have load_spec, delegate model loading to the provider
        if hasattr(provider, "load_spec"):
            # RPC providers and API-based providers (like ChatGPT Vision) that implement load_spec
            context_name = (
                "remote_model_loading"
                if isinstance(provider, TorchRpcInferenceProvider)
                else "api_provider_loading"
            )
            with PerformanceContext(logger, context_name, model_type=model_spec.__class__.__name__):
                provider.load_spec(model_spec)
            self.model = None  # Model is loaded by provider, not locally
        else:
            # For local providers, load the model locally
            with PerformanceContext(
                logger, "model_loading", model_type=model_spec.__class__.__name__
            ):
                self.model = self._convert_spec_to_model()

        logger.info(f"Initialized InferenceSession with {model_spec.__class__.__name__}")

    def _convert_spec_to_model(self):
        # Check model cache first
        cache_key = self._get_cache_key()
        cached_model = get_model_cache().get(cache_key)
        if cached_model is not None:
            logger.info(f"Using cached model: {cache_key}")
            return cached_model

        spec = self.model_spec
        loader = get_loader(spec)
        if loader is None:
            raise ModelLoadError(
                f"No registered model loader for spec type: {spec.__class__.__name__}"
            )

        try:
            model = loader(spec, self.provider, self)
            # Cache the loaded model
            get_model_cache().put(cache_key, model)
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}", cause=e)

    def _get_cache_key(self) -> str:
        """Generate a cache key for the model spec."""
        spec = self.model_spec
        key_parts = [spec.__class__.__name__]

        if hasattr(spec, "model_path"):
            key_parts.append(str(spec.model_path))
        if hasattr(spec, "repo_id"):
            key_parts.append(spec.repo_id)
        if hasattr(spec, "name") and spec.name:
            key_parts.append(spec.name)

        return ":".join(key_parts)

    @handle_inference_error
    def run(self, input_data, task=None, prompt=None):
        """Run inference with validation and performance monitoring.

        Args:
            input_data: Input data for inference
            task: Optional task type for task-specific processing
            prompt: Optional prompt for ChatGPT Vision models
        """
        with PerformanceContext(
            logger,
            "inference_run",
            input_shape=getattr(input_data, "shape", None),
            model_type=self.model_spec.__class__.__name__,
        ):
            # Validate and prepare input
            validated_input, warnings = validate_input(
                input_data,
                strict=False,  # Allow corrections for better user experience
            )

            if self.model:
                # Local or remote, but model object exists
                try:
                    preprocessed = self.model.preprocess(validated_input, task=task)
                    output = self.model.predict(preprocessed)
                    return self.model.postprocess(output, task=task)
                except Exception as e:
                    raise PredictionError(f"Model prediction failed: {e}", cause=e)
            elif hasattr(self.provider, "predict"):
                # If no model, but provider can predict (e.g., remote fallback)
                try:
                    # Try to pass prompt parameter if the provider's predict method supports it
                    import inspect

                    sig = inspect.signature(self.provider.predict)
                    kwargs = {}
                    if "task" in sig.parameters:
                        kwargs["task"] = task
                    if "prompt" in sig.parameters:
                        kwargs["prompt"] = prompt
                    return self.provider.predict(validated_input, **kwargs)
                except Exception as e:
                    raise PredictionError(f"Provider prediction failed: {e}", cause=e)
            else:
                raise InferenceError("No model or provider available for inference.")


class InferenceProvider(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def has_gpu(self) -> bool:
        """Return True if this provider has access to a GPU, else False."""
        return False

    def load_model(self, model_spec: ModelSpec):
        """Load a model given its specification."""
        pass


class LocalInferenceProvider(InferenceProvider):
    @handle_inference_error
    def __init__(self, device: str | None = None):
        super().__init__()
        config = get_config()

        if device is None:
            # Default to auto-detection if no device specified
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.config = config
        logger.info(f"Initialized LocalInferenceProvider with device: {device}")

    def has_gpu(self) -> bool:
        return self.device.startswith("cuda") and torch.cuda.is_available()


class RemoteInferenceProvider(InferenceProvider):
    def __init__(self, grpc_channel):
        super().__init__()
        self.grpc_channel = grpc_channel  # Should be a grpc.Channel instance
        self.model_spec: ModelSpec = None
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.grpc_channel)
        self.model_id = None

    def load_spec(self, model_spec: ModelSpec):
        # Only store the spec; actual Model is created by InferenceSession
        self.model_spec = model_spec
        fmt_map = {
            OnnxModelSpec: inference_pb2.ONNX,
            PytorchModelSpec: inference_pb2.PYTORCH,
            TorchHubModelSpec: inference_pb2.TORCHHUB,
            HuggingFaceModelSpec: inference_pb2.HUGGINGFACE,
        }
        model_format = fmt_map.get(type(model_spec), inference_pb2.MODEL_FORMAT_UNSPECIFIED)
        artifact_bytes = b""
        source = (
            getattr(model_spec, "model_path", "")
            or getattr(model_spec, "repo", "")
            or getattr(model_spec, "repo_id", "")
        )
        if isinstance(model_spec, OnnxModelSpec):
            with open(model_spec.model_path, "rb") as f:
                artifact_bytes = f.read()
        spec_msg = inference_pb2.ModelSpec(
            model_id="",  # let server assign
            format=model_format,
            name=getattr(model_spec, "name", "") or getattr(model_spec, "repo_id", ""),
            source=source,
            model_class=getattr(model_spec, "model_class", ""),
            artifact=artifact_bytes,
        )
        load_req = inference_pb2.LoadModelRequest(spec=spec_msg, overwrite=False)
        resp = self.stub.LoadModel(load_req)
        if not resp.success:
            raise RuntimeError(f"Remote load failed: {resp.message}")
        self.model_id = resp.model_id

    def predict(self, input_data):
        arr = input_data
        if not hasattr(arr, "shape"):
            # Attempt to convert generic input to numpy array
            arr = np.array(arr)
        shape = list(arr.shape)
        dtype = str(arr.dtype)
        if not self.model_id:
            raise RuntimeError(
                "Remote model not loaded (missing model_id). Did you call load_spec?"
            )
        request = inference_pb2.PredictRequest(
            model_id=self.model_id,
            input=arr.tobytes(),
            shape=shape,
            dtype=dtype,
        )
        response = self.stub.Predict(request)
        # Allow legitimately empty output arrays (e.g. zero-sized batch) but verify shape consistency
        if response.output is None:
            meta = (
                f"server_shape={list(response.shape)} server_dtype={response.dtype}"
                if response.shape
                else "no_shape_info"
            )
            raise RuntimeError(
                f"Missing output buffer from remote prediction ({meta}). Check server logs."
            )
        try:
            out = np.frombuffer(response.output, dtype=response.dtype).reshape(
                tuple(response.shape)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to deserialize remote output: shape={response.shape} dtype={response.dtype}: {e}"
            ) from e
        return out

    def has_gpu(self) -> bool:
        # Use Capabilities for GPU detection
        resp = self.stub.Capabilities(inference_pb2.CapabilitiesRequest())
        for cap in resp.capabilities:
            if cap.key == "cuda":
                return cap.value.lower() == "true"
        return False


class TorchRpcInferenceProvider(InferenceProvider):
    """Inference provider that delegates PyTorch model execution to a torch.distributed.rpc worker.

    This provider assumes an external process has started the RPC framework or will be started
    via helper utilities. For simplicity we allow lazy init on first predict if not already done.
    """

    def __init__(
        self,
        worker_name: str = "worker",
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        device: str | None = None,
    ):
        super().__init__()
        self.worker_name = worker_name
        self.master_addr = master_addr
        self.master_port = master_port
        self.initialized = False
        self._driver_initialized = False
        self.model_spec: ModelSpec = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def _init_rpc(self):
        if self.initialized:
            return
        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.master_port))
        # rank 0 acts as driver; skip if an RPC agent already exists (test harness may have initialized it)
        if not torch_rpc._is_current_rpc_agent_set():  # type: ignore[attr-defined]
            # Ensure process group is initialized using public API to avoid deprecated implicit construction
            if dist.is_available() and not dist.is_initialized():
                if not self._driver_initialized:
                    init_method = f"tcp://{self.master_addr}:{self.master_port}"
                    logger.debug("[RPC-DRIVER] init_process_group backend=gloo %s", init_method)
                    dist.init_process_group(
                        backend="gloo", rank=0, world_size=2, init_method=init_method
                    )
                    self._driver_initialized = True
            # Use explicit TensorPipe backend options with init_method to avoid deprecated implicit PG usage
            init_method = f"tcp://{self.master_addr}:{self.master_port}"
            opts = torch_rpc.TensorPipeRpcBackendOptions(init_method=init_method)
            logger.debug("[RPC-DRIVER] Initializing RPC driver at %s", init_method)
            torch_rpc.init_rpc("driver", rank=0, world_size=2, rpc_backend_options=opts)
        self.initialized = True

    def load_spec(self, model_spec: ModelSpec):
        """Load model specification on the RPC worker."""
        self.model_spec = model_spec
        if not self.initialized:
            self._init_rpc()

        # Send model spec to worker for loading
        model_spec_dict = {
            "model_class": model_spec.model_class,
            "model_path": model_spec.model_path,
            "name": model_spec.name,
        }

        from ..server.rpc_worker import _rpc_load_model

        result = torch_rpc.rpc_sync(self.worker_name, _rpc_load_model, args=(model_spec_dict,))
        if not result.get("success", False):
            raise RuntimeError(
                f"Failed to load model on RPC worker: {result.get('error', 'Unknown error')}"
            )

        logger.info(f"Model {model_spec.name} loaded successfully on RPC worker")

    def predict(self, input_data, task=None):
        if not self.initialized:
            self._init_rpc()
        arr = np.array(input_data) if not isinstance(input_data, np.ndarray) else input_data
        tensor = torch.from_numpy(arr) if not torch.is_tensor(input_data) else input_data
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        from ..server.rpc_worker import DEFAULT_MODEL_ID, _rpc_infer

        # The provider's single model: preloaded by run_worker or set by load_spec.
        result = torch_rpc.rpc_sync(
            self.worker_name, _rpc_infer, args=(DEFAULT_MODEL_ID, tensor, self.device)
        )
        if torch.is_tensor(result):
            return result.detach().cpu().numpy()
        return result

    def has_gpu(self) -> bool:
        return self.device.startswith("cuda") and torch.cuda.is_available()

    def ping(self, timeout: float = 5.0):
        if not self.initialized:
            self._init_rpc()
        began = time.time()
        while time.time() - began < timeout:
            try:
                from ..server.rpc_worker import _rpc_status

                resp = torch_rpc.rpc_sync(self.worker_name, _rpc_status, args=(), timeout=1.0)
                if resp.get("status") == "ok" and resp.get("model_loaded"):
                    return resp
            except Exception:
                time.sleep(0.05)
        raise TimeoutError("RPC ping timed out after %.2fs" % timeout)

    def close(self):
        if self.initialized:
            try:
                # Best-effort remote stop so worker exits loop gracefully
                try:
                    from ..server.rpc_worker import _rpc_stop

                    torch_rpc.rpc_sync(self.worker_name, _rpc_stop, args=(), timeout=2.0)
                except Exception:
                    logger.debug(
                        "_rpc_stop remote call failed or timed out; proceeding with shutdown"
                    )
                torch_rpc.shutdown()
            except Exception:
                pass
            finally:
                self.initialized = False

    # Context manager support
    def __enter__(self):
        if not self.initialized:
            self._init_rpc()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# --- Loader Implementations ---
def load_torchhub(spec: TorchHubModelSpec, provider, session):
    """Loader for TorchHubModelSpec.

    Remote strategy: export model to ONNX then delegate to RemoteInferenceProvider using ONNX path.
    """
    if isinstance(provider, LocalInferenceProvider):
        device = "cuda" if provider.has_gpu() else "cpu"

        # Set torch hub cache directory if provider has model_cache_dir configured
        hub_kwargs = dict(spec.hub_kwargs or {})
        try:
            base_cache = provider.config.inference.model_cache_dir
            if base_cache:
                hub_cache = Path(base_cache) / "torchhub"
                hub_cache.mkdir(parents=True, exist_ok=True)
                # torch.hub respects TORCH_HOME environment variable
                import os

                original_torch_home = os.environ.get("TORCH_HOME")
                os.environ["TORCH_HOME"] = str(hub_cache)
                logger.debug("Using TorchHub cache: %s", hub_cache)
        except Exception as exc:
            logger.warning("Failed to set TorchHub cache dir: %s", exc)
            original_torch_home = None

        try:
            model = torch.hub.load(spec.repo, spec.name, *(spec.hub_args or ()), **hub_kwargs)
        finally:
            # Restore original TORCH_HOME
            if original_torch_home is not None:
                import os

                if original_torch_home:
                    os.environ["TORCH_HOME"] = original_torch_home
                else:
                    os.environ.pop("TORCH_HOME", None)

        model.eval().to(device)
        return PytorchModel(model, provider)
    elif isinstance(provider, RemoteInferenceProvider):
        device = "cpu"
        model = torch.hub.load(
            spec.repo, spec.name, *(spec.hub_args or ()), **(spec.hub_kwargs or {})
        )
        model.eval().to(device)
        dummy_input = torch.randn(1, *getattr(model, "input_shape", DEFAULT_INPUT_SHAPE))
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.onnx.export(model, dummy_input, tmp_path, export_params=True, opset_version=DEFAULT_ONNX_OPSET_VERSION)
            onnx_spec = OnnxModelSpec(model_path=str(tmp_path), name=spec.name)
            session.model_spec = onnx_spec
            provider.load_spec(onnx_spec)
        finally:
            # Best-effort cleanup; remote provider already read bytes during load_spec
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return RemoteModel(provider)
    else:
        raise ValueError("Unknown inference provider for TorchHubModelSpec")


def load_pytorch(spec: PytorchModelSpec, provider, session):
    """Load PyTorch model with optimized memory management and error handling."""
    with PerformanceContext(logger, "pytorch_model_loading", model_path=spec.model_path):
        if isinstance(provider, LocalInferenceProvider):
            device = provider.device

            try:
                model = spec.model_class()

                # Optimize loading for device
                map_location = (
                    device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                )

                # Check cache directory first if configured
                model_path = Path(spec.model_path)
                if not model_path.is_absolute() or not model_path.exists():
                    try:
                        base_cache = provider.config.inference.model_cache_dir
                        if base_cache:
                            cached_path = Path(base_cache) / "pytorch" / model_path.name
                            if cached_path.exists():
                                logger.debug("Loading PyTorch model from cache: %s", cached_path)
                                model_path = cached_path
                    except Exception as exc:
                        logger.debug("Cache lookup failed: %s", exc)

                state_dict = torch.load(str(model_path), map_location=map_location, weights_only=True)
                model.load_state_dict(state_dict, strict=False)

                if map_location.startswith("cuda") and torch.cuda.is_available():
                    model = model.to(map_location)

                model.eval()
                return PytorchModel(model, provider)

            except FileNotFoundError:
                from ..exceptions import ModelNotFoundError

                raise ModelNotFoundError(spec.model_path)
            except Exception as e:
                from ..exceptions import ModelLoadError

                raise ModelLoadError(f"Failed to load PyTorch model: {e}", cause=e)

        elif isinstance(provider, (RemoteInferenceProvider, TorchRpcInferenceProvider)):
            provider.load_spec(spec)
            return RemoteModel(provider)
        else:
            raise ValueError("Unknown inference provider for PytorchModelSpec")


def load_huggingface(spec: HuggingFaceModelSpec, provider, session):
    """Loader for HuggingFace hub models. Remote strategy currently treats as opaque remote model."""
    if isinstance(provider, LocalInferenceProvider):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ModelLoadError(
                "HuggingFace support requires the huggingface_hub package"
            ) from exc

        # Resolve cache directory precedence: explicit spec value, provider config, environment fallback
        cache_dir = spec.cache_dir
        if not cache_dir:
            try:
                # Use provider's model_cache_dir with huggingface subdirectory
                base_cache = provider.config.inference.model_cache_dir
                if base_cache:
                    cache_dir = str(Path(base_cache) / "huggingface")
            except Exception:  # pragma: no cover - defensive
                cache_dir = None
        # Normalize cache_dir: handle None, empty string, or string "None"
        if cache_dir:
            cache_dir = str(cache_dir).strip()
            if cache_dir.lower() == "none" or not cache_dir:
                cache_dir = None
        if cache_dir:
            try:
                cache_path = Path(cache_dir).expanduser()
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_dir = str(cache_path)
            except Exception as exc:
                logger.warning("Failed to create HuggingFace cache dir %s: %s", cache_dir, exc)

        snapshot_kwargs: dict[str, object] = {
            "repo_id": spec.repo_id,
            "cache_dir": cache_dir,
        }
        if spec.revision:
            snapshot_kwargs["revision"] = spec.revision
        if spec.token:
            snapshot_kwargs["token"] = spec.token
            logger.debug(
                "Using HuggingFace token for %s (token length: %d)", spec.repo_id, len(spec.token)
            )
        else:
            logger.warning(
                "No token provided for HuggingFace model %s - may fail for gated repos",
                spec.repo_id,
            )

        # Try loading from cache first (local_files_only=True), fall back to download if needed
        try:
            snapshot_kwargs["local_files_only"] = True
            repo_path = snapshot_download(**snapshot_kwargs)
            logger.info("Loaded HuggingFace model %s from cache", spec.repo_id)
        except Exception as cache_exc:
            # Cache miss or error, try downloading
            logger.debug("Cache miss for %s: %s. Downloading...", spec.repo_id, cache_exc)
            try:
                snapshot_kwargs["local_files_only"] = False
                repo_path = snapshot_download(**snapshot_kwargs)
                logger.info("Downloaded HuggingFace model %s", spec.repo_id)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to download HuggingFace model {spec.repo_id}: {exc}", cause=exc
                ) from exc

        weight_candidate = None
        if spec.filename:
            candidate_path = Path(repo_path) / spec.filename
            if candidate_path.exists():
                weight_candidate = candidate_path
            else:
                logger.warning(
                    "Specified weight file %s not found in %s; relying on Transformers defaults",
                    spec.filename,
                    repo_path,
                )

        # Load transformer model and processor from the resolved snapshot path
        logger.warning(
            "Loading HuggingFace model %s with trust_remote_code=True. "
            "Only load models from repositories you trust.",
            spec.repo_id,
        )
        model_kwargs: dict[str, object] = {"trust_remote_code": True}
        processor_kwargs = dict(model_kwargs)
        # Prefer torch checkpoints when safetensors are absent in the snapshot.
        try:
            safetensors_path = Path(repo_path) / "model.safetensors"
            torch_bin_path = Path(repo_path) / "pytorch_model.bin"
            if not safetensors_path.exists() and torch_bin_path.exists():
                model_kwargs.setdefault("use_safetensors", False)
        except Exception:
            pass
        if spec.local_files_only:
            model_kwargs["local_files_only"] = True
            processor_kwargs["local_files_only"] = True
        if spec.token:
            model_kwargs["token"] = spec.token
            processor_kwargs["token"] = spec.token

        try:
            # Check for model class overrides
            override = _get_override_classes(spec.repo_id)
            if override:
                model_class, processor_class = override
                logger.debug("Using override for %s: %s", spec.repo_id, model_class.__name__)
                model = model_class.from_pretrained(str(repo_path), **model_kwargs)
                processor = processor_class.from_pretrained(str(repo_path), **processor_kwargs)
                model.eval()
                logger.info("Successfully loaded %s using %s", spec.repo_id, model_class.__name__)
            else:
                # Use AutoModel with trust_remote_code for all other models
                logger.debug(
                    "Loading HuggingFace model from %s with trust_remote_code=True", repo_path
                )
                model = AutoModel.from_pretrained(str(repo_path), **model_kwargs)
                try:
                    processor = AutoProcessor.from_pretrained(str(repo_path), **processor_kwargs)
                except Exception:
                    # Fallback to AutoImageProcessor or custom processor when AutoProcessor is absent
                    try:
                        from transformers import AutoImageProcessor

                        processor = AutoImageProcessor.from_pretrained(
                            str(repo_path), trust_remote_code=True
                        )
                    except Exception:
                        # Last resort: attempt to import processor_class from repo_path
                        try:
                            import importlib.util

                            proc_file = Path(repo_path) / "modeling_exampleconv.py"
                            if proc_file.exists():
                                spec_mod = importlib.util.spec_from_file_location(
                                    "modeling_exampleconv", proc_file
                                )
                                if spec_mod and spec_mod.loader:
                                    module = importlib.util.module_from_spec(spec_mod)
                                    spec_mod.loader.exec_module(module)
                                    processor_cls = getattr(module, "ExampleImageProcessor", None)
                                    if processor_cls:
                                        processor = processor_cls()
                        except Exception:
                            processor = None
                model.eval()
                logger.info("Successfully loaded HuggingFace model: %s", model.__class__.__name__)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load HuggingFace model from {repo_path}: {exc}", cause=exc
            ) from exc

        if weight_candidate:
            logger.info(
                "Loaded HuggingFace model from locally cached weights: %s",
                weight_candidate,
            )
        # Place the model on the provider's device (from_pretrained loads on CPU); HuggingFaceModel
        # derives its own .device from the parameters, so inputs follow the move.
        target_device = provider.device if provider.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if callable(getattr(model, "to", None)):
            model.to(target_device)
            logger.info("HuggingFace model %s placed on %s", spec.repo_id, target_device)
        else:
            # Remote-code models need not be torch modules; they stay where they loaded.
            logger.debug("HuggingFace model %s has no .to(); device move skipped", spec.repo_id)
        return HuggingFaceModel(model, processor, provider, repo_path=str(repo_path))
    elif isinstance(provider, RemoteInferenceProvider):
        return RemoteModel(provider)
    else:
        raise ValueError("Unknown inference provider for HuggingFaceModelSpec")


def load_onnx(spec: OnnxModelSpec, provider, session):
    """Loader for ONNX models choosing execution providers based on GPU availability."""
    if isinstance(provider, LocalInferenceProvider):
        if provider and hasattr(provider, "has_gpu") and provider.has_gpu():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Check cache directory first if configured
        model_path = Path(spec.model_path)
        if not model_path.is_absolute() or not model_path.exists():
            try:
                base_cache = provider.config.inference.model_cache_dir
                if base_cache:
                    cached_path = Path(base_cache) / "onnx" / model_path.name
                    if cached_path.exists():
                        logger.debug("Loading ONNX model from cache: %s", cached_path)
                        model_path = cached_path
            except Exception as exc:
                logger.debug("Cache lookup failed: %s", exc)

        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(model_path), providers=providers)
        return OnnxModel(ort_session, provider)
    elif isinstance(provider, RemoteInferenceProvider):
        return RemoteModel(provider)
    else:
        raise ValueError("Unknown inference provider for OnnxModelSpec")


# Register loaders and specs with unified registry
registry = get_registry()
registry.register_spec("torchhub", TorchHubModelSpec, load_torchhub)
registry.register_spec("pytorch", PytorchModelSpec, load_pytorch)
registry.register_spec("huggingface", HuggingFaceModelSpec, load_huggingface)
registry.register_spec("onnx", OnnxModelSpec, load_onnx)
registry.register_spec("ultralytics", UltralyticsModelSpec, load_ultralytics_yolo)
