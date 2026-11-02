# nxmndr

(named after / pronounced: **Anaximander**)

Backend and experimentation ground for multi-format vision inference (PyTorch, ONNX, HuggingFace) with local, remote (gRPC), and private PyTorch worker execution paths.

## Installation

### Editable Install (Recommended for Development)

Create a virtual environment (conda or venv) and install in editable mode:

```bash
# Using conda
conda create -n nxmndr python=3.11 -y
conda activate nxmndr

# Install with desired extras
pip install -e .                    # core only
pip install -e .[server]            # + server deps (aiohttp, azure-identity, openai)
pip install -e .[dev]               # + dev tools (pytest, ruff, mypy, etc.)
pip install -e .[server,dev]        # all extras
```

### Using requirements.txt Files

For users preferring flat requirements files:

```bash
pip install -r requirements.txt              # core runtime deps
pip install -r requirements-server.txt       # server deployment deps
pip install -r requirements-dev.txt          # dev tools (includes requirements.txt)
```

> **Note:** `pyproject.toml` is the source of truth for dependencies. The requirements files are maintained for legacy workflows.

### CUDA Note

If you need CUDA, install the appropriate Torch wheel *before* installing `nxmndr` so the dependency (`torch>=2.1.0`) is satisfied by the GPU build and not replaced by a CPU wheel.

Preferred order (example for CUDA 12.1):

```bash
# 1. Install a CUDA-enabled PyTorch wheel
pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Install nxmndr with desired extras (will reuse existing torch)
pip install -e .[server]
```

Verification:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Console Scripts

The package installs convenient entry points:

| Command | Purpose |
|---------|---------|
| `nxmndr-server` | gRPC inference server (+ HTTP Azure proxy with `--azure-proxy`) |
| `python -m nxmndr` | Multi-command launcher (see `--help`) |

### Examples Directory

Example scripts live under `examples/` (not installed as console scripts):
* `examples/client.py` (gRPC client utility)
* `examples/example_chatgpt_vision.py` (Azure ChatGPT Vision usage)

Legacy direct invocation via `python -m server.server` still works when running from source.

## Quickstart (Local PyTorch)

```python
from nxmndr.inference import InferenceSession, LocalInferenceProvider
from nxmndr.models import PytorchModelSpec
import torch.nn as nn

# Define a simple model (or import your own)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

spec = PytorchModelSpec(model_class=MyModel, model_path='tst/example_model/example_model.pth', name='example')
provider = LocalInferenceProvider()
session = InferenceSession(spec, provider)

import numpy as np
inp = np.zeros((1,3,32,32), dtype=np.float32)
out = session.run(inp)
print(out.shape)
```

## Remote Inference (gRPC)

1. Start the server:

```bash
nxmndr-server                                  # gRPC on 127.0.0.1:50051
nxmndr-server --azure-proxy --http-port 8080   # also the HTTP proxy, on 127.0.0.1:8080
```

To accept connections from other machines, read [Security](#security) first.

2. Load an ONNX model and predict:

```python
import grpc, numpy as np
from nxmndr.inference import InferenceSession, RemoteInferenceProvider
from nxmndr.models import OnnxModelSpec

channel = grpc.insecure_channel('localhost:50051')
provider = RemoteInferenceProvider(channel)
spec = OnnxModelSpec(model_path='tst/example_model/example_model.onnx', name='remote-test')
provider.load_spec(spec)
session = InferenceSession(spec, provider)
out = session.run(np.zeros((1,3,32,32), dtype=np.float32))
print(out.shape)
```

## Security

The gRPC server and the OpenAI / Azure OpenAI proxy have no authentication. You must trust every client that can connect to them:

- A client that loads a model can make the server run code that the client supplies. `LoadModel` takes uploaded weight bytes (`artifact`) or a path on the server (`source`). With `model_class = "ultralytics_yolo"`, `ultralytics.YOLO` opens a `.pt` checkpoint: it unpickles the file with `torch.load(weights_only=False)` and, by default, pip-installs a missing package that the checkpoint names (ultralytics 8.4.142, the tested version). Hugging Face models load with `trust_remote_code=True`, and TorchHub models run code from their repository (`torch.hub.load`).
- The proxy uses the operator's credentials (`OPENAI_API_KEY` or the Azure identity) for every caller.

For this reason, the servers listen on `127.0.0.1` (this machine only) by default. Earlier versions listened on every interface ([CHANGELOG.md](CHANGELOG.md)). To use a server from another machine, keep the default and open an SSH tunnel:

```bash
ssh -N -L 50051:127.0.0.1:50051 user@server-host   # then connect to 127.0.0.1:50051
```

Loopback and the tunnel keep out other hosts, not other accounts on the server host: every local user of that host can connect to a loopback port. The same holds on the client machine: `ssh -L` opens the forwarded port on its loopback, so every local user of the client machine can use the tunnel while it is open. On a host shared with other users, run the server or the tunnel only if you trust every account on it; otherwise run it on a host where only you and users you trust can log in.

To listen on the network, give the address explicitly. Do this only on a network where you trust every host that can reach the port:

```bash
nxmndr-server --host 0.0.0.0   # gRPC, and with --azure-proxy the HTTP proxy, on every interface
```

`python -m nxmndr server`, `python -m nxmndr grpc` and `python -m nxmndr.server.azure_openai_proxy` take the same `--host` option. With `--host ::` (or `[::]`) the gRPC server listens on IPv4 and IPv6, but the HTTP proxy listens on IPv6 only, so IPv4 clients reach the gRPC server and not the proxy. The gRPC server uses plaintext; for TLS, put a TLS endpoint in front of it.

## Private PyTorch Worker

A local worker process is spawned server-side for generic PyTorch specs. It communicates only over an anonymous pipe inherited from the server. Calls and responses contain JSON and typed tensor bytes, not pickled Python objects. No TCPStore, Gloo, TensorPipe, readiness socket, or network RPC listener is started, regardless of `MASTER_ADDR` or `MASTER_PORT`.

The spawned worker builds every PyTorch model on the server's first GPU (the first entry of its device plan; CPU without CUDA) and applies the server's cuDNN policy (`NXMNDR_CUDNN`). One inference call in the worker (a unary `Predict` or one stream tile) may take up to 60 s; a slower call fails that request or tile. A model load in the worker may take up to 300 s; a load that fails or times out in the server is dropped in the worker when its build ends.

Custom model classes must be importable and registered with `register_pytorch_model` before the worker starts. The trusted parent passes the class allowlist at process creation; load messages carry names only and cannot import arbitrary modules. A class supplied directly in the first `PytorchModelSpec` is also included. Dense CPU tensor values (including nested dictionaries, lists, and tuples) cross the channel; serialized messages are limited to 512 MiB.

### Migrating External Torch RPC

`NXMNDR_RPC_EXTERNAL_WORKER=1`, `run_worker(...)`, and `python -m nxmndr.server.rpc_worker` now fail closed. There is no unsafe opt-in: Torch RPC deserializes Python objects before application validation, and loopback rendezvous alone does not secure its other listeners.

Run `nxmndr-server` on the compute host instead, and use `RemoteInferenceProvider` or the canonical gRPC client. Keep gRPC on a trusted network or behind an authenticated TLS endpoint; this change does not add authentication to the public gRPC API.

`TorchRpcInferenceProvider` retains its name for compatibility but now starts its own local worker. Its legacy address and port arguments no longer select a network peer; non-loopback addresses are refused. `ping()` checks worker readiness, not whether a model has been loaded. Use `load_spec()` or `InferenceSession` before prediction. Closing a provider stops its worker, and another provider can subsequently start in the same process.

### Minimal Example (Conceptual)

```python
from nxmndr.inference import InferenceSession, TorchRpcInferenceProvider
from nxmndr.models import PytorchModelSpec
from my_models import MyNet
import numpy as np

if __name__ == "__main__":  # required by multiprocessing's spawn start method
    spec = PytorchModelSpec(model_class=MyNet, model_path="model.pth", name="net")
    provider = TorchRpcInferenceProvider()
    try:
        session = InferenceSession(spec, provider)
        out = session.run(np.zeros((1, 3, 32, 32), dtype=np.float32))
    finally:
        provider.close()
```

## Protobuf API (High-Level)

Service: `InferenceService`

Endpoints:
- `LoadModel(LoadModelRequest) -> LoadModelResponse` : Uploads or references a model artifact; returns `model_id`. Equal specs share one loaded model and one `model_id` (`effective_metadata` has `model_cache_hit=true`); a local file or directory `source` is loaded again after its size or modification time on the server host changes (for a worker on another host, see External Worker Mode). `overwrite=true` loads a fresh copy only when no session, call or earlier `LoadModel` caller holds the resident one; otherwise it fails with `FAILED_PRECONDITION`. A `LoadModel` whose client is gone when the load finishes (its deadline passed or it cancelled) keeps no reference and frees nothing: the model stays cached until it is evicted, as one that only sessions loaded does. One that its client gives up on after that moment keeps its reference.
- `ListModels(ListModelsRequest) -> ListModelsResponse` : Enumerate loaded models.
- `Predict(PredictRequest) -> PredictResponse` : Run a batch prediction (includes `model_id`).
- `UnloadModel(UnloadModelRequest) -> UnloadModelResponse` : Releases one `LoadModel` reference, also while a session or call uses the model. When no reference is left, the server frees the model at once if no session or call uses it, and otherwise as soon as the last session or call that uses it ends, unless a `LoadModel` takes a new reference first. The server keeps at most `NXMNDR_MODEL_CACHE_CAPACITY` models (default 10): a new model at full capacity evicts the least recently used one that no session or call uses, even if its `LoadModel` callers have not unloaded it. `Predict` with an evicted `model_id` fails with `NOT_FOUND`; `LoadModel` loads it again.
- `Capabilities` / `Health` : Introspection & readiness.
- `StreamPredict` : Bidirectional streaming for tiled/chunked inference with session support.

Changes since the previous version that clients and operators must act on: [CHANGELOG.md](CHANGELOG.md).

### Model Formats
Supported enums: `PYTORCH`, `ONNX`, `HUGGINGFACE`, `TORCHHUB`.

## Testing

Tests live in `tst/`:

```bash
pytest -q
```

Key tests:
- `tst/unit/test_local.py` local PyTorch
- `tst/integration/test_remote_onnx.py` remote ONNX load/predict/unload
- `tst/integration/test_server.py` lifecycle & capabilities
- `tst/integration/test_rpc_pytorch.py` private PyTorch worker smoke test
- `tst/integration/test_stream_predict.py` streaming prediction

`tst/integration/test_model_catalog_service.py` loads and runs one catalog model per output family by default. `NXMNDR_TEST_CATALOG_SWEEP=1` runs every served catalog model; that run is slow and needs several GB of memory for the largest models (`torchvision:vit_h_14` has about 2.5 GB of weights).

## Roadmap (Excerpt)
- Structured error envelopes (use `ErrorStatus`).
- Model warm-up and metrics.
- Artifact checksum validation & allowlist for safe `model_class` usage.
- Multi-GPU load balancing improvements.

## See also
- https://github.com/0xk1h0/ONNX_gRPC
- https://github.com/ikeboo/ezonnx
