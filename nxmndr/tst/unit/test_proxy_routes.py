# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""nxmndr.server.proxy_routes: route constants, identity and upstream shapes.

Standard library only, so a client imports it cheaply."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nxmndr.gpt.models import ModelEndpointSpec
from nxmndr.server import proxy_routes as routes
from nxmndr.exceptions import ValidationError

SRC = Path(__file__).resolve().parents[2] / "src"


def test_direct_openai_shape_for_any_host_when_the_identity_is_explicit():
    req = routes.upstream_request("http://127.0.0.1:8765/", "gpt-image-1", routes.OPERATION_IMAGE_EDITS,
                                  provider="openai", api_version="2025-04-01-preview")
    assert req.url == "http://127.0.0.1:8765/v1/images/edits"
    assert req.params == {} and req.fields == {"model": "gpt-image-1"}


def test_direct_azure_shape_for_any_host_when_the_identity_is_explicit():
    req = routes.upstream_request("https://api.openai.com", "edits", routes.OPERATION_CHAT_COMPLETIONS,
                                  provider="azure_openai", api_version="2025-04-01-preview")
    assert req.url == "https://api.openai.com/openai/deployments/edits/chat/completions"
    assert req.params == {"api-version": "2025-04-01-preview"} and req.fields == {}


def test_host_detection_stays_the_default():
    assert routes.resolve_provider("https://api.openai.com/") == "openai"
    assert routes.resolve_provider("https://API.OpenAI.com") == "openai"
    assert routes.resolve_provider("https://res.openai.azure.com") == "azure_openai"
    assert routes.resolve_provider("http://127.0.0.1:9") == "azure_openai"
    assert routes.upstream_request("https://api.openai.com", "m", "images/edits", provider="",
                                   api_version="v").url == "https://api.openai.com/v1/images/edits"
    with pytest.raises(ValueError, match="provider must be one of"):
        routes.resolve_provider("https://api.openai.com", "azure")


def test_proxy_client_urls():
    assert routes.proxy_url("http://127.0.0.1:8080/", routes.ROUTE_IMAGE_EDITS, deployment="gpt-image-1") == (
        "http://127.0.0.1:8080/openai/deployments/gpt-image-1/images/edits"
    )
    assert routes.proxy_url("http://proxy:8080/base", routes.ROUTE_MODELS) == "http://proxy:8080/base/models"
    # a deployment name is one path segment
    assert routes.proxy_url("http://p", routes.ROUTE_GENERATE_IMAGE, deployment="a/b c") == (
        "http://p/generate/image/a%2Fb%20c"
    )
    assert (routes.MODELS_TYPE_PARAM, routes.ENDPOINT_TYPE_VISION) == ("type", "vision")


def test_model_endpoint_spec_carries_an_explicit_identity():
    spec = ModelEndpointSpec("http://127.0.0.1:1", "gpt-image-1", "2025-04-01-preview", "vision", "openai")
    spec.validate()
    assert spec.to_dict()["provider"] == "openai"
    assert ModelEndpointSpec.from_json(json.dumps(spec.to_dict())) == spec
    assert ModelEndpointSpec.from_json(json.dumps({"endpoint_url": "https://x", "deployment_name": "d"})).provider == ""
    with pytest.raises(ValidationError):
        ModelEndpointSpec("https://x", "d", "v", "vision", "azure").validate()


def test_the_routes_module_imports_nothing_heavy():
    probe = (
        "import json, sys; from nxmndr.server import proxy_routes; "
        "print(json.dumps(sorted(m for m in ('aiohttp', 'openai', 'torch', 'numpy', 'nxmndr.server.server', "
        "'nxmndr.server.azure_openai_proxy', 'nxmndr.gpt') if m in sys.modules)))"
    )
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", probe], env=env,
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == []
