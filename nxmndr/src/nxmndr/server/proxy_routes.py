# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Routes and request shapes of the OpenAI / Azure OpenAI proxy (standard library only).

``nxmndr.server.azure_openai_proxy`` registers its client-facing routes and builds
its upstream requests from these values. Clients of the proxy, such as the QGIS
plugin's worker handler for Quick Labels image edits, import them instead of
repeating the route strings: ``from nxmndr.server import proxy_routes`` does not
load the server, aiohttp or the ML runtime.

Provider identity is explicit when given (``openai`` or ``azure_openai``, the same
vocabulary as the plugin's ``ProviderConfig.cloud_provider``) and independent of
the endpoint's host, so any URL, including a loopback stub, can serve either
identity. Only when no identity is given is it detected from the host:
``api.openai.com`` is the OpenAI platform, every other host is Azure OpenAI.

Upstream request shapes:

- ``openai``: ``POST <endpoint>/v1/<operation>``, no ``api-version``, the deployment
  name sent as the ``model`` field, ``Authorization: Bearer <OpenAI API key>``.
- ``azure_openai``: ``POST <endpoint>/openai/deployments/<deployment>/<operation>
  ?api-version=<version>``, ``Authorization: Bearer <Azure token>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import quote, urlparse

PROVIDER_OPENAI = "openai"
PROVIDER_AZURE_OPENAI = "azure_openai"
PROVIDERS = (PROVIDER_OPENAI, PROVIDER_AZURE_OPENAI)

# Hosts of the OpenAI platform, consulted only when no identity is given.
OPENAI_PLATFORM_HOSTS = frozenset({"api.openai.com"})

# Operations: the upstream path suffix of each forwarded request.
OPERATION_CHAT_COMPLETIONS = "chat/completions"
OPERATION_IMAGE_EDITS = "images/edits"

# Client-facing routes of the proxy (aiohttp patterns, relative to its base URL).
ROUTE_CHAT_COMPLETIONS = "/openai/deployments/{deployment}/chat/completions"
ROUTE_IMAGE_EDITS = "/openai/deployments/{deployment}/images/edits"
ROUTE_GENERATE_IMAGE = "/generate/image/{deployment}"
ROUTE_MODELS = "/models"
ROUTE_HEALTH = "/health"

# GET ROUTE_MODELS?type=<ENDPOINT_TYPE_CHAT|ENDPOINT_TYPE_VISION> filters the listing.
MODELS_TYPE_PARAM = "type"
ENDPOINT_TYPE_CHAT = "chat"
ENDPOINT_TYPE_VISION = "vision"

API_VERSION_PARAM = "api-version"
MODEL_FIELD = "model"


def proxy_url(base_url: str, route: str, **route_params: str) -> str:
    """Absolute client URL of a proxy route, for example
    ``proxy_url("http://127.0.0.1:8080/", ROUTE_IMAGE_EDITS, deployment="gpt-image-1")``.
    Route parameters are percent-encoded as one path segment each."""

    path = route.format(**{k: quote(str(v), safe="") for k, v in route_params.items()})
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def detect_provider(endpoint_url: str) -> str:
    """The identity implied by the endpoint's host (the default when none is given)."""

    host = (urlparse(endpoint_url).hostname or "").lower()
    return PROVIDER_OPENAI if host in OPENAI_PLATFORM_HOSTS else PROVIDER_AZURE_OPENAI


def resolve_provider(endpoint_url: str, provider: str = "") -> str:
    """``provider`` when given (validated), else :func:`detect_provider`."""

    if provider:
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS} or empty, got {provider!r}")
        return provider
    return detect_provider(endpoint_url)


@dataclass(frozen=True)
class UpstreamRequest:
    """Where and how the proxy forwards one operation."""

    url: str
    params: Dict[str, str] = field(default_factory=dict)  # query parameters
    fields: Dict[str, str] = field(default_factory=dict)  # added to the JSON body or form


def upstream_request(
    endpoint_url: str, deployment: str, operation: str, *, provider: str, api_version: str
) -> UpstreamRequest:
    """The upstream request shape for ``provider`` (module docstring)."""

    base = endpoint_url.rstrip("/")
    if resolve_provider(endpoint_url, provider) == PROVIDER_OPENAI:
        return UpstreamRequest(url=f"{base}/v1/{operation}", fields={MODEL_FIELD: deployment})
    return UpstreamRequest(
        url=f"{base}/openai/deployments/{deployment}/{operation}",
        params={API_VERSION_PARAM: api_version},
    )


__all__ = [
    "PROVIDER_OPENAI",
    "PROVIDER_AZURE_OPENAI",
    "PROVIDERS",
    "OPENAI_PLATFORM_HOSTS",
    "OPERATION_CHAT_COMPLETIONS",
    "OPERATION_IMAGE_EDITS",
    "ROUTE_CHAT_COMPLETIONS",
    "ROUTE_IMAGE_EDITS",
    "ROUTE_GENERATE_IMAGE",
    "ROUTE_MODELS",
    "ROUTE_HEALTH",
    "MODELS_TYPE_PARAM",
    "ENDPOINT_TYPE_CHAT",
    "ENDPOINT_TYPE_VISION",
    "API_VERSION_PARAM",
    "MODEL_FIELD",
    "UpstreamRequest",
    "detect_provider",
    "proxy_url",
    "resolve_provider",
    "upstream_request",
]
