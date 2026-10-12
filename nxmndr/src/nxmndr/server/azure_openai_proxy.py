# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Azure OpenAI API Proxy Server with Dynamic Endpoint Discovery

This lightweight forwarding server acts as a proxy for Azure OpenAI API endpoints,
automatically discovering and routing requests to multiple configured endpoints
based on deployment names and model types. It handles authentication using the
server's DefaultAzureCredential and provides intelligent fallback routing.

Key Features:
- Dynamic endpoint discovery from environment variables
- Separate routing for chat and vision models
- Intelligent fallback to default endpoints
- Centralized authentication with Azure credentials
- Support for multiple endpoints of each type

Supported API Endpoints:
- POST /openai/deployments/{deployment}/chat/completions - Chat completions (routes to chat endpoints)
- POST /openai/deployments/{deployment}/images/edits - Image editing (routes to vision endpoints)
- POST /generate/image/{deployment} - Unified image generation (delegates to chat completions for chat models, image edits for vision models)
- GET /models - List all registered ModelEndpointSpec objects (returns all by default, supports ?type=chat|vision filter)
- GET /health - Health check with endpoint configuration summary

Environment Variable Pattern (<PREFIX>_DEPLOYMENT_NAME):
The server automatically discovers endpoints by scanning environment variables ending in '_DEPLOYMENT_NAME'.
For each prefix found, it attempts to build a complete ModelEndpointSpec configuration.

Required per endpoint:
- <PREFIX>_DEPLOYMENT_NAME: Azure OpenAI deployment name (e.g., CHAT_DEPLOYMENT_NAME=gpt-4o)
- <PREFIX>_ENDPOINT_URL: Azure OpenAI endpoint URL (fallback: ENDPOINT_URL)

Optional per endpoint:
- <PREFIX>_API_VERSION: API version (fallback: API_VERSION, default: 2025-01-01-preview)
- <PREFIX>_TYPE: Model type "chat" or "vision" (default: "chat")
- <PREFIX>_PROVIDER: "openai" or "azure_openai" (default: detected from the host)

Global Fallbacks:
- ENDPOINT_URL: Default endpoint URL when specific endpoint URLs not configured
- API_VERSION: Default API version when specific versions not configured

OpenAI Platform (non-Azure) endpoints:
An endpoint is served as the OpenAI platform when ``<PREFIX>_PROVIDER=openai``, or,
when no provider is set, when its URL host is ``api.openai.com``
(``<PREFIX>_PROVIDER=azure_openai`` forces Azure OpenAI for any host). For such
endpoints the proxy authenticates with the
``OPENAI_API_KEY`` environment variable (no Azure credential is needed), forwards to
``/v1/chat/completions`` or ``/v1/images/edits`` instead of the
``/openai/deployments/{deployment}/...`` paths, omits ``api-version``, and sends the
deployment name as the ``model`` field. The client-facing routes are unchanged, so
the QGIS plugin needs no reconfiguration beyond pointing at this proxy. Routes,
identities and upstream shapes are defined once in ``nxmndr.server.proxy_routes``.

  export OPENAI_API_KEY="sk-..."
  export VISION_DEPLOYMENT_NAME="gpt-image-1"
  export VISION_ENDPOINT_URL="https://api.openai.com"
  export VISION_TYPE="vision"

Routing Logic:
1. Extract deployment name from request URL path
2. Find matching endpoint by deployment name and required type (chat/vision)
3. If no match found, use default endpoint for the type but log the fallback
4. Forward request to the selected endpoint with proper authentication

Configuration Examples:

Single Endpoint Per Type:
  export CHAT_DEPLOYMENT_NAME="gpt-4o"
  export VISION_DEPLOYMENT_NAME="gpt-4o-vision"
  export ENDPOINT_URL="https://myopenai.openai.azure.com"
  export CHAT_TYPE="chat"
  export VISION_TYPE="vision"

Multiple Endpoints:
  export CHAT1_DEPLOYMENT_NAME="gpt-4o"
  export CHAT1_ENDPOINT_URL="https://chat1.openai.azure.com"
  export CHAT1_TYPE="chat"

  export CHAT2_DEPLOYMENT_NAME="gpt-4o-turbo"
  export CHAT2_ENDPOINT_URL="https://chat2.openai.azure.com"
  export CHAT2_TYPE="chat"

  export VISION1_DEPLOYMENT_NAME="gpt-4o-vision"
  export VISION1_ENDPOINT_URL="https://vision.openai.azure.com"
  export VISION1_TYPE="vision"
"""

import asyncio
import logging
import os
import time
from typing import List, Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout, web
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from nxmndr.gpt.models import ModelEndpointSpec

from . import proxy_routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Hosts that are served by the OpenAI platform rather than Azure OpenAI (used only
# when an endpoint has no explicit provider).
OPENAI_PLATFORM_HOSTS = proxy_routes.OPENAI_PLATFORM_HOSTS


def is_openai_platform(endpoint_spec: ModelEndpointSpec) -> bool:
    """Return True if the endpoint targets the OpenAI platform instead of Azure OpenAI."""
    provider = proxy_routes.resolve_provider(endpoint_spec.endpoint_url, endpoint_spec.provider)
    return provider == proxy_routes.PROVIDER_OPENAI


class AzureOpenAIProxy:
    """Proxy server for Azure OpenAI API calls with credential forwarding."""

    def __init__(self):
        """Initialize the Azure OpenAI proxy by discovering endpoints from environment variables."""
        # Discover all ModelEndpointSpec instances from environment variables
        discovered_endpoints = self._discover_model_endpoints()

        # Store endpoints as dictionaries with deployment names as keys
        self.model_endpoints = {ep.deployment_name: ep for ep in discovered_endpoints}
        self.chat_endpoints = {
            ep.deployment_name: ep for ep in discovered_endpoints if ep.type == "chat"
        }
        self.vision_endpoints = {
            ep.deployment_name: ep for ep in discovered_endpoints if ep.type == "vision"
        }

        # Set default endpoints (prioritize GPT models)
        self.default_chat_endpoint = self._select_preferred_endpoint(
            list(self.chat_endpoints.values())
        )
        self.default_vision_endpoint = self._select_preferred_endpoint(
            list(self.vision_endpoints.values())
        )

        # Warn (rather than fail) when a type has no endpoint; requests for that type return 503.
        # Discovery already guarantees at least one endpoint overall.
        if not self.default_chat_endpoint:
            logger.warning(
                "No chat endpoints configured; chat completions requests will be rejected. "
                "Set <PREFIX>_DEPLOYMENT_NAME with <PREFIX>_TYPE=chat to enable them."
            )
        if not self.default_vision_endpoint:
            logger.warning(
                "No vision endpoints configured; image edits requests will be rejected. "
                "Set <PREFIX>_DEPLOYMENT_NAME with <PREFIX>_TYPE=vision to enable them."
            )

        # Azure credential provider is created lazily on first Azure request, so a proxy
        # that only fronts OpenAI-platform endpoints can start without Azure credentials.
        self._azure_token_provider = None

        # HTTP client session
        self.session: Optional[ClientSession] = None

        # Log discovered endpoints
        logger.info(f"Initialized Azure OpenAI proxy with {len(self.model_endpoints)} endpoints:")
        for deployment_name, endpoint in self.model_endpoints.items():
            provider = "openai-platform" if is_openai_platform(endpoint) else "azure"
            logger.info(f"  {deployment_name}: {endpoint} [{provider}]")
        logger.info(
            f"Default chat endpoint: "
            f"{self.default_chat_endpoint.deployment_name if self.default_chat_endpoint else 'none'}"
        )
        logger.info(
            f"Default vision endpoint: "
            f"{self.default_vision_endpoint.deployment_name if self.default_vision_endpoint else 'none'}"
        )

    def _discover_model_endpoints(self) -> List[ModelEndpointSpec]:
        """Discover ModelEndpointSpec instances from environment variables."""
        endpoints = []
        deployment_keys = {}

        # Find all environment variables ending with _DEPLOYMENT_NAME
        for key, value in os.environ.items():
            if key.endswith("_DEPLOYMENT_NAME"):
                prefix = key[: -len("_DEPLOYMENT_NAME")]
                deployment_keys[prefix] = value

        # For each prefix, try to build a complete ModelEndpointSpec
        for prefix, deployment_name in sorted(
            deployment_keys.items()
        ):  # Sort for consistent ordering
            try:
                # Look for corresponding endpoint URL and other settings
                endpoint_url_key = f"{prefix}_ENDPOINT_URL"
                api_version_key = f"{prefix}_API_VERSION"
                type_key = f"{prefix}_TYPE"
                provider_key = f"{prefix}_PROVIDER"

                endpoint_url = os.getenv(endpoint_url_key)
                if not endpoint_url:
                    # Try fallback patterns
                    endpoint_url = os.getenv("ENDPOINT_URL")

                if not endpoint_url:
                    logger.warning(
                        f"Skipping {prefix}: No endpoint URL found (looked for {endpoint_url_key} or ENDPOINT_URL)"
                    )
                    continue

                api_version = (
                    os.getenv(api_version_key) or os.getenv("API_VERSION") or "2025-01-01-preview"
                )
                endpoint_type = os.getenv(type_key, "chat").lower()

                # Validate endpoint type
                if endpoint_type not in ["chat", "vision"]:
                    logger.warning(
                        f"Skipping {prefix}: Invalid type '{endpoint_type}', must be 'chat' or 'vision'"
                    )
                    continue

                # Create ModelEndpointSpec
                endpoint_spec = ModelEndpointSpec(
                    endpoint_url=endpoint_url.rstrip("/"),
                    deployment_name=deployment_name,
                    api_version=api_version,
                    type=endpoint_type,
                    provider=os.getenv(provider_key, "").strip().lower(),
                )

                # Validate the spec
                endpoint_spec.validate()
                endpoints.append(endpoint_spec)
                logger.debug(f"Discovered endpoint: {endpoint_spec}")

            except Exception as e:
                logger.error(f"Error creating endpoint spec for {prefix}: {e}")
                continue

        if not endpoints:
            raise ValueError(
                "No valid model endpoints discovered. Please set environment variables with pattern: <PREFIX>_DEPLOYMENT_NAME"
            )

        return endpoints

    def _select_preferred_endpoint(
        self, endpoints: List[ModelEndpointSpec]
    ) -> Optional[ModelEndpointSpec]:
        """Select the preferred endpoint from a list, prioritizing GPT models."""
        if not endpoints:
            return None

        # First, try to find any endpoint with deployment name starting with "gpt"
        gpt_endpoints = [ep for ep in endpoints if ep.deployment_name.lower().startswith("gpt")]
        if gpt_endpoints:
            logger.debug(
                f"Found {len(gpt_endpoints)} GPT model(s), selecting: {gpt_endpoints[0].deployment_name}"
            )
            return gpt_endpoints[0]

        # Fall back to the first endpoint if no GPT models found
        logger.debug(
            f"No GPT models found, selecting first available: {endpoints[0].deployment_name}"
        )
        return endpoints[0]

    def get_endpoint_by_deployment(
        self, deployment_name: str, endpoint_type: str
    ) -> Optional[ModelEndpointSpec]:
        """
        Find an endpoint by deployment name and type.

        Args:
            deployment_name: The deployment name to search for
            endpoint_type: The endpoint type ("chat" or "vision")

        Returns:
            ModelEndpointSpec if found, None otherwise
        """
        # Select the appropriate endpoint dictionary by type
        if endpoint_type == "chat":
            endpoints_dict = self.chat_endpoints
        elif endpoint_type == "vision":
            endpoints_dict = self.vision_endpoints
        else:
            endpoints_dict = self.model_endpoints

        # Direct dictionary lookup
        endpoint = endpoints_dict.get(deployment_name)
        if endpoint:
            logger.debug(
                f"Found endpoint for deployment '{deployment_name}' of type '{endpoint_type}': {endpoint}"
            )
            return endpoint

        logger.debug(
            f"No endpoint found for deployment '{deployment_name}' of type '{endpoint_type}'"
        )
        return None

    async def start_session(self):
        """Start the HTTP client session."""
        if self.session is None:
            timeout = ClientTimeout(total=300)  # 5 minute timeout for long operations
            self.session = ClientSession(timeout=timeout)
            logger.info("HTTP client session started")

    async def close_session(self):
        """Close the HTTP client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("HTTP client session closed")

    @property
    def token_provider(self):
        """Azure bearer-token provider, created on first use."""
        if self._azure_token_provider is None:
            self._azure_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        return self._azure_token_provider

    def _get_auth_headers(self, endpoint_spec: ModelEndpointSpec) -> dict:
        """Get authentication headers for the given endpoint (API key for OpenAI, Azure token otherwise)."""
        if is_openai_platform(endpoint_spec):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"Endpoint {endpoint_spec.deployment_name} targets the OpenAI platform but "
                    "OPENAI_API_KEY is not set"
                )
            token = api_key
        else:
            token = self.token_provider()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _forward_request(
        self,
        endpoint_spec: ModelEndpointSpec,
        method: str,
        operation: str,
        request_data: dict = None,
        files: dict = None,
        params: dict = None,
    ) -> tuple[int, dict]:
        """
        Forward request to Azure OpenAI endpoint.

        Args:
            endpoint_spec: The ModelEndpointSpec to use for the request
            method: HTTP method (GET, POST, etc.)
            operation: proxy_routes.OPERATION_CHAT_COMPLETIONS or OPERATION_IMAGE_EDITS
            request_data: JSON request body
            files: File uploads for multipart requests
            params: Query parameters

        Returns:
            Tuple of (status_code, response_data)
        """
        await self.start_session()

        # Upstream URL, query and body fields for the endpoint's provider identity:
        # Azure OpenAI requires api-version (the OpenAI platform rejects it); the
        # OpenAI platform selects the model per request instead of per deployment URL.
        upstream = proxy_routes.upstream_request(
            endpoint_spec.endpoint_url,
            endpoint_spec.deployment_name,
            operation,
            provider=endpoint_spec.provider,
            api_version=endpoint_spec.api_version,
        )
        url = upstream.url

        if params is None:
            params = {}
        params.update(upstream.params)

        # Get authentication headers
        headers = self._get_auth_headers(endpoint_spec)

        if upstream.fields:
            if request_data is None:
                request_data = {}
            for key, value in upstream.fields.items():
                request_data[key] = value  # the routed endpoint decides, not the client

        try:
            logger.info(f"Forwarding {method} request to: {url}")

            # Handle different request types
            if files:
                # Multipart form request (for image uploads)
                # Remove Content-Type header for multipart - aiohttp will set it
                headers.pop("Content-Type", None)

                # Prepare form data
                data = aiohttp.FormData()
                if request_data:
                    for key, value in request_data.items():
                        data.add_field(key, str(value))

                for key, file_data in files.items():
                    data.add_field(
                        key,
                        file_data["content"],
                        filename=file_data.get("filename", "file"),
                        content_type=file_data.get("content_type", "application/octet-stream"),
                    )

                async with self.session.request(
                    method, url, data=data, headers=headers, params=params
                ) as response:
                    response_data = await response.json()
                    return response.status, response_data
            else:
                # JSON request
                async with self.session.request(
                    method, url, json=request_data, headers=headers, params=params
                ) as response:
                    response_data = await response.json()
                    return response.status, response_data

        except asyncio.CancelledError:
            logger.info("Request forwarding cancelled by client")
            raise  # Re-raise to propagate cancellation up the chain
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            return 500, {"error": {"message": f"Proxy error: {str(e)}", "type": "proxy_error"}}

    async def handle_chat_completions(self, request):
        """Handle chat completions API requests."""
        deployment = None
        try:
            # Extract deployment name from path
            deployment = request.match_info["deployment"]

            # Find the appropriate chat endpoint
            endpoint_spec = self.get_endpoint_by_deployment(deployment, "chat")
            if not endpoint_spec:
                # Fall back to default chat endpoint but use the requested deployment name
                endpoint_spec = self.default_chat_endpoint
                if not endpoint_spec:
                    return web.json_response(
                        {
                            "error": {
                                "message": "No chat endpoints are configured on this proxy",
                                "type": "configuration_error",
                            }
                        },
                        status=503,
                    )
                logger.info(
                    f"Deployment '{deployment}' not found, using default chat endpoint: {endpoint_spec.deployment_name}"
                )

            logger.info(f"Handling chat completions request for deployment: {deployment}")
            logger.info(f"Using endpoint: {endpoint_spec}")

            # Get request JSON data
            request_data = await request.json()
            logger.debug(f"Chat completions request data keys: {list(request_data.keys())}")

            # Log message count and model parameters
            if "messages" in request_data:
                logger.debug(f"Processing {len(request_data['messages'])} messages")
            if "max_tokens" in request_data:
                logger.debug(f"Max tokens: {request_data['max_tokens']}")
            if "temperature" in request_data:
                logger.debug(f"Temperature: {request_data['temperature']}")

            # Forward using the selected endpoint
            # Use the actual endpoint's deployment name, not the requested one
            status, response_data = await self._forward_request(
                endpoint_spec, "POST", proxy_routes.OPERATION_CHAT_COMPLETIONS, request_data
            )

            logger.info(f"Chat completions request completed with status: {status}")
            if status == 200 and "usage" in response_data:
                usage = response_data["usage"]
                logger.info(
                    f"Token usage - prompt: {usage.get('prompt_tokens', 'N/A')}, "
                    f"completion: {usage.get('completion_tokens', 'N/A')}, "
                    f"total: {usage.get('total_tokens', 'N/A')}"
                )

            return web.json_response(response_data, status=status)

        except asyncio.CancelledError:
            logger.info(
                f"Chat completions request cancelled by client for deployment '{deployment}'"
            )
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.error(f"Error handling chat completions for deployment '{deployment}': {e}")
            logger.exception("Full traceback:")
            return web.json_response(
                {"error": {"message": str(e), "type": "request_error"}}, status=500
            )

    async def handle_image_edits(self, request):
        """Handle image editing API requests."""
        deployment = None
        try:
            # Extract deployment name from path
            deployment = request.match_info["deployment"]

            # Find the appropriate vision endpoint
            endpoint_spec = self.get_endpoint_by_deployment(deployment, "vision")
            if not endpoint_spec:
                # Fall back to default vision endpoint but use the requested deployment name
                endpoint_spec = self.default_vision_endpoint
                if not endpoint_spec:
                    return web.json_response(
                        {
                            "error": {
                                "message": "No vision endpoints are configured on this proxy",
                                "type": "configuration_error",
                            }
                        },
                        status=503,
                    )
                logger.info(
                    f"Deployment '{deployment}' not found, using default vision endpoint: {endpoint_spec.deployment_name}"
                )

            logger.info(f"Handling image edits request for deployment: {deployment}")
            logger.info(f"Using endpoint: {endpoint_spec}")

            # Parse multipart form data
            reader = await request.multipart()

            form_data = {}
            files = {}
            image_size = 0

            async for field in reader:
                if field.name == "image":
                    # Handle image file
                    content = await field.read()
                    image_size = len(content)
                    files["image"] = {
                        "content": content,
                        "filename": getattr(field, "filename", "image.png"),
                        "content_type": field.headers.get("Content-Type", "image/png"),
                    }
                    logger.debug(
                        f"Received image file: {files['image']['filename']}, "
                        f"size: {image_size} bytes, type: {files['image']['content_type']}"
                    )
                else:
                    # Handle other form fields
                    value = await field.text()
                    form_data[field.name] = value
                    logger.debug(
                        f"Form field '{field.name}': {value[:100]}{'...' if len(value) > 100 else ''}"
                    )

            logger.info(
                f"Processing image edit with {len(form_data)} form fields and image size {image_size} bytes"
            )

            # Forward using the selected endpoint
            # Use the actual endpoint's deployment name, not the requested one
            status, response_data = await self._forward_request(
                endpoint_spec, "POST", proxy_routes.OPERATION_IMAGE_EDITS, form_data, files
            )

            logger.info(f"Image edits request completed with status: {status}")
            if status == 200 and "data" in response_data:
                logger.info(f"Successfully generated {len(response_data['data'])} image(s)")

            return web.json_response(response_data, status=status)

        except asyncio.CancelledError:
            logger.info(f"Image edits request cancelled by client for deployment '{deployment}'")
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.error(f"Error handling image edits for deployment '{deployment}': {e}")
            logger.exception("Full traceback:")
            return web.json_response(
                {"error": {"message": str(e), "type": "request_error"}}, status=500
            )

    async def handle_image_generation(self, request):
        """Unified image generation endpoint that delegates to existing handlers based on model type."""
        deployment = None
        try:
            # Extract deployment name from path
            deployment = request.match_info["deployment"]

            logger.info(f"Handling unified image generation request for deployment: {deployment}")

            # Find the endpoint (try both chat and vision types)
            endpoint_spec = self.get_endpoint_by_deployment(
                deployment, "vision"
            ) or self.get_endpoint_by_deployment(deployment, "chat")

            if not endpoint_spec:
                # Fall back to default vision endpoint, then default chat endpoint
                endpoint_spec = self.default_vision_endpoint or self.default_chat_endpoint
                if not endpoint_spec:
                    return web.json_response(
                        {
                            "error": {
                                "message": "No endpoints are configured on this proxy",
                                "type": "configuration_error",
                            }
                        },
                        status=503,
                    )
                logger.info(
                    f"Deployment '{deployment}' not found, using default endpoint: {endpoint_spec.deployment_name}"
                )

            logger.info(f"Using endpoint: {endpoint_spec} (type: {endpoint_spec.type})")

            # Route based on endpoint type - delegate to existing handlers
            if endpoint_spec.type == "vision":
                # Delegate to existing image edits handler
                logger.info(f"Delegating to image edits handler for vision model: {deployment}")
                return await self.handle_image_edits(request)
            else:
                # Delegate to existing chat completions handler
                logger.info(f"Delegating to chat completions handler for chat model: {deployment}")
                return await self.handle_chat_completions(request)

        except asyncio.CancelledError:
            logger.info(
                f"Image generation request cancelled by client for deployment '{deployment}'"
            )
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.error(f"Error handling image generation for deployment '{deployment}': {e}")
            logger.exception("Full traceback:")
            return web.json_response(
                {"error": {"message": str(e), "type": "request_error"}}, status=500
            )

    async def handle_list_models(self, request):
        """Handle list registered models API requests."""
        try:
            # Get optional type filter from query parameters
            model_type = request.query.get(proxy_routes.MODELS_TYPE_PARAM)  # 'chat', 'vision', or None for all

            logger.info(f"Handling list models request (type filter: {model_type or 'all'})")

            # Always return all models by default, but allow filtering
            if model_type == "chat":
                filtered_models = list(self.chat_endpoints.values())
                default_model = self.default_chat_endpoint
            elif model_type == "vision":
                filtered_models = list(self.vision_endpoints.values())
                default_model = self.default_vision_endpoint
            else:
                # Return all models by default
                filtered_models = list(self.model_endpoints.values())
                # Put both defaults first
                chat_default = self.default_chat_endpoint
                vision_default = self.default_vision_endpoint
                defaults = [m for m in [chat_default, vision_default] if m and m in filtered_models]
                non_defaults = [m for m in filtered_models if m not in defaults]
                sorted_models = defaults + non_defaults

                # Convert to response format and return early
                response_data = {
                    "data": [model.to_dict() for model in sorted_models],
                    "count": len(sorted_models),
                    "filter": model_type or "all",
                }

                logger.info(f"Returning {len(sorted_models)} registered models (all types)")
                return web.json_response(response_data, status=200)

            # Sort models to put default first (for filtered requests)
            if model_type and default_model and default_model in filtered_models:
                # Move default to front
                sorted_models = [default_model] + [m for m in filtered_models if m != default_model]
            else:
                sorted_models = filtered_models

            # Convert to response format
            response_data = {
                "data": [model.to_dict() for model in sorted_models],
                "count": len(sorted_models),
                "filter": model_type or "all",
            }

            logger.info(
                f"Returning {len(sorted_models)} registered models (default first: {default_model.deployment_name if default_model and default_model in filtered_models else 'none'})"
            )
            return web.json_response(response_data, status=200)

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            logger.exception("Full traceback:")
            return web.json_response(
                {"error": {"message": str(e), "type": "request_error"}}, status=500
            )

    async def handle_health(self, request):
        """Health check endpoint."""
        logger.debug("Health check requested")
        client_ip = request.remote
        user_agent = request.headers.get("User-Agent", "Unknown")
        logger.debug(f"Health check from {client_ip}, User-Agent: {user_agent}")

        health_data = {
            "status": "healthy",
            "total_endpoints": len(self.model_endpoints),
            "chat_endpoints": len(self.chat_endpoints),
            "vision_endpoints": len(self.vision_endpoints),
            "default_chat_endpoint": self.default_chat_endpoint.to_dict()
            if self.default_chat_endpoint
            else None,
            "default_vision_endpoint": self.default_vision_endpoint.to_dict()
            if self.default_vision_endpoint
            else None,
            "timestamp": int(time.time()),
        }

        logger.debug("Health check completed successfully")
        return web.json_response(health_data)


def setup_azure_proxy_routes(app: web.Application, proxy: AzureOpenAIProxy):
    """
    Setup Azure OpenAI proxy routes on an existing aiohttp application.

    This method can be reused by other servers to add Azure proxy functionality.

    Args:
        app: aiohttp web application to add routes to
        proxy: AzureOpenAIProxy instance with configured endpoints
    """
    # Store proxy instance for cleanup
    app["azure_proxy"] = proxy

    # Azure OpenAI API compatible routes
    app.router.add_post(proxy_routes.ROUTE_CHAT_COMPLETIONS, proxy.handle_chat_completions)
    app.router.add_post(proxy_routes.ROUTE_IMAGE_EDITS, proxy.handle_image_edits)

    # Unified client-facing routes
    app.router.add_post(proxy_routes.ROUTE_GENERATE_IMAGE, proxy.handle_image_generation)

    # Proxy management routes
    app.router.add_get(proxy_routes.ROUTE_MODELS, proxy.handle_list_models)  # List registered models
    app.router.add_get(proxy_routes.ROUTE_HEALTH, proxy.handle_health)

    # Add cleanup handler
    async def cleanup_azure_proxy(app):
        if "azure_proxy" in app:
            await app["azure_proxy"].close_session()

    app.on_cleanup.append(cleanup_azure_proxy)


def create_app() -> web.Application:
    """
    Create the proxy web application.

    The application automatically discovers endpoints from environment variables.

    Returns:
        Configured aiohttp web application
    """
    proxy = AzureOpenAIProxy()
    app = web.Application()

    # Setup Azure proxy routes using the reusable method
    setup_azure_proxy_routes(app, proxy)

    return app


def http_bind_host(host: str) -> str:
    """``host`` as aiohttp binds it: without the brackets of an IPv6 address."""
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


async def run_server(host: str = "127.0.0.1", port: int = 8080):
    """
    Run the Azure OpenAI proxy server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    # Create and run app
    app = create_app()

    logger.info(f"Starting Azure OpenAI proxy server on {host}:{port}")

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, http_bind_host(host), port)
    await site.start()

    logger.info("Server started successfully")

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await runner.cleanup()


def main():
    """CLI entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Azure OpenAI API Proxy Server - Automatically discovers endpoints from environment variables"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). The proxy has no authentication and "
        "spends its own credentials: expose it only on a trusted network.",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")

    args = parser.parse_args()

    try:
        asyncio.run(run_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
