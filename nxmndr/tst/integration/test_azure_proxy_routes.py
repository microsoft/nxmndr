# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The OpenAI / Azure OpenAI proxy over loopback HTTP.

A client calls the proxy's client-facing image-edit route, built with
``proxy_routes.proxy_url`` as the plugin's worker handler will, with no credential.
The proxy forwards to a loopback upstream stub that records each request, for an
endpoint with explicit identity ``openai`` (direct OpenAI shape), one with explicit
identity ``azure_openai`` (direct Azure shape) and one with none (host detection).
No external network, no real credential.
"""

from __future__ import annotations

import asyncio
import os

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from nxmndr.server import proxy_routes as routes
from nxmndr.server.azure_openai_proxy import AzureOpenAIProxy, setup_azure_proxy_routes

pytestmark = pytest.mark.integration

OPENAI_TEST_KEY = "test-openai-key-for-loopback"
AZURE_TEST_TOKEN = "test-azure-token-for-loopback"
PNG = b"\x89PNG\r\n\x1a\nnot-really-a-png"


@pytest.fixture(autouse=True)
def _proxy_env(monkeypatch):
    for key in list(os.environ):
        if key.endswith(("_DEPLOYMENT_NAME", "_ENDPOINT_URL", "_API_VERSION", "_TYPE", "_PROVIDER")) or key in (
            "ENDPOINT_URL", "API_VERSION", "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_TEST_KEY)


def _configure(monkeypatch, upstream_url):
    endpoints = {
        "OAI": ("gpt-image-1", "openai", None),
        "AZ": ("edits-az", "azure_openai", "2025-04-01-preview"),
        "DEF": ("edits-default", None, None),  # no identity: detected from the host
    }
    for prefix, (deployment, provider, api_version) in endpoints.items():
        monkeypatch.setenv(f"{prefix}_DEPLOYMENT_NAME", deployment)
        monkeypatch.setenv(f"{prefix}_ENDPOINT_URL", upstream_url)
        monkeypatch.setenv(f"{prefix}_TYPE", "vision")
        if provider:
            monkeypatch.setenv(f"{prefix}_PROVIDER", provider)
        if api_version:
            monkeypatch.setenv(f"{prefix}_API_VERSION", api_version)


async def _scenario(monkeypatch):
    seen = []

    async def record(request):
        form, image = {}, None
        async for part in await request.multipart():
            if part.name == "image":
                image = await part.read()
            else:
                form[part.name] = await part.text()
        seen.append({"path": request.path, "query": dict(request.query),
                     "auth": request.headers.get("Authorization"), "form": form, "image": image})
        return web.json_response({"data": [{"b64_json": "AAAA"}]})

    upstream = web.Application()
    upstream.router.add_post("/v1/images/edits", record)
    upstream.router.add_post("/openai/deployments/{deployment}/images/edits", record)
    async with TestServer(upstream, host="127.0.0.1") as up:
        _configure(monkeypatch, str(up.make_url("/")))
        proxy = AzureOpenAIProxy()
        proxy._azure_token_provider = lambda: AZURE_TEST_TOKEN  # no DefaultAzureCredential
        app = web.Application()
        setup_azure_proxy_routes(app, proxy)
        async with TestServer(app, host="127.0.0.1") as ps, aiohttp.ClientSession() as client:
            base = str(ps.make_url("/"))
            answers = {}
            for deployment in ("gpt-image-1", "edits-az", "edits-default"):
                form = aiohttp.FormData()
                form.add_field("prompt", "outline every field")
                form.add_field("size", "1024x1024")
                form.add_field("image", PNG, filename="input.png", content_type="image/png")
                url = routes.proxy_url(base, routes.ROUTE_IMAGE_EDITS, deployment=deployment)
                async with client.post(url, data=form) as resp:  # no client credential
                    answers[deployment] = (resp.status, await resp.json())
            listing_url = routes.proxy_url(base, routes.ROUTE_MODELS)
            params = {routes.MODELS_TYPE_PARAM: routes.ENDPOINT_TYPE_VISION}
            async with client.get(listing_url, params=params) as resp:
                listing = await resp.json()
    return seen, answers, listing


def test_the_proxy_forwards_each_identity_in_its_own_shape(monkeypatch):
    seen, answers, listing = asyncio.run(_scenario(monkeypatch))
    assert all(status == 200 and body["data"][0]["b64_json"] == "AAAA" for status, body in answers.values())
    openai_req, azure_req, default_req = seen
    common = {"prompt": "outline every field", "size": "1024x1024"}
    # direct OpenAI shape, although the upstream host is a loopback stub
    assert openai_req == {"path": "/v1/images/edits", "query": {}, "auth": f"Bearer {OPENAI_TEST_KEY}",
                          "form": {**common, "model": "gpt-image-1"}, "image": PNG}
    # direct Azure shape
    assert azure_req == {"path": "/openai/deployments/edits-az/images/edits",
                         "query": {"api-version": "2025-04-01-preview"}, "auth": f"Bearer {AZURE_TEST_TOKEN}",
                         "form": common, "image": PNG}
    # no identity and a host other than api.openai.com: Azure, as before
    assert default_req["path"] == "/openai/deployments/edits-default/images/edits"
    assert default_req["query"] == {"api-version": "2025-01-01-preview"}
    assert default_req["auth"] == f"Bearer {AZURE_TEST_TOKEN}"
    assert {e["deployment_name"]: e["provider"] for e in listing["data"]} == {
        "gpt-image-1": "openai", "edits-az": "azure_openai", "edits-default": "",
    }


async def _client_model_scenario(monkeypatch):
    seen = []

    async def record_form(request):
        form = {}
        async for part in await request.multipart():
            if part.name != "image":
                form[part.name] = await part.text()
        seen.append(form)
        return web.json_response({"data": [{"b64_json": "AAAA"}]})

    async def record_json(request):
        seen.append(await request.json())
        return web.json_response({"choices": []})

    upstream = web.Application()
    upstream.router.add_post("/v1/images/edits", record_form)
    upstream.router.add_post("/v1/chat/completions", record_json)
    async with TestServer(upstream, host="127.0.0.1") as up:
        for prefix, deployment, kind in (("IMG", "gpt-image-1", "vision"), ("CHAT", "gpt-4o", "chat")):
            monkeypatch.setenv(f"{prefix}_DEPLOYMENT_NAME", deployment)
            monkeypatch.setenv(f"{prefix}_ENDPOINT_URL", str(up.make_url("/")))
            monkeypatch.setenv(f"{prefix}_TYPE", kind)
            monkeypatch.setenv(f"{prefix}_PROVIDER", "openai")
        app = web.Application()
        setup_azure_proxy_routes(app, AzureOpenAIProxy())
        async with TestServer(app, host="127.0.0.1") as ps, aiohttp.ClientSession() as client:
            base = str(ps.make_url("/"))
            # an unconfigured deployment falls back to the default vision endpoint
            form = aiohttp.FormData()
            form.add_field("prompt", "outline every field")
            form.add_field("model", "my-vision")
            form.add_field("image", PNG, filename="input.png", content_type="image/png")
            url = routes.proxy_url(base, routes.ROUTE_IMAGE_EDITS, deployment="my-vision")
            async with client.post(url, data=form) as resp:
                assert resp.status == 200
            # a configured deployment, with a client asking for another model
            url = routes.proxy_url(base, routes.ROUTE_CHAT_COMPLETIONS, deployment="gpt-4o")
            async with client.post(url, json={"model": "o1-pro", "messages": []}) as resp:
                assert resp.status == 200
    return seen


def test_the_proxy_sends_the_configured_model_not_the_clients(monkeypatch):
    # The endpoint the proxy routed to, and whose credentials it spends, names the model.
    edit_form, chat_body = asyncio.run(_client_model_scenario(monkeypatch))
    assert edit_form == {"prompt": "outline every field", "model": "gpt-image-1"}
    assert chat_body == {"model": "gpt-4o", "messages": []}
