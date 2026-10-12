# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The endpoint's scheme selects the client's transport.

Intercepted channel factories, no network: ``https://`` opens a secure channel
with the default root certificates (or the given credentials) on the host's port,
443 by default, for the first channel and for every channel made after a reset
(control calls and streams share it); ``http://``, ``grpc://`` and bare
``host:port`` open a plaintext channel. The real TLS handshake, including
server-name validation, runs in tst/integration/test_client_tls.py.
"""

from __future__ import annotations

import pytest

from nxmndr import client as client_module
from nxmndr.client import GRPC_OPTIONS, InferenceGrpcClient, parse_endpoint


@pytest.fixture
def channels(monkeypatch):
    made = []
    default_creds = object()

    def secure(target, credentials, options=None):
        made.append(("secure", target, credentials, options))
        return object()

    def insecure(target, options=None):
        made.append(("insecure", target, None, options))
        return object()

    monkeypatch.setattr(client_module.grpc, "secure_channel", secure)
    monkeypatch.setattr(client_module.grpc, "insecure_channel", insecure)
    monkeypatch.setattr(client_module.grpc, "ssl_channel_credentials", lambda *a, **k: default_creds)
    return made, default_creds


def test_https_selects_a_secure_channel_with_default_roots(channels):
    made, default_creds = channels
    client = InferenceGrpcClient("https://example.invalid:443")
    client._get_channel()
    assert made == [("secure", "example.invalid:443", default_creds, GRPC_OPTIONS)]

    client.close()  # a closed client reopens its channel with TLS on the next call
    client._get_channel()
    assert made[-1] == ("secure", "example.invalid:443", default_creds, GRPC_OPTIONS)
    assert all(kind == "secure" for kind, *_ in made)


def test_https_without_port_uses_443_and_given_credentials_win(channels):
    made, default_creds = channels
    InferenceGrpcClient("https://example.invalid")._get_channel()
    explicit = object()
    InferenceGrpcClient("HTTPS://Example.invalid:8443/", credentials=explicit)._get_channel()
    assert made == [
        ("secure", "example.invalid:443", default_creds, GRPC_OPTIONS),
        ("secure", "Example.invalid:8443", explicit, GRPC_OPTIONS),
    ]


@pytest.mark.parametrize(
    "endpoint, target",
    [
        ("127.0.0.1:50051", "127.0.0.1:50051"),
        ("grpc://127.0.0.1:50051", "127.0.0.1:50051"),
        ("http://localhost:50051", "localhost:50051"),
        ("http://localhost", "localhost:80"),
        ("[::1]:50051", "[::1]:50051"),
    ],
)
def test_plaintext_endpoints_select_an_insecure_channel(channels, endpoint, target):
    made, _ = channels
    InferenceGrpcClient(endpoint)._get_channel()
    assert made == [("insecure", target, None, GRPC_OPTIONS)]


def test_parse_endpoint_rules():
    assert parse_endpoint("https://[::1]:8443").target == "[::1]:8443"
    assert parse_endpoint("https://[::1]").host == "::1"
    assert parse_endpoint("unix:///tmp/nxmndr.sock").target == "unix:///tmp/nxmndr.sock"
    for bad in ("https://host/v1", "https://:443", "https://host:port", "https://user@host:443"):
        with pytest.raises(ValueError):
            parse_endpoint(bad)
