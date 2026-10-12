# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The canonical client over a real local TLS server, and over plaintext loopback.

A throwaway CA and leaf certificates are generated in the test with
``cryptography`` (a requirement of azure-identity in the backend's ``server``
extra). The server is the real ``InferenceService`` behind ``add_secure_port`` on
127.0.0.1; nothing leaves the machine.

- ``https://127.0.0.1:<port>`` with the CA as root certificate: control calls and a
  stream reach the service over TLS.
- The same endpoint against a certificate issued for another name fails the
  handshake: the server name is validated against the endpoint's host.
- Without credentials the default roots apply, so the private CA is not trusted;
  and an ``http://`` endpoint does not speak TLS. Neither falls back to plaintext
  success.
- A plaintext loopback server (the local worker's transport) still works for bare
  ``host:port``, ``grpc://`` and ``http://``.
"""

from __future__ import annotations

import datetime
import ipaddress
from concurrent import futures

import grpc
import numpy as np
import pytest

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from nxmndr.client import InferenceGrpcClient, InferenceGrpcError
from nxmndr.inference import inference_pb2_grpc
from nxmndr.server.server import GRPC_OPTIONS, InferenceService

pytestmark = pytest.mark.integration


def _name(common_name):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _pem(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def _key_pem(key):
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _builder(subject, issuer, public_key):
    now = datetime.datetime.now(datetime.timezone.utc)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
    )


@pytest.fixture(scope="module")
def pki():
    """A private CA, a leaf for 127.0.0.1 and a leaf for another name, all PEM."""

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("nxmndr test CA")
    ca = (
        _builder(ca_name, ca_name, ca_key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )

    def leaf(common_name, san):
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (
            _builder(_name(common_name), ca_name, key.public_key())
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        return _key_pem(key), _pem(cert)

    return {
        "ca": _pem(ca),
        "loopback": leaf("127.0.0.1", [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
        "other": leaf("other.invalid", [x509.DNSName("other.invalid")]),
    }


def _serve(tmp_path, credentials=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_OPTIONS)
    service = InferenceService(model_cache_dir=tmp_path / "models", max_cores=1)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    if credentials is None:
        port = server.add_insecure_port("127.0.0.1:0")
    else:
        port = server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials([credentials]))
    server.start()
    return server, service, port


@pytest.fixture
def servers(tmp_path):
    started = []

    def start(credentials=None):
        server, service, port = _serve(tmp_path, credentials)
        started.append((server, service))
        return port

    yield start
    for server, service in started:
        server.stop(0).wait()
        service.shutdown(grace=0)


def _client(endpoint, **kwargs):
    return InferenceGrpcClient(endpoint, timeout=10, max_attempts=1, **kwargs)


def test_tls_round_trip_with_server_name_validation(pki, servers):
    port = servers(pki["loopback"])
    trust = grpc.ssl_channel_credentials(root_certificates=pki["ca"])
    with _client(f"https://127.0.0.1:{port}", credentials=trust) as client:
        caps = client.capabilities()
        assert caps["stream_context_version"] == "1"
        # A stream over the same TLS channel reaches the service: an unknown model is
        # the service's NOT_FOUND, not a transport failure.
        with pytest.raises(InferenceGrpcError) as stream_err:
            list(client.stream_predict(model_id="nope", samples=[np.zeros((2, 2, 1), np.float32)]))
        assert stream_err.value.code == grpc.StatusCode.NOT_FOUND

    other_port = servers(pki["other"])  # reachable, but its certificate names other.invalid
    with _client(f"https://127.0.0.1:{other_port}", credentials=trust) as client:
        with pytest.raises(InferenceGrpcError) as mismatch:
            client.capabilities()
    assert mismatch.value.code == grpc.StatusCode.UNAVAILABLE
    assert "hostname verification" in str(mismatch.value).lower(), str(mismatch.value)  # grpcio 1.78.0
    # The same server and trust work when the name matches: only the name differed.
    with _client(f"https://127.0.0.1:{port}", credentials=trust) as client:
        assert client.capabilities()["stream_context_version"] == "1"


def test_https_never_falls_back_to_plaintext(pki, servers):
    port = servers(pki["loopback"])
    with _client(f"https://127.0.0.1:{port}") as client:  # default roots: private CA untrusted
        with pytest.raises(InferenceGrpcError) as untrusted:
            client.capabilities()
    assert untrusted.value.code == grpc.StatusCode.UNAVAILABLE
    assert "certificate_verify_failed" in str(untrusted.value).lower(), str(untrusted.value)
    with _client(f"http://127.0.0.1:{port}") as client:  # plaintext to a TLS-only server
        with pytest.raises(InferenceGrpcError):
            client.capabilities()

    plain_port = servers()  # and https to a plaintext server fails rather than downgrading
    with _client(f"https://127.0.0.1:{plain_port}") as client:
        with pytest.raises(InferenceGrpcError) as plain:
            client.capabilities()
    assert plain.value.code == grpc.StatusCode.UNAVAILABLE


@pytest.mark.parametrize("form", ["{}", "grpc://{}", "http://{}"])
def test_plaintext_loopback_still_works(servers, form):
    port = servers()
    with _client(form.format(f"127.0.0.1:{port}")) as client:
        assert client.capabilities()["stream_context_version"] == "1"
