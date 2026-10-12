# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InferenceGrpcClient failure handling over a real loopback gRPC transport.

A failed call is retried only when the server could not be reached, and a failure
never ends the other calls of the same client. A stream that ends releases its
request thread even when the caller keeps the response iterator.
"""

from __future__ import annotations

import sys
import threading
import time

import grpc
import numpy as np
import pytest

from nxmndr import client as nx_client
from nxmndr.client import InferenceGrpcError
from nxmndr.inference import inference_pb2
from tst.support.grpc_harness import running_server, wait_until
from tst.support.stream_doubles import CountingLoader, EchoModel, GatedModel

pytestmark = pytest.mark.integration


class FailingModel:
    """Raises inside predict; the server answers INTERNAL."""

    def __init__(self):
        self.calls = 0

    def predict(self, arr, return_embeddings=False):
        self.calls += 1
        raise RuntimeError("inference failed")


@pytest.fixture
def failing():
    return FailingModel()


@pytest.fixture
def gated():
    return GatedModel()


@pytest.fixture
def broken_loads():
    return []


@pytest.fixture
def loader(failing, gated, broken_loads):
    def broken():
        broken_loads.append(1)
        raise RuntimeError("load failed")

    return CountingLoader(
        {
            "double://fail": lambda: (failing, "onnx"),
            "double://gated": lambda: (gated, "onnx"),
            "double://broken": broken,
            "double://echo": lambda: (EchoModel(), "onnx"),
        }
    )


def _load(client, source):
    return client.load_model("ignored", {"format": "onnx", "source": source, "task": ""})


def _chip():
    return np.ones((4, 4, 3), np.float32)


def test_a_failing_predict_runs_the_model_once(tmp_path, loader, failing):
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        mid = _load(c, "double://fail")
        with pytest.raises(InferenceGrpcError) as err:
            c.predict(mid, _chip())
        assert err.value.code == grpc.StatusCode.INTERNAL
        assert failing.calls == 1


def test_a_failed_call_does_not_end_a_stream_on_the_same_client(tmp_path, loader, gated, broken_loads):
    # The failing call is a LoadModel, so it needs no inference slot while the
    # stream's tile holds one.
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        gated_id = _load(c, "double://gated")
        call = c.stream_predict(model_id=gated_id, samples=[_chip()], tile_ids=["t0"])
        got, errors = [], []

        def consume():
            try:
                got.extend(call)
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert gated.entered.wait(10), "the stream's tile never started"
        with pytest.raises(InferenceGrpcError) as err:
            _load(c, "double://broken")
        assert err.value.code == grpc.StatusCode.INTERNAL
        assert len(broken_loads) == 1
        gated.release.set()
        consumer.join(10)
        assert errors == [] and [r.metadata["tile_id"] for r in got] == ["t0"]


def test_failed_registry_calls_do_not_end_a_stream_on_the_same_client(
    tmp_path, loader, gated, monkeypatch
):
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        gated_id = _load(c, "double://gated")
        call = c.stream_predict(model_id=gated_id, samples=[_chip()], tile_ids=["t0"])
        got, errors = [], []

        def consume():
            try:
                got.extend(call)
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert gated.entered.wait(10), "the stream's tile never started"

        def broken_list(provider_id=None):
            raise RuntimeError("registry store failed")

        monkeypatch.setattr(h.service.registry_store, "list_entries", broken_list)
        with pytest.raises(InferenceGrpcError):
            c.list_model_registry()
        with pytest.raises(InferenceGrpcError):
            c.evict_model_registry_entry("no-such-entry")  # NOT_FOUND
        gated.release.set()
        consumer.join(10)
        assert errors == [] and [r.metadata["tile_id"] for r in got] == ["t0"]


def test_a_failed_stream_does_not_end_a_predict_on_the_same_client(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        gated_id = _load(c, "double://gated")
        result, errors = [], []

        def predict():
            try:
                result.append(c.predict(gated_id, _chip()))
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        caller = threading.Thread(target=predict)
        caller.start()
        assert gated.entered.wait(10), "the predict never started"
        with pytest.raises(InferenceGrpcError) as err:
            list(c.stream_predict(session_id="never-opened", samples=[_chip()], tile_ids=["t0"]))
        assert err.value.code == grpc.StatusCode.NOT_FOUND
        gated.release.set()
        caller.join(10)
        assert errors == [] and tuple(result[0].shape) == (4, 4, 3)


def _threads_waiting_for_a_window_slot():
    acquire = nx_client._TileWindow.acquire.__code__
    waiting = []
    for thread_id, frame in sys._current_frames().items():
        while frame is not None:
            if frame.f_code is acquire:
                waiting.append(thread_id)
                break
            frame = frame.f_back
    return waiting


def test_stream_end_releases_the_request_thread_while_the_caller_holds_the_iterator(
    tmp_path, loader
):
    with running_server(tmp_path, loader=loader) as h, h.client() as c, h.client() as control:
        mid = _load(c, "double://echo")
        polls = []
        c.open_session(session_id="sess-held", spec=inference_pb2.ModelSpec(model_id=mid))
        call = c.stream_predict(
            session_id="sess-held",
            samples=[np.zeros(4, np.uint8) for _ in range(20)],
            tile_ids=[f"t{i}" for i in range(20)],
            max_inflight=1,
            cancel_fn=lambda: polls.append(1) and False,
        )
        responses = iter(call)
        assert next(responses).metadata.get("error") is None
        control.cancel_session("sess-held", reason="test")
        last = None
        for resp in responses:  # leaving the loop keeps `responses` alive
            last = resp
            if resp.metadata.get("error_scope") == "stream":
                break
        assert last is not None and last.metadata["error_code"] == "cancelled"
        # Wait for the RPC status first: the window check alone can pass before it arrives.
        assert wait_until(call._call.done, timeout=5), "the server never ended the RPC"
        assert wait_until(lambda: not _threads_waiting_for_a_window_slot(), timeout=5), (
            "the request thread still waits for a window slot after the RPC ended"
        )

        def polling_stopped():
            seen = len(polls)
            time.sleep(4 * nx_client.CANCEL_POLL_SECONDS)
            return len(polls) == seen

        assert wait_until(polling_stopped, timeout=5), "cancel_fn is still polled after the RPC ended"
        del responses
