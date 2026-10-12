# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from pathlib import Path

import numpy as np
import pytest

from nxmndr.inference import inference_pb2, inference_pb2_grpc
from tst.integration.remote_test_utils import RemoteTestServer
from tst.example_model.modeling_exampleconv import export

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_stream_predict_round_trip(tmp_path):
    repo_root = Path.cwd()
    model_dir = repo_root / "tst" / "example_model"
    model_path = model_dir / "example_model.onnx"
    created_model = False
    if not model_path.exists():
        export(model_dir)
        created_model = True

    try:
        with RemoteTestServer() as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)

            spec = inference_pb2.ModelSpec(
                format=inference_pb2.ONNX,
                source=str(model_path),
                name="stream-test",
            )
            load_resp = stub.LoadModel(inference_pb2.LoadModelRequest(spec=spec))
            assert load_resp.success
            model_id = load_resp.model_id

            # A session stream needs a session opened with OpenSession (no fabricated sessions).
            open_resp = stub.OpenSession(
                inference_pb2.OpenSessionRequest(
                    session_id="sessB", spec=inference_pb2.ModelSpec(model_id=model_id)
                )
            )
            assert open_resp.status == "ok"

            arr = np.zeros((1, 3, 32, 32), dtype=np.float32)
            reqs = [
                inference_pb2.StreamPredictRequest(
                    model_id=model_id,
                    chunk=arr.tobytes(),
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    context={"tile_id": "tile1", "session_id": "sessB"},
                ),
                inference_pb2.StreamPredictRequest(
                    end_of_sequence=True, context={"tile_id": "tile1"}
                ),
                inference_pb2.StreamPredictRequest(
                    model_id=model_id,
                    chunk=arr.tobytes(),
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    context={"tile_id": "tile2", "session_id": "sessB"},
                ),
                inference_pb2.StreamPredictRequest(
                    end_of_sequence=True, context={"tile_id": "tile2"}
                ),
            ]

            responses = list(stub.StreamPredict(iter(reqs)))
            assert len(responses) == 2
            # every tile answered exactly once
            assert sorted(r.metadata.get("tile_id") for r in responses) == ["tile1", "tile2"]
            for resp in responses:
                assert resp.end_of_sequence is True
                assert resp.metadata.get("error") is None
                assert resp.metadata.get("model_id") == model_id
                out = np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(resp.shape)
                assert out.size > 0
                # Ensure device metadata is present
                assert resp.metadata.get("device_id")
                assert resp.metadata.get("device_type")
                assert resp.metadata.get("session_id") == "sessB"
                assert resp.metadata.get("tile_id") in {"tile1", "tile2"}
    finally:
        if created_model and model_path.exists():
            model_path.unlink()


@pytest.mark.integration
def test_session_open_stream_close_summary(tmp_path):
    repo_root = Path.cwd()
    model_dir = repo_root / "tst" / "example_model"
    model_path = model_dir / "example_model.onnx"
    created_model = False
    if not model_path.exists():
        export(model_dir)
        created_model = True

    try:
        with RemoteTestServer() as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)

            spec = inference_pb2.ModelSpec(
                format=inference_pb2.ONNX,
                source=str(model_path),
                name="stream-lifecycle",
            )
            load_resp = stub.LoadModel(inference_pb2.LoadModelRequest(spec=spec))
            assert load_resp.success
            model_id = load_resp.model_id

            open_resp = stub.OpenSession(
                inference_pb2.OpenSessionRequest(
                    session_id="sessLifecycle",
                    spec=inference_pb2.ModelSpec(
                        model_id=model_id,
                        format=inference_pb2.ONNX,
                        source=str(model_path),
                    ),
                    # Allow full-tile chunks for the example model (~12KB)
                    transport=inference_pb2.TransportCaps(max_inflight=8, chunk_bytes=16384),
                )
            )
            assert open_resp.status == "ok"
            assert open_resp.session_id == "sessLifecycle"
            assert open_resp.device_plan
            assert open_resp.max_inflight == 8

            arr = np.ones((1, 3, 32, 32), dtype=np.float32)
            reqs = [
                inference_pb2.StreamPredictRequest(
                    model_id=model_id,
                    chunk=arr.tobytes(),
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    context={"tile_id": "t1", "session_id": "sessLifecycle"},
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": "t1"}),
                inference_pb2.StreamPredictRequest(
                    model_id=model_id,
                    chunk=arr.tobytes(),
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    context={"tile_id": "t2", "session_id": "sessLifecycle"},
                ),
                inference_pb2.StreamPredictRequest(end_of_sequence=True, context={"tile_id": "t2"}),
            ]

            responses = list(stub.StreamPredict(iter(reqs)))
            assert len(responses) == 2
            for resp in responses:
                assert resp.metadata.get("session_id") == "sessLifecycle"
                assert resp.metadata.get("tile_id") in {"t1", "t2"}
                assert resp.metadata.get("device_id")
                # progress may be present when total_tiles known; ensure not crashing

            close_resp = stub.CloseSession(
                inference_pb2.CloseSessionRequest(session_id="sessLifecycle")
            )
            assert close_resp.status == "closed"
            assert close_resp.summary.session_id == "sessLifecycle"
            assert close_resp.summary.ok_tiles == 2
            assert close_resp.summary.failed_tiles == 0
    finally:
        if created_model and model_path.exists():
            model_path.unlink()


@pytest.mark.integration
def test_stream_tile_over_limit_and_cancel(tmp_path):
    repo_root = Path.cwd()
    model_dir = repo_root / "tst" / "example_model"
    model_path = model_dir / "example_model.onnx"
    created_model = False
    if not model_path.exists():
        export(model_dir)
        created_model = True

    try:
        with RemoteTestServer() as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)

            spec = inference_pb2.ModelSpec(
                format=inference_pb2.ONNX,
                source=str(model_path),
                name="stream-oversize",
            )
            load_resp = stub.LoadModel(inference_pb2.LoadModelRequest(spec=spec))
            assert load_resp.success
            model_id = load_resp.model_id

            # Enforce tiny tile limit but allow larger chunks
            open_resp = stub.OpenSession(
                inference_pb2.OpenSessionRequest(
                    session_id="sessOver",
                    spec=inference_pb2.ModelSpec(
                        model_id=model_id,
                        format=inference_pb2.ONNX,
                        source=str(model_path),
                    ),
                    options={"max_tile_bytes": "8"},
                    transport=inference_pb2.TransportCaps(max_inflight=8, chunk_bytes=1024),
                )
            )
            assert open_resp.status == "ok"

            big_chunk = b"x" * 16
            reqs = [
                inference_pb2.StreamPredictRequest(
                    model_id=model_id,
                    chunk=big_chunk,
                    shape=[1, 1, 1, 4],
                    dtype="float32",
                    context={"session_id": "sessOver", "tile_id": "tile-big"},
                ),
                inference_pb2.StreamPredictRequest(
                    end_of_sequence=True, context={"session_id": "sessOver"}
                ),
            ]

            resps = list(stub.StreamPredict(iter(reqs)))
            assert resps and resps[0].metadata.get("error")
            # Error message may vary based on whether batching or legacy path is used
            error_msg = resps[0].metadata.get("error")
            assert "exceeds" in error_msg and ("tile" in error_msg or "stream sample" in error_msg)
            assert resps[0].metadata.get("tile_id") == "tile-big"
            assert resps[0].metadata.get("session_id") == "sessOver"

            cancel_resp = stub.CancelSession(
                inference_pb2.CancelSessionRequest(session_id="sessOver", reason="test")
            )
            assert cancel_resp.status == "cancelled"
    finally:
        if created_model and model_path.exists():
            model_path.unlink()
