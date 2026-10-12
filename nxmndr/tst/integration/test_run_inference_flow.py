# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import rasterio
import pytest
from pathlib import Path

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2
from tst.integration.remote_test_utils import start_test_server
from tst.example_model.modeling_exampleconv import export


@pytest.mark.integration
def test_run_inference_flow_with_raster(tmp_path, synthetic_tif):
    repo_root = (Path(__file__).resolve().parents[2]).resolve()

    # Prepare example ONNX model
    model_dir = repo_root / "tst" / "integration" / "example_model"
    model_path = model_dir / "example_model.onnx"
    created_model = False
    if not model_path.exists():
        export(model_dir)
        created_model = True

    # Input raster
    image_path = synthetic_tif
    assert image_path.exists()

    server = None
    bound_port = None
    try:
        server, bound_port = start_test_server()
        client = InferenceGrpcClient(f"127.0.0.1:{bound_port}", timeout=10)

        model_id = client.load_model(
            model_id="integration-stream",
            spec={"format": "onnx", "source": str(model_path), "name": "integration-stream"},
        )

        tile_ids = []
        samples = []

        # Tile the raster into 32x32 patches matching the example model input
        with rasterio.open(image_path) as src:
            data = src.read()  # [C,H,W]
            _, h, w = data.shape
            tile_size = 32
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    window = data[:, y : y + tile_size, x : x + tile_size]
                    if window.shape[1] != tile_size or window.shape[2] != tile_size:
                        pad = np.zeros((window.shape[0], tile_size, tile_size), dtype=window.dtype)
                        pad[:, : window.shape[1], : window.shape[2]] = window
                        window = pad
                    arr = window.astype(np.float32) / 255.0
                    arr = np.expand_dims(arr, axis=0)  # N,C,H,W
                    samples.append(arr)
                    tile_ids.append(f"tile_{y}_{x}")

        manifest = inference_pb2.TileManifest(
            total_tiles=len(samples),
            tile_h=32,
            tile_w=32,
            stride_h=32,
            stride_w=32,
            window_h=32,
            window_w=32,
        )

        open_resp = client.open_session(
            session_id="sess_integ",
            spec=inference_pb2.ModelSpec(
                model_id=model_id,
                format=inference_pb2.ONNX,
                source=str(model_path),
            ),
            manifest=manifest,
            transport=inference_pb2.TransportCaps(max_inflight=4, chunk_bytes=32 * 1024 * 1024),
        )
        assert open_resp.status == "ok"

        responses = list(
            client.stream_predict(
                model_id=model_id,
                samples=samples,
                session_id="sess_integ",
                tile_ids=tile_ids,
                chunk_bytes=32 * 1024 * 1024,
                dtype="float32",
            )
        )

        assert len(responses) == len(samples)
        seen_tiles = set()
        for resp in responses:
            assert resp.end_of_sequence is True
            assert resp.metadata.get("session_id") == "sess_integ"
            assert resp.metadata.get("tile_id") in set(tile_ids)
            seen_tiles.add(resp.metadata.get("tile_id"))
            out = np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(resp.shape)
            assert out.size > 0

        assert seen_tiles == set(tile_ids)

        close_resp = client.close_session("sess_integ")
        assert close_resp.status == "closed"
        assert close_resp.summary.ok_tiles == len(samples)
        assert close_resp.summary.failed_tiles == 0
    finally:
        if server:
            server.stop(0)
        if created_model and model_path.exists():
            model_path.unlink()
