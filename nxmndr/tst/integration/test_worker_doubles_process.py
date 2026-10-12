# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The worker-process doubles (``tst.support.worker_doubles``) over real gRPC.

A separate Python process installs the doubles and runs ``create_server`` with its
default InferenceService, ModelManager and loaders; this process is only a client,
as QGIS is. Prompted SAM and Ultralytics YOLO sessions stream through the new
``tiles=`` client API, and the process's counters show each model and SAM variant
loaded once.
"""

from __future__ import annotations

import json
import os
import platform
import select
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nxmndr.client import InferenceGrpcClient
from nxmndr.inference import inference_pb2
from tst.support.ultralytics_double import write_checkpoint
from tst.support.worker_doubles import SAM_DOUBLE_REPO

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]  # the directory that contains tst/ and src/
H = W = 16
READY_TIMEOUT_S = 120.0  # bound only: the process imports torch and transformers first


def _command(stats, cache):
    cmd = [sys.executable, "-m", "tst.support.worker_doubles", "--stats", str(stats),
           "serve", "--cache-dir", str(cache)]
    if sys.platform == "darwin" and platform.machine() == "arm64" and shutil.which("arch"):
        cmd = ["arch", "-arm64", *cmd]  # never let a universal Python start as x86_64
    return cmd


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("NXMNDR_", "HF_", "HUGGINGFACE_", "OPENAI_"))}
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
        NXMNDR_CACHE_DIR=str(tmp_path / "worker-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return env


@pytest.fixture
def worker(tmp_path):
    stats = tmp_path / "stats.json"
    log = open(tmp_path / "worker.log", "w", encoding="utf-8")  # the server's logs, kept for failures
    proc = subprocess.Popen(
        _command(stats, tmp_path / "artifacts"), cwd=str(ROOT), env=_env(tmp_path),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
    )
    try:
        ready, _, _ = select.select([proc.stdout], [], [], READY_TIMEOUT_S)
        assert ready, "the worker-doubles server did not announce its endpoint"
        announced = json.loads(proc.stdout.readline())
        yield announced["endpoint"], proc, stats
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write("stop\n")
                proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()  # only the process this fixture started
                proc.wait(timeout=10)
        log.close()


def _array(resp):
    assert "error" not in resp.metadata, resp.metadata
    return np.frombuffer(resp.output, dtype=np.dtype(resp.dtype)).reshape(tuple(resp.shape))


def _stop(proc):
    proc.stdin.write("stop\n")
    proc.stdin.close()
    assert proc.wait(timeout=60) == 0


def test_prompted_sam_and_yolo_run_in_a_separate_worker_process(tmp_path, worker):
    endpoint, proc, stats_path = worker
    ckpt = write_checkpoint(tmp_path / "delineate.pt", group_mask=True)
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    sam_spec = inference_pb2.ModelSpec(format=inference_pb2.HUGGINGFACE, source=SAM_DOUBLE_REPO,
                                       task=inference_pb2.SEGMENTATION)
    yolo_spec = inference_pb2.ModelSpec(format=inference_pb2.PYTORCH, source=str(ckpt),
                                        model_class="ultralytics_yolo", task=inference_pb2.SEGMENTATION)
    sam_tiles = [
        ("text", rgb, {"sam_text_prompt": "field"}),
        ("box", rgb, {"sam_input_bbox": json.dumps([2, 2, 10, 10])}),
        ("points", rgb, {"sam_input_points": json.dumps([[3, 3], [12, 12]]),
                         "sam_input_labels": json.dumps([1, 0])}),
        ("none", rgb, None),
    ]
    with InferenceGrpcClient(endpoint, timeout=60, max_attempts=1) as client:
        assert client.capabilities()["instance_model_classes"] == "ultralytics_yolo"
        sam_loaded = client.load_model_result("", sam_spec)
        assert sam_loaded.effective_metadata["capability.sam"] == "sam3"

        sam_resps = {}
        for sid in ("sam-a", "sam-b"):  # two sessions, one resident SAM record
            opened = client.open_session(session_id=sid, spec=sam_spec, options={"task_type": "segmentation"})
            assert opened.status == "ok" and opened.model_cache_hit
            sam_resps[sid] = {r.metadata["tile_id"]: r for r in client.stream_predict(session_id=sid, tiles=sam_tiles)}
            assert client.close_session(sid).summary.ok_tiles == len(sam_tiles)

        opened = client.open_session(session_id="yolo", spec=yolo_spec)
        assert opened.status == "ok" and not opened.model_cache_hit
        blocks = np.zeros((H, W, 3), np.uint8)
        blocks[:8, :8, 0], blocks[:8, 8:, 0], blocks[12:, :, 0] = 1, 2, 3
        yolo_resps = list(client.stream_predict(
            session_id="yolo", tiles=[("r0_c0", blocks, None), ("r0_c1", np.zeros_like(blocks), None)]))
        assert client.close_session("yolo").summary.ok_tiles == 2

    for sid, resps in sam_resps.items():
        assert {t: r.metadata["sam_prompt"] for t, r in resps.items()} == {
            "text": "text", "box": "geometry", "points": "geometry", "none": "none"}, sid
        text = _array(resps["text"])  # the whole-chip group mask is dropped
        assert text.shape == (2, H, W)
        assert text[0][:, : W // 2].all() and not text[0][:, W // 2:].any()
        assert text[1][:, W // 2:].all() and not text[1][:, : W // 2].any()
        box = np.zeros((H, W), bool)
        box[2:10, 2:10] = True
        np.testing.assert_array_equal(_array(resps["box"]) != 0, box)
        point = np.zeros((H, W), bool)
        point[1:6, 1:6] = True  # positive point; the negative point removes nothing here
        np.testing.assert_array_equal(_array(resps["points"]) != 0, point)
        assert _array(resps["none"]).shape == (3, H, W)
    for resp in yolo_resps:
        assert resp.metadata["result_type"] == "segmentation_mask" and resp.dtype == "uint8"
    first = _array(yolo_resps[0])
    assert first.shape == (3, H, W)
    np.testing.assert_array_equal(first, np.stack([blocks[:, :, 0] == v for v in (1, 2, 3)]).astype(np.uint8))
    assert list(yolo_resps[1].shape) == [0, H, W]

    _stop(proc)
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # the server's first device
    stats = json.loads(stats_path.read_text())
    assert stats == {
        "yolo_checkpoints_opened": [str(ckpt)],
        "yolo_predict_options": [{"conf": 0.1, "iou": 0.7, "max_det": 300}],  # the adapter's defaults
        "sam_models_built": 1,
        "sam_unprompted_calls": 2,
        "sam_variant_loads": {f"sam3_tracker@{device}": 1},  # geometry variant: once per record
        "sam_variant_disposals": {f"sam3_tracker@{device}": 1},  # at server shutdown
    }


PROBE_MODULE = '''
import json, sys
from nxmndr.models import sam
print(json.dumps({"argv": sys.argv, "ultralytics": sys.modules["ultralytics"].__version__,
                  "variant_loader": type(sam.load_sam_variant).__name__}), flush=True)
'''


def test_run_mode_hands_a_clean_stdout_to_the_module(tmp_path):
    (tmp_path / "probe_mod.py").write_text(PROBE_MODULE)
    stats = tmp_path / "stats.json"
    cmd = _command(stats, tmp_path)
    cut = cmd.index("serve")
    cmd = cmd[:cut] + ["run", "probe_mod", "--flag", "value"]
    env = _env(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], str(tmp_path)])
    out = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                         timeout=READY_TIMEOUT_S, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert len(lines) == 1, out.stdout  # the installer's log lines went to stderr
    report = json.loads(lines[0])
    argv = report.pop("argv")
    assert argv[0].endswith("probe_mod.py") and argv[1:] == ["--flag", "value"]  # as python -m
    assert report == {"ultralytics": "0.0.0+nxmndr-test-double", "variant_loader": "CountingVariantLoader"}
    assert "Overriding existing spec registration" in out.stderr
    assert json.loads(stats.read_text())["sam_models_built"] == 0
