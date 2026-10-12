# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""server.py on the real ModelManager, over a real loopback gRPC transport.

Cases that need the server: the model token reaches the SAM variant loader through
ModelRecord.auth_token and never the record metadata or the registry file (the
registry is written to disk from that metadata); the Hugging Face revision is
normalized like the cache key; a session closed between the open check and the tile
dispatch fails the stream instead of raising out of touch_session.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nxmndr.inference import inference_pb2
from nxmndr.models import sam as sam_support
from nxmndr.server.registry import ModelRegistryStore
from tst.support.grpc_harness import running_server
from tst.support.stream_doubles import (
    CountingLoader,
    CountingVariantLoader,
    EchoModel,
    FakeHFSamModel,
)

pytestmark = pytest.mark.integration

TOKEN = "hf_FollowUpDummyTokenValue"  # a placeholder, not a credential
BOX = json.dumps([2.0, 2.0, 10.0, 10.0])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "NXMNDR_REMOTE_HOST", "NXMNDR_REMOTE_PORT"):
        monkeypatch.delenv(var, raising=False)


class _RecordingLoader(CountingLoader):
    """CountingLoader that also keeps the positional arguments of every call."""

    def __init__(self, factories):
        super().__init__(factories)
        self.args = []

    def __call__(self, *args):
        self.args.append(args)
        return super().__call__(*args)


def test_token_reaches_sam_through_the_record_and_never_metadata_or_registry(
    tmp_path, monkeypatch
):
    log = []
    variants = CountingVariantLoader(log)
    monkeypatch.setattr(sam_support, "load_sam_variant", variants)
    loader = _RecordingLoader({"double://sam3": lambda: (FakeHFSamModel(log), "huggingface")})
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        registry_path = tmp_path / "registry" / "model_registry.json"
        h.service.registry_store = ModelRegistryStore(
            registry_path, artifact_root=tmp_path / "artifacts"
        )
        resp = c._get_stub().LoadModel(
            inference_pb2.LoadModelRequest(
                spec=inference_pb2.ModelSpec(
                    format=inference_pb2.HUGGINGFACE,
                    source="double://sam3",
                    version="  rev-2 ",
                    token=TOKEN,
                    task=inference_pb2.SEGMENTATION,
                )
            )
        )
        assert resp.success, resp.message
        mid = resp.model_id

        [(model_spec, key, metadata, _)] = loader.args
        assert model_spec.revision == "rev-2" and key.revision == "rev-2"  # key and load agree
        assert model_spec.token == TOKEN  # the loader may use it
        assert key.auth_scope and TOKEN not in repr(key)
        assert TOKEN not in json.dumps(metadata, default=str)
        record = h.manager.get(mid)
        assert record.auth_token == TOKEN
        assert TOKEN not in json.dumps(record.metadata, default=str)
        assert TOKEN not in repr(record)

        opened = c.open_session(session_id="s-token", spec=inference_pb2.ModelSpec(model_id=mid))
        assert opened.status == "ok"
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        [tile] = list(
            c.stream_predict(
                session_id="s-token",
                samples=[image],
                tile_ids=["t0"],
                options={"sam_input_bbox": BOX},
            )
        )
        assert tile.metadata.get("error") is None, tile.metadata
        assert variants.tokens == [TOKEN]  # the geometry variant got the record's token

        text = registry_path.read_text(encoding="utf-8")
        assert mid in text  # the registry was written ...
        assert TOKEN not in text  # ... without the token


def test_session_closed_between_open_check_and_dispatch_fails_the_stream(tmp_path):
    echo = EchoModel()
    loader = CountingLoader({"double://echo": lambda: (echo, "onnx")})
    with running_server(tmp_path, loader=loader) as h, h.client() as c:
        mid = c.load_model("ignored", {"format": "onnx", "source": "double://echo"})
        assert c.open_session(
            session_id="s-race", spec=inference_pb2.ModelSpec(model_id=mid)
        ).status == "ok"

        real_touch = h.manager.touch_session

        def cancel_then_touch(session_id):
            # A CancelSession that lands after the per-message open check.
            h.manager.close_session(session_id, reason="cancelled")
            return real_touch(session_id)

        h.manager.touch_session = cancel_then_touch
        responses = list(
            c.stream_predict(
                session_id="s-race", samples=[np.ones((4, 4, 3), np.float32)], tile_ids=["t0"]
            )
        )
        assert [r.metadata.get("error_code") for r in responses] == ["cancelled"]
        assert responses[0].metadata.get("error_scope") == "stream"
        assert echo.seen == []  # nothing was dispatched
        assert h.service._session_state("s-race") == "cancelled"
        assert h.manager.pin_count(mid) == 0  # the execution lease was released
