# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Client encoding and server decoding agree with the frozen stream context v1 fixture.

The fixture is copied verbatim from the plugin's contracts package; see
tst/fixtures/contracts/PROVENANCE.md.
"""

import hashlib
import json
from pathlib import Path

import pytest

from nxmndr import client
from nxmndr.models import sam
from nxmndr.server import dispatch

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "contracts" / "stream_context_v1.json"
FIXTURE_SHA256 = "212eeb8207afc688c012ba56733824b1842b556c26c698e2841e8fa8f1466e06"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


def test_fixture_is_the_recorded_copy():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_constants_match(fixture):
    assert list(client.RESERVED_CONTEXT_KEYS) == fixture["reserved_context_keys"]
    assert list(dispatch.RESERVED_CONTEXT_KEYS) == fixture["reserved_context_keys"]
    assert client.TILE_OPTION_PREFIX == dispatch.TILE_OPTION_PREFIX == fixture["tile_option_prefix"]
    assert list(dispatch.SESSION_CONTROL_OPTION_KEYS) == fixture["session_control_option_keys"]
    assert fixture["capability"] == {
        "key": client.STREAM_CONTEXT_CAPABILITY,
        "value": dispatch.STREAM_CONTEXT_VERSION,
    }
    server_codes = {v for k, v in vars(dispatch).items() if k.startswith("ERROR_")}
    assert server_codes == set(fixture["error_codes"])
    assert {dispatch.SCOPE_TILE, dispatch.SCOPE_STREAM} == set(fixture["error_scopes"])


def test_client_encodes_fixture_contexts(fixture):
    for tile in fixture["tiles"]:
        first = client.encode_tile_context(
            tile_id=tile["tile_id"], session_id="sess-sam-1", options=tile["tile_options"]
        )
        later = client.encode_tile_context(tile_id=tile["tile_id"], session_id="sess-sam-1")
        assert first == tile["first_message_context"]
        assert later == tile["later_message_context"]


def test_server_decodes_fixture_contexts(fixture):
    for tile in fixture["tiles"]:
        decoded = dispatch.decode_tile_context(tile["first_message_context"])
        assert (decoded.session_id, decoded.tile_id) == ("sess-sam-1", tile["tile_id"])
        assert decoded.options == tile["tile_options"]
        assert dispatch.effective_tile_options(fixture["session_options"], decoded.options) == (
            tile["effective_options"]
        )
        assert dispatch.decode_tile_context(tile["later_message_context"]).options == {}


def test_invalid_contexts_are_rejected(fixture):
    for case in fixture["context_invalid"]:
        with pytest.raises(dispatch.MalformedOptionsError):
            dispatch.decode_tile_context(case["context"])
    with pytest.raises(ValueError):
        client.encode_tile_context(tile_id="t", options={"tile_id": "x"})
    with pytest.raises(ValueError):
        client.encode_tile_context(tile_id="t", options={"Bad-Key": "x"})


def test_sam_option_fixture_cases(fixture):
    for case in fixture["sam_options_valid"]:
        sam.parse_sam_prompt(case["options"])
    for case in fixture["sam_options_invalid"]:
        with pytest.raises(sam.SamPromptError):
            sam.parse_sam_prompt(case["options"])
