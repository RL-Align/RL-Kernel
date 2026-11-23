# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check the integrity and row semantics of the sealed packed-layout fixtures.

These CPU tests run offline, independently of ``H3PackedLayout``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent
UPSTREAM = json.loads((ROOT / "upstream_cases.json").read_text(encoding="utf-8"))
HAND = json.loads((ROOT / "hand_written_case.json").read_text(encoding="utf-8"))
CASES = {case["case_id"]: case for case in UPSTREAM["cases"]}
CASE_IDS = {
    "fl2va_none",
    "fl2va_first",
    "fl2va_last",
    "fl2va_first_last",
    "fl2va_text_visual_tag",
    "fl2va_last_16_frames",
    "fl2va_same_timestep",
    "fl2va_block_tiny",
    "ref2va_image",
    "ref2va_audio",
    "ref2va_silent_video",
    "ref2va_video_audio",
    "ref2va_mixed",
}
# Seal data and case parameters independently of the generator's current environment.
CASES_SHA256 = "51efe871dd712632675098c08f326e61831f144c5ecbcd4966cde93e1929b6d4"


def decode(field):
    dtype = {"float32": "<f4", "float64": "<f8", "int64": "<i8"}[field["dtype"]]
    if "data_le_hex" in field:
        values = np.frombuffer(bytes.fromhex(field["data_le_hex"]), dtype=dtype)
    else:
        values = np.asarray(field["values"], dtype=dtype)
    return values.reshape(field["shape"])


def test_upstream_provenance_and_sealed_case_matrix():
    assert UPSTREAM["format"] == "h3-packed-layout-upstream.v1"
    assert UPSTREAM["checkpoint_revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
    source = UPSTREAM["upstream"]
    assert source["commit"] == "f53d552036a0d1bd5570782a39cd40cfabf112bc"
    assert source["sha256"] == ("530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff")
    assert source["git_blob_sha1"] == "c670467a93075ae841c7b7cdd250ceb6fa85eda9"
    assert UPSTREAM["environment"]["device"] == "cpu"
    assert set(CASES) == CASE_IDS
    assert len(UPSTREAM["cases"]) == len(CASE_IDS)
    payload = json.dumps(UPSTREAM["cases"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == CASES_SHA256


@pytest.mark.parametrize("state", ["before", "after"])
def test_hand_written_scalar_scatter_and_source_order_outputs(state):
    case = HAND[state]
    packed = [None] * HAND["sequence_length"]
    row_ids = [None] * HAND["sequence_length"]
    for stream in ("text", "audio", "video"):
        for source_row, packed_row in enumerate(case[f"{stream}_indices"]):
            assert packed[packed_row] is None, "overlapping source destinations"
            packed[packed_row] = HAND["source_payloads"][stream][source_row]
            row_ids[packed_row] = HAND["source_row_ids"][stream][source_row]
    assert packed == case["packed_payload"]
    assert row_ids == case["row_ids"]
    for stream in ("video", "audio"):
        selected = [packed[row] for row in case[f"{stream}_indices"]]
        assert selected == HAND["expected_selected_payloads"][stream]
    assert HAND["source_roles"]["video"] == ["condition", "target"]


def test_hand_written_non_self_inverse_permutation_preserves_each_identity():
    before, after = HAND["before"], HAND["after"]
    perm, inverse = HAND["perm_new_to_old"], HAND["inverse_old_to_new"]
    assert perm != inverse
    assert sorted(perm) == list(range(HAND["sequence_length"]))
    for new_row, old_row in enumerate(perm):
        assert inverse[old_row] == new_row
        for field in (
            "row_ids",
            "packed_payload",
            "token_tags",
            "timestep_indices",
            "adaln_indices",
            "position_ids",
        ):
            assert after[field][new_row] == before[field][old_row]
    for stream in ("text", "audio", "video"):
        assert after[f"{stream}_indices"] == [inverse[row] for row in before[f"{stream}_indices"]]
    for state in (before, after):
        for row, (time, tag) in enumerate(
            zip(state["timestep_indices"], state["token_tags"], strict=True)
        ):
            assert state["adaln_indices"][row] == time * 3 + tag
    assert [after["packed_payload"][row] for row in inverse] == before["packed_payload"]


@pytest.mark.parametrize("case_id", sorted(CASE_IDS))
def test_sealed_upstream_shapes_and_single_source_ownership(case_id):
    case = CASES[case_id]
    expected = case["expected"]
    fields = expected["fields"]
    size = expected["sequence_length"]
    assert fields["position_ids"]["dtype"] == "float64"
    assert decode(fields["position_ids"]).shape == (size, 3)
    assert np.isfinite(decode(fields["position_ids"])).all()
    for name in ("token_tags", "timestep_indices"):
        assert fields[name]["dtype"] == "int64"
        assert decode(fields[name]).shape == (size,)
    owners = [None] * size
    for stream in ("text", "video", "audio"):
        selector = fields[f"{stream}_indices"]
        assert selector["dtype"] == "int64"
        assert len(selector["shape"]) == 1
        for slot, row in enumerate(decode(selector)):
            assert 0 <= row < size
            assert owners[row] is None
            owners[row] = (stream, slot)
    assert all(owner is not None for owner in owners)
    assert len(decode(fields["text_indices"])) == len(case["layout_inputs"]["text_token_tags"])


@pytest.mark.parametrize("case_id", sorted(CASE_IDS))
def test_sealed_times_and_tags_follow_source_slots_not_packed_positions(case_id):
    case = CASES[case_id]
    expected, inputs = case["expected"], case["layout_inputs"]
    fields = {name: decode(field) for name, field in expected["fields"].items()}
    times, references = fields["timestep"], fields["timestep_indices"]
    assert times.dtype == np.dtype("<f4")
    assert np.isfinite(times).all() and ((0 <= times) & (times <= 1)).all()
    assert (times[1:] > times[:-1]).all()
    assert ((0 <= references) & (references < len(times))).all()
    for stream in ("text", "video", "audio"):
        for slot, row in enumerate(fields[f"{stream}_indices"]):
            if stream == "text":
                wanted_tag = inputs["text_token_tags"][slot]
                wanted_time = case["timestep_inputs"]["video_timestep"]
            else:
                wanted_tag = inputs[f"{stream}_tag"]
                count = expected[f"num_condition_{stream}_rows"]
                time_key = f"condition_{stream}_timestep" if slot < count else f"{stream}_timestep"
                wanted_time = case["timestep_inputs"][time_key]
            assert fields["token_tags"][row] == wanted_tag
            assert times[references[row]] == np.float32(wanted_time)


@pytest.mark.parametrize("case_id", ["fl2va_text_visual_tag", "ref2va_image", "ref2va_mixed"])
def test_text_stream_visual_rows_are_not_selected_video_outputs(case_id):
    fields = {name: decode(field) for name, field in CASES[case_id]["expected"]["fields"].items()}
    row = int(fields["text_indices"][1])
    assert fields["token_tags"][row] == 0
    assert row not in fields["video_indices"]
    assert row not in fields["audio_indices"]


def test_coverage_includes_equal_times_four_distinct_times_and_fractional_coordinates():
    assert CASES["fl2va_same_timestep"]["expected"]["fields"]["timestep"]["shape"] == [1]
    assert CASES["ref2va_mixed"]["expected"]["fields"]["timestep"]["shape"] == [4]
    positions = decode(CASES["fl2va_last_16_frames"]["expected"]["fields"]["position_ids"])
    assert np.any(positions != np.floor(positions))


def test_existing_tiny_position_and_tag_gold_is_preserved():
    fields = CASES["fl2va_block_tiny"]["expected"]["fields"]
    payload = decode(fields["position_ids"]).tobytes() + decode(fields["token_tags"]).tobytes()
    # Already sealed in tests/models/minimax_h3/test_h3_one_block.py before this work.
    assert hashlib.sha256(payload).hexdigest() == (
        "5329a0a13a9996c4fea7a0e676b65aa24302d65bea2e2248ba68d12184cc7c37"
    )


def test_generator_refuses_unknown_source_without_writing_output(tmp_path):
    source = tmp_path / "before_denoise.py"
    output = tmp_path / "fixture.json"
    source.write_text("raise RuntimeError('untrusted source must not execute')\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "make_upstream_fixtures.py"),
            "--upstream-file",
            str(source),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode != 0
    assert "upstream source hash mismatch" in result.stderr
    assert "RuntimeError: untrusted source" not in result.stderr
    assert not output.exists()
