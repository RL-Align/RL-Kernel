# SPDX-License-Identifier: Apache-2.0
"""CPU verifier controls do not represent model or GPU consistency results."""

import hashlib
import json
import struct

import pytest

from rl_engine.integrations.common.response_bitwise import compare_fp32_bytes, verify_saved_pair


def test_one_bit_negative_control_and_signed_zero_are_detected():
    original = struct.pack("<ff", -1.0, 0.0)
    modified = bytearray(original)
    modified[0] ^= 1
    result = compare_fp32_bytes(original, modified, 2)
    assert result["status"] == "FAIL" and result["mismatch_count"] == 1
    assert result["first_difference"]["token_index"] == 0
    signed_zero = compare_fp32_bytes(struct.pack("<f", 0.0), struct.pack("<f", -0.0), 1)
    assert signed_zero["status"] == "FAIL" and signed_zero["max_abs_diff"] == 0


@pytest.mark.parametrize(
    "data,count",
    [
        (b"", 0),
        (b"\0" * 3, 1),
        (struct.pack("<f", float("nan")), 1),
        (struct.pack("<f", float("inf")), 1),
    ],
)
def test_incomplete_empty_or_nonfinite_scores_fail(data, count):
    with pytest.raises(ValueError):
        compare_fp32_bytes(data, data, count)


def test_replay_rejects_missing_samples_changed_snapshot_and_shared_cache(tmp_path):
    suite = dict(
        samples=[
            dict(
                id="A",
                prompt_ids=[1, 2],
                response_ids=[3],
                position_ids=[0, 1, 2],
                mask="full-causal-unpadded-all-response-tokens",
            )
        ]
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    data = struct.pack("<f", -1.25)
    (tmp_path / "left.f32").write_bytes(data)
    (tmp_path / "right.f32").write_bytes(data)
    common = dict(
        input_suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        snapshot_sha256="a" * 64,
        dtype="float32",
        encoding="little-endian-ieee754",
        temperature=1,
        top_k=-1,
        top_p=1,
        producer=dict(engine="unit-test-control"),
    )

    def write(path, filename, **changes):
        value = dict(
            common,
            rows=[
                dict(
                    id="A",
                    response_ids=[3],
                    expected_tokens=1,
                    file=filename,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            ],
        )
        value.update(changes)
        path.write_text(json.dumps(value))

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write(left, "left.f32")
    write(right, "right.f32")
    result = verify_saved_pair(suite_path, left, right, "a" * 64)
    assert (
        result["status"] == "PASS" and result["expected_tokens"] == result["compared_tokens"] == 1
    )
    write(right, "right.f32", rows=[])
    with pytest.raises(ValueError, match="missing, duplicate"):
        verify_saved_pair(suite_path, left, right, "a" * 64)
    write(right, "right.f32", snapshot_sha256="b" * 64)
    with pytest.raises(ValueError, match="same snapshot"):
        verify_saved_pair(suite_path, left, right, "a" * 64)
    write(right, "left.f32")
    with pytest.raises(ValueError, match="same score file"):
        verify_saved_pair(suite_path, left, right, "a" * 64)
