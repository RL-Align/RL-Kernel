# SPDX-License-Identifier: Apache-2.0
"""The launcher must propagate numerical rejection after a zero-exit engine."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.operators.dsv4_4l.verify_result import verify_result


def accepted_result():
    return dict(
        before_passed=True,
        after_passed=True,
        pure_rlk_passed=True,
        update_executed=True,
        same_rollout_process=True,
        expanded_coverage_passed=True,
        rows=[
            dict(
                expected_tokens=2,
                compared_tokens=2,
                mismatch_count=0,
                dtype="torch.float32",
                all_finite=True,
            )
        ],
    )


def test_complete_result_is_accepted():
    assert verify_result(accepted_result()) == 1


@pytest.mark.parametrize(
    "flag",
    [
        "before_passed",
        "after_passed",
        "pure_rlk_passed",
        "update_executed",
        "same_rollout_process",
        "expanded_coverage_passed",
    ],
)
def test_failed_stage_is_rejected(flag):
    result = accepted_result()
    result[flag] = False
    with pytest.raises(ValueError, match=flag):
        verify_result(result)


@pytest.mark.parametrize(
    "change",
    [
        dict(compared_tokens=1),
        dict(mismatch_count=1),
        dict(expected_tokens=0, compared_tokens=0),
        dict(all_finite=False),
        dict(dtype="torch.bfloat16"),
    ],
)
def test_invalid_comparison_is_rejected(change):
    result = accepted_result()
    result["rows"][0].update(change)
    with pytest.raises(ValueError, match="row 0"):
        verify_result(result)


@pytest.mark.parametrize("result", [{}, dict(accepted_result(), rows=[])])
def test_missing_acceptance_evidence_is_rejected(result):
    with pytest.raises(ValueError):
        verify_result(result)


def test_cli_exits_nonzero_for_a_completed_but_mismatched_cycle(tmp_path):
    result = accepted_result()
    result["rows"][0]["mismatch_count"] = 1
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    script = Path(__file__).resolve().parents[2] / "examples/operators/dsv4_4l/verify_result.py"
    process = subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True
    )
    assert process.returncode != 0
    assert "row 0" in process.stderr
    assert "CYCLE_ACCEPTANCE_PASS" not in process.stdout
