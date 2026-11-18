# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete numerical acceptance even when the engine process exits zero."""

import json
import sys
from pathlib import Path


def verify_result(result):
    for name in (
        "before_passed",
        "after_passed",
        "pure_rlk_passed",
        "update_executed",
        "same_rollout_process",
        "expanded_coverage_passed",
    ):
        if result.get(name) is not True:
            raise ValueError(f"four-layer acceptance incomplete or failed: {name}")
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("four-layer acceptance requires nonempty comparison rows")
    for index, row in enumerate(rows):
        expected = row.get("expected_tokens")
        compared = row.get("compared_tokens")
        mismatch = row.get("mismatch_count")
        if (
            type(expected) is not int
            or expected <= 0
            or type(compared) is not int
            or compared != expected
            or type(mismatch) is not int
            or mismatch != 0
            or row.get("all_finite") is not True
            or row.get("dtype") != "torch.float32"
        ):
            raise ValueError(f"invalid original-FP32 response comparison at row {index}")
    return len(rows)


if __name__ == "__main__":
    count = verify_result(json.loads(Path(sys.argv[1]).read_text()))
    print(f"CYCLE_ACCEPTANCE_PASS comparison_rows={count}")
