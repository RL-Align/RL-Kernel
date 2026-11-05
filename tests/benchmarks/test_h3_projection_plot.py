# SPDX-License-Identifier: Apache-2.0
"""Check plot data preparation without a plotting runtime or GPU."""

import math

from benchmarks.operators.gemm.plot_h3_projections import _accuracy_series, _load_cases


def test_accuracy_uses_all_families_and_marks_unavailable_ratios():
    def accuracy(ratio):
        return {"tensor": "y", "transform": "fp32_reference", **ratio}

    cases = [
        {
            "rows": 8,
            "input_family": "random",
            "checks": [accuracy({"max_error_over_tolerance": 0.2})],
        },
        {
            "rows": 8,
            "input_family": "checkpoint_fixture",
            "checks": [accuracy({"error": "nonfinite tensor"})],
        },
        {
            "rows": 16,
            "input_family": "checkpoint_fixture",
            "checks": [accuracy({"max_error_over_tolerance": 1.4})],
        },
        {"rows": 32, "input_family": "cancellation", "checks": []},
    ]

    sizes, values, unavailable = _accuracy_series(cases, "y")

    assert sizes == [8, 16, 32]
    assert values[:2] == [0.2, 1.4]
    assert math.isnan(values[2])
    assert unavailable == {8, 32}


def test_duplicate_size_keeps_failed_task_accuracy_and_full_report_timing():
    op = "h3_attention_o_gemm"
    size = 32768
    full_check = {"tensor": "y", "transform": "fp32_reference", "max_error_over_tolerance": 0.2}
    full = {"semantic_op": op, "input_family": "random", "rows": size, "checks": [full_check]}
    failed_check = {**full_check, "max_error_over_tolerance": 1.4, "passed": False}
    task = {**full, "checks": [failed_check]}
    bundle = {
        "artifacts": {
            "full.json": {"content": {"cases": [full]}},
            "tasks.json": {"content": {"cases": [task]}},
        }
    }

    cases, timing = _load_cases(bundle)

    assert _accuracy_series(cases, "y") == ([size], [1.4], set())
    assert timing == {op: {size: full}}
