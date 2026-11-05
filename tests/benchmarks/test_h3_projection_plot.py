# SPDX-License-Identifier: Apache-2.0
"""Check plot data preparation without a plotting runtime or GPU."""

import math

import pytest

from benchmarks.operators.gemm.plot_h3_projections import (
    _accuracy_series,
    _load_cases,
    _load_reports,
)


def test_accuracy_uses_all_families_and_distinguishes_failed_ratios():
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
            "checks": [accuracy({"passed": False, "error": "nonfinite tensor"})],
        },
        {
            "rows": 16,
            "input_family": "checkpoint_fixture",
            "checks": [accuracy({"max_error_over_tolerance": 1.4, "passed": False})],
        },
        {"rows": 32, "input_family": "cancellation", "checks": []},
        {
            "rows": 64,
            "input_family": "random",
            "checks": [accuracy({"max_error_over_tolerance": math.inf, "passed": False})],
        },
    ]

    sizes, values, failed, unavailable = _accuracy_series(cases, "y")

    assert sizes == [8, 16, 32, 64]
    assert values[:2] == [0.2, 1.4]
    assert all(math.isnan(value) for value in values[2:])
    assert failed == {8, 64}
    assert unavailable == {32}


@pytest.mark.parametrize("wrapped", [False, True])
def test_reports_keep_failed_task_accuracy_and_prefer_full_timing(wrapped):
    op = "h3_attention_o_gemm"
    size = 32768
    full_check = {"tensor": "y", "transform": "fp32_reference", "max_error_over_tolerance": 0.2}
    full = {"semantic_op": op, "input_family": "random", "rows": size, "checks": [full_check]}
    failed_check = {**full_check, "max_error_over_tolerance": 1.4, "passed": False}
    task = {**full, "checks": [failed_check]}
    full_report = {"schema": "h3-projections-qualification-v1", "cases": [full]}
    tasks_report = {"schema": "h3-projections-qualification-v1", "cases": [task]}
    if wrapped:
        bundle = {
            "artifacts": {
                "tasks.json": {"content": tasks_report},
                "full.json": {"content": full_report},
            }
        }
        reports = _load_reports(bundle)
    else:
        reports = _load_reports(full_report) + _load_reports(tasks_report)

    cases, timing = _load_cases(reports)

    assert _accuracy_series(cases, "y") == ([size], [1.4], set(), set())
    assert timing == {op: {size: full}}
