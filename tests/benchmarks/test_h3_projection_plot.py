# SPDX-License-Identifier: Apache-2.0
"""Check plot data preparation without a plotting runtime or GPU."""

import math

from benchmarks.operators.gemm.plot_h3_projections import _accuracy_series


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
