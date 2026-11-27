# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The SGLang rollout chart tool on reports the 8-GPU run does not produce: one TP size, all-zero
or absent values, and a bound output that is not equal."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))

import plot_h3_sglang_rollout as plot  # noqa: E402

NATIVE = ("native_max_abs_diff", "bound_equals_canonical")
PREFIX = ("native_prefix_max_abs_diff", "bound_prefix_equals_rows")
EQUAL = "RL-Kernel bound output: 0 (bitwise) for every output at every TP size"


def _report(sizes=(1,), value=0.0, bound=True, prefix=True) -> dict:
    rank = {"native_max_abs_diff": value, "bound_equals_canonical": bound}
    if prefix:
        rank |= {"native_prefix_max_abs_diff": value, "bound_prefix_equals_rows": bound}
    # Each TP size gets its own rows with its own ranks, so a change at one TP leaves the others.
    return {
        "tp": {
            str(tp): {"rows": {k: {f"rank{r}": dict(rank) for r in range(tp)} for k in plot.LABELS}}
            for tp in sizes
        }
    }


@pytest.mark.parametrize("fields", [NATIVE, PREFIX])
def test_one_tp_size_with_all_zero_values_is_plotted(tmp_path, fields):
    sizes, table = plot.collect(_report(), *fields)
    out = tmp_path / "chart.png"
    plot.plot(sizes, table, title="t", xlabel="x", out=out)
    assert out.stat().st_size > 0
    assert "| refiner final norm | 0 | 0 (bitwise) at every TP |" in plot.markdown(
        sizes, table, value_header="v", bound_header="b"
    )
    assert plot.bound_line(sizes, table) == EQUAL


def test_a_report_without_prefix_rows_is_plotted_as_no_measurements(tmp_path):
    sizes, table = plot.collect(_report(sizes=(1, 2), prefix=False), *PREFIX)
    out = tmp_path / "chart.png"
    plot.plot(sizes, table, title="t", xlabel="x", out=out)
    assert out.stat().st_size > 0
    assert plot.bound_line(sizes, table) == "RL-Kernel bound output: not measured"
    assert plot.markdown(sizes, table, value_header="v", bound_header="b") == (
        "No measurements in this report."
    )


def test_the_subtitle_names_a_bound_output_that_is_not_equal():
    report = _report(sizes=(1, 2), value=2.0**-8)
    assert plot.bound_line(*plot.collect(report, *NATIVE)) == EQUAL
    report["tp"]["2"]["rows"]["fc2"]["rank1"]["bound_equals_canonical"] = False
    sizes, table = plot.collect(report, *NATIVE)
    assert plot.bound_line(sizes, table) == "RL-Kernel bound output: NOT EQUAL for fc2"
    lines = plot.markdown(sizes, table, value_header="v", bound_header="b").splitlines()
    assert "| `fc2` | 0.00390625 | 0.00390625 | NOT EQUAL |" in lines
    assert "| `fc1` | 0.00390625 | 0.00390625 | 0 (bitwise) at every TP |" in lines
