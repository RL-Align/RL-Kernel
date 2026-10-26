# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A heatmap cell counts every forward output of the operator, not only the first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))

from h3_bitwise_reuse import _compare_case  # noqa: E402
from plot_h3_bitwise_reuse import _cell, _panel  # noqa: E402


def _out(differ: int, numel: int = 100) -> dict:
    return {"equal": differ == 0, "differ": differ, "numel": numel}


def test_cell_sums_over_all_outputs():
    assert _cell({"out": [_out(0), _out(0), _out(0)]}) == 0.0
    # a difference that is only in the fifth of six outputs is not hidden by output 0
    assert _cell({"out": [_out(0)] * 4 + [_out(30)] + [_out(0)]}) == pytest.approx(30 / 600)


def test_cell_marks_shape_or_dtype_mismatch_as_full_difference():
    assert _cell({"out": [_out(0), {"note": "shape mismatch"}]}) == 1.0


def test_cell_without_data_is_not_applicable():
    assert _cell(None) is None
    assert _cell({"out": []}) is None
    assert _cell({"out": [None]}) is None


def test_a_case_the_reference_never_saved_is_unavailable_and_does_not_abort_the_comparison():
    import torch

    run = {"out": [torch.ones(3)], "grads": {"x": torch.zeros(2)}}
    ref = {"out": [torch.ones(3)], "grads": {"x": torch.zeros(2)}}
    done = _compare_case(run, ref)
    assert done["out"][0]["equal"] and done["grads"]["x"]["equal"]
    missing = _compare_case(run, None)
    assert "unavailable" in missing and _cell(missing) is None


def test_missing_comparisons_are_drawn_grey_not_as_bitwise_equal():
    import matplotlib.pyplot as plt

    def result(frac_differ):
        return {"out": [_out(frac_differ)], "grads": {}}

    report = {
        "op": "probe",
        "rl_kernel_commit": "0123456789",
        "cases": ["a", "b"],
        "results": {
            "rl_kernel": {"vs diffusers": {"a": result(0), "b": {"unavailable": "no run"}}},
            "diffusers": {},
        },
    }
    fig, ax = plt.subplots()
    _panel(ax, report)
    grid = ax.images[0].get_array()
    assert grid.mask[0, 1] and not grid.mask[0, 0]  # the made comparison is data, the other masked
    assert ax.images[0].cmap(ax.images[0].norm(0.0)) != ax.images[0].cmap._rgba_bad
    texts = [t.get_text() for t in ax.texts]
    assert "n/a" in texts and "0" in texts
    plt.close(fig)
