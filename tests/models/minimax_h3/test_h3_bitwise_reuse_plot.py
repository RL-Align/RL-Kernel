# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A heatmap cell counts every forward output of the operator, not only the first.

Also: the TP evidence figure for a report that has no TP > 1 run.
"""

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


def test_a_different_number_of_outputs_is_a_difference_not_a_shorter_comparison():
    import torch

    ones = torch.ones(3)
    run = {"out": [ones, ones], "grads": {"x": torch.zeros(2)}}
    ref = {"out": [ones], "grads": {"x": torch.zeros(2)}}
    done = _compare_case(run, ref)
    # the extra output is not dropped (a zip would call this case bitwise equal) ...
    assert done["out"] == [{"equal": False, "note": "2 vs 1 outputs"}]
    assert _cell(done) == 1.0
    # ... and the gradients are still compared.
    assert done["grads"]["x"]["equal"]
    same = _compare_case(run, {"out": [ones, ones], "grads": {"x": torch.zeros(2)}})
    assert [o["equal"] for o in same["out"]] == [True, True]


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


def _tp_run(tp: int) -> dict:
    perf = {
        "num_timesteps": 3,
        "tp_forward_us": 110.0,
        "tp_forward_backward_us": 2000.0,
        "ws1_forward_us": 100.0,
        "ws1_forward_backward_us": 1900.0,
    }
    equal = dict.fromkeys(("table", "d_temb", "d_weight_shard", "d_bias_shard"), True)
    return {
        "tp": tp,
        "ranks": [{"rank": 0, "equality": [{"num_timesteps": 3, **equal}], "perf": [perf]}],
    }


def _tp_report(worlds: list[int]) -> dict:
    return {
        "op": "tp_adaln_3mod",
        "environment": {"gpu": "gpu", "torch": "t", "cuda": "c"},
        "rl_kernel_commit": "0123456789",
        "model_revision": "0123456789",
        "runs": [_tp_run(tp) for tp in worlds],
    }


@pytest.mark.parametrize("worlds", [[1], [1, 2, 4], [2, 4]])
def test_tp_figure_for_reports_with_and_without_tp_above_one(worlds):
    import matplotlib.pyplot as plt
    from plot_h3_evidence import plot_tp_adaln

    fig = plot_tp_adaln(_tp_report(worlds))
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert ("no run with TP > 1 in this report" in texts) == (worlds == [1])
    plt.close(fig)


def _sp_run(sp: int) -> dict:
    case = {
        "layout": "block",
        "seq": 64,
        "batch": 1,
        "equal": dict.fromkeys(("out", "d_residual", "d_y", "d_weight", "d_table"), True),
        "sp_forward_backward_us": 1500.0,
        "ws1_forward_backward_us": 1900.0,
    }
    return {"sp": sp, "ranks": [{"rank": 0, "cases": [case]}]}


def _sp_report(worlds: list[int]) -> dict:
    naive = {"d_weight": 0.5, "d_table": 0.25}
    return {
        "op": "sp_norm_adaln",
        "environment": {"gpu": "gpu", "gpus": 1, "torch": "t", "cuda": "c"},
        "rl_kernel_commit": "0123456789",
        "model_revision": "0123456789",
        "runs": [_sp_run(sp) for sp in worlds],
        "naive_sp": {"sp": max(worlds), "block": naive, "interleaved": naive},
    }


@pytest.mark.parametrize("worlds", [[1], [1, 2, 4], [2, 4]])
def test_sp_figure_for_reports_with_and_without_sp_above_one(worlds):
    import matplotlib.pyplot as plt
    from plot_h3_evidence import plot_sp_norm

    fig = plot_sp_norm(_sp_report(worlds))
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert ("no run with SP > 1 in this report" in texts) == (worlds == [1])
    plt.close(fig)
