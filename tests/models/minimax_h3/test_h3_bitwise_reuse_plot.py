# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A heatmap cell counts every forward output of the operator, not only the first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))

from plot_h3_bitwise_reuse import _cell  # noqa: E402


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
