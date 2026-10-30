# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The block plot draws the error panels also when a node matches the golden bitwise."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.validation.models import plot_h3_block as plot  # noqa: E402


def _panel(cand, prov):
    fig, ax = plt.subplots()
    try:
        plot._err_panel(ax, [f"n{i}" for i in range(len(cand))], cand, prov, "t")
        return [t.get_text() for t in ax.get_yticklabels()], ax.get_ylim()
    finally:
        plt.close(fig)


def test_a_zero_error_is_drawn_at_the_floor_and_labelled_zero():
    labels, _ = _panel([0.0, 1e-3], [2e-3, 1e-2])
    assert labels[0] == "0"
    assert labels[-1] == "1e-2"


def test_panels_without_zero_errors_keep_their_decade_labels():
    labels, _ = _panel([1e-4, 1e-3], [2e-4, 1e-2])
    assert "0" not in labels
    assert labels[0] == "1e-4"
