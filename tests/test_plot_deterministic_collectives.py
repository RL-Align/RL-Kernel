# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks import plot_deterministic_collectives as plotter  # noqa: E402


def _write_report(path: Path, world_size: int, sizes: list[int], latency: float) -> None:
    """Write a report whose latencies identify each size after sorting."""
    report = {
        "world_size": world_size,
        "gpu": "test GPU",
        "torch": "test",
        "cuda": "test",
        "dtype": "float32",
        "rows": [
            {
                "operation": "all_gather",
                "input_bytes_per_rank": size,
                "deterministic_us": latency + size / 1024,
                "nccl_us": size / 1024,
            }
            for size in sizes
        ],
    }
    path.write_text(json.dumps(report))


@pytest.mark.parametrize(
    ("after_world_size", "after_sizes", "message"),
    [
        (4, [1024, 2048], "different world sizes"),
        (2, [1024, 4096], "different input sizes for all_gather"),
        (2, [1024], "different input sizes for all_gather"),
    ],
)
def test_mismatched_reports_fail_before_plotting(
    tmp_path, monkeypatch, after_world_size, after_sizes, message
):
    """Reject differing rank counts or input sweeps before creating a figure."""
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    _write_report(before_path, 2, [1024, 2048], 10)
    _write_report(after_path, after_world_size, after_sizes, 5)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_deterministic_collectives.py",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--output",
            str(tmp_path / "figure.png"),
        ],
    )

    def unexpected_subplots(*args, **kwargs):
        """Fail if invalid reports reach the plotting stage."""
        pytest.fail("mismatched reports must be rejected before creating a figure")

    monkeypatch.setattr(plotter.plt, "subplots", unexpected_subplots)
    with pytest.raises(SystemExit, match=message):
        plotter.main()


def test_reordered_rows_keep_latencies_aligned(tmp_path, monkeypatch):
    """Plot matching sweeps in size order even when source rows are reordered."""
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    output = tmp_path / "figure.png"
    _write_report(before_path, 2, [2048, 1024], 10)
    _write_report(after_path, 2, [1024, 2048], 5)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_deterministic_collectives.py",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--output",
            str(output),
        ],
    )
    figure, axes = plotter.plt.subplots(1, 1, squeeze=False)
    monkeypatch.setattr(plotter.plt, "subplots", lambda *args, **kwargs: (figure, axes))
    try:
        plotter.main()
        before, after, reference = axes[0, 0].get_lines()
        assert list(before.get_xdata()) == [1, 2]
        assert list(before.get_ydata()) == [11, 12]
        assert list(after.get_xdata()) == [1, 2]
        assert list(after.get_ydata()) == [6, 7]
        assert list(reference.get_ydata()) == [1, 2]
        assert output.is_file()
    finally:
        plotter.plt.close(figure)
