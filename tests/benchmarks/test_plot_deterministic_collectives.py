# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks.distributed import plot_deterministic_collectives as plotter  # noqa: E402


def _write_report(
    path: Path,
    world_size: int,
    sizes: list[int],
    latency: float,
    slope: float = 1 / 1024,
    nccl_offset: float = 0.0,
) -> None:
    """Write a report whose latencies identify each size after sorting.

    ``slope`` sets how deterministic latency grows with size (negative: it shrinks, so the
    latency order differs from the size order); ``nccl_offset`` tells two reports' NCCL apart.
    """
    report = {
        "world_size": world_size,
        "gpu": "test GPU",
        "torch": "test",
        "cuda": "test",
        "dtype": "float32",
        "warmup": 1,
        "iterations": 2,
        "rows": [
            {
                "operation": "all_gather",
                "input_bytes_per_rank": size,
                "deterministic_us": latency + slope * size,
                "nccl_us": size / 1024 + nccl_offset,
            }
            for size in sizes
        ],
    }
    path.write_text(json.dumps(report))


@pytest.mark.parametrize(
    ("after_world_size", "after_sizes", "after_metadata", "message"),
    [
        (4, [1024, 2048], {}, "different world sizes"),
        (2, [1024, 4096], {}, "different input sizes for all_gather"),
        (2, [1024], {}, "different input sizes for all_gather"),
        (2, [1024, 2048], {"gpu": "different GPU"}, "different gpu values"),
        (2, [1024, 2048], {"dtype": "bf16"}, "different dtype values"),
    ],
)
def test_mismatched_reports_fail_before_plotting(
    tmp_path, monkeypatch, after_world_size, after_sizes, after_metadata, message
):
    """Reject differing workloads or input sweeps before creating a figure."""
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    _write_report(before_path, 2, [1024, 2048], 10)
    _write_report(after_path, after_world_size, after_sizes, 5)
    report = json.loads(after_path.read_text())
    report.update(after_metadata)
    after_path.write_text(json.dumps(report))
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


@pytest.mark.parametrize("field", ["torch", "cuda", "dtype"])
def test_environment_must_match_across_world_sizes(tmp_path, monkeypatch, field):
    """Reject a multi-panel figure whose reports record different environments."""
    paths = {name: tmp_path / f"{name}.json" for name in ("b2", "a2", "b4", "a4")}
    for name, world_size in (("b2", 2), ("a2", 2), ("b4", 4), ("a4", 4)):
        _write_report(paths[name], world_size, [1024, 2048], 10 if name[0] == "b" else 5)
    # Change both reports of the second pair: each pair stays consistent (so the pair check
    # passes) and only the comparison with the first world size can reject it.
    for name in ("b4", "a4"):
        report = json.loads(paths[name].read_text())
        report[field] = "different"
        paths[name].write_text(json.dumps(report))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_deterministic_collectives.py",
            "--before",
            str(paths["b2"]),
            str(paths["b4"]),
            "--after",
            str(paths["a2"]),
            str(paths["a4"]),
            "--output",
            str(tmp_path / "figure.png"),
        ],
    )

    def unexpected_subplots(*args, **kwargs):
        """Fail if reports from different environments reach the plotting stage."""
        pytest.fail("reports from different environments must be rejected")

    monkeypatch.setattr(plotter.plt, "subplots", unexpected_subplots)
    with pytest.raises(
        SystemExit,
        match=f"{re.escape(str(paths['a4']))} and {re.escape(str(paths['a2']))} "
        f"have different {field} values",
    ):
        plotter.main()


@pytest.mark.parametrize(
    ("before_operation", "after_operation"),
    [
        ("all_reduce", "all_gather"),
        ("all_gather", "all_reduce"),
        ("all_reduce", "all_reduce"),
    ],
)
def test_missing_operation_fails_before_plotting(
    tmp_path, monkeypatch, before_operation, after_operation
):
    """Reject a selected operation absent from either or both reports."""
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    for path, operation in ((before_path, before_operation), (after_path, after_operation)):
        _write_report(path, 2, [1024, 2048], 10)
        report = json.loads(path.read_text())
        for row in report["rows"]:
            row["operation"] = operation
        path.write_text(json.dumps(report))
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
        """Fail if an empty selected series reaches the plotting stage."""
        pytest.fail("reports missing the selected operation must fail before plotting")

    monkeypatch.setattr(plotter.plt, "subplots", unexpected_subplots)
    with pytest.raises(SystemExit, match="need rows for all_gather in both reports"):
        plotter.main()


def test_reordered_rows_keep_latencies_aligned(tmp_path, monkeypatch):
    """Plot matching sweeps in size order even when source rows are reordered."""
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    output = tmp_path / "figure.png"
    # Latency falls with size and the NCCL series differ, so sorting by latency, or taking the
    # reference from the before report, would change the asserted series.
    _write_report(before_path, 2, [2048, 1024], 100, slope=-1 / 1024, nccl_offset=100)
    _write_report(after_path, 2, [1024, 2048], 50, slope=-1 / 1024)
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
        assert list(before.get_ydata()) == [99, 98]
        assert list(after.get_xdata()) == [1, 2]
        assert list(after.get_ydata()) == [49, 48]
        assert list(reference.get_ydata()) == [1, 2]
        assert output.is_file()
    finally:
        plotter.plt.close(figure)
