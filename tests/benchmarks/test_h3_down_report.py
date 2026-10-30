# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Inspect rendered accuracy series, including real nonfinite comparison failures."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from rl_engine.validation.operators.h3_down_validation import comparison

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "h3_report_under_test", ROOT / "benchmarks/operators/gemm/plot_h3_down_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_failure_is_flagged_and_never_plotted_as_zero(value):
    failed = comparison(torch.tensor([value]), torch.zeros(1), "forward_accuracy")
    assert not failed["passed"] and failed["error"] == "nonfinite tensor"
    assert "max_error_over_tolerance" not in failed
    checks = [dict(failed, tensor="y", transform="fp32_reference")]
    checks += [
        dict(tensor=tensor, transform="fp32_reference", passed=True, max_error_over_tolerance=0.5)
        for tensor in ("dx", "dw")
    ]
    fig, axis = plt.subplots()
    try:
        report.plot_reference_accuracy(axis, [{"rows": 129, "checks": checks}])
        lines = {line.get_label(): line for line in axis.lines}
        assert math.isnan(lines["y"].get_ydata()[0])
        assert lines["dx"].get_ydata()[0] == lines["dw"].get_ydata()[0] == 0.5
        (flag,) = axis.collections
        assert flag.get_label() == "y: failed / unavailable"
        assert flag.get_offsets()[0, 0] == 129
        assert flag.get_offset_transform() == axis.get_xaxis_transform()
        assert flag.get_offsets()[0, 1] > 1  # Outside the numerical plot height.
    finally:
        plt.close(fig)


def test_measured_failure_keeps_error_ratio_and_missing_checks_are_unavailable():
    checks = [
        dict(tensor="y", transform="fp32_reference", passed=False, max_error_over_tolerance=2.5),
        dict(tensor="dx", transform="fp32_reference", passed=True, max_error_over_tolerance=0.0),
    ]
    fig, axis = plt.subplots()
    try:
        report.plot_reference_accuracy(axis, [{"rows": 33, "checks": checks}])
        lines = {line.get_label(): line for line in axis.lines}
        assert lines["y"].get_ydata()[0] == 2.5
        assert lines["dx"].get_ydata()[0] == 0.0
        assert math.isnan(lines["dw"].get_ydata()[0])
        assert {flag.get_label() for flag in axis.collections} == {
            "y: failed / unavailable",
            "dw: failed / unavailable",
        }
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("backend", "prefix"),
    [
        ("rlkernel.h3_ffn_down.triton.fp32_tiles.v1", "candidate"),
        ("rlkernel.h3_ffn_down.sm90.midtree.v1", "native"),
        (None, "candidate"),
    ],
)
def test_report_preserves_backend_identity_in_table_and_plot(
    tmp_path, monkeypatch, backend, prefix
):
    case = {
        "input_family": "random",
        "rows": 33,
        "checks": [],
        "benchmarks": {
            f"{prefix}_forward": {"median_ms": 2.0},
            f"{prefix}_forward_backward": {"median_ms": 6.0},
            "torch_bf16_forward": {"median_ms": 1.0},
            "torch_bf16_forward_backward": {"median_ms": 3.0},
        },
    }
    payload = {"status": "failed", "cases": [case]}
    if backend is not None:
        payload["backend"] = backend
    source = tmp_path / "report.json"
    source.write_text(json.dumps(payload))
    output = tmp_path / "rendered"
    monkeypatch.setattr(
        sys, "argv", ["plot_h3_down_report.py", str(source), "--output-dir", str(output)]
    )
    timing_series = []

    def inspect_figure(figure, *args, **kwargs):
        timing_series.extend(
            (line.get_label(), list(line.get_ydata())) for line in figure.axes[0].lines
        )

    monkeypatch.setattr(plt.Figure, "savefig", inspect_figure)
    report.main()

    label = f"Candidate ({backend})" if backend else "Candidate"
    summary = (output / "summary.md").read_text()
    assert f"| {label} fwd ms |" in summary
    assert "| random | 33 | 0 | 0 | 2.0000 | 1.0000 |" in summary
    assert "Run status: **failed**." in summary
    assert timing_series == [
        (f"{label} forward", [2.0]),
        ("torch BF16 forward", [1.0]),
        (f"{label} forward+backward", [6.0]),
        ("torch BF16 forward+backward", [3.0]),
    ]
