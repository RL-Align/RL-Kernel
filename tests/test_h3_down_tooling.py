# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI error evidence and report rendering checks without a GPU."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_validator_refuses_cpu_and_preserves_error_evidence(tmp_path, monkeypatch):
    import torch

    spec = importlib.util.spec_from_file_location(
        "h3_validator_under_test", ROOT / "scripts/validate_h3_ffn_down.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    # Exercise the CPU branch without reinitializing this host's WSL CUDA driver.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report_path = tmp_path / "report.json"
    status = validator.main(
        [
            "--rows",
            "1",
            "--families",
            "random",
            "--output",
            str(report_path),
        ],
    )
    assert status == 2
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert not report["acceptance_complete"]
    assert not report["cases"]
    assert "SM90 GPU is required" in report["error"]
    assert report["provenance"]["contract_version"] == "ws1-c1-v2"
    assert report["provenance"]["source_hashes"]


def test_failed_run_renders_summary_without_inventing_charts(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps({"status": "error", "cases": [], "error": "no SM90 GPU"}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plot_h3_down_report.py"),
            str(source),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "**error**" in (tmp_path / "summary.md").read_text()
    assert "no SM90 GPU" in (tmp_path / "summary.md").read_text()
    assert list(tmp_path.glob("*.png")) == []


def test_plot_pipeline_with_explicitly_synthetic_test_data(tmp_path):
    checks = [
        {
            "judgment": "forward_accuracy" if name == "y" else "gradient_accuracy",
            "tensor": name,
            "transform": "fp32_reference",
            "passed": True,
            "max_error_over_tolerance": 0.5,
            "max_abs_error": 0.01,
        }
        for name in ("y", "dx", "dw")
    ]
    keys = (
        "candidate_forward",
        "candidate_forward_backward",
        "torch_bf16_forward",
        "torch_bf16_forward_backward",
    )
    source = tmp_path / "synthetic-test-report.json"
    source.write_text(
        json.dumps(
            {
                "status": "test_fixture",
                "cases": [
                    {
                        "rows": 1,
                        "input_family": "synthetic_test_data",
                        "checks": checks,
                        "benchmarks": {key: {"median_ms": 1.0} for key in keys},
                    }
                ],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plot_h3_down_report.py"),
            str(source),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "synthetic_test_data.png").read_bytes().startswith(b"\x89PNG")
    assert "synthetic_test_data" in (tmp_path / "checks.csv").read_text()
