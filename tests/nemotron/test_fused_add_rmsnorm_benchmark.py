# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark CLI, measurement scopes, correctness gates, and report generation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm import NativeFusedAddRMSNormOp


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_fused_add_rmsnorm.py"
    spec = importlib.util.spec_from_file_location("fused_rmsnorm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_requires_no_gpu_and_writes_no_reports(benchmark, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    folder = tmp_path / "reports"
    benchmark.main(["--dry-run", "--output-dir", str(folder)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["case_count"] == 15
    assert plan["measurement_count"] == 45
    assert plan["shapes"] == [[1, 2688], [32, 2688], [8192, 2688], [16384, 2688], [65536, 2688]]
    assert not folder.exists()


@pytest.mark.parametrize("argument", ["--warmup", "--repeat"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_counts(benchmark, argument, value):
    with pytest.raises(SystemExit):
        benchmark._parse_args([argument, value])


@pytest.mark.parametrize("shape", ["0x128", "3x0", "-1x8", "3x4x5", "invalid"])
def test_cli_rejects_invalid_shapes(benchmark, shape):
    with pytest.raises(SystemExit):
        benchmark._parse_args([f"--shapes={shape}"])


def test_cli_deduplicates_plan_and_preserves_order(benchmark):
    args = benchmark._parse_args(
        [
            "--shapes",
            "3x129",
            "3x129",
            "8x2688",
            "--dtypes",
            "bf16",
            "bf16",
            "fp32",
            "--modes",
            "backward",
            "backward",
            "forward",
        ]
    )
    assert args.output_dir == Path("reports/fused-add-rmsnorm")
    assert benchmark._case_plan(args) == {
        "dtypes": ["bf16", "fp32"],
        "shapes": [[3, 129], [8, 2688]],
        "modes": ["backward", "forward"],
        "case_count": 4,
        "measurement_count": 8,
    }


@pytest.mark.parametrize(
    "mode,expected_calls", [("forward", 2), ("backward", 1), ("forward_backward", 2)]
)
def test_workloads_have_correct_autograd_scope_without_accumulation(
    benchmark, mode, expected_calls
):
    inputs = (torch.randn(3, 7), torch.randn(3, 7), torch.randn(7))
    upstream = (torch.randn(3, 7), torch.randn(3, 7))
    native = NativeFusedAddRMSNormOp()
    calls = []

    def op(*values):
        calls.append((torch.is_grad_enabled(), values))
        return native(*values)

    fn = benchmark._make_workload(op, inputs, upstream, mode)
    assert fn() is None
    assert fn() is None
    assert len(calls) == expected_calls
    for enabled, leaves in calls:
        assert enabled == (mode != "forward")
        for original, leaf in zip(inputs, leaves, strict=True):
            assert leaf.data_ptr() == original.data_ptr()
            assert leaf.grad is None and original.grad is None
            assert leaf.requires_grad == (mode != "forward")


@pytest.mark.parametrize("broken_branch", [None, "y", "updated_residual"])
def test_accuracy_gate_checks_gradients_of_both_outputs(benchmark, broken_branch):
    inputs = (torch.randn(5, 7), torch.randn(5, 7), torch.randn(7))
    upstream = (torch.randn(5, 7), torch.randn(5, 7))
    native = NativeFusedAddRMSNormOp()

    def candidate(*values):
        y, updated = native(*values)
        # Preserve the value exactly while corrupting one output's derivative.
        if broken_branch == "y":
            y = y * 0.5 + y.detach() * 0.5
        if broken_branch == "updated_residual":
            updated = updated * 0.5 + updated.detach() * 0.5
        return y, updated

    if broken_branch:
        with pytest.raises(AssertionError):
            benchmark._check_accuracy(native, candidate, inputs, upstream)
    else:
        errors = benchmark._check_accuracy(native, candidate, inputs, upstream)
        assert set(errors) == {"y", "updated_residual", "grad_x", "grad_residual", "grad_weight"}
        assert all(entry["max_abs_error"] == 0 for entry in errors.values())


@pytest.mark.parametrize("latency", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timer_results_cannot_be_reported(benchmark, latency):
    profiler = SimpleNamespace(_time_kernel=lambda fn: (None, latency, None))
    with pytest.raises(RuntimeError, match="Invalid measured latency"):
        benchmark._measure(profiler, lambda: None)


def test_no_gpu_fails_before_writing_results(benchmark, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA or ROCm"):
        benchmark.main(["--output-dir", str(tmp_path / "reports")])
    assert not (tmp_path / "reports").exists()


def test_report_preserves_completion_latency_and_configuration(benchmark, tmp_path):
    args = benchmark._parse_args(
        ["--shapes", "16384x2688", "--dtypes", "bf16", "--modes", "forward_backward"]
    )
    payload = {
        "complete": False,
        "environment": {
            "gpu": "test GPU",
            "torch": "test",
            "triton": "test",
            "backend": "cuda",
            "runtime": "test",
        },
        "config": {"warmup": 10, "repeat": 50, "seed": 434},
        "case_plan": benchmark._case_plan(args),
        "results": [],
    }
    benchmark._save(payload, tmp_path)
    assert "Status: incomplete; 0/1 comparisons" in (tmp_path / "report.md").read_text()
    payload["results"].append(
        {
            "dtype": "bf16",
            "shape": [16384, 2688],
            "mode": "forward_backward",
            "triton_plan": {
                "strategy": "fused",
                "block_rows": 64,
                "block_cols": 64,
                "num_warps": 4,
            },
            "native": {"median_ms": 4.0, "std_ms": 0.1, "peak_extra_mib": 800.0},
            "triton": {"median_ms": 1.0, "std_ms": 0.02, "peak_extra_mib": 200.0},
            "speedup": 4.0,
        }
    )
    payload["complete"] = True
    benchmark._save(payload, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text()) == payload
    report = (tmp_path / "report.md").read_text()
    assert "Status: complete; 1/1 comparisons" in report
    assert "fused | forward_backward | 4.000000 | 1.000000 | 4.00x" in report
    assert "No torch.compile or CUDA Graph timing" in report
    assert "Extra peak allocation excludes inputs and prebuilt graphs" in report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")
def test_gpu_benchmark_compares_public_ops_and_records_accuracy(benchmark, tmp_path):
    payload = benchmark.run_benchmark(
        benchmark._parse_args(
            [
                "--shapes",
                "33x129",
                "--dtypes",
                "bf16",
                "--warmup",
                "1",
                "--repeat",
                "2",
                "--output-dir",
                str(tmp_path),
            ]
        )
    )
    assert payload["complete"]
    assert len(payload["results"]) == payload["case_plan"]["measurement_count"] == 3
    assert {result["mode"] for result in payload["results"]} == {
        "forward",
        "backward",
        "forward_backward",
    }
    for result in payload["results"]:
        assert result["train_inference_bitwise"]
        assert result["triton_plan"] == {
            "strategy": "tiled",
            "block_rows": 64,
            "block_cols": 64,
            "num_warps": 8,
        }
        assert set(result["errors"]) == {
            "y",
            "updated_residual",
            "grad_x",
            "grad_residual",
            "grad_weight",
        }
        for provider in ("native", "triton"):
            assert result[provider]["median_ms"] > 0
            assert result[provider]["std_ms"] >= 0
            assert result[provider]["peak_extra_mib"] >= 0
    assert json.loads((tmp_path / "results.json").read_text()) == payload
    assert (tmp_path / "report.md").is_file()
