# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark timing boundaries, correctness gates and report completeness."""

import json
import sys

import pytest
import torch

from benchmarks import benchmark_softcapped_selected_logprob as benchmark
from benchmarks.benchmark_softcapped_selected_logprob import (
    _check_accuracy,
    _make_workload,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp


@pytest.mark.parametrize("mode", ("forward", "backward", "forward_backward"))
def test_workload_timing_and_random_upstream(mode):
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ids = torch.tensor([2, 0])
    upstream = torch.tensor([0.5, -2.0])
    native = NativeSoftcappedSelectedLogprobOp()
    gold_leaf = logits.clone().requires_grad_(True)
    expected = torch.autograd.grad(native(gold_leaf, ids), gold_leaf, grad_outputs=upstream)[0]
    calls, gradients = [], []

    def observed(leaf, token_ids):
        calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled() and len(calls) == 1:
            leaf.register_hook(lambda grad: gradients.append(grad.clone()))
        return native(leaf, token_ids)

    fn = _make_workload(observed, logits, ids, upstream, mode)
    assert len(calls) == (1 if mode == "backward" else 0)
    assert fn() is None and fn() is None
    assert len(calls) == (1 if mode == "backward" else 2)
    if mode == "forward":
        assert calls == [False, False] and not gradients
    else:
        assert len(gradients) == 2
        for grad in gradients:
            torch.testing.assert_close(grad, expected)
    assert logits.grad is None and not logits.requires_grad


def test_benchmark_rejects_cpu():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA or AMD ROCm GPU"):
        run_benchmark(args)


def test_accuracy_gate_rejects_incorrect_gradient():
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, token_ids):
            ctx.shape = logits.shape
            return NativeSoftcappedSelectedLogprobOp()(logits, token_ids)

        @staticmethod
        def backward(ctx, grad_output):
            return torch.ones(ctx.shape, device=grad_output.device), None

    with pytest.raises(AssertionError):
        _check_accuracy(
            NativeSoftcappedSelectedLogprobOp(),
            WrongGradient.apply,
            torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            torch.tensor([2, 0]),
            torch.tensor([0.5, -2.0]),
        )


@pytest.mark.parametrize(
    "arguments", [["--shapes", "0x10"], ["--shapes", "2x3x4"], ["--repeat", "0"]]
)
def test_benchmark_rejects_invalid_measurements(arguments):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(arguments)


def test_list_cases_needs_no_gpu_or_report_files(monkeypatch, capsys, tmp_path):
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--list-cases",
            "--dtypes",
            "bf16",
            "--shapes",
            "1x1025",
            "16x262144",
            "--modes",
            "forward",
            "--output-dir",
            str(output_dir),
        ],
    )

    def unexpected_gpu_run(args):
        pytest.fail("--list-cases must not enter the GPU benchmark")

    monkeypatch.setattr(benchmark, "run_benchmark", unexpected_gpu_run)
    benchmark.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["shapes"] == [[1, 1025], [16, 262144]]
    assert plan["case_count"] == plan["measurement_count"] == 2
    assert plan["modes"] == ["forward"]
    assert not output_dir.exists()


def test_report_preserves_incomplete_status_and_slowdowns(tmp_path):
    measurement = {"median_ms": 2.0, "std_ms": 0.1, "peak_extra_mib": 1.0}
    report = {
        "environment": {
            "gpu": "test GPU",
            "torch": "test",
            "triton": "test",
            "backend": "cuda",
            "runtime": "test",
            "warmup": 10,
            "repeat": 50,
            "seed": 415,
        },
        "case_plan": {"measurement_count": 2},
        "complete": False,
        "results": [
            {
                "dtype": "bf16",
                "shape": [1, 1025],
                "mode": "forward",
                "triton_strategy": "row",
                "native": dict(measurement, median_ms=1.0),
                "triton": measurement,
                "speedup": 0.5,
            }
        ],
    }
    benchmark._write_report(report, tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved == report
    markdown = (tmp_path / "report.md").read_text()
    assert "Status: incomplete; 1/2 measurements" in markdown
    assert "0.50x" in markdown and "| row |" in markdown
    assert "automatic strategy selection" in markdown
    assert "no torch.compile or CUDA Graph capture" in markdown


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", ("fp16", "bf16", "fp32"))
def test_public_auto_benchmark_gpu_smoke(tmp_path, dtype):
    args = build_arg_parser().parse_args(
        [
            "--dtypes",
            dtype,
            "--shapes",
            "3x1025",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report = run_benchmark(args)
    assert report["complete"]
    assert json.loads((tmp_path / "results.json").read_text()) == report
    assert {row["mode"] for row in report["results"]} == set(benchmark.MODES)
    assert report["environment"]["operator"] == "softcapped_selected_logprob"
    for row in report["results"]:
        assert row["triton_strategy"] == "row"
        assert row["speedup"] > 0
        assert row["native"]["std_ms"] is not None
        assert row["triton"]["std_ms"] is not None
        assert set(row["accuracy"]) == {"output", "gradient"}
