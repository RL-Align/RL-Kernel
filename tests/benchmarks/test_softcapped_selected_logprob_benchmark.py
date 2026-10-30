# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark timing boundaries, correctness gates and report completeness."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from benchmarks.operators.logprob import benchmark_softcapped_selected_logprob as benchmark
from benchmarks.operators.logprob.benchmark_softcapped_selected_logprob import (
    _check_accuracy,
    _make_workload,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)


@pytest.mark.parametrize("mode", ("forward", "backward", "forward_backward"))
@pytest.mark.parametrize("softcap", (1.0, 30.0))
def test_workload_timing_and_random_upstream(mode, softcap):
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ids = torch.tensor([2, 0])
    upstream = torch.tensor([0.5, -2.0])
    native = NativeSoftcappedSelectedLogprobOp()
    gold_leaf = logits.clone().requires_grad_(True)
    expected = torch.autograd.grad(
        native(gold_leaf, ids, softcap=softcap), gold_leaf, grad_outputs=upstream
    )[0]
    calls, gradients, received_caps = [], [], []

    def observed(leaf, token_ids, *, softcap):
        calls.append(torch.is_grad_enabled())
        received_caps.append(softcap)
        if torch.is_grad_enabled() and len(calls) == 1:
            leaf.register_hook(lambda grad: gradients.append(grad.clone()))
        return native(leaf, token_ids, softcap=softcap)

    fn = _make_workload(observed, logits, ids, upstream, mode, softcap=softcap)
    assert len(calls) == (1 if mode == "backward" else 0)
    assert fn() is None and fn() is None
    assert len(calls) == (1 if mode == "backward" else 2)
    assert received_caps == [softcap] * len(calls)
    if mode == "forward":
        assert calls == [False, False] and not gradients
    else:
        assert len(gradients) == 2
        for grad in gradients:
            torch.testing.assert_close(grad, expected)
    assert logits.grad is None and not logits.requires_grad


def test_benchmark_rejects_cpu():
    args = build_arg_parser().parse_args(["--device", "cpu", "--softcap", "30"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA or AMD ROCm GPU"):
        run_benchmark(args)


def test_accuracy_gate_rejects_incorrect_output():
    """A correct gradient must not hide an output error against the FP32 reference."""
    reference = NativeSoftcappedSelectedLogprobOp()
    logits = torch.tensor([[1.0, -2.0, 3.0]])
    ids, upstream = torch.tensor([0]), torch.tensor([0.5])

    def wrong_output(x, tokens, *, softcap):
        return reference(x, tokens, softcap=softcap) + 0.1

    with pytest.raises(AssertionError):
        _check_accuracy(reference, wrong_output, logits, ids, upstream, softcap=50.0)


@pytest.mark.parametrize("mismatch", (None, "output", "gradient"))
def test_strategy_gate_checks_all_gradient_bytes(mismatch):
    """Cross-strategy checks cannot stop at identical forward values."""
    native = NativeSoftcappedSelectedLogprobOp()
    logits = torch.tensor([[1.0, -2.0, 3.0]])
    ids, upstream = torch.tensor([0]), torch.tensor([0.5])

    def alternative(x, tokens, *, softcap):
        output = native(x, tokens, softcap=softcap)
        if mismatch == "output":
            return output + 0.1
        if mismatch == "gradient":
            return output.detach() + 0.5 * (output - output.detach())
        return output

    def check():
        return _check_accuracy(
            native,
            native,
            logits,
            ids,
            upstream,
            softcap=50.0,
            strategy_candidates=(("alternative", alternative),),
        )

    if mismatch is None:
        check()
    else:
        with pytest.raises(AssertionError, match=f"Bitwise mismatch: alternative {mismatch}"):
            check()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_accuracy_reference_keeps_fp32_gradients_and_quantized_inputs(dtype):
    """The untimed reference must not round its gradient to the candidate dtype."""
    logits = torch.tensor([[0.1, -1.3, 2.7]], dtype=dtype)
    ids, upstream = torch.tensor([0]), torch.tensor([0.5])
    native = NativeSoftcappedSelectedLogprobOp()
    seen = {}

    def observed_reference(x, tokens, *, softcap):
        seen["reference_dtype"] = x.dtype
        torch.testing.assert_close(x, logits.float(), atol=0, rtol=0)
        x.register_hook(lambda gradient: seen.update(gradient_dtype=gradient.dtype))
        return native(x, tokens, softcap=softcap)

    def observed_candidate(x, tokens, *, softcap):
        seen["candidate_dtype"] = x.dtype
        return native(x, tokens, softcap=softcap)

    _check_accuracy(observed_reference, observed_candidate, logits, ids, upstream, softcap=30.0)
    assert seen == {
        "reference_dtype": torch.float32,
        "gradient_dtype": torch.float32,
        "candidate_dtype": dtype,
    }


@pytest.mark.parametrize("deviation", (None, 0.1))
def test_forward_comparison_rotates_providers_and_keeps_round_samples(monkeypatch, deviation):
    """Every provider takes each timing position once, even for one-sample timing."""
    order = []
    received_caps = []

    def workload(op, logits, ids, upstream, mode, *, softcap):
        assert mode == "forward"
        received_caps.append(softcap)
        return op

    def measure(profiler, op):
        order.append(op)
        return {"median_ms": float(len(order)), "std_ms": deviation, "peak_extra_mib": 2.0}

    monkeypatch.setattr(benchmark, "_make_workload", workload)
    monkeypatch.setattr(benchmark, "_measure", measure)
    providers = [
        benchmark._Provider(name, name) for name in ("native", "triton", "row", "parallel")
    ]
    result = benchmark._measure_strategy_comparison(None, providers, None, None, None, softcap=20.0)
    assert order == [
        "native",
        "triton",
        "row",
        "parallel",
        "triton",
        "row",
        "parallel",
        "native",
        "row",
        "parallel",
        "native",
        "triton",
        "parallel",
        "native",
        "triton",
        "row",
    ]
    assert received_caps == [20.0] * 16
    assert result["native"]["median_ms"] == 9.5
    for measurement in result.values():
        assert len(measurement["rounds"]) == 4
        assert measurement["std_ms"] == deviation


def test_benchmark_entry_point_runs_outside_checkout(tmp_path):
    """The relocated script must find the shared profiler and canonical reference."""
    root = Path(__file__).resolve().parents[2]
    script = root / "benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["RL_KERNEL_LOG_STREAM"] = "stderr"
    result = subprocess.run(
        [sys.executable, str(script), "--softcap", "30", "--list-cases"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["softcap"] == 30.0
    assert plan["case_count"] == 15
    assert plan["measurement_count"] == 45
    assert not plan["compare_forward_strategies"]
    assert plan["forward_timing_rounds"] == 1
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("softcap", (1.0, 30.0))
def test_accuracy_gate_rejects_incorrect_gradient(softcap):
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, token_ids, softcap):
            ctx.shape = logits.shape
            return NativeSoftcappedSelectedLogprobOp()(logits, token_ids, softcap=softcap)

        @staticmethod
        def backward(ctx, grad_output):
            return torch.ones(ctx.shape, device=grad_output.device), None, None

    def candidate(logits, token_ids, *, softcap):
        return WrongGradient.apply(logits, token_ids, softcap)

    with pytest.raises(AssertionError):
        _check_accuracy(
            NativeSoftcappedSelectedLogprobOp(),
            candidate,
            torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            torch.tensor([2, 0]),
            torch.tensor([0.5, -2.0]),
            softcap=softcap,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--shapes", "0x10"],
        ["--shapes", "2x3x4"],
        ["--repeat", "0"],
        ["--rounds", "0"],
        ["--pipeline-stages", "0"],
        ["--pipeline-stages", "-1"],
        ["--softcap", "0"],
        ["--softcap", "-1"],
        ["--softcap", "nan"],
        ["--softcap", "inf"],
    ],
)
def test_benchmark_rejects_invalid_measurements(arguments):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--softcap", "30", *arguments])


def test_benchmark_requires_explicit_softcap():
    """Listing cases must not silently select a model's cap either."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--list-cases"])


def test_list_cases_needs_no_gpu_or_report_files(monkeypatch, capsys, tmp_path):
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--list-cases",
            "--compare-strategies",
            "--softcap",
            "50",
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
    assert plan["softcap"] == 50.0
    assert plan["shapes"] == [[1, 1025], [16, 262144]]
    assert plan["case_count"] == plan["measurement_count"] == 2
    assert plan["modes"] == ["forward"]
    assert plan["compare_forward_strategies"]
    assert plan["forward_timing_rounds"] == 4
    assert plan["pipeline_stages"] == [1, 2, 3, 4, 5, 6, 8]
    assert plan["provider_measurement_count"] == 22
    assert not output_dir.exists()


@pytest.mark.parametrize("compare_strategies", (False, True))
def test_report_preserves_incomplete_status_and_slowdowns(tmp_path, compare_strategies):
    measurement = {
        "median_ms": 2.0,
        "std_ms": 0.1,
        "peak_extra_mib": 1.0,
        "round_spread_ratio": 1.1,
    }
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
            "softcap": 50.0,
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
    if compare_strategies:
        report["results"][0]["strategy_comparison"] = {
            "providers": {"row": measurement, "parallel": dict(measurement, median_ms=1.0)},
            "best_explicit": "parallel",
            "auto_over_best_explicit": 2.0,
        }
    benchmark._write_report(report, tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved == report
    markdown = (tmp_path / "report.md").read_text()
    assert "Status: incomplete; 1/2 measurements" in markdown
    assert "0.50x" in markdown and "| row |" in markdown
    assert "Softcap: 50.0" in markdown
    assert "automatic strategy selection" in markdown
    assert "no torch.compile or CUDA Graph capture" in markdown
    assert (
        "Eager NativeSoftcappedSelectedLogprobOp vs TritonSoftcappedSelectedLogprobOp" in markdown
    )
    assert "FP32 device-tensor divisor" in markdown
    assert "FP32 log_softmax" in markdown
    assert "creation and transfer are included in forward and forward+backward timing" in markdown
    assert "Backward reuses the divisor saved in the prebuilt graph" in markdown
    if compare_strategies:
        assert "Forward schedule tuning" in markdown
        assert "2.000x" in markdown
    else:
        assert "Forward schedule tuning" not in markdown


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("softcap", (30.0, 50.0))
def test_public_auto_benchmark_gpu_smoke(tmp_path, dtype, softcap):
    args = build_arg_parser().parse_args(
        [
            "--softcap",
            str(softcap),
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
    assert report["environment"]["softcap"] == softcap
    assert report["environment"]["baseline"] == (
        "eager NativeSoftcappedSelectedLogprobOp; FP32 device-tensor division; FP32 log_softmax"
    )
    assert report["case_plan"]["softcap"] == softcap
    assert report["complete"]
    assert json.loads((tmp_path / "results.json").read_text()) == report
    assert {row["mode"] for row in report["results"]} == set(benchmark.MODES)
    assert report["environment"]["operator"] == "softcapped_selected_logprob"
    for row in report["results"]:
        assert row["triton_strategy"] == "row"
        assert row["train_inference_bitwise"]
        assert set(row["strategy_bitwise"]) == {"row", "parallel", "row_pipelined"}
        assert row["row_pipeline_stages"] > 0
        assert row["speedup"] > 0
        assert row["native"]["std_ms"] is not None
        assert row["triton"]["std_ms"] is not None
        assert set(row["accuracy"]) == {"output", "gradient"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", ("fp16", "bf16", "fp32"))
def test_forward_strategy_comparison_gpu_smoke(tmp_path, dtype):
    args = build_arg_parser().parse_args(
        [
            "--compare-strategies",
            "--softcap",
            "20",
            "--dtypes",
            dtype,
            "--shapes",
            "2x31745",
            "--warmup",
            "1",
            "--repeat",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report = run_benchmark(args)
    assert report["complete"] and len(report["results"]) == 3
    for row in report["results"]:
        assert row["triton_strategy"] == "parallel"
        if row["mode"] == "backward":
            assert "strategy_comparison" not in row
            continue
        comparison = row["strategy_comparison"]
        assert comparison["auto_over_best_explicit"] > 0
        names = {
            "row",
            "parallel",
            *{f"row_pipelined_s{s}" for s in benchmark.DEFAULT_PIPELINE_STAGES},
        }
        assert set(comparison["providers"]) == names
        assert row["saved_statistics_bitwise"]
        for name in names:
            assert len(comparison["providers"][name]["rounds"]) == 4
            assert comparison["providers"][name]["std_ms"] is None
        assert len(row["triton"]["rounds"]) == len(row["native"]["rounds"]) == 4
    assert json.loads((tmp_path / "results.json").read_text()) == report


@pytest.mark.parametrize("fail", (False, True))
def test_pipeline_override_restores_rules_and_caches_on_exit(fail):
    """A candidate cannot leak its depth into AUTO or the next timing round."""
    original_rules = {"measured-range": "original-config"}
    original_default = object()
    cache_clear = Mock()
    core = SimpleNamespace(
        ROW_PIPELINE_CONFIGS=original_rules,
        DEFAULT_ROW_PIPELINE_CONFIG=original_default,
        RowPipelineConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        select_softcapped_logprob_pipeline_config=SimpleNamespace(cache_clear=cache_clear),
    )
    try:
        with benchmark._pipeline_depth(core, 8):
            assert core.ROW_PIPELINE_CONFIGS == {}
            assert core.DEFAULT_ROW_PIPELINE_CONFIG.pipeline_stages == 8
            assert cache_clear.call_count == 1
            if fail:
                raise RuntimeError("compile failed")
    except RuntimeError:
        assert fail
    assert core.ROW_PIPELINE_CONFIGS is original_rules
    assert core.DEFAULT_ROW_PIPELINE_CONFIG is original_default
    assert cache_clear.call_count == 2
    with benchmark._pipeline_depth(core, None):
        assert core.ROW_PIPELINE_CONFIGS is original_rules
        assert core.DEFAULT_ROW_PIPELINE_CONFIG is original_default
    assert cache_clear.call_count == 2


def test_tuning_plan_counts_all_depths_and_deduplicates_candidates():
    args = build_arg_parser().parse_args(
        [
            "--softcap",
            "30",
            "--compare-strategies",
            "--pipeline-stages",
            "1",
            "3",
            "3",
            "8",
            "--dtypes",
            "bf16",
            "--shapes",
            "2x1025",
            "--rounds",
            "7",
        ]
    )
    plan = benchmark._case_plan(args)
    assert plan["pipeline_stages"] == [1, 3, 8]
    assert plan["comparison_providers"] == [
        "native",
        "triton",
        "row",
        "parallel",
        "row_pipelined_s1",
        "row_pipelined_s3",
        "row_pipelined_s8",
    ]
    assert plan["measurement_count"] == 3
    assert plan["provider_measurement_count"] == 7 + 2 + 7
    assert plan["forward_timing_rounds"] == 7


def test_tuning_config_is_active_during_timing_not_inside_workload(monkeypatch):
    """The timer sees the public callable directly, while its chosen depth is active."""
    core = SimpleNamespace(
        ROW_PIPELINE_CONFIGS={},
        DEFAULT_ROW_PIPELINE_CONFIG=SimpleNamespace(pipeline_stages=3),
        RowPipelineConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        select_softcapped_logprob_pipeline_config=SimpleNamespace(cache_clear=Mock()),
    )
    native, pipelined = object(), object()
    observed = []

    def workload(op, *args, mode, softcap):
        return op

    def measure(profiler, op):
        observed.append((op, core.DEFAULT_ROW_PIPELINE_CONFIG.pipeline_stages))
        return {"median_ms": 1.0, "std_ms": 0.0, "peak_extra_mib": 0.0}

    monkeypatch.setattr(
        benchmark, "_make_workload", lambda op, x, ids, g, mode, **kw: workload(op, mode=mode, **kw)
    )
    monkeypatch.setattr(benchmark, "_measure", measure)
    benchmark._measure_strategy_comparison(
        None,
        [benchmark._Provider("native", native), benchmark._Provider("stage8", pipelined, 8)],
        None,
        None,
        None,
        softcap=30,
        core=core,
        mode="forward_backward",
        rounds=2,
    )
    assert observed == [(native, 3), (pipelined, 8), (pipelined, 8), (native, 3)]


@pytest.mark.parametrize("statistic_delta", (0.0, 1.0))
def test_accuracy_gate_checks_saved_statistics_even_when_outputs_and_gradients_match(
    statistic_delta,
):
    """A hidden saved-state mismatch cannot pass just because this gradient ignores it."""

    class SavedStatistics(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, ids, delta):
            maximum = logits.max(dim=-1).values + delta
            ctx.save_for_backward(logits, ids, maximum, torch.zeros_like(maximum))
            return logits[:, 0].clone()

        @staticmethod
        def backward(ctx, upstream):
            gradient = torch.zeros_like(ctx.saved_tensors[0])
            gradient[:, 0] = upstream
            return gradient, None, None

    def reference(logits, ids, *, softcap):
        return logits[:, 0]

    def candidate(logits, ids, *, softcap):
        return SavedStatistics.apply(logits, ids, 0.0)

    def alternative(logits, ids, *, softcap):
        return SavedStatistics.apply(logits, ids, statistic_delta)

    def check():
        return _check_accuracy(
            reference,
            candidate,
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([0]),
            torch.tensor([0.5]),
            softcap=30,
            strategy_candidates=[("alternative", alternative)],
            check_saved_statistics=True,
        )

    if statistic_delta:
        with pytest.raises(AssertionError, match="alternative row_max"):
            check()
    else:
        check()
