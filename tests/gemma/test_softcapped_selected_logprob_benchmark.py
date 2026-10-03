# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for benchmark timing, correctness gates and dispatch evidence."""

import json

import pytest
import torch

from benchmarks.benchmark_softcapped_selected_logprob import (
    DISPATCH_ROWS,
    DISPATCH_VOCAB_SIZES,
    IMPLEMENTATIONS,
    _case_plan,
    _check_accuracy,
    _check_forward_variants,
    _dispatch_summary,
    _forward_choice,
    _make_workload,
    _markdown,
    _measure_case,
    _resolve_cases,
    _summarize_measurements,
    _write_report,
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


@pytest.mark.parametrize("std_ms", (0.1, None))
def test_report_discloses_split_baseline_and_regression(std_ms):
    env = dict(
        gpu="test GPU",
        backend="cuda",
        runtime="test",
        torch="test",
        triton="test",
        git_commit="test",
        git_tracked_changes="",
        warmup=1,
        repeat=2 if std_ms is not None else 1,
        block_size=1024,
        num_warps=4,
    )
    measurement = dict(median_ms=2.0, std_ms=std_ms, peak_extra_mib=0.5)
    parallel_std = 0.2 if std_ms is not None else None
    report = {
        "environment": env,
        "results": [
            {
                "dtype": "fp32",
                "shape": [1, 1025],
                "mode": "forward",
                "speedup": 0.5,
                "split_speedup": 0.75,
                "parallel_speedup": 0.8,
                "native": measurement,
                "split_triton": measurement,
                "triton": measurement,
                "triton_parallel": dict(median_ms=2.5, std_ms=parallel_std, peak_extra_mib=0.75),
            }
        ],
    }
    markdown = _markdown(report)
    assert "TritonBatchInvariantLogpOp" in markdown
    assert "0.50x" in markdown and "0.75x" in markdown
    assert "0.80x" in markdown
    expected_std = "0.200000" if parallel_std is not None else "N/A"
    assert f"2.500000 ± {expected_std}" in markdown
    assert "Row/parallel" in markdown and "ordered merge" in markdown
    assert "allocates scratch and launches both kernels inside" in markdown
    assert "standard deviation" in markdown and "extra MiB" in markdown


def test_benchmark_can_time_forward_only():
    args = build_arg_parser().parse_args(["--modes", "forward", "--shapes", "1x262144", "4x262144"])
    assert args.modes == ["forward"]
    assert args.shapes == [(1, 262144), (4, 262144)]


def test_exact_gate_checks_saved_statistics_even_when_outputs_and_gradients_match():
    class WithSavedStatistics(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, token_ids, offset):
            ctx.save_for_backward(logits, token_ids, logits.sum(dim=1) + offset)
            return logits.sum(dim=1)

        @staticmethod
        def backward(ctx, upstream):
            logits, _, _ = ctx.saved_tensors
            return upstream[:, None].expand_as(logits), None, None

    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ids = torch.tensor([2, 0])
    upstream = torch.tensor([0.5, -2.0])

    def row_op(x, token_ids):
        return WithSavedStatistics.apply(x, token_ids, 0.0)

    def wrong_statistics(x, token_ids):
        return WithSavedStatistics.apply(x, token_ids, 1.0)

    checks = _check_forward_variants(row_op, row_op, logits, ids, upstream)
    assert checks == {"output": True, "log_sum_exp": True, "gradient": True}
    with pytest.raises(AssertionError, match="differ in log_sum_exp"):
        _check_forward_variants(row_op, wrong_statistics, logits, ids, upstream)


@pytest.mark.parametrize("shape", ("1", "1x2x3", "0x1024", "1x-1"))
def test_benchmark_rejects_invalid_geometry(shape):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--shapes", shape])


def test_dispatch_sweep_covers_dtypes_row_parallelism_and_vocabulary_tails():
    args = _resolve_cases(build_arg_parser().parse_args(["--suite", "dispatch"]))
    plan = _case_plan(args)
    assert args.dtypes == ["fp16", "bf16", "fp32"]
    assert args.modes == ["forward"]
    assert args.rounds == 4 and args.repeat == 100 and args.warmup == 25
    assert len(args.shapes) == len(DISPATCH_ROWS) * len(DISPATCH_VOCAB_SIZES) == 85
    assert plan["correctness_cases"] == plan["timing_cases"] == 255
    for rows in (1, 4, 16, 64, 256):
        for vocab in (512, 1023, 1024, 1025, 4097, 32769, 128256, 151936, 262144):
            assert (rows, vocab) in args.shapes


def test_original_benchmark_defaults_remain_available():
    args = _resolve_cases(build_arg_parser().parse_args([]))
    assert _case_plan(args)["timing_cases"] == 45
    assert args.rounds == 1 and args.repeat == 50 and args.warmup == 10


def test_custom_dispatch_grid_deduplicates_and_preserves_timing_overrides():
    args = _resolve_cases(
        build_arg_parser().parse_args(
            [
                "--suite",
                "dispatch",
                "--rows",
                "16",
                "1",
                "1",
                "--vocab-sizes",
                "8193",
                "8192",
                "--dtypes",
                "bf16",
                "bf16",
                "--modes",
                "forward",
                "forward_backward",
                "--rounds",
                "8",
                "--repeat",
                "200",
                "--warmup",
                "50",
            ]
        )
    )
    assert args.shapes == [(1, 8192), (1, 8193), (16, 8192), (16, 8193)]
    assert args.dtypes == ["bf16"]
    assert args.modes == ["forward", "forward_backward"]
    assert args.rounds == 8 and args.repeat == 200 and args.warmup == 50
    assert _resolve_cases(args).shapes == args.shapes


def test_explicit_shapes_override_dispatch_preset():
    args = _resolve_cases(
        build_arg_parser().parse_args(["--suite", "dispatch", "--shapes", "4x12345"])
    )
    assert args.shapes == [(4, 12345)]


@pytest.mark.parametrize("axis", ("--rows", "--vocab-sizes"))
def test_ambiguous_geometry_is_rejected(axis):
    args = build_arg_parser().parse_args(["--shapes", "1x1024", axis, "16"])
    with pytest.raises(ValueError, match="either --shapes"):
        _resolve_cases(args)


@pytest.mark.parametrize("flag", ("--rows", "--vocab-sizes", "--rounds"))
def test_dispatch_rejects_nonpositive_parameters(flag):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([flag, "0"])


def _measurements(row_ms, parallel_ms, relative_std=0.01):
    def summary(medians):
        return _summarize_measurements(
            [
                dict(median_ms=value, std_ms=value * relative_std, peak_extra_mib=0.5)
                for value in medians
            ]
        )

    return {"triton": summary(row_ms), "triton_parallel": summary(parallel_ms)}


@pytest.mark.parametrize(
    "row_ms, parallel_ms, relative_std, candidate",
    [
        ([2.0] * 4, [1.0] * 4, 0.01, "parallel"),
        ([1.0] * 4, [2.0] * 4, 0.01, "row"),
        ([1.01] * 4, [1.0] * 4, 0.01, "inconclusive"),  # Too close.
        ([1.02, 0.98, 1.02, 0.98], [1.0] * 4, 0.01, "inconclusive"),
        ([2.0] * 4, [1.0] * 4, 0.50, "inconclusive"),  # Large within-round noise.
        ([2.0, 2.0, 3.0, 2.0], [1.0] * 4, 0.01, "inconclusive"),  # Run drift.
        ([2.0], [1.0], 0.01, "inconclusive"),  # One round is not enough.
        ([2.0] * 5, [1.0] * 5, 0.01, "inconclusive"),  # Unbalanced timing order.
    ],
)
def test_forward_candidates_reject_noise_and_marginal_gains(
    row_ms, parallel_ms, relative_std, candidate
):
    measurements = _measurements(row_ms, parallel_ms, relative_std)
    result = _forward_choice(measurements)
    assert result["candidate"] == candidate
    assert len(result["row_over_parallel_per_round"]) == len(row_ms)


def test_missing_sample_std_cannot_produce_a_dispatch_candidate():
    measurements = _measurements([2.0] * 4, [1.0] * 4)
    measurements["triton"]["rounds"][0]["std_ms"] = None
    assert _forward_choice(measurements)["candidate"] == "inconclusive"


def test_measurement_rounds_balance_order_and_keep_the_worst_memory_peak(monkeypatch):
    from benchmarks import benchmark_softcapped_selected_logprob as benchmark

    calls = []

    def workload(op, *args):
        return op

    def measure(profiler, name):
        calls.append(name)
        round_number = (len(calls) - 1) // 4 + 1
        return dict(median_ms=round_number, std_ms=round_number / 10, peak_extra_mib=round_number)

    monkeypatch.setattr(benchmark, "_make_workload", workload)
    monkeypatch.setattr(benchmark, "_measure", measure)
    measurements = _measure_case(
        None, {name: name for name in IMPLEMENTATIONS}, None, None, None, "forward", 4
    )
    for position in range(4):
        assert set(calls[position::4]) == set(IMPLEMENTATIONS)
    for measurement in measurements.values():
        assert measurement["median_ms"] == 2.5
        assert measurement["std_ms"] == 0.4
        assert measurement["peak_extra_mib"] == 4
        assert len(measurement["rounds"]) == 4


@pytest.mark.parametrize(
    "choices, expected",
    [
        (["parallel", "parallel"], "parallel"),
        (["row", "row"], "row"),
        (["row", "parallel"], "depends_on_rows"),
        (["parallel", "inconclusive"], "inconclusive"),
        (["parallel"], "inconclusive"),  # Partial run; M=16 is still missing.
    ],
)
def test_dispatch_summary_does_not_hide_row_dependence_or_missing_cases(choices, expected):
    report = {
        "case_plan": {"shapes": [[1, 1024], [16, 1024]]},
        "results": [
            dict(
                dtype="bf16",
                shape=[rows, 1024],
                mode="forward",
                forward_choice={"candidate": choice},
            )
            for rows, choice in zip((1, 16), choices, strict=False)
        ],
    }
    # A backward-only timing difference must not choose the forward path.
    report["results"].append(dict(dtype="fp32", shape=[1, 1024], mode="backward"))
    summary = _dispatch_summary(report)
    assert len(summary) == 1
    assert summary[0]["candidate"] == expected
    assert summary[0]["missing_rows"] == ([16] if len(choices) == 1 else [])


def test_list_cases_needs_no_gpu_and_writes_no_reports(monkeypatch, capsys, tmp_path):
    from benchmarks import benchmark_softcapped_selected_logprob as benchmark

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--suite", "dispatch", "--list-cases", "--output-dir", str(tmp_path)],
    )

    def unexpected_run(args):
        pytest.fail("--list-cases should not run a GPU benchmark")

    monkeypatch.setattr(benchmark, "run_benchmark", unexpected_run)
    benchmark.main()
    assert json.loads(capsys.readouterr().out)["timing_cases"] == 255
    assert not list(tmp_path.iterdir())


def test_partial_report_is_marked_incomplete_and_explains_selection_limits(tmp_path):
    env = dict(
        gpu="test",
        backend="cuda",
        runtime="test",
        torch="test",
        triton="test",
        git_commit="test",
        git_tracked_changes="",
        warmup=25,
        repeat=100,
        block_size=1024,
        num_warps=4,
        rounds=4,
    )
    report = {
        "environment": env,
        "complete": False,
        "case_plan": dict(suite="dispatch", correctness_cases=2, timing_cases=2),
        "results": [],
        "dispatch_summary": [
            {
                "dtype": "fp32",
                "vocab_size": 1024,
                "candidate": "inconclusive",
                "rows_by_candidate": {"row": [1], "parallel": [], "inconclusive": []},
                "missing_rows": [16],
            }
        ],
    }
    _write_report(report, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text()) == report
    markdown = (tmp_path / "report.md").read_text()
    assert "report complete: `False`" in markdown
    assert "0/2 completed timing cases" in markdown
    assert "inconclusive | 1 | — | — | 16" in markdown
    assert "screening heuristic, not a confidence interval" in markdown
    assert "No runtime dispatch rule is changed" in markdown
