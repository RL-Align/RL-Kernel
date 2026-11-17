# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for benchmark workload construction and accuracy gating, not timing."""

import argparse
import json

import pytest
import torch

from benchmarks.operators.rope.benchmark_qwen3_6_partial_rope import (
    MODES,
    _check_accuracy,
    _make_workload,
    _measure_interleaved,
    build_arg_parser,
    main,
    run_benchmark,
)
from rl_engine.reference.rope.qwen3_6_partial_rope import NativeQwen3_6PartialRoPEOp
from rl_engine.validation.operators.operator_inputs import make_operator_inputs


def test_dry_run_needs_no_gpu_or_triton(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["benchmark", "--dry-run", "--tokens", "1", "7"])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["shapes"] == [[1, 2, 256], [7, 2, 256], [1, 16, 256], [7, 16, 256]]
    assert plan["measured"] is False


def test_cpu_measurements_are_rejected():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="real CUDA/ROCm GPU"):
        run_benchmark(args)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("position_offset", (0, 11))
def test_workloads_and_accuracy_gate(dtype, position_offset):
    args = argparse.Namespace(
        batch=1,
        seq=3,
        n_heads=2,
        input_mode="random",
        seed=642,
        position_offset=position_offset,
    )
    inputs = make_operator_inputs("qwen3_6_partial_rope", args, dtype, torch.device("cpu"))
    grad_y = torch.ones_like(inputs["x"])
    op = NativeQwen3_6PartialRoPEOp()
    for mode in MODES:
        leaves = []

        def capture_leaf(**kwargs):
            if kwargs["x"].requires_grad:
                leaves.append(kwargs["x"])
            return op(**kwargs)

        workload = _make_workload(capture_leaf, inputs, grad_y, mode)
        assert workload() is None
        assert workload() is None  # Retained backward graph survives repeated timing calls.
        if mode != "forward":
            assert leaves
            assert all(leaf is not inputs["x"] and leaf.grad is None for leaf in leaves)
    assert set(_check_accuracy(op, inputs, grad_y)) == {"output", "gradient"}

    def wrong_rotation(**kwargs):
        return op(**kwargs) + 1

    with pytest.raises(AssertionError):
        _check_accuracy(wrong_rotation, inputs, grad_y)

    def wrong_backward(**kwargs):
        y = op(**kwargs)
        return 2 * y - y.detach()  # Correct values, incorrect derivative.

    with pytest.raises(AssertionError):
        _check_accuracy(wrong_backward, inputs, grad_y)


def test_interleaved_samples_record_balanced_order(monkeypatch):
    """Check scheduling/report aggregation with a fake clock, not GPU performance."""
    calls = []
    elapsed = iter(range(1, 9))

    class FakeEvent:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return next(elapsed)

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    profiler = argparse.Namespace(warmup=2, repeat=4, device="cuda:0", _sync=lambda: None)
    rows = _measure_interleaved(
        profiler,
        {
            "pytorch": lambda: calls.append("pytorch"),
            "triton": lambda: calls.append("triton"),
        },
    )
    assert calls == ["pytorch", "triton", "triton", "pytorch"] * 3
    for name, positions in (("pytorch", [0, 1, 0, 1]), ("triton", [1, 0, 1, 0])):
        row = rows[name]
        assert [sample["order_index"] for sample in row["samples"]] == positions
        assert [sample["round"] for sample in row["samples"]] == [0, 1, 2, 3]
        assert row["median_us"] == 4500
        assert row["std_us"] > 0
