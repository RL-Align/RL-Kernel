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
def test_workloads_and_accuracy_gate(dtype):
    args = argparse.Namespace(batch=1, seq=3, n_heads=2, input_mode="random", seed=642)
    inputs = make_operator_inputs("qwen3_6_partial_rope", args, dtype, torch.device("cpu"))
    grad_y = torch.ones_like(inputs["x"])
    op = NativeQwen3_6PartialRoPEOp()
    for mode in MODES:
        workload = _make_workload(op, inputs, grad_y, mode)
        assert workload() is None
        assert workload() is None  # Retained backward graph survives repeated timing calls.
        assert inputs["x"].grad is None
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
