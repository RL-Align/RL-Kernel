# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for benchmark timing boundaries and gradient semantics."""

import pytest
import torch

from benchmarks.benchmark_final_logit_softcap import (
    _check_accuracy,
    _make_workload,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.kernels.ops.pytorch.activation import NativeFinalLogitSoftcapOp


@pytest.mark.parametrize("mode", ["forward", "backward", "forward_backward"])
def test_workload_reuses_inputs_without_accumulating_gradients(mode):
    x = torch.tensor([-30.0, 0.0, 30.0])
    grad_y = torch.tensor([0.5, -2.0, 3.0])
    forward_calls, gradients = [], []
    native = NativeFinalLogitSoftcapOp()

    def observed_op(leaf):
        forward_calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled() and len(forward_calls) == 1:
            leaf.register_hook(lambda grad: gradients.append(grad.clone()))
        return native(leaf)

    fn = _make_workload(observed_op, x, grad_y, mode)
    assert len(forward_calls) == (1 if mode == "backward" else 0)
    assert fn() is None
    assert fn() is None
    assert len(forward_calls) == (1 if mode == "backward" else 2)
    if mode == "forward":
        assert forward_calls == [False, False]
        assert not gradients
    else:
        assert len(gradients) == 2
        for grad in gradients:
            torch.testing.assert_close(grad, grad_y / torch.cosh(x / 30.0).square())
    assert x.grad is None and not x.requires_grad
    assert torch.equal(x, torch.tensor([-30.0, 0.0, 30.0]))
    assert torch.equal(grad_y, torch.tensor([0.5, -2.0, 3.0]))


def test_benchmark_rejects_cpu_instead_of_timing_fallback():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA or AMD ROCm GPU"):
        run_benchmark(args)


def test_accuracy_gate_checks_random_upstream_gradient():
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return NativeFinalLogitSoftcapOp()(x)

        @staticmethod
        def backward(ctx, grad_y):
            return torch.ones_like(grad_y)

    x = torch.tensor([-30.0, 0.0, 30.0])
    with pytest.raises(AssertionError):
        _check_accuracy(
            NativeFinalLogitSoftcapOp(), WrongGradient.apply, x, torch.tensor([0.5, -2.0, 3.0])
        )
