# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check timing boundaries and both input-gradient paths without a GPU."""

import pytest
import torch

from benchmarks.benchmark_conditioning_noise_mix import (
    _check_accuracy,
    _make_workload,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.kernels.ops.pytorch.conditioning_noise_mix import NativeConditioningNoiseMixOp


@pytest.mark.parametrize("mode", ["forward", "backward", "forward_backward"])
def test_workload_reuses_inputs_without_accumulating_gradients(mode):
    sample = torch.tensor([[1.0, -2.0, 3.0]])
    noise = torch.tensor([[-4.0, 5.0, -6.0]])
    timestep = torch.tensor([0.25])
    grad_y = torch.tensor([[0.5, -2.0, 3.0]])
    inputs = (sample, timestep, noise, grad_y)
    originals = [value.clone() for value in inputs]
    calls, gradients = [], [[], []]
    leaves = []
    native = NativeConditioningNoiseMixOp()

    def observed_op(sample, timestep, noise):
        calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled() and len(calls) == 1:
            leaves.extend((sample, noise))
            sample.register_hook(lambda grad: gradients[0].append(grad.clone()))
            noise.register_hook(lambda grad: gradients[1].append(grad.clone()))
        return native(sample, timestep, noise)

    fn = _make_workload(observed_op, *inputs, mode)
    assert len(calls) == (1 if mode == "backward" else 0)
    assert fn() is None
    assert fn() is None
    assert len(calls) == (1 if mode == "backward" else 2)
    if mode == "forward":
        assert calls == [False, False]
        assert gradients == [[], []]
    else:
        assert len(leaves) == 2
        assert all(value.grad is None for value in leaves)
        for observed, expected in zip(gradients, (grad_y * timestep, grad_y * (1 - timestep))):
            assert len(observed) == 2
            for grad in observed:
                torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    for actual, original in zip(inputs, originals):
        assert actual.grad is None and not actual.requires_grad
        assert torch.equal(actual, original)


def test_benchmark_rejects_cpu_instead_of_timing_fallback():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        run_benchmark(args)


def test_accuracy_gate_rejects_wrong_noise_gradient():
    native = NativeConditioningNoiseMixOp()

    def wrong_noise_gradient(sample, timestep, noise):
        if noise.requires_grad:
            noise.register_hook(torch.ones_like)
        return native(sample, timestep, noise)

    with pytest.raises(AssertionError, match="d_noise"):
        _check_accuracy(
            native,
            wrong_noise_gradient,
            torch.tensor([[1.0, -2.0, 3.0]]),
            torch.tensor([0.25]),
            torch.tensor([[-4.0, 5.0, -6.0]]),
            torch.tensor([[0.5, -2.0, 3.0]]),
        )


def test_accuracy_gate_rejects_wrong_output():
    native = NativeConditioningNoiseMixOp()

    def wrong_output(sample, timestep, noise):
        return native(sample, timestep, noise) + 1

    with pytest.raises(AssertionError, match="output"):
        _check_accuracy(
            native,
            wrong_output,
            torch.ones(1, 3),
            torch.tensor([0.25]),
            torch.zeros(1, 3),
            torch.tensor([[0.5, -2.0, 3.0]]),
        )
