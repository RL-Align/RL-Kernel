# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``timestep_sinusoid_h3``: FP32 [cos | sin] timestep features.

Three properties, each checked separately:

* provider parity: the CUDA kernel is bitwise equal to the diffusers
  ``get_timestep_embedding`` replay on the same device (an elementwise op
  with the same FP32 operation sequence);
* accuracy: within the FP32 elementwise contract of an FP64 golden;
* invariance: a row's bytes do not depend on how many timesteps share the
  launch, their order, or repeated runs.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.h3.timestep_sinusoid import NativeH3TimestepSinusoidOp
from rl_engine.testing.h3_cases import h3_timesteps
from rl_engine.testing.h3_provider import provider_time_proj

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.timestep_sinusoid import (
        H3TimestepSinusoidCudaOp,
        h3_sinusoid_cuda_available,
    )

    if not h3_sinusoid_cuda_available():
        pytest.skip("rl_engine._C lacks h3_timestep_sinusoid_forward")
    return H3TimestepSinusoidCudaOp()


class TestReference:
    def test_layout_is_cos_then_sin(self):
        t = torch.tensor([0.0, 1.0])
        out = NativeH3TimestepSinusoidOp().forward(t)
        assert out.shape == (2, 256) and out.dtype == torch.float32
        assert torch.equal(out[0, :128], torch.ones(128))
        assert torch.equal(out[0, 128:], torch.zeros(128))
        # k = 0 has frequency 1: cos(1), sin(1) at t = 1.
        assert out[1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-7)
        assert out[1, 128].item() == pytest.approx(math.sin(1.0), abs=1e-7)

    def test_provider_path_matches_diffusers_cpu(self):
        t = h3_timesteps(9, device="cpu")
        ours = NativeH3TimestepSinusoidOp().forward(t)
        assert torch.equal(ours, provider_time_proj(t, 256))

    def test_golden_close_to_provider_path(self):
        t = h3_timesteps(33, device="cpu")
        op = NativeH3TimestepSinusoidOp()
        torch.testing.assert_close(op.forward(t), op.forward_fp32(t), atol=1e-6, rtol=0)

    @pytest.mark.parametrize(
        "bad, error",
        [
            (torch.tensor([[0.5]]), ValueError),  # not 1-D
            (torch.empty(0), ValueError),  # empty
            (torch.tensor([1, 0]), TypeError),  # integer
            (torch.tensor([0.5, 1.5]), ValueError),  # H10: t outside [0, 1]
            (torch.tensor([500.0]), ValueError),  # H10: t * 1000 convention
            (torch.tensor([float("nan")]), ValueError),
        ],
    )
    def test_rejects_invalid_timesteps(self, bad, error):
        with pytest.raises(error):
            NativeH3TimestepSinusoidOp().forward(bad)

    def test_rejects_odd_channels(self):
        with pytest.raises(ValueError):
            NativeH3TimestepSinusoidOp().forward(torch.tensor([0.5]), num_channels=255)


@requires_cuda
class TestCuda:
    @pytest.mark.parametrize("num", [1, 2, 3, 7, 64, 1000, 4097])
    def test_bitwise_equal_to_diffusers_cuda_path(self, num):
        t = h3_timesteps(num, seed=num)
        out = _cuda_op()(t)
        assert out.dtype == torch.float32 and out.shape == (num, 256)
        assert torch.equal(out, provider_time_proj(t, 256))
        assert torch.equal(out, NativeH3TimestepSinusoidOp().forward(t))

    @pytest.mark.parametrize("num_channels", [2, 8, 96, 256, 320])
    def test_other_channel_counts_match_reference(self, num_channels):
        t = h3_timesteps(5)
        out = _cuda_op()(t, num_channels=num_channels)
        ref = NativeH3TimestepSinusoidOp().forward(t, num_channels=num_channels)
        assert torch.equal(out, ref)

    def test_fp32_contract_against_fp64_golden(self):
        t = h3_timesteps(257)
        out = _cuda_op()(t)
        gold = NativeH3TimestepSinusoidOp().forward_fp32(t)
        # tolerance_contract.json forward_accuracy / elementwise / float32.
        torch.testing.assert_close(out, gold, atol=1e-5, rtol=1e-5)
        assert (out - gold).abs().max().item() <= 2.0**-23

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_timesteps_are_upcast_first(self, dtype):
        t = h3_timesteps(6).to(dtype)
        assert torch.equal(_cuda_op()(t), provider_time_proj(t, 256))

    def test_batch_position_and_repeat_invariance(self):
        op = _cuda_op()
        base = torch.tensor(0.3141592, device="cuda")
        single = op(base.view(1))
        for num in (2, 5, 64, 1023):
            for pos in (0, num // 2, num - 1):
                t = h3_timesteps(num, seed=num)
                t[pos] = base
                assert torch.equal(op(t)[pos], single[0]), (num, pos)
        t = h3_timesteps(129, seed=11)
        first = op(t)
        for _ in range(3):
            assert torch.equal(op(t), first)
        perm = torch.randperm(129, generator=torch.Generator().manual_seed(0)).cuda()
        assert torch.equal(op(t[perm]), first[perm])

    def test_non_contiguous_input(self):
        t = h3_timesteps(20)
        strided = t[::2]
        assert not strided.is_contiguous()
        assert torch.equal(_cuda_op()(strided), _cuda_op()(strided.contiguous()))

    def test_rejects_cpu_input(self):
        with pytest.raises(ValueError):
            _cuda_op()(torch.tensor([0.5]))

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _cuda_op()(torch.tensor([0.0, 999.0], device="cuda"))

    def test_backward_matches_autograd_reference(self):
        t = h3_timesteps(17).requires_grad_(True)
        t_ref = t.detach().clone().requires_grad_(True)
        grad = torch.randn(17, 256, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        _cuda_op()(t).backward(grad)
        NativeH3TimestepSinusoidOp().forward_fp32(t_ref).backward(grad)
        torch.testing.assert_close(t.grad, t_ref.grad, atol=1e-4, rtol=1e-4)

    def test_backward_is_row_invariant(self):
        op = _cuda_op()
        grad_row = torch.randn(256, device="cuda", generator=torch.Generator("cuda").manual_seed(2))

        def grad_of(t, pos):
            t = t.clone().requires_grad_(True)
            grad = torch.zeros(t.shape[0], 256, device="cuda")
            grad[pos] = grad_row
            op(t).backward(grad)
            return t.grad[pos]

        single = grad_of(torch.tensor([0.7], device="cuda"), 0)
        t = h3_timesteps(40)
        t[13] = 0.7
        assert torch.equal(grad_of(t, 13), single)

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("timestep_sinusoid_h3", device="cuda")
        assert type(op).__name__ == "H3TimestepSinusoidCudaOp"
