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
    """Construct the CUDA sinusoid operator, skipping a missing native forward entrypoint."""

    from rl_engine.kernels.ops.cuda.h3.timestep_sinusoid import (
        H3TimestepSinusoidCudaOp,
        h3_sinusoid_cuda_available,
    )

    if not h3_sinusoid_cuda_available():
        pytest.skip("rl_engine._C lacks h3_timestep_sinusoid_forward")
    return H3TimestepSinusoidCudaOp()


class TestReference:
    def test_layout_is_cos_then_sin(self):
        """Check FP32 cosine-then-sine layout and the unit-frequency boundary values."""

        t = torch.tensor([0.0, 1.0])
        out = NativeH3TimestepSinusoidOp().forward(t)
        assert out.shape == (2, 256) and out.dtype == torch.float32
        assert torch.equal(out[0, :128], torch.ones(128))
        assert torch.equal(out[0, 128:], torch.zeros(128))
        # k = 0 has frequency 1: cos(1), sin(1) at t = 1.
        assert out[1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-7)
        assert out[1, 128].item() == pytest.approx(math.sin(1.0), abs=1e-7)

    def test_provider_path_matches_diffusers_cpu(self):
        """Require CPU sinusoid values to match the diffusers provider bitwise."""

        t = h3_timesteps(9, device="cpu")
        ours = NativeH3TimestepSinusoidOp().forward(t)
        assert torch.equal(ours, provider_time_proj(t, 256))

    def test_golden_close_to_provider_path(self):
        """Keep the CPU provider within FP32 error of the high-precision sinusoid golden."""

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
        """Reject empty, malformed, non-floating, non-finite, or out-of-range timesteps."""

        with pytest.raises(error):
            NativeH3TimestepSinusoidOp().forward(bad)

    def test_rejects_odd_channels(self):
        """Reject odd feature counts that cannot split into equal cosine and sine halves."""

        with pytest.raises(ValueError):
            NativeH3TimestepSinusoidOp().forward(torch.tensor([0.5]), num_channels=255)


@requires_cuda
class TestCuda:
    @pytest.mark.parametrize("num", [1, 2, 3, 7, 64, 1000, 4097])
    def test_bitwise_equal_to_diffusers_cuda_path(self, num):
        """Require CUDA sinusoid values to match both provider replays for varied batch sizes."""

        t = h3_timesteps(num, seed=num)
        out = _cuda_op()(t)
        assert out.dtype == torch.float32 and out.shape == (num, 256)
        assert torch.equal(out, provider_time_proj(t, 256))
        assert torch.equal(out, NativeH3TimestepSinusoidOp().forward(t))

    @pytest.mark.parametrize("num_channels", [2, 8, 96, 256, 320])
    def test_other_channel_counts_match_reference(self, num_channels):
        """Preserve bitwise reference parity across supported even channel counts."""

        t = h3_timesteps(5)
        out = _cuda_op()(t, num_channels=num_channels)
        ref = NativeH3TimestepSinusoidOp().forward(t, num_channels=num_channels)
        assert torch.equal(out, ref)

    def test_fp32_contract_against_fp64_golden(self):
        """Bound CUDA sinusoid error against the high-precision golden by FP32 precision."""

        t = h3_timesteps(257)
        out = _cuda_op()(t)
        gold = NativeH3TimestepSinusoidOp().forward_fp32(t)
        # tolerance_contract.json forward_accuracy / elementwise / float32.
        torch.testing.assert_close(out, gold, atol=1e-5, rtol=1e-5)
        assert (out - gold).abs().max().item() <= 2.0**-23

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_timesteps_are_upcast_first(self, dtype):
        """Match provider FP32 upcasting for BF16 and FP16 timesteps."""

        t = h3_timesteps(6).to(dtype)
        assert torch.equal(_cuda_op()(t), provider_time_proj(t, 256))

    def test_batch_position_and_repeat_invariance(self):
        """Keep row bits unchanged across batch positions, repeated calls, and permutations."""

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
        """Treat strided timesteps identically to their contiguous copies."""

        t = h3_timesteps(20)
        strided = t[::2]
        assert not strided.is_contiguous()
        assert torch.equal(_cuda_op()(strided), _cuda_op()(strided.contiguous()))

    def test_rejects_cpu_input(self):
        """Reject CPU timesteps at the CUDA sinusoid boundary."""

        with pytest.raises(ValueError):
            _cuda_op()(torch.tensor([0.5]))

    def test_rejects_out_of_range(self):
        """Reject scaled timesteps outside the declared unit interval."""

        with pytest.raises(ValueError):
            _cuda_op()(torch.tensor([0.0, 999.0], device="cuda"))

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
    def test_native_entrypoint_rejects_invalid_values(self, bad):
        """Reject invalid values in native and wrapper calls even when neighbors are valid."""

        from rl_engine.kernels.ops.base import _C

        _cuda_op()
        # Valid neighbors must not hide an invalid timestep in the native call.
        t = torch.tensor([0.0, bad, 1.0], device="cuda")
        with pytest.raises(ValueError, match=r"finite and lie in \[0, 1\]"):
            _C.h3_timestep_sinusoid_forward(t)
        with pytest.raises(ValueError, match=r"finite and lie in \[0, 1\]"):
            _cuda_op()(t)

    def test_native_entrypoint_accepts_boundaries(self):
        """Accept both interval endpoints and match provider values in the native call."""

        from rl_engine.kernels.ops.base import _C

        _cuda_op()
        t = torch.tensor([0.0, 0.5, 1.0], device="cuda")
        assert torch.equal(_C.h3_timestep_sinusoid_forward(t), provider_time_proj(t, 256))

    def test_trusted_path_supports_cuda_graph(self):
        """Capture and replay the prevalidated sinusoid path without changing its output."""

        op = _cuda_op()
        t = h3_timesteps(7)
        expected = op(t)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            op.forward(t, check_range=False)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op.forward(t, check_range=False)
        graph.replay()
        assert torch.equal(out, expected)

    def test_backward_matches_autograd_reference(self):
        """Match timestep gradients to the high-precision autograd sinusoid replay."""

        t = h3_timesteps(17).requires_grad_(True)
        t_ref = t.detach().clone().requires_grad_(True)
        grad = torch.randn(17, 256, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        _cuda_op()(t).backward(grad)
        NativeH3TimestepSinusoidOp().forward_fp32(t_ref).backward(grad)
        torch.testing.assert_close(t.grad, t_ref.grad, atol=1e-4, rtol=1e-4)

    def test_backward_is_row_invariant(self):
        """Keep a timestep gradient bitwise unchanged when unrelated rows join the batch."""

        op = _cuda_op()
        grad_row = torch.randn(256, device="cuda", generator=torch.Generator("cuda").manual_seed(2))

        def grad_of(t, pos):
            """Differentiate one selected timestep using a fixed upstream feature-gradient row."""

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
        """Resolve the sinusoid registry entry to its dedicated CUDA implementation."""

        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("timestep_sinusoid_h3", device="cuda")
        assert type(op).__name__ == "H3TimestepSinusoidCudaOp"
