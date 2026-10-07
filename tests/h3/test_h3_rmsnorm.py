# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``h3_rmsnorm``: RMSNorm (eps 1e-5) and its fused AdaLN modulation.

* forward is bitwise equal to ``nn.RMSNorm`` and to diffusers'
  ``norm(x) * (1.0 + scale[i]) + shift[i]`` (block: ``adaln_indices``;
  ``norm_out``: ``timestep_indices``) on the pinned norm weights;
* rows are independent of batch size and position;
* backward is deterministic and checked against an FP64 golden;
* malformed inputs fail closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.h3.rmsnorm import NativeH3RMSNormOp
from rl_engine.testing.h3_cases import h3_packed_layout
from rl_engine.testing.h3_provider import provider_norm_modulate

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
NORMS = (
    "transformer_blocks.0.norm1.weight",
    "transformer_blocks.0.norm2.weight",
    "token_refiner.final_norm.weight",
    "norm_out.norm.weight",
)


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.rmsnorm import H3RMSNormCudaOp, h3_rmsnorm_available

    if not h3_rmsnorm_available():
        pytest.skip("rl_engine._C lacks h3_rmsnorm_*")
    return H3RMSNormCudaOp()


def _x(shape, dtype=torch.bfloat16, seed=0, scale=2.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(device=device, dtype=dtype)


def _table(rows, hidden, chunks, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    table = (torch.randn(rows, chunks * hidden, generator=g) * 0.5).to(dtype).cuda()
    return table.chunk(chunks, dim=-1)  # strided (R, H) views, like the AdaLN outputs


def _fp64_modulated(x, weight, shift, scale, index, eps=1e-5):
    x64 = x.double()
    n = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps) * weight.double()
    return n * (1 + scale.double().index_select(0, index)) + shift.double().index_select(0, index)


class TestReference:
    def test_golden_close_to_provider(self):
        x, w = _x((3, 64), torch.float32, device="cpu"), torch.rand(64) + 0.5
        op = NativeH3RMSNormOp()
        torch.testing.assert_close(op.forward(x, w), op.forward_fp32(x, w), atol=1e-6, rtol=1e-6)

    def test_modulated_with_prevalidated_indices(self):
        x, w = _x((2, 8), torch.float32, device="cpu"), torch.ones(8)
        shift, scale = torch.randn(3, 8), torch.randn(3, 8)
        index = torch.tensor([2, 0])
        op = NativeH3RMSNormOp()
        expected = op.forward(x, w) * (1 + scale[index]) + shift[index]
        torch.testing.assert_close(
            op.forward_modulated(x, w, shift, scale, index, check_range=False),
            expected,
        )

    def test_rejects_bad_inputs(self):
        op = NativeH3RMSNormOp()
        x, w = torch.randn(2, 8), torch.ones(8)
        with pytest.raises(TypeError):
            op.forward(x, w.double())
        with pytest.raises(TypeError):
            op.forward(x, w.bfloat16())
        with pytest.raises(ValueError):
            op.forward(x, torch.ones(7))
        with pytest.raises(ValueError):
            op.forward(x, w, eps=0.0)
        shift, scale = torch.zeros(3, 8), torch.zeros(3, 8)
        with pytest.raises(IndexError):
            op.forward_modulated(x, w, shift, scale, torch.tensor([0, 3]))
        with pytest.raises(ValueError):  # one index per position S
            op.forward_modulated(x, w, shift, scale, torch.tensor([0]))


@requires_cuda
class TestCudaForward:
    @pytest.mark.parametrize("name", NORMS)
    def test_bitwise_equal_to_nn_rmsnorm_on_pinned_weights(self, h3_weights_cpu, name):
        weight = h3_weights_cpu[name].cuda()
        module = torch.nn.RMSNorm(5376, eps=1e-5).cuda().bfloat16()
        module.weight.data.copy_(weight)
        x = _x((2, 777, 5376), seed=1)
        assert torch.equal(_cuda_op()(x, weight), module(x))

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("hidden", [12, 256, 4096, 5376])
    def test_bitwise_equal_to_f_rms_norm(self, dtype, hidden):
        x = _x((5, 33, hidden), dtype, seed=hidden, scale=30.0)
        w = (torch.rand(hidden, device="cuda") * 2).to(dtype)
        ref = torch.nn.functional.rms_norm(x, (hidden,), w, 1e-5)
        assert torch.equal(_cuda_op()(x, w), ref)

    def test_block_modulation_bitwise_equal_to_diffusers(self, h3_weights_cpu):
        weight = h3_weights_cpu["transformer_blocks.0.norm1.weight"].cuda()
        shift, scale, *_ = _table(9, 5376, 6, seed=2)  # (3T, H) rows, T = 3
        ti, tags = h3_packed_layout(4097, 3, seed=2)
        index = ti * 3 + tags
        x = _x((2, 4097, 5376), seed=2)
        ours = _cuda_op().forward_modulated(x, weight, shift, scale, index)
        assert torch.equal(ours, provider_norm_modulate(x, weight, shift, scale, index))

    def test_norm_out_modulation_bitwise_equal_to_diffusers(self, h3_weights_cpu):
        weight = h3_weights_cpu["norm_out.norm.weight"].cuda()
        shift, scale = _table(3, 5376, 2, seed=3)  # (T, H) rows indexed by timestep
        ti, _ = h3_packed_layout(1000, 3, seed=3)
        x = _x((1, 1000, 5376), seed=3)
        ours = _cuda_op().forward_modulated(x, weight, shift, scale, ti)
        assert torch.equal(ours, provider_norm_modulate(x, weight, shift, scale, ti))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_modulation_bitwise_in_every_dtype(self, dtype):
        """FP32 must not contract the separately rounded eager ops into an FMA."""

        w = (torch.rand(5376, device="cuda") + 0.5).to(dtype)
        shift, scale, *_ = _table(9, 5376, 6, dtype=dtype, seed=9)
        ti, tags = h3_packed_layout(333, 3, seed=9)
        index = ti * 3 + tags
        x = _x((2, 333, 5376), dtype, seed=9)
        ours = _cuda_op().forward_modulated(x, w, shift, scale, index)
        assert torch.equal(ours, provider_norm_modulate(x, w, shift, scale, index))

    def test_rows_are_batch_and_position_invariant(self):
        w = (torch.rand(5376, device="cuda") + 0.5).bfloat16()
        shift, scale, *_ = _table(9, 5376, 6, seed=4)
        ti, tags = h3_packed_layout(300, 3, seed=4)
        index = ti * 3 + tags
        x = _x((3, 300, 5376), seed=4)
        full = _cuda_op().forward_modulated(x, w, shift, scale, index)
        part = _cuda_op().forward_modulated(x[2:3, 50:90], w, shift, scale, index[50:90])
        assert torch.equal(part[0], full[2, 50:90])

    def test_rejects_unsupported(self):
        w = torch.ones(6, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="multiple of 4"):
            _cuda_op()(_x((2, 6)), w)
        with pytest.raises(ValueError):
            _cuda_op()(_x((2, 8)).cpu(), torch.ones(8).bfloat16())


@requires_cuda
class TestCudaBackward:
    def _grads(self, fn, tensors, grad):
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(*leaves).backward(grad)
        return [leaf.grad for leaf in leaves]

    def test_modulated_backward_against_fp64(self):
        g = torch.Generator(device="cpu").manual_seed(5)
        w = (torch.rand(5376, generator=g) + 0.5).bfloat16().cuda()
        table = (torch.randn(9, 2 * 5376, generator=g) * 0.5).bfloat16().cuda()
        ti, tags = h3_packed_layout(2049, 3, seed=5)
        index = ti * 3 + tags
        x = _x((2, 2049, 5376), seed=5)
        grad = _x((2, 2049, 5376), seed=6, scale=1.0)
        op = _cuda_op()

        def ours(x_, w_, t_):
            shift, scale = t_.chunk(2, dim=-1)
            return op.forward_modulated(x_, w_, shift, scale, index)

        def golden(x_, w_, t_):
            shift, scale = t_.chunk(2, dim=-1)
            return _fp64_modulated(x_, w_, shift, scale, index)

        got = self._grads(ours, (x, w, table), grad)
        ref = self._grads(golden, (x.double(), w.double(), table.double()), grad.double())
        for name, g_, r in zip(("dx", "dweight", "dtable"), got, ref):
            rel = ((g_.double() - r).abs().max() / r.abs().max()).item()
            # Two BF16 roundings remain: the eager VJP uses round(1 + scale), and
            # every gradient is rounded once to BF16 (2^-9 relative each).
            assert rel < 1e-2, (name, rel)
        again = self._grads(ours, (x, w, table), grad)
        assert all(torch.equal(a, b) for a, b in zip(got, again))

    def test_plain_backward_against_fp64_and_row_local_dx(self):
        w = (torch.rand(5376, generator=torch.Generator().manual_seed(7)) + 0.5).bfloat16().cuda()
        x = _x((4, 64, 5376), seed=7)
        grad = _x((4, 64, 5376), seed=8, scale=1.0)
        op = _cuda_op()
        got = self._grads(lambda a, b: op(a, b), (x, w), grad)

        def golden(a, b):
            return a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-5) * b

        ref = self._grads(golden, (x.double(), w.double()), grad.double())
        for g, r in zip(got, ref):
            assert ((g.double() - r).abs().max() / r.abs().max()).item() < 5e-3
        part = self._grads(lambda a, b: op(a, b), (x[1:2, 10:20], w), grad[1:2, 10:20])
        assert torch.equal(part[0][0], got[0][1, 10:20])

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        assert (
            type(KernelRegistry().get_op("h3_rmsnorm", device="cuda")).__name__ == "H3RMSNormCudaOp"
        )
