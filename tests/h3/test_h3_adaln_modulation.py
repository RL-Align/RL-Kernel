# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fused AdaLN modulation (projection + row gather) with an FP32 table gradient.

The forward must be bitwise equal to the two RFC #420 ops run separately; the
backward must be deterministic and must not re-round the table gradient to
BF16 (so ``d_temb`` reaches FP64-golden accuracy).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.testing.h3_cases import h3_packed_layout

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _ops():
    from rl_engine.kernels.ops.cuda.h3.adaln_modulation import H3AdaLNModulationCudaOp
    from rl_engine.kernels.ops.cuda.h3.adaln_projection import H3AdaLNProjectionCudaOp
    from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import (
        H3AdaLNRowGatherCudaOp,
        adaln_row_gather_available,
    )

    if not adaln_row_gather_available():
        pytest.skip("rl_engine._C lacks the H3 kernels")
    return H3AdaLNModulationCudaOp(), H3AdaLNProjectionCudaOp(), H3AdaLNRowGatherCudaOp()


def _params(hidden, dim, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_out = 18 * hidden
    weight = (torch.randn(n_out, dim, generator=g) / dim**0.5).bfloat16().cuda()
    bias = (torch.randn(n_out, generator=g) * 0.1).bfloat16().cuda()
    temb = (torch.randn(3, dim, generator=g) * 2).cuda()
    return temb, weight, bias


def _grads(seq, hidden, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(seq, hidden, device="cuda", generator=g).bfloat16() for _ in range(6)]


def _leaf_grads(fn, temb, weight, bias, grads):
    leaves = [t.detach().clone().requires_grad_(True) for t in (temb, weight, bias)]
    torch.autograd.backward(list(fn(*leaves)), grads)
    return [leaf.grad for leaf in leaves]


@requires_cuda
class TestFusedModulation:
    def test_forward_bitwise_equal_to_separate_ops(self):
        fused, proj, gather = _ops()
        temb, weight, bias = _params(64, 2688)
        ti, tags = h3_packed_layout(1000, 3)
        ours = fused(temb, weight, bias, ti, tags)
        separate = gather.gather_chunks(proj(temb, weight, bias), ti, tags)
        assert all(torch.equal(a, b) for a, b in zip(ours, separate))

    def test_backward_keeps_table_gradient_fp32(self, h3_weights_cpu):
        fused, proj, gather = _ops()
        weight = h3_weights_cpu["transformer_blocks.0.adaln_proj.linear.weight"].cuda()
        bias = h3_weights_cpu["transformer_blocks.0.adaln_proj.linear.bias"].cuda()
        temb = torch.randn(3, 2688, device="cuda") * 2
        ti, tags = h3_packed_layout(4097, 3, seed=1)
        grads = _grads(4097, 5376, seed=1)
        ours = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)
        separate = _leaf_grads(
            lambda t, w, b: gather.gather_chunks(proj(t, w, b), ti, tags),
            temb,
            weight,
            bias,
            grads,
        )

        ref = [t.detach().double().requires_grad_(True) for t in (temb, weight, bias)]
        act = ref[0] * torch.sigmoid(ref[0])
        act = act + (act.to(torch.bfloat16).double() - act).detach()  # identity-VJP cast
        table = F.linear(act, ref[1], ref[2]).view(-1, 6 * 5376)
        index = ti * 3 + tags
        torch.autograd.backward(
            [chunk.index_select(0, index) for chunk in table.chunk(6, dim=-1)],
            [g.double() for g in grads],
        )

        def rel(grad, golden):
            return ((grad.double() - golden).abs().max() / golden.abs().max()).item()

        # d_temb: no BF16 rounding anywhere on its path, so it reaches FP32 accuracy.
        assert rel(ours[0], ref[0].grad) < 1e-5
        assert rel(ours[0], ref[0].grad) < rel(separate[0], ref[0].grad) / 100
        # dW, db: only their own final BF16 rounding remains.
        for grad, golden in zip(ours[1:], (ref[1].grad, ref[2].grad)):
            assert (grad == golden.to(torch.bfloat16)).float().mean() > 0.9999

    def test_backward_repeat_bitwise(self):
        fused, _, _ = _ops()
        temb, weight, bias = _params(64, 2688, seed=3)
        ti, tags = h3_packed_layout(3000, 3, seed=3)
        grads = _grads(3000, 64, seed=3)
        first = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)
        again = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)
        assert all(torch.equal(a, b) for a, b in zip(first, again))

    def test_rejects_bad_indices(self):
        fused, _, _ = _ops()
        temb, weight, bias = _params(8, 16)
        ti, tags = h3_packed_layout(10, 3)
        with pytest.raises(IndexError):
            fused(temb, weight, bias, ti + 3, tags)
        with pytest.raises(TypeError):
            fused(temb.bfloat16(), weight, bias, ti, tags)
