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

from rl_engine.validation.models.h3_cases import h3_packed_layout

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _ops():
    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_modulation import (
        H3AdaLNModulationCudaOp,
    )
    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_projection import (
        H3AdaLNProjectionCudaOp,
    )
    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
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


def test_pre_sm80_device_is_refused_at_construction_so_the_registry_can_fall_back(monkeypatch):
    from rl_engine.backends.cuda.model_specific.minimax_h3 import adaln_modulation as module

    monkeypatch.setattr(module, "det_linear_available", lambda: True)
    monkeypatch.setattr(module, "adaln_row_gather_available", lambda: True)
    monkeypatch.setattr(module, "det_linear_device_supported", lambda: False)
    with pytest.raises(RuntimeError, match="SM80"):
        module.H3AdaLNModulationCudaOp()


@requires_cuda
class TestFusedModulation:
    def test_indices_on_another_device_than_temb_are_rejected(self):
        fused, _, _ = _ops()
        temb, weight, bias = _params(64, 2688)
        ti, tags = h3_packed_layout(1000, 3)
        with pytest.raises(ValueError, match="rows are on"):
            fused(temb, weight, bias, ti.cpu(), tags.cpu(), check_range=False)

    def test_forward_bitwise_equal_to_separate_ops(self):
        fused, proj, gather = _ops()
        temb, weight, bias = _params(64, 2688)
        ti, tags = h3_packed_layout(1000, 3)
        ours = fused(temb, weight, bias, ti, tags)
        separate = gather.gather_chunks(proj(temb, weight, bias), ti, tags)
        assert all(torch.equal(a, b) for a, b in zip(ours, separate, strict=True))

    def test_backward_keeps_table_gradient_fp32(self, h3_weights_cpu):
        fused, proj, gather = _ops()
        weight = h3_weights_cpu["transformer_blocks.0.adaln_proj.linear.weight"].cuda()
        bias = h3_weights_cpu["transformer_blocks.0.adaln_proj.linear.bias"].cuda()
        g = torch.Generator(device="cuda").manual_seed(0)
        temb = torch.randn(3, 2688, device="cuda", generator=g) * 2
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
        # The declared cast rounds the FP32 SiLU, as the kernel and diffusers do: an
        # FP64 SiLU can round to the other BF16 neighbour near a tie, which shifts a
        # whole dW column (about 1 temb draw in 30 dropped below 0.9999 that way).
        act_bf16 = F.silu(temb.float()).to(torch.bfloat16).double()
        act = act + (act_bf16 - act).detach()  # identity-VJP cast
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
        for grad, golden in zip(ours[1:], (ref[1].grad, ref[2].grad), strict=True):
            assert (grad == golden.to(torch.bfloat16)).float().mean() > 0.9999

    @pytest.mark.parametrize("frozen", ["bias", "weight", "temb", "bias_and_temb"])
    def test_frozen_inputs_get_no_gradient_and_the_others_are_unchanged(self, frozen):
        fused, _, _ = _ops()
        temb, weight, bias = _params(64, 2688, seed=5)
        ti, tags = h3_packed_layout(1500, 3, seed=5)
        grads = _grads(1500, 64, seed=5)
        full = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)

        leaves = [t.detach().clone() for t in (temb, weight, bias)]
        names = ("temb", "weight", "bias")
        for leaf, name in zip(leaves, names, strict=True):
            leaf.requires_grad_(name not in frozen)
        torch.autograd.backward(list(fused(*leaves, ti, tags)), grads)
        for leaf, name, expected in zip(leaves, names, full, strict=True):
            if name in frozen:
                assert leaf.grad is None
            else:
                assert torch.equal(leaf.grad, expected)

    def test_backward_repeat_bitwise(self):
        fused, _, _ = _ops()
        temb, weight, bias = _params(64, 2688, seed=3)
        ti, tags = h3_packed_layout(3000, 3, seed=3)
        grads = _grads(3000, 64, seed=3)
        first = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)
        again = _leaf_grads(lambda t, w, b: fused(t, w, b, ti, tags), temb, weight, bias, grads)
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))

    def test_rejects_bad_indices(self):
        fused, _, _ = _ops()
        temb, weight, bias = _params(8, 16)
        ti, tags = h3_packed_layout(10, 3)
        with pytest.raises(IndexError):
            fused(temb, weight, bias, ti + 3, tags)
        with pytest.raises(TypeError):
            fused(temb.bfloat16(), weight, bias, ti, tags)
