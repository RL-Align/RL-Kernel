# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``adaln_projection_3mod``: SiLU + 2688 -> 96768 projection, 6 x 3 table.

* layout: the six outputs and the three modality rows per timestep land
  exactly where diffusers puts them (RFC probe H4);
* mixed precision: SiLU in FP32, one cast to BF16; an early cast (probe H7)
  is rejected at the API and is measurably different numerically;
* accuracy: forward and gradients against the declared-cast golden on the
  pinned block-0 weights;
* invariance and determinism: a timestep's 18 modulation rows do not depend
  on the other timesteps, and repeated runs are bitwise equal.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM
from rl_engine.kernels.ops.pytorch.h3.adaln_projection import NativeH3AdaLNProjectionOp
from rl_engine.testing.h3_provider import provider_adaln_modulation

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
WEIGHT = "transformer_blocks.0.adaln_proj.linear.weight"
BIAS = "transformer_blocks.0.adaln_proj.linear.bias"


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.adaln_projection import H3AdaLNProjectionCudaOp
    from rl_engine.kernels.ops.cuda.h3.det_linear import det_linear_available

    if not det_linear_available():
        pytest.skip("rl_engine._C lacks h3_det_linear_*")
    return H3AdaLNProjectionCudaOp()


def _temb(num, dim=2688, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(num, dim, generator=g) * 2).to(device)


def _synthetic(hidden, dim, dtype=torch.bfloat16, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_out = H3_ADALN_CHUNKS * H3_MODALITY_NUM * hidden
    weight = (torch.randn(n_out, dim, generator=g) / dim**0.5).to(dtype)
    bias = (torch.randn(n_out, generator=g) * 0.1).to(dtype)
    return weight.to(device), bias.to(device)


def _flat(outputs):
    return torch.cat([out.reshape(-1) for out in outputs])


class TestReference:
    def test_provider_layout_matches_diffusers(self):
        temb = _temb(2, dim=16, device="cpu")
        weight, bias = _synthetic(8, 16, device="cpu")
        ours = NativeH3AdaLNProjectionOp().forward(temb, weight, bias)
        theirs = provider_adaln_modulation(temb, weight, bias, hidden_size=8)
        assert len(ours) == 6
        for a, b in zip(ours, theirs):
            assert a.shape == (6, 8) and torch.equal(a, b)

    def test_rejects_bf16_temb(self):
        # RFC probe H7: casting temb before the SiLU.
        weight, bias = _synthetic(8, 16, device="cpu")
        with pytest.raises(TypeError, match="float32"):
            NativeH3AdaLNProjectionOp().forward(_temb(2, 16, device="cpu").bfloat16(), weight, bias)

    def test_rejects_bad_shapes(self):
        weight, bias = _synthetic(8, 16, device="cpu")
        op = NativeH3AdaLNProjectionOp()
        with pytest.raises(ValueError):  # not a multiple of 6 * 3 rows
            op.forward(_temb(2, 16, device="cpu"), weight[:-1], bias[:-1])
        with pytest.raises(ValueError):
            op.forward(_temb(2, 15, device="cpu"), weight, bias)
        with pytest.raises(ValueError):
            op.forward(_temb(0, 16, device="cpu"), weight, bias)
        with pytest.raises(TypeError):
            op.forward(_temb(2, 16, device="cpu"), weight, bias.float())


@requires_cuda
class TestLayout:
    def test_channel_identity_lands_in_its_chunk_and_modality_row(self):
        """Probe H4: bias o = m*6H + c*H + h tagged (m, c); zero weight passes it through."""

        hidden, dim, num = 16, 32, 3
        n_out = H3_ADALN_CHUNKS * H3_MODALITY_NUM * hidden
        o = torch.arange(n_out)
        modality, chunk = o // (H3_ADALN_CHUNKS * hidden), (o // hidden) % H3_ADALN_CHUNKS
        bias = (modality * 10 + chunk).to(torch.bfloat16).cuda()
        weight = torch.zeros(n_out, dim, dtype=torch.bfloat16, device="cuda")
        outputs = _cuda_op()(_temb(num, dim), weight, bias)
        assert len(outputs) == 6
        for c, out in enumerate(outputs):
            assert out.shape == (num * 3, hidden)
            for t in range(num):
                for m in range(3):
                    expected = torch.full((hidden,), float(m * 10 + c), device="cuda")
                    assert torch.equal(out[t * 3 + m].float(), expected), (c, t, m)

    def test_outputs_are_views_of_one_table_like_diffusers(self):
        weight, bias = _synthetic(16, 32)
        outputs = _cuda_op()(_temb(2, 32), weight, bias)
        base = outputs[0].untyped_storage().data_ptr()
        assert all(out.untyped_storage().data_ptr() == base for out in outputs)
        assert outputs[1].data_ptr() - outputs[0].data_ptr() == 16 * outputs[0].element_size()


@pytest.fixture(scope="module")
def block0(h3_weights_cpu):
    return h3_weights_cpu[WEIGHT].cuda(), h3_weights_cpu[BIAS].cuda()


@requires_cuda
class TestCudaRealWeights:
    @pytest.mark.parametrize("num", [1, 2, 3, 4])
    def test_forward_is_correctly_rounded_declared_golden(self, block0, num):
        weight, bias = block0
        temb = _temb(num, seed=num)
        outputs = _cuda_op()(temb, weight, bias)
        golden = NativeH3AdaLNProjectionOp().forward_fp32(temb, weight, bias)
        for out, gold in zip(outputs, golden):
            assert out.dtype == torch.bfloat16 and out.shape == (3 * num, 5376)
            # tolerance_contract.json forward_accuracy / reduction / bfloat16.
            torch.testing.assert_close(out.float(), gold, atol=5e-2, rtol=2e-2)
        # Beyond the contract: almost every element is the correctly rounded
        # golden value; the rest are 1-ULP ties of the FP32 accumulation.
        match = (_flat(outputs) == _flat(golden).bfloat16()).float().mean().item()
        assert match > 0.999

    def test_early_cast_is_numerically_detectable(self, block0):
        """Probe H7: a BF16 cast before the SiLU moves about half the outputs."""

        weight, bias = block0
        temb = _temb(3, seed=7)
        op = NativeH3AdaLNProjectionOp()
        ours = _flat(_cuda_op()(temb, weight, bias))
        declared = _flat(op.forward_fp32(temb, weight, bias)).bfloat16()
        early = _flat(op.forward_fp32(temb.bfloat16().float(), weight, bias)).bfloat16()
        assert (ours == declared).float().mean() > 0.999
        assert (ours == early).float().mean() < 0.8

    def test_rows_are_batch_and_position_invariant(self, block0):
        weight, bias = block0
        op = _cuda_op()
        temb = _temb(5, seed=3)
        full = op(temb, weight, bias)
        for i in range(5):
            single = op(temb[i : i + 1], weight, bias)
            for s, f in zip(single, full):
                assert torch.equal(s, f[3 * i : 3 * i + 3])
        swapped = op(temb.flip(0), weight, bias)
        for s, f in zip(swapped, full):
            assert torch.equal(s.view(5, 3, -1).flip(0), f.view(5, 3, -1))

    def test_backward_against_golden(self, block0):
        weight, bias = block0
        temb = _temb(2, seed=5)
        g = torch.Generator(device="cuda").manual_seed(0)
        grads = [torch.randn(6, 5376, device="cuda", generator=g).bfloat16() for _ in range(6)]
        leaves = [t.detach().clone().requires_grad_(True) for t in (temb, weight, bias)]
        torch.autograd.backward(list(_cuda_op()(*leaves)), grads)

        ref = [t.detach().double().requires_grad_(True) for t in (temb, weight, bias)]
        act = ref[0] * torch.sigmoid(ref[0])
        act = act + (act.to(torch.bfloat16).double() - act).detach()  # declared cast, identity VJP
        table = F.linear(act, ref[1], ref[2]).view(-1, 6 * 5376).chunk(6, dim=-1)
        torch.autograd.backward(list(table), [gr.double() for gr in grads])
        # tolerance_contract.json gradient_accuracy / reduction: FP32 temb grad
        # and BF16 weight/bias grads (rounded once from an FP32 accumulation).
        torch.testing.assert_close(leaves[0].grad.double(), ref[0].grad, atol=1e-4, rtol=1e-4)
        for leaf, r in zip(leaves[1:], ref[1:]):
            torch.testing.assert_close(leaf.grad.double(), r.grad, atol=1e-1, rtol=2e-2)
            assert (leaf.grad == r.grad.to(torch.bfloat16)).float().mean() > 0.999


@requires_cuda
class TestCudaSynthetic:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("hidden, dim", [(24, 16), (40, 72), (8, 2688)])
    def test_other_sizes(self, dtype, hidden, dim):
        weight, bias = _synthetic(hidden, dim, dtype=dtype, seed=hidden)
        temb = _temb(3, dim, seed=hidden)
        tol = dict(atol=5e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)
        for out, gold in zip(
            _cuda_op()(temb, weight, bias),
            NativeH3AdaLNProjectionOp().forward_fp32(temb, weight, bias),
        ):
            torch.testing.assert_close(out.float(), gold, **tol)

    def test_repeat_forward_backward_bitwise(self):
        weight, bias = _synthetic(64, 2688, seed=9)
        temb = _temb(3, seed=9)
        grads = [torch.randn(9, 64, device="cuda").bfloat16() for _ in range(6)]
        runs = []
        for _ in range(3):
            leaves = [t.detach().clone().requires_grad_(True) for t in (temb, weight, bias)]
            outs = _cuda_op()(*leaves)
            torch.autograd.backward(list(outs), grads)
            runs.append([*[o.detach() for o in outs], *[leaf.grad for leaf in leaves]])
        for later in runs[1:]:
            for a, b in zip(runs[0], later):
                assert torch.equal(a, b)

    def test_temb_grad_rows_are_batch_invariant(self):
        weight, bias = _synthetic(64, 2688, seed=11)
        temb = _temb(4, seed=11)
        grads = [torch.randn(12, 64, device="cuda").bfloat16() for _ in range(6)]

        def d_temb(rows):
            leaf = temb[rows].detach().clone().requires_grad_(True)
            outs = _cuda_op()(leaf, weight, bias)
            torch.autograd.backward(
                list(outs), [g.view(4, 3, -1)[rows].reshape(-1, 64) for g in grads]
            )
            return leaf.grad

        full = d_temb(slice(0, 4))
        for i in range(4):
            assert torch.equal(d_temb(slice(i, i + 1))[0], full[i])

    @pytest.mark.parametrize("num", [8, 9, 17])
    def test_rows_invariant_across_tensor_core_row_tiles(self, num):
        """The BF16 path computes rows in tiles of 8; a row's bits must not depend on its tile."""

        weight, bias = _synthetic(64, 2688, seed=13)
        temb = _temb(num, seed=num)
        full = _cuda_op()(temb, weight, bias)
        for i in (0, 7, num - 1):
            single = _cuda_op()(temb[i : i + 1], weight, bias)
            for s_out, f_out in zip(single, full):
                assert torch.equal(s_out, f_out[3 * i : 3 * i + 3])

    def test_rejects_cpu(self):
        weight, bias = _synthetic(8, 16, device="cpu")
        with pytest.raises(ValueError):
            _cuda_op()(_temb(2, 16, device="cpu"), weight, bias)

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("adaln_projection_3mod", device="cuda")
        assert type(op).__name__ == "H3AdaLNProjectionCudaOp"
