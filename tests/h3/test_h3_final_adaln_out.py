# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``final_adaln_out``: norm_out shift/scale projection + final norm/modulation.

* layout: ``shift`` is the first half of ``norm_out.linear``'s output, rows
  are indexed by ``timestep_indices`` (not ``adaln_indices``);
* forward matches diffusers' ``norm_out`` to the projection's 1-ULP ties
  (the norm and modulation are bitwise; the GEMV tree differs from cuBLAS);
* backward is deterministic, with the table gradient kept in FP32;
* rows are batch/position invariant; malformed inputs fail closed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.h3.final_adaln_out import NativeH3FinalAdaLNOutOp
from rl_engine.testing.h3_cases import h3_packed_layout
from rl_engine.testing.h3_provider import provider_final_adaln_out

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.final_adaln_out import H3FinalAdaLNOutCudaOp
    from rl_engine.kernels.ops.cuda.h3.rmsnorm import h3_rmsnorm_available

    if not h3_rmsnorm_available():
        pytest.skip("rl_engine._C lacks h3_rmsnorm_*")
    return H3FinalAdaLNOutCudaOp()


@pytest.fixture(scope="module")
def norm_out(h3_weights_cpu):
    return [
        h3_weights_cpu[name].cuda()
        for name in ("norm_out.norm.weight", "norm_out.linear.weight", "norm_out.linear.bias")
    ]


def _inputs(seq, num_timesteps=3, seed=0, batch=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(batch, seq, 5376, generator=g) * 2).bfloat16().cuda()
    temb = (torch.randn(num_timesteps, 2688, generator=g) * 2).cuda()
    ti, _ = h3_packed_layout(seq, num_timesteps, seed=seed)
    return x, temb, ti


class TestReference:
    def test_rejects_bad_inputs(self):
        op = NativeH3FinalAdaLNOutOp()
        x, nw = torch.zeros(1, 4, 8), torch.ones(8)
        temb, w, b = torch.zeros(2, 16), torch.zeros(16, 16), torch.zeros(16)
        ti = torch.tensor([0, 1, 1, 0])
        with pytest.raises(TypeError):  # BF16 temb (probe H7)
            op(x, nw, temb.bfloat16(), w, b, ti)
        with pytest.raises(ValueError):  # not 2H rows
            op(x, nw, temb, w[:10], b[:10], ti)
        with pytest.raises(IndexError):
            op(x, nw, temb, w, b, ti + 2)


@requires_cuda
class TestCuda:
    def test_shift_first_layout_and_timestep_indexing(self):
        """Zero weights: the bias halves are (shift, scale); rows follow timestep_indices."""

        hidden, seq = 16, 6
        x = torch.randn(1, seq, hidden, device="cuda").bfloat16()
        nw = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.zeros(2 * hidden, 32, device="cuda", dtype=torch.bfloat16)
        b = torch.cat([torch.full((hidden,), 1.0), torch.full((hidden,), -1.0)]).bfloat16().cuda()
        ti = torch.tensor([0, 1, 0, 1, 1, 0], device="cuda")
        out = _cuda_op()(x, nw, torch.zeros(2, 32, device="cuda"), w, b, ti)
        # scale = -1 -> norm(x) * 0 + shift = 1 everywhere
        assert torch.equal(out, torch.ones_like(out))

    @pytest.mark.parametrize("seq, num_timesteps", [(1, 1), (257, 2), (4097, 3)])
    def test_forward_against_diffusers_and_golden(self, norm_out, seq, num_timesteps):
        x, temb, ti = _inputs(seq, num_timesteps, seed=seq)
        ours = _cuda_op()(x, *norm_out[:1], temb, *norm_out[1:], ti)
        theirs = provider_final_adaln_out(x, norm_out[0], temb, *norm_out[1:], ti)
        golden = NativeH3FinalAdaLNOutOp().forward_fp32(x, norm_out[0], temb, *norm_out[1:], ti)
        assert (ours == theirs).float().mean() > 0.999
        torch.testing.assert_close(ours.float(), golden, atol=5e-2, rtol=2e-2)

    def test_rows_are_batch_and_position_invariant(self, norm_out):
        x, temb, ti = _inputs(600, seed=1, batch=2)
        full = _cuda_op()(x, norm_out[0], temb, *norm_out[1:], ti)
        part = _cuda_op()(x[1:2, 200:260], norm_out[0], temb, *norm_out[1:], ti[200:260])
        assert torch.equal(part[0], full[1, 200:260])

    def test_backward_deterministic_and_fp32_table_gradient(self, norm_out):
        x, temb, ti = _inputs(4097, seed=2)
        grad = torch.randn_like(x)

        def grads(fn, dtype=None):
            tensors = [x, norm_out[0], temb, *norm_out[1:]]
            leaves = [
                (t if dtype is None else t.to(dtype)).detach().clone().requires_grad_(True)
                for t in tensors
            ]
            fn(*leaves).backward(grad if dtype is None else grad.to(dtype))
            return [leaf.grad for leaf in leaves]

        def golden(x_, nw, t, w, b):
            act = t * torch.sigmoid(t)
            act = act + (act.to(torch.bfloat16).double() - act).detach()
            shift, scale = F.linear(act, w, b).chunk(2, dim=-1)
            n = x_ * torch.rsqrt(x_.square().mean(-1, keepdim=True) + 1e-5) * nw
            return n * (1 + scale.index_select(0, ti)) + shift.index_select(0, ti)

        ours = grads(lambda *t: _cuda_op()(*t, ti))
        again = grads(lambda *t: _cuda_op()(*t, ti))
        ref = grads(golden, torch.float64)
        assert all(torch.equal(a, b) for a, b in zip(ours, again))
        for name, g, r in zip(("dx", "d_norm_weight", "d_temb", "dW", "db"), ours, ref):
            rel = ((g.double() - r).abs().max() / r.abs().max()).item()
            assert rel < 1e-2, (name, rel)  # BF16 output rounding and round(1 + scale) only

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("final_adaln_out", device="cuda")
        assert type(op).__name__ == "H3FinalAdaLNOutCudaOp"
