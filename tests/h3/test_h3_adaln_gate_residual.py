# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``adaln_gate_residual``: ``residual + gate[row] * sublayer_output``.

* forward and ``d_sublayer`` are bitwise equal to diffusers' eager
  expression in every dtype (the gate row is gathered in the kernel);
* ``d_gate`` is a deterministic FP32 segment sum, checked against FP64;
* rows are elementwise-independent; malformed inputs fail closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.h3.gate_residual import NativeH3GateResidualOp
from rl_engine.testing.h3_cases import h3_packed_layout
from rl_engine.testing.h3_provider import provider_gate_residual

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.gate_residual import (
        H3GateResidualCudaOp,
        h3_gate_residual_available,
    )

    if not h3_gate_residual_available():
        pytest.skip("rl_engine._C lacks h3_gate_residual_*")
    return H3GateResidualCudaOp()


def _case(seq, hidden=5376, dtype=torch.bfloat16, seed=0, batch=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    table = (torch.randn(9, 6 * hidden, generator=g) * 0.5).to(dtype).cuda()
    residual = torch.randn(batch, seq, hidden, generator=g).to(dtype).cuda()
    y = (torch.randn(batch, seq, hidden, generator=g) * 3).to(dtype).cuda()
    ti, tags = h3_packed_layout(seq, 3, seed=seed)
    return table, residual, y, ti * 3 + tags


def _gate(table, chunk=2):
    return table.view(table.shape[0], 6, -1)[:, chunk]  # gate_msa (2) / gate_mlp (5) view


class TestReference:
    def test_rejects_bad_inputs(self):
        op = NativeH3GateResidualOp()
        res, y, gate = torch.zeros(4, 8), torch.zeros(4, 8), torch.zeros(3, 8)
        with pytest.raises(IndexError):
            op(res, y, gate, torch.tensor([0, 1, 2, 3]))
        with pytest.raises(ValueError):
            op(res, y[:, :4], gate, torch.tensor([0, 1, 2, 0]))
        with pytest.raises(TypeError):
            op(res, y, gate.bfloat16(), torch.tensor([0, 1, 2, 0]))
        with pytest.raises(ValueError):  # S mismatch
            op(res, y, gate, torch.tensor([0, 1]))


@requires_cuda
class TestCuda:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("seq, hidden", [(4097, 5376), (33, 12), (17, 13)])
    def test_forward_bitwise_equal_to_diffusers(self, dtype, seq, hidden):
        table, residual, y, index = _case(seq, hidden, dtype, seed=seq)
        for chunk in (2, 5):
            ours = _cuda_op()(residual, y, _gate(table, chunk), index)
            assert torch.equal(
                ours, provider_gate_residual(residual, _gate(table, chunk), index, y)
            )

    def test_rows_are_batch_and_position_invariant(self):
        table, residual, y, index = _case(500, seed=1)
        full = _cuda_op()(residual, y, _gate(table), index)
        part = _cuda_op()(residual[1:2, 100:150], y[1:2, 100:150], _gate(table), index[100:150])
        assert torch.equal(part[0], full[1, 100:150])

    def test_backward(self):
        table, residual, y, index = _case(4097, seed=2)
        grad = torch.randn_like(residual)

        def grads(fn, dtype=None):
            tensors = [t if dtype is None else t.to(dtype) for t in (residual, y, table)]
            leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
            fn(leaves[0], leaves[1], _gate(leaves[2])).backward(
                grad if dtype is None else grad.to(dtype)
            )
            return [leaf.grad for leaf in leaves]

        ours = grads(lambda r, yy, gg: _cuda_op()(r, yy, gg, index))
        again = grads(lambda r, yy, gg: _cuda_op()(r, yy, gg, index))
        theirs = grads(lambda r, yy, gg: provider_gate_residual(r, gg, index, yy))
        golden = grads(lambda r, yy, gg: provider_gate_residual(r, gg, index, yy), torch.float64)
        assert all(torch.equal(a, b) for a, b in zip(ours, again))
        assert torch.equal(ours[0], theirs[0])  # d_residual
        assert torch.equal(ours[1], theirs[1])  # d_sublayer: exact product, one rounding
        rel = ((ours[2].double() - golden[2]).abs().max() / golden[2].abs().max()).item()
        assert rel < 5e-3  # only the BF16 rounding of the table gradient
        assert torch.count_nonzero(ours[2].view(9, 6, -1)[:, [0, 1, 3, 4, 5]]) == 0

    def test_rejects_cpu_and_bad_index(self):
        table, residual, y, index = _case(10, hidden=16)
        with pytest.raises(IndexError):
            _cuda_op()(residual, y, _gate(table), index + 9)
        with pytest.raises(ValueError):
            _cuda_op()(residual.cpu(), y.cpu(), _gate(table).cpu(), index.cpu())

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("adaln_gate_residual", device="cuda")
        assert type(op).__name__ == "H3GateResidualCudaOp"
