# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``adaln_gate_residual``: ``residual + gate[row] * sublayer_output``.

* forward and ``d_sublayer`` are bitwise equal to diffusers' eager
  expression in every dtype (the gate row is gathered in the kernel);
* ``d_gate`` is a deterministic FP32 segment sum, checked against FP64;
* rows are elementwise-independent; malformed inputs fail closed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.reference.minimax_h3.gate_residual import NativeH3GateResidualOp
from rl_engine.validation.models.h3_cases import h3_packed_layout
from rl_engine.validation.models.h3_provider import provider_gate_residual

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    from rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual import (
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
    def test_gate_backward_is_bitwise_the_per_row_scan_it_replaced(self):
        from rl_engine.reference.minimax_h3.gate_residual import _DeterministicGateResidual

        g = torch.Generator(device="cpu").manual_seed(5)
        residual, y = torch.randn(2, 300, 16, generator=g), torch.randn(2, 300, 16, generator=g)
        gate = torch.randn(7, 16, generator=g)
        index = torch.randint(0, 5, (300,), generator=g)  # rows 5 and 6 are never referenced
        grad = torch.randn(2, 300, 16, generator=g)
        leaf = gate.clone().requires_grad_(True)
        _DeterministicGateResidual.apply(residual, y, leaf, index).backward(grad)

        per_position = (grad * y).reshape(-1, 300, 16).sum(dim=0)
        expected = torch.zeros(7, 16)
        for row in range(7):  # the loop of the earlier implementation
            positions = torch.nonzero(index == row).flatten()
            if positions.numel():
                expected[row] = per_position.index_select(0, positions).sum(dim=0)
        assert torch.equal(leaf.grad, expected)

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

    def test_residual_gradient_keeps_the_residual_shape_for_batched_and_strided_gradients(self):
        """A (B, S, H) residual gets a (B, S, H) gradient even when the incoming one is a view."""

        table, residual, y, index = _case(300, hidden=64, seed=5)
        leaves = [t.detach().clone().requires_grad_(True) for t in (residual, y, table)]
        out = _cuda_op()(leaves[0], leaves[1], _gate(leaves[2]), index)
        # a non-contiguous cotangent: the transpose of a (B, H, S) tensor
        grad = torch.randn(2, 64, 300, device="cuda").bfloat16().transpose(1, 2)
        assert not grad.is_contiguous() and grad.shape == out.shape
        out.backward(grad)
        ref = [t.detach().clone().requires_grad_(True) for t in (residual, y, table)]
        provider_gate_residual(ref[0], _gate(ref[2]), index, ref[1]).backward(grad)
        assert leaves[0].grad.shape == residual.shape and leaves[1].grad.shape == y.shape
        assert torch.equal(leaves[0].grad, ref[0].grad)
        assert torch.equal(leaves[1].grad, ref[1].grad)

    def test_a_gate_table_whose_rows_overlap_is_copied_not_read_in_place(self):
        """An expanded table (row stride 0) is a legal gate: it is copied to dense rows first."""

        from rl_engine.backends.extension import _C

        hidden = 64
        row = (torch.randn(1, hidden, device="cuda") * 0.5).bfloat16()
        expanded = row.expand(9, hidden)  # stride (0, 1): all nine rows alias one
        assert expanded.stride(0) == 0
        _, residual, y, index = _case(100, hidden=hidden, seed=6)
        index = index % 9
        dense = expanded.contiguous()
        got = _cuda_op()(residual, y, expanded, index)
        assert torch.equal(got, _cuda_op()(residual, y, dense, index))
        assert torch.equal(got, provider_gate_residual(residual, dense, index, y))
        # the native entry point refuses the layout instead of reading overlapping rows
        res2, y2 = residual.reshape(-1, hidden), y.reshape(-1, hidden)
        with pytest.raises(RuntimeError, match="row stride"):
            _C.h3_gate_residual_forward(res2, y2, expanded, index)
        _C.h3_gate_residual_forward(res2, y2, dense, index)  # the dense copy is accepted

    def test_rejects_cpu_and_bad_index(self):
        table, residual, y, index = _case(10, hidden=16)
        with pytest.raises(IndexError):
            _cuda_op()(residual, y, _gate(table), index + 9)
        with pytest.raises(ValueError):
            _cuda_op()(residual.cpu(), y.cpu(), _gate(table).cpu(), index.cpu())

    def test_registry_dispatches_cuda(self):
        from rl_engine.runtime.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("adaln_gate_residual", device="cuda")
        assert type(op).__name__ == "H3GateResidualCudaOp"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
@pytest.mark.parametrize("bad_row", [-1, 2], ids=["negative", "past-table"])
def test_gate_row_outside_the_table_asserts_in_the_kernel(entrypoint, bad_row):
    """check_range=False callers get a device-side assert, not a read past the gate table."""

    probe = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine.backends.extension import _C

        if not hasattr(_C, "h3_gate_residual_forward"):
            sys.exit(77)
        entrypoint, bad_row = sys.argv[1], int(sys.argv[2])
        x = torch.randn(4, 8, device="cuda")
        gate = torch.ones(2, 8, device="cuda")  # a table of two rows
        index = torch.tensor([0, bad_row], dtype=torch.int64, device="cuda")
        if entrypoint == "forward":
            _C.h3_gate_residual_forward(x, x, gate, index)
        else:
            i64 = dict(dtype=torch.int64, device="cuda")
            _C.h3_gate_residual_backward(
                torch.ones_like(x), x, gate, index,
                torch.tensor([0, 2, 1, 3], **i64), torch.tensor([0, 2], **i64),
                torch.tensor([2, 4], **i64), torch.tensor([0, 1, 2], **i64),
            )
        torch.cuda.synchronize()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, entrypoint, str(bad_row)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode == 77:
        pytest.skip("rl_engine._C lacks h3_gate_residual_*")
    assert completed.returncode != 0, "the kernel accepted a gate row outside the table"
    assert (
        "device-side assert triggered" in completed.stderr
    ), f"expected a CUDA bounds assertion, got:\n{completed.stdout}\n{completed.stderr}"


def _gate_backward_in_child(mutation: str) -> subprocess.CompletedProcess:
    probe = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine.backends.extension import _C

        if not hasattr(_C, "h3_gate_residual_backward"):
            sys.exit(77)
        mutation = sys.argv[1]
        x = torch.randn(4, 8, device="cuda")
        gate = torch.ones(2, 8, device="cuda")
        index = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
        i64 = dict(dtype=torch.int64, device="cuda")
        sorted_pos = torch.tensor([0, 2, 1, 3], **i64)
        begin, end = torch.tensor([0, 2], **i64), torch.tensor([2, 4], **i64)
        first = torch.tensor([0, 1, 2], **i64)
        if mutation == "tile_end_past_rows":
            end = torch.tensor([2, 9], **i64)
        elif mutation == "position_past_rows":
            sorted_pos = torch.tensor([0, 2, 1, 4], **i64)
        elif mutation == "segments_past_tiles":
            first = torch.tensor([0, 1, 5], **i64)
        elif mutation == "no_tiles":
            begin, end = begin[:0], end[:0]
        elif mutation == "short_sorted_pos":
            sorted_pos = sorted_pos[:3]
        elif mutation == "no_boundary":
            first = first[:0]
        grad = torch.ones_like(x)
        _C.h3_gate_residual_backward(grad, x, gate, index, sorted_pos, begin, end, first)
        torch.cuda.synchronize()
        """
    )
    return subprocess.run(
        [sys.executable, "-c", probe, mutation],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_gate_backward_stops_on_tile_metadata_that_does_not_describe_the_rows():
    control = _gate_backward_in_child("none")
    if control.returncode == 77:
        pytest.skip("rl_engine._C lacks h3_gate_residual_*")
    assert control.returncode == 0, control.stderr
    for mutation in (
        "tile_end_past_rows",
        "position_past_rows",
        "segments_past_tiles",
        "no_tiles",
    ):
        done = _gate_backward_in_child(mutation)
        assert done.returncode != 0, mutation
        assert "device-side assert triggered" in done.stderr, (mutation, done.stderr)
    # Lengths are checked on the host, before any launch.
    for mutation, message in (
        ("short_sorted_pos", "sorted_pos must have M entries"),
        ("no_boundary", "seg_first_tile must hold R + 1 entries"),
    ):
        done = _gate_backward_in_child(mutation)
        assert done.returncode != 0 and message in done.stderr, (mutation, done.stderr)
