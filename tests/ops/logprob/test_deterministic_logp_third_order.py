# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native order-three LogP derivatives, including mixed upstream derivatives."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.cuda.logprob.logp import DeterministicLogpCUDAOp

from .test_deterministic_logp_higher_order import ATOL, DTYPES, bits, call, inputs, reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.fixture
def op(record_property):
    """Return a ``DeterministicLogpCUDAOp`` and record its backend in the test report."""
    record_property("backend", "DeterministicLogpCUDAOp")
    record_property("third_backward", "recursive probability/row-sum CUDA primitives")
    return DeterministicLogpCUDAOp()


def second(y, x, g, h):
    """Return differentiable grads of ``<J^T g, h>`` w.r.t. ``x`` and ``g`` (``J = dy/dx``)."""
    first = torch.autograd.grad(y, x, g, create_graph=True)[0]
    return torch.autograd.grad(first, (x, g), h, create_graph=True)


def third(y, x, g, h, a, b, branch="both"):
    """Return grads of ``<dx, a> + <dg, b>`` w.r.t. ``(x, g, h)``, where ``dx, dg = second(...)``.

    ``branch="x"`` zeroes ``b`` and ``branch="g"`` zeroes ``a``, keeping one second output.
    """
    dx, dg = second(y, x, g, h)
    if branch == "x":
        b = torch.zeros_like(b)
    if branch == "g":
        a = torch.zeros_like(a)
    return torch.autograd.grad((dx, dg), (x, g, h), (a, b))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", [1, 128, 129, 257, 4096, 4097, 128256])
@pytest.mark.parametrize("indexed", [False, True])
def test_third_reference(op, dtype, vocab, indexed, record_property):
    """Third-order grads w.r.t. ``x``, ``g`` and ``h`` match the PyTorch reference.

    Rows with out-of-range labels or not selected by ``rows`` get exactly zero gradients.
    """
    x, ids, g, h = inputs(dtype, vocab=vocab)
    h.requires_grad_()
    ids[1], ids[4] = -1, vocab
    rows = torch.tensor([0, 1, 3, 3, 5, -1, 99], device="cuda") if indexed else None
    a, b = h.detach().flip(-1), g.detach().flip(0)
    actual = third(call(op, x, ids, rows), x, g, h, a, b)
    expected = third(reference(x, ids, rows), x, g, h, a, b)
    record_property("input_dtype", str(dtype))
    record_property("atol", ATOL[dtype])
    record_property("rtol", 0)
    for name, got, ref in zip(("logits", "upstream", "hessian_vector"), actual, expected):
        record_property("max_" + name + "_error", (got.double() - ref.double()).abs().max().item())
        torch.testing.assert_close(got, ref, atol=ATOL[dtype], rtol=0)
        inactive = [1, 2, 4] if indexed else [1, 4]
        assert torch.count_nonzero(got[inactive]) == 0


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_third_fp64_finite_differences(op, indexed, out):
    """FP64 ``second`` passes ``gradcheck`` in ``x``, ``g`` and ``h``, validating third order."""
    x, ids, g, h = inputs(torch.float64, rows=3, vocab=5)
    h.requires_grad_()
    rows = torch.tensor([0, 2, 2, -1, 8], device="cuda") if indexed else None

    def fn(x, g, h):
        """Return ``second`` of the LogP call with the captured labels, rows and ``out`` mode."""
        return second(call(op, x, ids, rows, out), x, g, h)

    assert torch.autograd.gradcheck(fn, (x, g, h), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("branch", ["x", "g"])
def test_each_second_output(op, dtype, branch):
    """Grads of ``dx`` alone or ``dg`` alone match the reference with the other cotangent zeroed."""
    x, ids, g, h = inputs(dtype)
    h.requires_grad_()
    a, b = h.detach().flip(-1), g.detach().flip(0)
    dx, dg = second(call(op, x, ids), x, g, h)
    # Exercise autograd's zero materialization for the unused second output.
    actual = torch.autograd.grad(
        dx if branch == "x" else dg,
        (x, g, h),
        a if branch == "x" else b,
        allow_unused=True,
        materialize_grads=True,
    )
    expected = third(reference(x, ids), x, g, h, a, b, branch)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=ATOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_third_layout_bits(op, dtype, indexed, out):
    """Third-order grads are bitwise invariant to batch layout.

    Compares repeat runs, the selection without duplicate/invalid indices, row chunks of 1, 2 and
    4, a row permutation and a padded batch; when indexed, the unselected padding rows must get
    zero gradients.
    """
    x, ids, g, h = inputs(dtype, rows=7, vocab=4097)
    h.requires_grad_()
    a, b = h.detach().flip(-1), g.detach().flip(0)
    rows = torch.tensor([0, 2, 6, 2, -1, 99], device="cuda") if indexed else None

    def run(x, ids, g, h, a, b, rows):
        """Return ``third`` of the LogP call for these inputs, cotangents and row selection."""
        return third(call(op, x, ids, rows, out), x, g, h, a, b)

    base = run(x, ids, g, h, a, b, rows)
    for got, ref in zip(run(x, ids, g, h, a, b, rows), base):
        bits(got, ref)
    selected = torch.tensor([0, 2, 6], device="cuda") if indexed else None
    for got, ref in zip(run(x, ids, g, h, a, b, selected), base):
        bits(got, ref)
    for size in (1, 2, 4):
        chunks = []
        for start in range(0, 7, size):
            end = min(start + size, 7)
            local = rows[(rows >= start) & (rows < end)] - start if indexed else None
            chunks.append(
                run(
                    x[start:end],
                    ids[start:end],
                    g[start:end],
                    h[start:end],
                    a[start:end],
                    b[start:end],
                    local,
                )
            )
        for i, ref in enumerate(base):
            bits(torch.cat([c[i] for c in chunks]), ref)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device="cuda")
    selected_order = torch.argsort(order)[selected] if indexed else None
    for got, ref in zip(
        run(x[order], ids[order], g[order], h[order], a[order], b[order], selected_order), base
    ):
        bits(got, ref[order])
    px, pi, pg, ph = inputs(dtype, rows=13, vocab=4097)
    ph.requires_grad_()
    pa, pb = ph.detach().clone(), pg.detach().clone()
    with torch.no_grad():
        for target, original in zip((px, pi, pg, ph, pa, pb), (x, ids, g, h, a, b)):
            target[3:10].copy_(original)
    selected_pad = selected + 3 if indexed else None
    for got, ref in zip(run(px, pi, pg, ph, pa, pb, selected_pad), base):
        bits(got[3:10], ref)
        if indexed:
            assert torch.count_nonzero(got[:3]) == torch.count_nonzero(got[10:]) == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("selection", [None, [], [0, 2, 2, -1, 99]])
def test_third_noncontiguous_empty_out(op, dtype, empty, selection):
    """Third-order grads of ``out`` calls on transposed and empty inputs match the reference."""
    x, ids, g, h = inputs(dtype, rows=0 if empty else 6)
    x = x.reshape(2, -1, 17).transpose(0, 1)
    ids = ids.reshape(2, -1).transpose(0, 1)
    g = g.reshape(2, -1).transpose(0, 1)
    h = h.reshape(2, -1, 17).transpose(0, 1).requires_grad_()
    a, b = h.detach().flip(-1), g.detach().flip(0)
    rows = None if selection is None else torch.tensor(selection, device="cuda", dtype=torch.long)
    actual = third(call(op, x, ids, rows, out=True), x, g, h, a, b)
    expected = third(reference(x, ids, rows, out=True), x, g, h, a, b)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=ATOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("composed", [False, True])
def test_fourth_reference(op, dtype, composed):
    """Fourth-order directional derivatives of ``sum(y)`` and ``sum(y ** 2)`` match reference."""
    x, ids, _, h = inputs(dtype)
    y = call(op, x, ids)
    loss = y.square().sum() if composed else y.sum()
    first = torch.autograd.grad(loss, x, create_graph=True)[0]
    sec = torch.autograd.grad(first, x, h, create_graph=True)[0]
    tri = torch.autograd.grad(sec, x, h.flip(-1), create_graph=True)[0]
    actual = torch.autograd.grad(tri, x, h)[0]
    ref_y = reference(x, ids)
    loss = ref_y.square().sum() if composed else ref_y.sum()
    first = torch.autograd.grad(loss, x, create_graph=True)[0]
    sec = torch.autograd.grad(first, x, h, create_graph=True)[0]
    tri = torch.autograd.grad(sec, x, h.flip(-1), create_graph=True)[0]
    expected = torch.autograd.grad(tri, x, h)[0]
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype] * 8, rtol=0)


def test_missing_recursive_symbol(op):
    """A ``create_graph`` backward raises "Rebuild rl_engine._C" with only first-order symbols."""
    names = ("deterministic_logp_forward_with_lse", "deterministic_logp_backward")
    op._backend = SimpleNamespace(**{name: getattr(op._backend, name) for name in names})
    x, ids, g, h = inputs(torch.float32)
    with pytest.raises(RuntimeError, match="Rebuild rl_engine._C"):
        torch.autograd.grad(op(x, ids), x, g, create_graph=True)
