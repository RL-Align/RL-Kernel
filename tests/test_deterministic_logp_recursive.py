# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Recursive AD and torch.func acceptance; extend orders with RL_KERNEL_LOGP_MAX_ORDER."""

import json
import os

import pytest
import torch
from test_deterministic_logp_higher_order import ATOL, DTYPES, bits, reference

from rl_engine.kernels.ops.cuda.loss.logp import DeterministicLogpCUDAOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
MAX_ORDER = int(os.environ.get("RL_KERNEL_LOGP_MAX_ORDER", "8"))
if MAX_ORDER < 1:
    raise ValueError("RL_KERNEL_LOGP_MAX_ORDER must be positive")
HIGH_ATOL = {torch.float64: 1e-8, torch.float32: 1e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}


@pytest.fixture
def op(record_property):
    record_property("backend", "DeterministicLogpCUDAOp")
    record_property(
        "derivative_backend", "recursive custom CUDA probability/row-sum + ATen pointwise"
    )
    return DeterministicLogpCUDAOp()


def invoke(op, x, ids, rows=None, out=False):
    suffix = "fp64" if x.dtype == torch.float64 else "fp32"
    if out:
        buffer = x.new_full(
            ids.shape, 7.0, dtype=torch.float64 if suffix == "fp64" else torch.float32
        )
        name = ("indexed_out" if rows is not None else "out") + (
            "_fp64" if suffix == "fp64" else ""
        )
        args = (x, ids, rows, buffer) if rows is not None else (x, ids, buffer)
        result = getattr(op, name)(*args)
        return result
    name = ("indexed_" if rows is not None else "apply_") + suffix
    return getattr(op, name)(x, ids, rows) if rows is not None else getattr(op, name)(x, ids)


def data(dtype, rows=2, vocab=5):
    rng = torch.Generator(device="cuda").manual_seed(20261010)
    x = (torch.randn(rows, vocab, device="cuda", dtype=dtype, generator=rng) * 0.2).requires_grad_()
    ids = torch.arange(rows, device="cuda") % vocab
    return x, ids


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_recursive_orders(op, dtype, indexed, out, composed, record_property):
    x, ids = data(dtype)
    rows = torch.tensor([0, 0, -1, 9], device="cuda") if indexed else None
    weights = torch.tensor(
        [0.3, -0.4],
        device="cuda",
        dtype=torch.float64 if dtype == torch.float64 else torch.float32,
        requires_grad=True,
    )
    actual = invoke(op, x, ids, rows, out)
    ref = reference(x, ids, rows, out)
    if composed:
        actual, ref = actual.square(), ref.square()
    actual, ref = (actual * weights).sum(), (ref * weights).sum()
    direction = torch.linspace(-0.7, 0.8, x.numel(), device="cuda", dtype=dtype).reshape(x.shape)
    weight_direction = torch.tensor([0.2, -0.1], device="cuda", dtype=weights.dtype)
    errors = []
    for order in range(1, MAX_ORDER + 1):
        ag = torch.autograd.grad(
            actual, (x, weights), create_graph=True, allow_unused=True, materialize_grads=True
        )
        rg = torch.autograd.grad(
            ref, (x, weights), create_graph=True, allow_unused=True, materialize_grads=True
        )
        atol = (ATOL[dtype] * 8 if composed else ATOL[dtype]) if order <= 3 else HIGH_ATOL[dtype]
        error = max((a.double() - r.double()).abs().max().item() for a, r in zip(ag, rg))
        errors.append({"order": order, "max_error": error, "atol": atol, "rtol": 0})
        for a, r in zip(ag, rg):
            torch.testing.assert_close(a, r, atol=atol, rtol=0)
        actual = (ag[0] * direction).sum() + (ag[1] * weight_direction).sum()
        ref = (rg[0] * direction).sum() + (rg[1] * weight_direction).sum()
        direction = direction.flip(-1)
    record_property("input_dtype", str(dtype))
    record_property("orders", json.dumps(errors))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_func_transforms(op, dtype, indexed, out, record_property):
    x, ids = data(dtype)
    rows = torch.tensor([0, 0, -1, 9], device="cuda") if indexed else None

    def fn(z):
        return invoke(op, z, ids, rows, out)

    def ref(z):
        return reference(z, ids, rows, out)

    tangent = torch.linspace(-0.7, 0.8, x.numel(), device="cuda", dtype=dtype).reshape(x.shape)
    atol = ATOL[dtype]
    errors = {}
    checks = {
        "grad": lambda f: torch.func.grad(lambda z: f(z).sum())(x),
        "vjp": lambda f: torch.func.vjp(f, x)[1](torch.ones_like(f(x)))[0],
        "jvp": lambda f: torch.func.jvp(f, (x,), (tangent,))[1],
        "jacrev": lambda f: torch.func.jacrev(f)(x),
        "jacfwd": lambda f: torch.func.jacfwd(f)(x),
        "hessian": lambda f: torch.func.hessian(lambda z: f(z).sum())(x),
        "forward_over_reverse": lambda f: torch.func.jvp(
            torch.func.grad(lambda z: f(z).sum()), (x,), (tangent,)
        )[1],
        "forward_over_forward": lambda f: torch.func.jvp(
            lambda z: torch.func.jvp(f, (z,), (tangent,))[1], (x,), (tangent,)
        )[1],
        "reverse_over_reverse": lambda f: torch.func.jacrev(torch.func.grad(lambda z: f(z).sum()))(
            x
        ),
        "reverse_over_forward": lambda f: torch.func.grad(
            lambda z: torch.func.jvp(f, (z,), (tangent,))[1].sum()
        )(x),
    }
    for name, check in checks.items():
        a, r = check(fn), check(ref)
        torch.testing.assert_close(a, r, atol=atol, rtol=0)
        errors[name] = (a.double() - r.double()).abs().max().item()
    with torch.no_grad():
        torch.testing.assert_close(checks["jvp"](fn), checks["jvp"](ref), atol=atol, rtol=0)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, tangent)
        a = torch.autograd.forward_ad.unpack_dual(fn(dual)).tangent
        r = torch.autograd.forward_ad.unpack_dual(ref(dual)).tangent
        torch.testing.assert_close(a, r, atol=atol, rtol=0)
    # Adjoint identity, evaluated in FP64 for the check itself.
    _, jv = torch.func.jvp(fn, (x,), (tangent,))
    cotangent = torch.tensor([0.3, -0.4], device=x.device, dtype=jv.dtype)
    vjp = torch.func.vjp(fn, x)[1](cotangent)[0]
    torch.testing.assert_close(
        (jv.double() * cotangent.double()).sum(),
        (vjp.double() * tangent.double()).sum(),
        atol=atol * 2,
        rtol=0,
    )
    record_property("input_dtype", str(dtype))
    record_property("transform_max_errors", json.dumps(errors))


@pytest.mark.parametrize("out", [False, True])
@pytest.mark.parametrize("mapped_metadata", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_vmap_combinations(op, out, mapped_metadata, axis):
    x, ids = data(torch.float64)
    batch = torch.stack([x, x + 0.1, x - 0.1]).movedim(0, axis)
    labels = ids.expand(3, -1) if mapped_metadata else ids
    indices = (
        torch.tensor([[0, 0, -1], [1, 1, 9], [0, 1, 0]], device="cuda")
        if mapped_metadata
        else torch.tensor([0, 0, -1], device="cuda")
    )
    dims = (axis, 0 if mapped_metadata else None, 0 if mapped_metadata else None)

    def fn(z, i, r):
        return invoke(op, z, i, r, out)

    result = torch.vmap(fn, in_dims=dims)(batch, labels, indices)
    baseline = torch.stack(
        [
            fn(
                batch.select(axis, j),
                labels[j] if mapped_metadata else labels,
                indices[j] if mapped_metadata else indices,
            )
            for j in range(3)
        ]
    )
    bits(result, baseline)

    def scalar(z, i, r):
        return fn(z, i, r).sum()

    vg = torch.vmap(torch.func.grad(scalar), in_dims=dims)(batch, labels, indices)
    expected = torch.stack(
        [
            torch.func.grad(scalar)(
                batch.select(axis, j),
                labels[j] if mapped_metadata else labels,
                indices[j] if mapped_metadata else indices,
            )
            for j in range(3)
        ]
    )
    bits(vg, expected)

    def jvp(z, i, r):
        return torch.func.jvp(lambda t: fn(t, i, r), (z,), (torch.ones_like(z),))[1]

    bits(
        torch.vmap(jvp, in_dims=dims)(batch, labels, indices),
        torch.stack(
            [
                jvp(
                    batch.select(axis, j),
                    labels[j] if mapped_metadata else labels,
                    indices[j] if mapped_metadata else indices,
                )
                for j in range(3)
            ]
        ),
    )

    def fn_shared(z):
        return invoke(op, z, ids, torch.tensor([0, 0], device="cuda"), out)

    nested = x.expand(2, 3, *x.shape)
    bits(
        torch.vmap(torch.vmap(fn_shared))(nested),
        torch.stack([torch.stack([fn_shared(x) for _ in range(3)]) for _ in range(2)]),
    )
    empty = torch.vmap(fn_shared)(x.expand(0, *x.shape))
    assert empty.shape == (0, *ids.shape)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_fp64_checks(op, indexed, out):
    x, ids = data(torch.float64, rows=1, vocab=3)
    rows = torch.tensor([0, 0, -1], device="cuda") if indexed else None

    def fn(z):
        return invoke(op, z, ids, rows, out)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3, check_forward_ad=True)
    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
    t = torch.tensor([[0.2, -0.3, 0.4]], device="cuda", dtype=torch.float64)
    actual = torch.func.jvp(fn, (x,), (t,))[1]
    finite = (fn(x + 1e-6 * t) - fn(x - 1e-6 * t)) / (2e-6)
    torch.testing.assert_close(actual, finite, atol=1e-8, rtol=1e-6)


def test_mapped_out_storage_and_shared_rejection(op):
    x, ids = data(torch.float64)
    batch = x.expand(3, *x.shape).clone()
    buffers = torch.empty(3, *ids.shape, device="cuda", dtype=x.dtype)

    def fn(z, b):
        return op.out_fp64(z, ids, b)

    result = torch.vmap(fn)(batch, buffers)
    assert result.data_ptr() == buffers.data_ptr()
    torch.testing.assert_close(
        result, torch.vmap(lambda z: op.apply_fp64(z, ids))(batch), atol=0, rtol=0
    )
    with pytest.raises(RuntimeError):
        torch.vmap(fn, in_dims=(0, None))(batch, torch.empty_like(buffers[0]))
    captured = torch.empty_like(buffers[0])
    with pytest.raises(RuntimeError):
        torch.func.grad(lambda z: op.out_fp64(z, ids, captured).sum())(x)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_all_order_layout_bits(op, dtype, indexed, out):
    x, ids = data(dtype, rows=7, vocab=17)
    weights = torch.linspace(
        -0.4,
        0.5,
        7,
        device="cuda",
        dtype=torch.float64 if dtype == torch.float64 else torch.float32,
    )
    vectors = torch.linspace(-0.6, 0.7, x.numel(), device="cuda", dtype=dtype).reshape(x.shape)
    rows = torch.tensor([0, 2, 6, 2, -1, 99], device="cuda") if indexed else None

    def run(x, ids, weights, vectors, rows):
        scalar = (invoke(op, x, ids, rows, out) * weights).sum()
        grads = []
        for order in range(MAX_ORDER):
            grad = torch.autograd.grad(scalar, x, create_graph=order + 1 < MAX_ORDER)[0]
            grads.append(grad.detach())
            scalar = (grad * vectors).sum()
        return grads

    baseline = run(x, ids, weights, vectors, rows)
    for a, b in zip(run(x, ids, weights, vectors, rows), baseline):
        bits(a, b)
    unique = torch.tensor([0, 2, 6], device="cuda") if indexed else None
    for a, b in zip(run(x, ids, weights, vectors, unique), baseline):
        bits(a, b)
    for chunk in (1, 2, 4):
        chunks = []
        for start in range(0, 7, chunk):
            end = min(start + chunk, 7)
            local = rows[(rows >= start) & (rows < end)] - start if indexed else None
            chunks.append(
                run(x[start:end], ids[start:end], weights[start:end], vectors[start:end], local)
            )
        for order, ref in enumerate(baseline):
            bits(torch.cat([item[order] for item in chunks]), ref)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device="cuda")
    mapped = torch.argsort(order)[unique] if indexed else None
    for a, b in zip(run(x[order], ids[order], weights[order], vectors[order], mapped), baseline):
        bits(a, b[order])
    px, pi = data(dtype, rows=13, vocab=17)
    pw = weights.new_zeros(13)
    pv = vectors.new_zeros(13, 17)
    with torch.no_grad():
        for target, original in zip((px, pi, pw, pv), (x, ids, weights, vectors)):
            target[3:10].copy_(original)
    for a, b in zip(run(px, pi, pw, pv, unique + 3 if indexed else None), baseline):
        bits(a[3:10], b)
        assert torch.count_nonzero(a[:3]) == torch.count_nonzero(a[10:]) == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("empty", [False, True])
def test_transform_edge_inputs(op, dtype, empty):
    x, ids = data(dtype, rows=0 if empty else 6, vocab=5)
    x = x.reshape(2, 0 if empty else 3, 5).transpose(0, 1)
    ids = ids.reshape(2, 0 if empty else 3).transpose(0, 1)
    if not empty:
        ids[0, 0] = -1
        ids[1, 0] = 5
    rows = torch.tensor([0, 1, 2, 2, -1, 99], device="cuda")
    tangent = torch.ones_like(x) * 0.3
    for out in (False, True):

        def fn(z):
            return invoke(op, z, ids, rows, out)

        def ref(z):
            return reference(z, ids, rows, out)

        for actual, expected in zip(
            torch.func.jvp(fn, (x,), (tangent,)), torch.func.jvp(ref, (x,), (tangent,))
        ):
            torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=0)
        torch.testing.assert_close(
            torch.func.grad(lambda z: fn(z).sum())(x),
            torch.func.grad(lambda z: ref(z).sum())(x),
            atol=ATOL[dtype],
            rtol=0,
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_forward_orders(op, dtype, record_property):
    x, ids = data(dtype)
    rows = torch.tensor([0, 0, -1, 99], device="cuda")
    vector = torch.linspace(-0.7, 0.8, x.numel(), device=x.device, dtype=dtype).reshape(x.shape)

    def actual(z):
        return invoke(op, z, ids, rows, out=True)

    def expected(z):
        return reference(z, ids, rows, out=True)

    def differentiate(fn):
        def derivative(z):
            return torch.func.jvp(fn, (z,), (vector,))[1]

        return derivative

    errors = []
    for order in range(1, MAX_ORDER + 1):
        actual, expected = differentiate(actual), differentiate(expected)
        a, b = actual(x), expected(x)
        atol = ATOL[dtype] if order <= 3 else HIGH_ATOL[dtype]
        torch.testing.assert_close(a, b, atol=atol, rtol=0)
        errors.append(
            {
                "order": order,
                "max_error": (a.double() - b.double()).abs().max().item(),
                "atol": atol,
                "rtol": 0,
            }
        )
    record_property("input_dtype", str(dtype))
    record_property("orders", json.dumps(errors))
