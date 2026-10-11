# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native FP64, out buffers and second derivatives: numeric and bitwise contracts."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.cuda.logprob.logp import DeterministicLogpCUDAOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
ATOL = {
    torch.float32: 1e-5,
    torch.float16: 1e-3,
    torch.bfloat16: 8e-3,
    torch.float64: 1e-10,
}


@pytest.fixture
def op(record_property):
    result = DeterministicLogpCUDAOp()
    record_property("backend", type(result).__name__)
    record_property("second_backward", "recursive probability/row-sum CUDA primitives")
    return result


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def inputs(dtype, rows=6, vocab=17):
    rng = torch.Generator(device="cuda").manual_seed(20261010)
    x = torch.randn(rows, vocab, device="cuda", dtype=dtype, generator=rng).requires_grad_()
    ids = torch.arange(rows, device="cuda") % vocab
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    g = torch.linspace(-0.8, 0.9, rows, device="cuda", dtype=acc).requires_grad_()
    h = torch.randn(rows, vocab, device="cuda", dtype=dtype, generator=rng)
    return x, ids, g, h


def call(op, x, ids, rows=None, out=False):
    fp64 = x.dtype == torch.float64
    if out:
        buffer = torch.full(
            ids.shape,
            7.0,
            device=x.device,
            dtype=torch.float64 if fp64 else torch.float32,
        )
        name = "indexed_out" if rows is not None else "out"
        name += "_fp64" if fp64 else ""
        args = (x, ids, rows, buffer) if rows is not None else (x, ids, buffer)
        result = getattr(op, name)(*args)
        assert result.data_ptr() == buffer.data_ptr()
        return result
    name = ("indexed_" if rows is not None else "apply_") + ("fp64" if fp64 else "fp32")
    return getattr(op, name)(x, ids, rows) if rows is not None else getattr(op, name)(x, ids)


def reference(x, ids, rows=None, out=False):
    acc = x if x.dtype == torch.float64 else x.float()
    valid = (ids >= 0) & (ids < x.size(-1))
    result = acc.log_softmax(-1).gather(-1, ids.clamp(0, x.size(-1) - 1).unsqueeze(-1)).squeeze(-1)
    result = result.masked_fill(~valid, 0)
    if rows is not None:
        mask = torch.zeros(ids.numel(), device=x.device, dtype=torch.bool)
        valid_rows = rows[(rows >= 0) & (rows < ids.numel())].unique()
        mask[valid_rows] = True
        result = torch.where(mask.reshape(ids.shape), result, 7.0 if out else 0.0)
    return result


def derivatives(y, x, g, h):
    dx = torch.autograd.grad(y, x, g, create_graph=True)[0]
    dxx, dg = torch.autograd.grad(dx, (x, g), h)
    return dx.detach(), dxx, dg


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", [1, 128, 129, 257, 4096, 4097, 128256])
@pytest.mark.parametrize("indexed", [False, True])
def test_second_derivative_reference(op, dtype, vocab, indexed, record_property):
    x, ids, g, h = inputs(dtype, vocab=vocab)
    ids[1] = -1
    ids[4] = vocab
    rows = torch.tensor([0, 1, 3, 3, 5, -1, 99], device="cuda") if indexed else None
    y = call(op, x, ids, rows)
    expected = reference(x, ids, rows)
    actual_grads = derivatives(y, x, g, h)
    expected_grads = derivatives(expected, x, g, h)
    for name, a, b in zip(("gradient", "hvp", "upstream_derivative"), actual_grads, expected_grads):
        record_property("max_" + name + "_error", (a.double() - b.double()).abs().max().item())
        torch.testing.assert_close(a, b, atol=ATOL[dtype], rtol=0)
    record_property("atol", ATOL[dtype])
    record_property("rtol", 0)
    if dtype == torch.float64:
        torch.testing.assert_close(y, expected, atol=1e-10, rtol=0)
    with torch.no_grad():
        bits(call(op, x, ids, rows), y)
    inactive = [1, 2, 4] if indexed else [1, 4]
    assert torch.count_nonzero(actual_grads[0][inactive]) == 0
    assert torch.count_nonzero(actual_grads[1][inactive]) == 0
    assert torch.count_nonzero(actual_grads[2][inactive]) == 0


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_fp64_gradcheck(op, indexed, out):
    x, ids, _, _ = inputs(torch.float64, rows=3, vocab=5)
    rows = torch.tensor([0, 2, 2, -1, 7], device="cuda") if indexed else None

    def fn(x):
        return call(op, x, ids, rows, out)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("selected", [None, [], [0, 2, 2, -1, 99]])
@pytest.mark.parametrize("empty", [False, True])
def test_out_noncontiguous_empty_and_mask(op, dtype, selected, empty):
    x, ids, g, h = inputs(dtype, rows=0 if empty else 6)
    x = x.reshape(2, -1, 17).transpose(0, 1)
    ids = ids.reshape(2, -1).transpose(0, 1)
    g = g.reshape(2, -1).transpose(0, 1)
    h = h.reshape(2, -1, 17).transpose(0, 1)
    if not empty:
        ids[0, 0] = -1
    rows = None if selected is None else torch.tensor(selected, device="cuda", dtype=torch.long)
    y = call(op, x, ids, rows, out=True)
    ref = reference(x, ids, rows, out=True)
    torch.testing.assert_close(y, ref, atol=ATOL[dtype], rtol=0)
    for a, b in zip(derivatives(y, x, g, h), derivatives(ref, x, g, h)):
        torch.testing.assert_close(a, b, atol=ATOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_derivative_layout_bits(op, dtype, indexed, out):
    x, ids, g, h = inputs(dtype, rows=7, vocab=4097)
    selected = torch.tensor([0, 2, 6, 2, -1, 99], device="cuda") if indexed else None

    def run(x, ids, g, h, rows):
        return derivatives(call(op, x, ids, rows, out), x, g, h)

    baseline = run(x, ids, g, h, selected)
    for a, b in zip(run(x, ids, g, h, selected), baseline):
        bits(a, b)
    if indexed:
        for a, b in zip(run(x, ids, g, h, torch.tensor([0, 2, 6], device="cuda")), baseline):
            bits(a, b)
    for chunk in (1, 2, 4):
        results = []
        for start in range(0, 7, chunk):
            end = min(start + chunk, 7)
            rows = selected[(selected >= start) & (selected < end)] - start if indexed else None
            results.append(run(x[start:end], ids[start:end], g[start:end], h[start:end], rows))
        for i, b in enumerate(baseline):
            bits(torch.cat([r[i] for r in results]), b)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device="cuda")
    inverse = torch.argsort(order)
    valid = selected[(selected >= 0) & (selected < 7)] if indexed else None
    rows = inverse[valid] if indexed else None
    for a, b in zip(run(x[order], ids[order], g[order], h[order], rows), baseline):
        bits(a, b[order])
    padded, labels, weights, vectors = inputs(dtype, rows=13, vocab=4097)
    with torch.no_grad():
        padded[3:10].copy_(x)
        labels[3:10].copy_(ids)
        weights[3:10].copy_(g)
        vectors[3:10].copy_(h)
    for a, b in zip(
        run(padded, labels, weights, vectors, valid + 3 if indexed else None), baseline
    ):
        bits(a[3:10], b)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
def test_out_reuse_and_guards(op, dtype, indexed):
    x, ids, g, h = inputs(dtype)
    buffer = torch.zeros_like(ids, dtype=g.dtype)
    rows = torch.tensor([0, 2, 2], device="cuda")
    name = ("indexed_out" if indexed else "out") + ("_fp64" if dtype == torch.float64 else "")

    def run(output):
        return (
            getattr(op, name)(x, ids, rows, output)
            if indexed
            else getattr(op, name)(x, ids, output)
        )

    for _ in range(2):
        result = run(buffer)
        assert result.data_ptr() == buffer.data_ptr()
        derivatives(result, x, g, h)
        with pytest.raises(RuntimeError, match="must not require gradients"):
            run(buffer)
        buffer = buffer.detach()
    with pytest.raises(RuntimeError, match="must not require gradients"):
        run(buffer.requires_grad_())
    alias = x.detach().reshape(-1)[: ids.numel()]
    with pytest.raises(RuntimeError, match="share storage"):
        run(alias)
    with torch.no_grad():
        result = run(buffer)
        assert result.data_ptr() == buffer.data_ptr()


@pytest.mark.parametrize(
    "method",
    [
        "apply_fp64",
        "indexed_fp64",
        "out_fp64",
        "indexed_out_fp64",
        "online_fp64",
        "online_indexed_fp64",
        "online_out_fp64",
        "online_indexed_out_fp64",
    ],
)
def test_fp64_aliases(op, method):
    x, ids, g, h = inputs(torch.float64)
    args = [x, ids]
    if "indexed" in method:
        args.append(torch.arange(ids.numel(), device="cuda"))
    if "out" in method:
        args.append(torch.empty_like(ids, dtype=torch.float64))
    y = getattr(op, method)(*args)
    for a, b in zip(derivatives(y, x, g, h), derivatives(reference(x, ids), x, g, h)):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=0)
    args[0] = x.float()
    if "out" in method:
        args[-1] = args[-1].detach()
    with pytest.raises(TypeError, match="requires float64 logits"):
        getattr(op, method)(*args)


def test_fp64_not_float32_disguised(op):
    # FP32 loses these differences completely; true FP64 must retain them.
    x = torch.tensor(
        [[1e8, 1e8 + 1, 1e8 + 2]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    ids = torch.tensor([0], device="cuda")
    y = op.apply_fp64(x, ids)
    torch.testing.assert_close(y, reference(x, ids), atol=1e-12, rtol=0)
    with torch.no_grad():
        legacy = op(x, ids)
    assert legacy.dtype == torch.float32
    assert abs(legacy.item() - y.item()) > 0.1


def test_missing_symbols_and_third_order(op):
    x, ids, g, h = inputs(torch.float64)
    grad = torch.autograd.grad(op.apply_fp64(x, ids), x, g, create_graph=True)[0]
    second = torch.autograd.grad(grad, x, h.requires_grad_(), create_graph=True)[0]
    third = torch.autograd.grad(second.sum(), h)[0]
    torch.testing.assert_close(third, torch.zeros_like(h), atol=1e-10, rtol=0)
    op._backend = SimpleNamespace()
    with pytest.raises(RuntimeError, match="Rebuild rl_engine._C"):
        op.apply_fp64(x, ids)


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("out", [False, True])
def test_fp64_three_step_sgd(op, linear, indexed, out, record_property):
    x, ids, g, _ = inputs(torch.float64, rows=6, vocab=17)
    torch.manual_seed(123)
    if linear:
        model = torch.nn.Linear(5, 17, device="cuda", dtype=torch.float64)
        ref_model = torch.nn.Linear(5, 17, device="cuda", dtype=torch.float64)
        ref_model.load_state_dict(model.state_dict())
        features = torch.randn(6, 5, device="cuda", dtype=torch.float64)
        parameters, ref_parameters = list(model.parameters()), list(ref_model.parameters())
    else:
        parameters = [torch.nn.Parameter(x.detach().clone())]
        ref_parameters = [torch.nn.Parameter(x.detach().clone())]
    opt = torch.optim.SGD(parameters, lr=0.03)
    ref_opt = torch.optim.SGD(ref_parameters, lr=0.03)
    rows = torch.tensor([0, 2, 2, 5], device="cuda") if indexed else None
    error = 0
    for _ in range(3):
        opt.zero_grad()
        ref_opt.zero_grad()
        logits = model(features) if linear else parameters[0]
        ref_logits = ref_model(features) if linear else ref_parameters[0]
        (call(op, logits, ids, rows, out) * g.detach()).sum().backward()
        (reference(ref_logits, ids, rows, out) * g.detach()).sum().backward()
        opt.step()
        ref_opt.step()
        for a, b in zip(parameters, ref_parameters):
            error = max(error, (a - b).abs().max().item())
            torch.testing.assert_close(a, b, atol=1e-10, rtol=0)
    record_property("max_parameter_error", error)


@pytest.mark.parametrize("dtype", DTYPES)
def test_composed_loss_second_derivative(op, dtype):
    x, ids, _, h = inputs(dtype)
    y = call(op, x, ids)
    expected = reference(x, ids)
    grad = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    ref_grad = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
    actual = torch.autograd.grad(grad, x, h)[0]
    ref = torch.autograd.grad(ref_grad, x, h)[0]
    # Compound loss amplifies low-precision rounding relative to a direct VJP.
    torch.testing.assert_close(actual, ref, atol=ATOL[dtype] * 8, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("composed", [False, True])
def test_third_order_with_constant_vector(op, dtype, composed):
    x, ids, _, h = inputs(dtype)
    y = call(op, x, ids)
    loss = y.square().sum() if composed else y.sum()
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    second = torch.autograd.grad(grad, x, h, create_graph=True)[0]
    vector = torch.randn_like(x)
    actual = torch.autograd.grad(second, x, vector)[0]
    ref_y = reference(x, ids)
    ref_loss = ref_y.square().sum() if composed else ref_y.sum()
    first = torch.autograd.grad(ref_loss, x, create_graph=True)[0]
    ref_second = torch.autograd.grad(first, x, h, create_graph=True)[0]
    expected = torch.autograd.grad(ref_second, x, vector)[0]
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype] * 8, rtol=0)
