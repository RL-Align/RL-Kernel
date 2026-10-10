# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA router accuracy, invariance, boundary and consumer qualification."""

import pytest
import torch

pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_SM90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="NVIDIA SM90 required",
)


@pytest.mark.parametrize("n", [0, 1, 7, 128, 1025])
def test_scores_forward_backward(n):
    pytest.importorskip("triton")
    from rl_engine.kernels.ops.pytorch.nemotron_router import route_scores
    from rl_engine.kernels.ops.triton.nemotron_router import route_scores_cuda

    torch.manual_seed(434)
    s = torch.rand(n, 128, device="cuda", requires_grad=True)
    bias = torch.randn(128, device="cuda") * 0.01
    ids, weights = route_scores_cuda(s, bias)
    rid, rw = route_scores(s, bias)
    assert torch.equal(ids, rid)
    torch.testing.assert_close(weights, rw, atol=2e-7, rtol=2e-6)
    g = torch.randn_like(weights)
    actual = torch.autograd.grad(weights, s, g, retain_graph=True)[0]
    expected = torch.autograd.grad(rw, s, g)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    if n:
        row = n // 2
        one = s[row : row + 1].detach().clone().requires_grad_()
        oi, ow = route_scores_cuda(one, bias)
        og = torch.autograd.grad(ow, one, g[row : row + 1])[0]
        assert torch.equal(oi[0], ids[row])
        assert torch.equal(ow.view(torch.int32)[0], weights.view(torch.int32)[row])
        assert torch.equal(og.view(torch.int32)[0], actual.view(torch.int32)[row])


def test_cuda_ties_and_zero():
    pytest.importorskip("triton")
    from rl_engine.kernels.ops.triton.nemotron_router import route_scores_cuda

    for value in (0.0, 0.5):
        scores = torch.full((17, 128), value, device="cuda", requires_grad=True)
        ids, weights = route_scores_cuda(scores, torch.zeros(128, device="cuda"))
        assert torch.equal(ids, torch.arange(6, device="cuda").expand(17, -1))
        assert torch.isfinite(torch.autograd.grad(weights.sum(), scores)[0]).all()


def test_cuda_near_ties_bias_and_invalid_values():
    from rl_engine.kernels.ops.triton.nemotron_router import route_scores_cuda

    scores = torch.full((1, 128), 0.5, device="cuda")
    scores[0, 7:9] = torch.nextafter(scores[0, 7:9], torch.ones(2, device="cuda"))
    bias = torch.zeros(128, device="cuda")
    bias[80] = 1
    ids, weights = route_scores_cuda(scores, bias)
    assert ids.tolist() == [[0, 1, 2, 7, 8, 80]]
    # The large correction selects expert 80 but must not boost its weight.
    assert weights[0, -1] == weights[0, 0]
    for value in (float("nan"), float("inf"), -1.0, 2.0):
        bad = scores.clone()
        bad[0, 0] = value
        with pytest.raises(ValueError, match="finite sigmoid"):
            route_scores_cuda(bad, bias)


def _reference(x, w, b):
    s = (x.double() @ w.double().t()).sigmoid()
    ids = torch.argsort(s + b.double(), descending=True, stable=True)[:, :6]
    ids = ids.sort(1).values
    values = s.gather(1, ids)
    weights = values / (values.sum(1, keepdim=True) + 1e-20) * 2.5
    perm = torch.argsort(ids.flatten(), stable=True)
    offsets = torch.cat((ids.new_zeros(1), torch.bincount(ids.flatten(), minlength=128).cumsum(0)))
    return ids, weights, perm, offsets, x.double()[perm // 6]


def _bits(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@_SM90
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [0, 1, 7, 17, 129, 513, 4096])
def test_full_reference_and_batch_invariance(n, dtype):
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    torch.manual_seed(434)
    x = torch.randn(n, 2688, device="cuda", dtype=dtype).requires_grad_()
    w = (torch.randn(128, 2688, device="cuda") * 0.01).requires_grad_()
    b = torch.randn(128, device="cuda") * 0.001
    result = nemotron_router_cuda(x, w, b)
    xr = x.detach().double().requires_grad_()
    wr = w.detach().double().requires_grad_()
    ref = _reference(xr, wr, b)
    for i in (0, 2, 3):
        assert torch.equal(result[i], ref[i])
    torch.testing.assert_close(result[1].double(), ref[1], atol=5e-7, rtol=3e-6)
    assert torch.equal(result[4].double(), ref[4])
    gw = torch.randn_like(result[1])
    gp = torch.randn_like(result[4]) * 0.1
    dx, dw = torch.autograd.grad((result[1], result[4]), (x, w), (gw, gp))
    drx, drw = torch.autograd.grad((ref[1], ref[4]), (xr, wr), (gw.double(), gp.double()))
    # Each BF16 input-gradient branch rounds before their autograd addition.
    torch.testing.assert_close(
        dx.double(),
        drx,
        atol=0.008 if dtype == torch.bfloat16 else 2e-6,
        rtol=0.012 if dtype == torch.bfloat16 else 3e-5,
    )
    torch.testing.assert_close(dw.double(), drw, atol=2e-5, rtol=3e-4)
    if n:
        inv = torch.argsort(result[2])
        for token in sorted({0, n // 2, n - 1}):
            xx = x.detach()[token : token + 1].clone().requires_grad_()
            one = nemotron_router_cuda(xx, w, b)
            _bits(result[0][token : token + 1], one[0])
            _bits(result[1][token : token + 1], one[1])
            gp_token = gp[inv[token * 6 : (token + 1) * 6]]
            da = torch.autograd.grad((one[1], one[4]), xx, (gw[token : token + 1], gp_token))[0]
            _bits(dx[token : token + 1], da)
        repeat = nemotron_router_cuda(x, w, b)
        dx2, dw2 = torch.autograd.grad((repeat[1], repeat[4]), (x, w), (gw, gp))
        _bits(dx, dx2)
        _bits(dw, dw2)


@_SM90
def test_full_ties_and_payload_only_backward():
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    x = torch.randn(43, 2688, device="cuda", requires_grad=True)
    w = torch.zeros(128, 2688, device="cuda", requires_grad=True)
    b = torch.zeros(128, device="cuda")
    ids, weights, perm, offsets, packed = nemotron_router_cuda(x, w, b)
    assert torch.equal(ids, torch.arange(6, device="cuda").expand(43, -1))
    assert torch.equal(perm, torch.arange(258, device="cuda").reshape(43, 6).t().flatten())
    assert torch.equal(offsets[:7], torch.arange(7, device="cuda") * 43)
    assert (offsets[7:] == 258).all()
    dx = torch.autograd.grad(packed.sum(), x)[0]
    assert torch.equal(dx, torch.full_like(x, 6))
    # Weight-only backward must work when the dispatch output is unused.
    result = nemotron_router_cuda(x, w, b)
    dx, dw = torch.autograd.grad(result[1].sum(), (x, w))
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()


@_SM90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("branch", ["route", "payload", "both"])
def test_joint_backward_preserves_branch_rounding(dtype, branch):
    from rl_engine.kernels.ops.triton.nemotron_router import (
        _Dispatch,
        _ProjectRoute,
        nemotron_router_cuda,
    )

    torch.manual_seed(5434)
    x = torch.randn(43, 2688, device="cuda", dtype=dtype).requires_grad_(branch != "route")
    w = (torch.randn(128, 2688, device="cuda") * 0.02).requires_grad_(branch != "payload")
    bias = torch.zeros(128, device="cuda")
    gw = torch.randn(43, 6, device="cuda")
    gp = torch.randn(43 * 6, 2688, device="cuda", dtype=dtype)
    actual = nemotron_router_cuda(x, w, bias)
    ids, weights = _ProjectRoute.apply(x, w, bias)
    perm, offsets, packed = _Dispatch.apply(x, ids)
    separate = (ids, weights, perm, offsets, packed)
    inputs = tuple(v for v in (x, w) if v.requires_grad)
    indices = (1, 4) if branch == "both" else ((1,) if branch == "route" else (4,))
    cots = (gw, gp) if branch == "both" else ((gw,) if branch == "route" else (gp,))
    da = torch.autograd.grad(tuple(actual[i] for i in indices), inputs, cots)
    ds = torch.autograd.grad(tuple(separate[i] for i in indices), inputs, cots)
    for a, s in zip((*actual, *da), (*separate, *ds)):
        _bits(a, s)


@_SM90
def test_unsupported_geometry_fails():
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    with pytest.raises(ValueError):
        nemotron_router_cuda(
            torch.empty(3, 128, device="cuda"),
            torch.empty(128, 128, device="cuda"),
            torch.empty(128, device="cuda"),
        )


@_SM90
def test_registry_real_provider_forward_backward():
    from rl_engine.kernels.ops.triton.nemotron_router import NemotronRouterOp
    from rl_engine.kernels.registry import KernelRegistry

    registry = KernelRegistry()
    op = registry.get_op("nemotron_router_dispatch", device="cuda")
    assert isinstance(op, NemotronRouterOp)
    x = torch.zeros(2, 2688, device="cuda", requires_grad=True)
    w = torch.zeros(128, 2688, device="cuda", requires_grad=True)
    b = torch.zeros(128, device="cuda")
    ids, weights, perm, offsets, packed = op(x, w, b)
    assert torch.equal(ids, torch.arange(6, device="cuda").expand(2, -1))
    assert offsets[-1].item() == 12
    dx, dw = torch.autograd.grad((weights.sum() + packed.sum()), (x, w))
    assert torch.equal(dx, torch.full_like(x, 6))
    assert torch.equal(dw, torch.zeros_like(w))


@_SM90
@pytest.mark.parametrize("n", [1, 42, 43, 85, 86, 8193, 65536])
@pytest.mark.parametrize("concentrated", [False, True])
def test_integer_dispatch_permutation_boundaries(n, concentrated):
    """Independent stable-sort oracle at either side of 256-route block tails."""
    import triton

    from rl_engine.kernels.ops.triton.nemotron_router import _dispatch_counts, _dispatch_map

    torch.manual_seed(434 + n)
    ids = torch.rand(n, 128, device="cuda").argsort(1)[:, :6].sort(1).values.contiguous()
    if concentrated:
        # Include the largest legal expert, adjacent to the masked sentinel.
        ids[:] = torch.tensor([0, 1, 2, 125, 126, 127], device="cuda")
    routes = n * 6
    blocks = triton.cdiv(routes, 256)
    counts = torch.empty((128, blocks), device="cuda", dtype=torch.int64)
    _dispatch_counts[(blocks,)](ids, counts, n, blocks)
    prefix = counts.cumsum(1)
    offsets = torch.cat((ids.new_zeros(1), prefix[:, -1].cumsum(0)))
    permutation = torch.full((routes,), -1, device="cuda", dtype=torch.int64)
    inverse = torch.full_like(permutation, -1)
    _dispatch_map[(blocks,)](ids, prefix, offsets, permutation, inverse, n, blocks)
    expected = torch.argsort(ids.flatten(), stable=True)
    assert torch.equal(permutation, expected)
    assert torch.equal(inverse[permutation], torch.arange(routes, device="cuda"))
    expected_counts = torch.bincount(ids.flatten(), minlength=128)
    assert torch.equal(offsets[1:] - offsets[:-1], expected_counts)


@_SM90
@pytest.mark.parametrize("seed", [0, 17, 2026])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 33, 257, 8193])
def test_row_relocation_and_gradient(seed, dtype, n):
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    torch.manual_seed(seed)
    x = torch.randn(n, 2688, device="cuda", dtype=dtype).requires_grad_()
    w = torch.randn(128, 2688, device="cuda") * 0.01
    bias = torch.randn(128, device="cuda") * 0.001
    out = nemotron_router_cuda(x, w, bias)
    gradient = torch.randn_like(out[1])
    dx = torch.autograd.grad(out[1], x, gradient)[0]
    # Reverse row order moves rows between fixed output tiles and tile lanes.
    xx = x.detach().flip(0).contiguous().requires_grad_()
    other = nemotron_router_cuda(xx, w, bias)
    _bits(out[0].flip(0), other[0])
    _bits(out[1].flip(0), other[1])
    dx_other = torch.autograd.grad(other[1], xx, gradient.flip(0).contiguous())[0]
    _bits(dx.flip(0), dx_other)
    for index in sorted({0, min(n - 1, 31), min(n - 1, 32), n - 1}):
        row = x.detach()[index : index + 1].clone().requires_grad_()
        single = nemotron_router_cuda(row, w, bias)
        _bits(out[0][index : index + 1], single[0])
        _bits(out[1][index : index + 1], single[1])
        drow = torch.autograd.grad(single[1], row, gradient[index : index + 1])[0]
        _bits(dx[index : index + 1], drow)
    # Every flat route occurs exactly once, including multi-block expert segments.
    _bits(out[2].sort().values, torch.arange(n * 6, device="cuda"))
    assert out[3][0].item() == 0 and out[3][-1].item() == n * 6
    assert (out[3][1:] >= out[3][:-1]).all()
    _bits(out[4], x.detach()[out[2] // 6])


@_SM90
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_projection_cancellation_and_dynamic_range(scale):
    from rl_engine.kernels.ops.triton.nemotron_router import _Projection

    # Products cancel in adjacent pairs. Perturbations prevent a trivial zero-only test.
    torch.manual_seed(912)
    x = torch.ones(65, 2688, device="cuda") * scale
    x[:, ::2] *= -1
    w = torch.randn(128, 1344, device="cuda").repeat_interleave(2, dim=1)
    w[:, 0] += 0.125
    result = _Projection.apply(x, w)
    reference = x.double() @ w.double().t()
    torch.testing.assert_close(result.double(), reference, atol=3e-5 * scale, rtol=3e-4)
    _bits(result[32:33], _Projection.apply(x[32:33].contiguous(), w))


@_SM90
@pytest.mark.parametrize("n", [1, 129])
def test_graph_replay_observes_input_and_weight_updates(n):
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    torch.manual_seed(45)
    x = torch.randn(n, 2688, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 2688, device="cuda") * 0.01
    bias = torch.zeros(128, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            nemotron_router_cuda(x, w, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = nemotron_router_cuda(x, w, bias)
    for _ in range(3):
        x.normal_()
        w.normal_(std=0.02)
        bias.normal_(std=0.001)
        graph.replay()
        expected = nemotron_router_cuda(x, w, bias)
        for actual, reference in zip(captured, expected):
            _bits(actual, reference)


@_SM90
def test_outer_autocast_preserves_fp32_contract():
    from rl_engine.kernels.ops.triton.nemotron_router import nemotron_router_cuda

    torch.manual_seed(9)
    x = torch.randn(33, 2688, device="cuda", requires_grad=True)
    w = (torch.randn(128, 2688, device="cuda") * 0.01).requires_grad_()
    bias = torch.zeros(128, device="cuda")
    baseline = nemotron_router_cuda(x, w, bias)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = nemotron_router_cuda(x, w, bias)
    for a, b in zip(actual, baseline):
        _bits(a, b)
    g = torch.randn_like(actual[1])
    ga = torch.autograd.grad(actual[1], (x, w), g)
    gb = torch.autograd.grad(baseline[1], (x, w), g)
    for a, b in zip(ga, gb):
        _bits(a, b)


@_SM90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_maximum_batch_last_payload_and_gradient(dtype):
    from rl_engine.kernels.ops.triton.nemotron_router import MAX_TOKENS, nemotron_router_cuda

    torch.manual_seed(734)
    x = torch.randn(MAX_TOKENS, 2688, device="cuda", dtype=dtype).requires_grad_()
    w = torch.randn(128, 2688, device="cuda") * 0.01
    bias = torch.zeros(128, device="cuda")
    output = nemotron_router_cuda(x, w, bias)
    # Check every packed row, including addresses near the supported index bound.
    _bits(output[4], x.detach()[output[2] // 6])
    row = x.detach()[-1:].clone().requires_grad_()
    single = nemotron_router_cuda(row, w, bias)
    _bits(output[0][-1:], single[0])
    _bits(output[1][-1:], single[1])
    upstream = torch.zeros_like(output[1])
    upstream[-1] = torch.arange(6, device="cuda")
    dx = torch.autograd.grad(output[1], x, upstream)[0]
    single_dx = torch.autograd.grad(single[1], row, upstream[-1:])[0]
    _bits(dx[-1:], single_dx)
    assert not torch.count_nonzero(dx[:-1])


@_SM90
def test_excess_tokens_rejected_before_allocation():
    from rl_engine.kernels.ops.triton.nemotron_router import MAX_TOKENS, nemotron_router_cuda

    # A broadcast view exercises the metadata guard without allocating a huge input.
    x = torch.zeros(1, device="cuda").expand(MAX_TOKENS + 1, 2688)
    w = torch.zeros(128, 2688, device="cuda")
    bias = torch.zeros(128, device="cuda")
    with pytest.raises(ValueError, match="at most 65536 tokens"):
        nemotron_router_cuda(x, w, bias)


@_SM90
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("concentrated", [False, True])
def test_registry_nonlinear_consumer_forward_backward(dtype, concentrated):
    from rl_engine.kernels.registry import KernelRegistry

    torch.manual_seed(4434)
    x = torch.randn(33, 2688, device="cuda", dtype=dtype).requires_grad_()
    w = (torch.randn(128, 2688, device="cuda") * 0.01).requires_grad_()
    bias = torch.randn(128, device="cuda") * 0.01
    if concentrated:
        bias.zero_()
        bias[:6] = 4
    scale = torch.linspace(0.5, 1.5, 128, device="cuda")
    op = KernelRegistry().get_op("nemotron_router_dispatch", device=x.device)
    ids, weights, permutation, offsets, packed = op(x, w, bias)
    assert ids.dtype == permutation.dtype == offsets.dtype == torch.int64
    assert weights.dtype == torch.float32 and packed.dtype == dtype
    spans = offsets.tolist()  # Correctness-only consumer; not a benchmark.
    chunks = [
        torch.relu(packed[spans[e] : spans[e + 1]].float() * scale[e] + 0.01).square()
        for e in range(128)
    ]
    expert_output = torch.cat(chunks)
    slots = expert_output[permutation.argsort()].reshape(33, 6, 2688)
    actual = (slots * weights[:, :, None]).sum(1) + x.float().tanh()

    # Independent token-major FP64 expression never uses packed data or offsets.
    xr, wr = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    scores = (xr @ wr.t()).sigmoid()
    gold_ids = torch.argsort(scores + bias.double(), descending=True, stable=True)[:, :6]
    gold_ids = gold_ids.sort(1).values
    assert torch.equal(ids, gold_ids)
    values = scores.gather(1, gold_ids)
    gold_weights = values / (values.sum(1, keepdim=True) + 1e-20) * 2.5
    expert_values = torch.relu(xr[:, None, :] * scale.double()[gold_ids, None] + 0.01).square()
    expected = (expert_values * gold_weights[:, :, None]).sum(1) + xr.tanh()
    g = torch.randn_like(actual)
    dx, dw = torch.autograd.grad(actual, (x, w), g)
    rx, rw = torch.autograd.grad(expected, (xr, wr), g.double())
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=3e-6)
    torch.testing.assert_close(dw.double(), rw, atol=5e-5, rtol=3e-4)
    error = (dx.double() - rx).norm() / rx.norm()
    assert error < (0.01 if dtype == torch.bfloat16 else 1e-5)
