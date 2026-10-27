# SPDX-License-Identifier: Apache-2.0
"""CPU math and autograd checks; these do not establish GPU bitwise acceptance."""

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.clamp_swiglu import split_packed, validate_backward
from rl_engine.ops.moe import clamp_swiglu as api
from rl_engine.reference.moe.clamp_swiglu import ClampSwiGLUReference as Reference


def _math64(g, u, p):
    if p is not None:
        g, u = g.clamp(max=10.0), u.clamp(-10.0, 10.0)
    raw = torch.nn.functional.silu(g) * u
    return raw if p is None else raw * p[:, None]


@pytest.fixture
def diagnostic_api(monkeypatch):
    # Injection is test-only; the public operator never dispatches to this reference.
    monkeypatch.setattr(api, "RocmClampSwiGLUWeighted", Reference)
    return api


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_forward_and_vjp_against_independent_math(dtype, weighted):
    rng = torch.Generator().manual_seed(46)
    g = (torch.randn(4, 17, generator=rng) * 8).to(dtype)
    u = (torch.randn(4, 17, generator=rng) * 8).to(dtype)
    # Keep equality-boundary subgradient checks separate from smooth math checks.
    g[g == 10] = 9
    u[u.abs() == 10] = 9
    p = torch.tensor([0.0, 1e-8, 0.37, 1.0]) if weighted else None
    dy = torch.randn(4, 17, generator=rng)
    y, state = Reference.forward(g, u, p)
    dg, du = Reference.backward(dy, g, u, p, state)
    g64, u64 = g.double().requires_grad_(), u.double().requires_grad_()
    y64 = _math64(g64, u64, None if p is None else p.double())
    expected = torch.autograd.grad(y64, (g64, u64), dy.double())
    assert y.dtype == torch.bfloat16
    assert dg.dtype == du.dtype == torch.float32
    torch.testing.assert_close(y.float(), y64.detach().float(), rtol=0.004, atol=1e-7)
    for actual, want in zip((dg, du), expected):
        torch.testing.assert_close(actual.double(), want, rtol=2e-5, atol=3e-6)


def test_smooth_finite_difference():
    # Differentiate the mathematical activation, not the BF16 output staircase.
    g = torch.tensor([[-2.0, 0.3, 2.0, 11.0]], dtype=torch.float64)
    u = torch.tensor([[0.4, -2.0, 12.0, 0.5]], dtype=torch.float64)
    p = torch.tensor([0.37], dtype=torch.float64)
    _, saved = Reference.forward(g.float(), u.float(), p.float())
    grads = Reference.backward(torch.ones_like(g).float(), g.float(), u.float(), p.float(), saved)
    eps = 1e-5
    for which, actual in enumerate(grads):
        for j in range(g.shape[1]):
            plus, minus = [g.clone(), u.clone()], [g.clone(), u.clone()]
            plus[which][0, j] += eps
            minus[which][0, j] -= eps
            finite = (_math64(*plus, p).sum() - _math64(*minus, p).sum()) / (2 * eps)
            assert actual[0, j].item() == pytest.approx(finite.item(), rel=1e-5, abs=1e-6)


def test_clamp_boundary_subgradient():
    ten = torch.tensor(10.0)
    inside = torch.nextafter(ten, torch.tensor(0.0)).item()
    outside = torch.nextafter(ten, torch.tensor(20.0)).item()
    g = torch.tensor([[inside, 10.0, outside, 2.0, 2.0, 2.0, 2.0]])
    u = torch.tensor([[1.0, 1.0, 1.0, -10.0, 10.0, -inside, inside]])
    p = torch.ones(1)
    _, state = Reference.forward(g, u, p)
    dg, du = Reference.backward(torch.ones_like(g), g, u, p, state)
    assert dg[0, 0] > 0
    assert torch.equal(dg[0, 1:3], torch.zeros(2))
    assert torch.equal(du[0, 3:5], torch.zeros(2))
    assert torch.all(du[0, 5:] > 0)


def test_weight_once_and_no_intermediate_bf16_round():
    # An early BF16 cast and a second route multiply both change these output bytes.
    g = torch.tensor([[1.234, -0.743, 3.123]])
    u = torch.tensor([[2.345, 1.237, 0.583]])
    p = torch.tensor([0.371])
    y, saved = Reference.forward(g, u, p)
    assert torch.equal(y, (saved.unweighted * p[:, None]).bfloat16())
    assert not torch.equal(y, (saved.unweighted.bfloat16().float() * p[:, None]).bfloat16())
    assert not torch.equal(y, (saved.unweighted * p[:, None].square()).bfloat16())


def test_zero_tiny_weights_keep_unweighted_values():
    g, u = torch.full((3, 5), 2.0), torch.full((3, 5), 3.0)
    p = torch.tensor([0.0, 1e-12, 0.5])
    y, state = Reference.forward(g, u, p)
    assert torch.equal(y[0], torch.zeros_like(y[0]))
    assert torch.all(y[1] > 0)
    assert torch.equal(state.unweighted[0], state.unweighted[2])
    assert torch.all(state.unweighted[0] > 0)
    dg, du = Reference.backward(torch.ones_like(g), g, u, p, state)
    assert torch.equal(dg[0], torch.zeros(5))
    assert torch.equal(du[0], torch.zeros(5))


def test_shared_variant_has_no_clamp():
    g, u = torch.tensor([[20.0, 2.0]]), torch.tensor([[20.0, -20.0]])
    shared, state = Reference.forward(g, u, None)
    routed, _ = Reference.forward(g, u, torch.ones(1))
    assert torch.all(shared.abs() > routed.abs())
    dg, du = Reference.backward(torch.ones_like(g), g, u, None, state)
    assert torch.all(dg != 0) and torch.all(du != 0)


def test_bf16_overflow_is_not_silently_saturated():
    g, u = torch.tensor([[10.0]]), torch.tensor([[10.0]])
    y, state = Reference.forward(g, u, torch.tensor([torch.finfo(torch.float32).max]))
    assert torch.isposinf(y).all()
    assert torch.isfinite(state.unweighted).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_autograd_packed_separate_save_recompute(diagnostic_api, dtype, weighted):
    rng = torch.Generator().manual_seed(9)
    # Slice both dimensions to exercise noncontiguous gate/up and packed inputs.
    source = torch.randn(6, 28, generator=rng).to(dtype)[::2, ::2]
    p = torch.tensor([0.0, 0.37, 1e-8]) if weighted else None
    dy = torch.randn(3, 7, generator=rng).bfloat16()
    outputs, gradients = [], []
    for packed in (False, True):
        for recompute in (False, True):
            x = source.detach().requires_grad_()
            g, u = x.chunk(2, dim=1)
            if packed:
                y = diagnostic_api.clamp_swiglu_weighted_packed(x, p, recompute=recompute)
            else:
                y = diagnostic_api.clamp_swiglu_weighted(g, u, p, recompute=recompute)
            dx = torch.autograd.grad(y, x, dy)[0]
            _, state = Reference.forward(g, u, p)
            expected = torch.cat(Reference.backward(dy, g, u, p, state), dim=1).to(dtype)
            assert torch.equal(dx, expected)
            outputs.append(y)
            gradients.append(dx)
    assert all(torch.equal(outputs[0], y) for y in outputs)
    assert all(torch.equal(gradients[0], dx) for dx in gradients)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("mutated", ["gate", "up", "weight"])
def test_autograd_detects_stale_saved_inputs(diagnostic_api, recompute, mutated):
    g, u = torch.ones(2, 4, requires_grad=True), torch.ones(2, 4, requires_grad=True)
    p = torch.ones(2)
    y = diagnostic_api.clamp_swiglu_weighted(g, u, p, recompute=recompute)
    with torch.no_grad():
        {"gate": g, "up": u, "weight": p}[mutated].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


def test_route_gradient_is_not_silently_dropped(diagnostic_api):
    g, u, p = torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, requires_grad=True)
    with pytest.raises(NotImplementedError, match="task 49"):
        diagnostic_api.clamp_swiglu_weighted(g, u, p)
    with torch.no_grad():
        y = diagnostic_api.clamp_swiglu_weighted(g, u, p)
    assert not y.requires_grad


def test_empty_rows_and_expanded_weights(diagnostic_api):
    for m in (0, 3):
        g = torch.ones(m, 7, requires_grad=True)
        u = torch.ones(m, 7, requires_grad=True)
        p = torch.tensor([0.5]).expand(m)
        y = diagnostic_api.clamp_swiglu_weighted(g, u, p)
        dg, du = torch.autograd.grad(y, (g, u), torch.ones_like(y))
        assert y.shape == dg.shape == du.shape == (m, 7)


@pytest.mark.parametrize("m", [0, 1])
def test_production_rejects_cpu_even_empty(m):
    with pytest.raises(RuntimeError, match="requires ROCm"):
        api.clamp_swiglu_weighted(torch.ones(m, 4), torch.ones(m, 4), torch.ones(m))


@pytest.mark.parametrize(
    "g,u,p,error",
    [
        (torch.ones(2), torch.ones(2), None, ValueError),
        (torch.ones(2, 0), torch.ones(2, 0), None, ValueError),
        (torch.ones(2, 4), torch.ones(2, 3), None, ValueError),
        (torch.ones(2, 4).half(), torch.ones(2, 4).half(), None, TypeError),
        (torch.ones(2, 4), torch.ones(2, 4).bfloat16(), None, TypeError),
        (torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 1), ValueError),
        (torch.ones(2, 4), torch.ones(2, 4), torch.ones(2).bfloat16(), TypeError),
    ],
)
def test_invalid_inputs(g, u, p, error):
    with pytest.raises(error):
        Reference.forward(g, u, p)


@pytest.mark.parametrize("shape", [(2,), (3, 5), (3, 0)])
def test_invalid_packed_shape(shape):
    with pytest.raises(ValueError, match="packed input"):
        split_packed(torch.empty(shape))


def test_backward_validation():
    g, u = torch.ones(2, 3), torch.ones(2, 3)
    _, state = Reference.forward(g, u, None)
    with pytest.raises(ValueError, match="dh"):
        validate_backward(torch.ones(2, 4), g, u, None, state)
    with pytest.raises(TypeError, match="dh"):
        validate_backward(g.half(), g, u, None, state)
    with pytest.raises(ValueError, match="state.silu"):
        validate_backward(g, g, u, None, replace(state, silu=state.silu.bfloat16()))
