# SPDX-License-Identifier: Apache-2.0
"""CPU math and contract checks; these do not certify ROCm execution."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient
from rl_engine.contracts.operators.route_weight_grad import (
    validate_inputs,
    validate_training_inputs,
)
from rl_engine.ops.moe.route_weight_grad import route_weight_grad
from rl_engine.reference.moe.route_weight_grad import route_weight_grad_reference as reference


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 33), (1, 1), (3, 255), (3, 256), (3, 257), (3, 2049)])
def test_reference_accuracy_strides_and_repeat(dtype, shape):
    m, f = shape
    rng = torch.Generator().manual_seed(490)
    dh = torch.randn(m, f * 2, generator=rng).to(dtype)[:, ::2]
    silu = torch.randn(m, f * 2, generator=rng)[:, ::2]
    up = (torch.randn(m, f * 2, generator=rng) * 20).to(dtype)[:, ::2]
    actual = reference(dh, silu, up)
    terms64 = (dh.double() * silu.double()) * up.double().clamp(-10, 10)
    # Absolute error scales with the sum of magnitudes, including cancellation.
    bound = 20 * torch.finfo(torch.float32).eps * terms64.abs().sum(1)
    assert torch.all((actual.double() - terms64.sum(1)).abs() <= bound)
    assert actual.shape == (m,) and actual.dtype == torch.float32
    for repeat in (
        reference(dh, silu, up),
        reference(dh.contiguous(), silu.contiguous(), up.contiguous()),
    ):
        assert torch.equal(actual.view(torch.uint8), repeat.view(torch.uint8))


def test_explicit_pair_tree_and_ascending_partial_merge():
    dh = torch.zeros(1, 769)
    dh[0, :4] = torch.tensor([1e20, 1, -1e20, 1])
    ones = torch.ones_like(dh)
    assert reference(dh, ones, ones).item() == 0  # serial summation would give 1
    dh.zero_()
    dh[0, [0, 256, 512]] = torch.tensor([1e20, -1e20, 3])
    assert reference(dh, ones, ones).item() == 3  # reversing partials would give 0


def test_product_association_and_overflow_are_not_hidden():
    # (max * 2) * 0.25 overflows; max * (2 * 0.25) does not.
    dh = torch.tensor([[torch.finfo(torch.float32).max]])
    assert torch.isposinf(reference(dh, torch.full_like(dh, 2), torch.full_like(dh, 0.25))).all()
    dh = torch.tensor([[float("inf"), -float("inf")]])
    assert torch.isnan(reference(dh, torch.ones_like(dh), torch.ones_like(dh))).all()
    assert torch.isneginf(reference(dh[:, :1], torch.ones(1, 1), -torch.ones(1, 1))).all()


@pytest.mark.parametrize("weight", [0.0, 1e-30, 0.37, 1.0])
def test_independent_fp64_finite_difference_including_zero_weight(weight):
    # Differentiate the smooth pre-cast forward. The BF16 cast uses an STE;
    # finite differencing the discrete BF16 output would be the wrong oracle.
    g = torch.tensor([[10.0, 11.0, -1.0, 2.0]], dtype=torch.float64)
    u = torch.tensor([[-11.0, -10.0, 10.0, 11.0]], dtype=torch.float64)
    dh = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float64)
    s = torch.nn.functional.silu(g.clamp(max=10))

    def loss(p):
        return ((s * u.clamp(-10, 10)) * p * dh).sum()

    step = 1e-4
    fd = (loss(weight + step) - loss(weight - step)) / (2 * step)
    actual = reference(dh.float(), s.float(), u.float())[0]
    torch.testing.assert_close(actual.double(), fd, rtol=2e-6, atol=2e-6)
    assert actual != 0


def test_tail_lanes_do_not_change_the_row():
    dh = torch.zeros(2, 257)
    dh[:, 256] = torch.tensor([3.0, -7.0])
    assert torch.equal(reference(dh, torch.ones_like(dh), torch.ones_like(dh)), dh[:, 256])


def test_adapter_preserves_rows_and_ignores_weight_values(monkeypatch):
    adapter = RocmRouteWeightGradient()
    monkeypatch.setattr(adapter, "reduce", reference)
    g, u = torch.zeros(3, 5), torch.arange(15).reshape(3, 5).float()
    dh, state = torch.ones_like(g), SimpleNamespace(silu=torch.ones_like(g))
    expected = reference(dh, state.silu, u)
    for p in (torch.zeros(3), torch.full((3,), 1e-30), torch.ones(3)):
        assert torch.equal(adapter.backward(dh, g, u, p, state), expected)
    perm = torch.tensor([2, 0, 1])
    got = adapter.backward(
        dh[perm], g[perm], u[perm], torch.zeros(3), SimpleNamespace(silu=state.silu[perm])
    )
    assert torch.equal(got, expected[perm])


@pytest.mark.parametrize(
    "case", ["empty_width", "shape", "dh_dtype", "silu_dtype", "up_dtype", "layout"]
)
def test_invalid_operator_inputs(case):
    dh, s, u = torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3)
    if case == "empty_width":
        dh, s, u = (torch.empty(2, 0) for _ in range(3))
    elif case == "shape":
        s = s[:1]
    elif case == "dh_dtype":
        dh = dh.double()
    elif case == "silu_dtype":
        s = s.bfloat16()
    elif case == "up_dtype":
        u = u.int()
    else:
        s = s.to_sparse()
    with pytest.raises((ValueError, TypeError)):
        validate_inputs(dh, s, u)


@pytest.mark.parametrize(
    "case", ["shared", "weight_shape", "weight_dtype", "gate_shape", "gate_dtype"]
)
def test_invalid_training_inputs(case):
    dh, g, u, p = torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3), torch.ones(2)
    if case == "shared":
        p = None
    elif case == "weight_shape":
        p = p[:, None]
    elif case == "weight_dtype":
        p = p.bfloat16()
    elif case == "gate_shape":
        g = g[:, :1]
    else:
        g = g.bfloat16()
    with pytest.raises(ValueError):
        validate_training_inputs(dh, g, u, p, SimpleNamespace(silu=torch.ones_like(u)))


def test_no_implicit_cpu_fallback():
    x = torch.ones(1, 3)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        route_weight_grad(x, x, x)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        RocmRouteWeightGradient().validate(SimpleNamespace(ffn=3), x.device)


def test_unknown_architecture_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx000")
    )
    with pytest.raises(RuntimeError, match="no candidate for gfx000"):
        RocmRouteWeightGradient().validate(SimpleNamespace(ffn=3), torch.device("cuda"))
