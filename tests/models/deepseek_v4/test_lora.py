# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU math and autograd tests. Reference injection is not ROCm qualification."""

import pytest
import torch

import rl_engine.ops.moe.lora_delta as lora_op
from rl_engine.backends.rocm.moe.lora_delta import RocmLoRADelta
from rl_engine.contracts.operators.lora import validate_lora_inputs
from rl_engine.models.deepseek_v4.lora import lora_projection
from rl_engine.reference.moe.lora_delta import ReferenceLoRADelta


def operands(m=7, k=13, n=11, rank=3):
    gen = torch.Generator().manual_seed(45)
    return tuple(
        (torch.randn(shape, generator=gen) * 0.125).to(torch.bfloat16)
        for shape in ((m, k), (rank, k), (n, rank))
    )


@pytest.fixture
def reference_autograd(monkeypatch):
    # Only the autograd wrapper is under test here. Production refuses CPU.
    monkeypatch.setattr(lora_op, "RocmLoRADelta", ReferenceLoRADelta)


def test_rocm_composition_with_independent_cpu_gemm(monkeypatch):
    # Exercise the production composition while substituting only device/GEMM.
    # Actual Triton arithmetic is covered by the separate ROCm acceptance suite.
    def fp64_mm(self, lhs, rhs):
        assert not torch.is_grad_enabled()
        return (lhs.double() @ rhs.double()).float()

    monkeypatch.setattr(RocmLoRADelta, "_check_device", lambda self, device: None)
    monkeypatch.setattr(RocmLoRADelta, "_mm", fp64_mm)
    x, a, b = (t.requires_grad_() for t in operands())
    ref = ReferenceLoRADelta()
    expected_y, u = ref.forward(x, a, b, 0.375)
    dy = torch.ones_like(expected_y) * 0.2
    expected_grads = ref.backward(dy, x, a, b, 0.375, u)
    y = lora_op.shared_grouped_lora_delta(x, a, b, 0.375, recompute=True)
    y.backward(dy)
    torch.testing.assert_close(y, expected_y, rtol=2e-6, atol=1e-7)
    for tensor, target in zip((x, a, b), expected_grads):
        torch.testing.assert_close(tensor.grad, target.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 8, 33])
@pytest.mark.parametrize("m", [0, 1, 17])
def test_staged_math_and_casts(m, rank):
    x, a, b = operands(m=m, rank=rank)
    alpha = 0.375
    op = ReferenceLoRADelta()
    y, u = op.forward(x, a, b, alpha)
    # FP64 matmul is independent of the reference's serial FP32 reduction.
    expected_u = (x.double() @ a.double().t()).to(torch.bfloat16)
    expected_y = (expected_u.double() @ b.double().t()) * alpha
    torch.testing.assert_close(u, expected_u, rtol=0, atol=0)
    torch.testing.assert_close(y.double(), expected_y, rtol=2e-6, atol=1e-7)
    assert y.dtype == torch.float32 and u.dtype == torch.bfloat16
    dy = torch.ones_like(y) * 0.2
    dx, da, db = op.backward(dy, x, a, b, alpha, u)
    dv = (dy.double() * alpha).to(torch.bfloat16).double()
    du = (dv @ b.double()).to(torch.bfloat16).double()
    expected = (du @ a.double(), du.t() @ x.double(), dv.t() @ expected_u.double())
    for actual, target in zip((dx, da, db), expected):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), target, rtol=2e-6, atol=1e-7)


def test_math_gradient_and_finite_difference():
    # Finite differences apply to the smooth LoRA formula, not BF16's staircase.
    # Powers-of-two data makes the staged casts exact at this evaluation point.
    # Non-exact rounding is covered separately by test_staged_math_and_casts.
    x, a, b = (t.sign() * 0.125 for t in operands())
    doubles = tuple(t.double().requires_grad_() for t in (x, a, b))
    alpha = 0.5
    dy = torch.full((x.shape[0], b.shape[0]), 0.25)

    def loss(*args):
        xx, aa, bb = args
        return (((xx @ aa.t()) @ bb.t()) * alpha * dy.double()).sum()

    mathematical = torch.autograd.grad(loss(*doubles), doubles)
    for operand, grad in zip(doubles, mathematical):
        index = (0, 0)
        original = operand[index].item()
        eps = 1e-5
        with torch.no_grad():
            operand[index] = original + eps
            plus = loss(*doubles).item()
            operand[index] = original - eps
            minus = loss(*doubles).item()
            operand[index] = original
        assert (plus - minus) / (2 * eps) == pytest.approx(grad[index].item(), abs=1e-10)
    ref = ReferenceLoRADelta()
    _, u = ref.forward(x, a, b, alpha)
    rounded = ref.backward(dy, x, a, b, alpha, u)
    for actual, smooth in zip(rounded, mathematical):
        torch.testing.assert_close(actual.double(), smooth, rtol=0, atol=0)


@pytest.mark.parametrize("recompute", [False, True])
def test_autograd_matches_explicit_vjp(reference_autograd, recompute):
    x, a, b = (t.requires_grad_() for t in operands())
    alpha = 0.375
    expected_y, u = ReferenceLoRADelta().forward(x, a, b, alpha)
    dy = torch.linspace(-0.2, 0.2, expected_y.numel()).reshape_as(expected_y)
    expected_grads = ReferenceLoRADelta().backward(dy, x, a, b, alpha, u)
    y = lora_op.shared_grouped_lora_delta(x, a, b, alpha, recompute=recompute)
    y.backward(dy)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    for tensor, expected in zip((x, a, b), expected_grads):
        torch.testing.assert_close(tensor.grad, expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("changed", [0, 1, 2])
def test_mutation_between_forward_and_backward_is_rejected(reference_autograd, recompute, changed):
    tensors = tuple(t.requires_grad_() for t in operands())
    y = lora_op.shared_grouped_lora_delta(*tensors, 0.5, recompute=recompute)
    with torch.no_grad():
        tensors[changed].add_(0.125)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


def test_two_adapter_updates_are_visible(reference_autograd):
    x, a, b = (t.requires_grad_() for t in operands())
    previous = None
    for _ in range(3):
        y = lora_op.shared_grouped_lora_delta(x, a, b, 0.5, recompute=True)
        expected, _ = ReferenceLoRADelta().forward(x, a, b, 0.5)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(y, previous)
        previous = y.detach().clone()
        y.sum().backward()
        with torch.no_grad():
            a.add_(a.grad, alpha=-0.1)
            b.add_(b.grad, alpha=-0.1)
        for t in (x, a, b):
            t.grad = None


@pytest.mark.parametrize("mode", ["base", "lora", "combined"])
def test_projection_preserves_frozen_base_dx(reference_autograd, mode):
    x, a, b = (t.requires_grad_() for t in operands())
    # This tests composition, not the future quantized base GEMM implementation.
    weight = torch.eye(b.shape[0], x.shape[1]) * 0.5
    base = x.float() @ weight.t()
    use_lora = mode != "base"
    y = lora_projection(
        x,
        a if use_lora else None,
        b if use_lora else None,
        0.5,
        base_output=base if mode != "lora" else None,
    )
    delta, u = ReferenceLoRADelta().forward(x, a, b, 0.5)
    expected_y = base if mode == "base" else delta if mode == "lora" else base + delta
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    y.sum().backward()
    dx = ReferenceLoRADelta().backward(torch.ones_like(y), x, a, b, 0.5, u)[0]
    # Separate BF16 autograd edges merge at X; do not move this rounding point.
    base_dx = (torch.ones_like(base) @ weight).bfloat16()
    expected_dx = base_dx if mode == "base" else dx.bfloat16()
    if mode == "combined":
        expected_dx = expected_dx + base_dx
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    assert weight.grad is None
    assert (a.grad is not None) == use_lora
    assert (b.grad is not None) == use_lora


def test_alpha_is_effective_multiplier(reference_autograd):
    x, a, b = operands(rank=8)
    full = lora_op.shared_grouped_lora_delta(x, a, b, 1.0)
    half = lora_op.shared_grouped_lora_delta(x, a, b, 0.5)
    torch.testing.assert_close(half, full * 0.5, rtol=0, atol=0)
    zero = lora_op.shared_grouped_lora_delta(x, a, b, 0.0)
    assert torch.count_nonzero(zero) == 0


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), 1e100])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        validate_lora_inputs(*operands(), alpha)


@pytest.mark.parametrize("alpha", [True, torch.tensor(1.0), "1"])
def test_invalid_alpha_type(alpha):
    with pytest.raises(TypeError, match="alpha"):
        validate_lora_inputs(*operands(), alpha)


def test_invalid_shapes_and_dtypes():
    x, a, b = operands()
    with pytest.raises(TypeError, match="BF16"):
        validate_lora_inputs(x.float(), a, b, 1.0)
    for bad_x, bad_a, bad_b in ((x[0], a, b), (x[:, :-1], a, b), (x, a[:0], b[:, :0])):
        with pytest.raises(ValueError):
            validate_lora_inputs(bad_x, bad_a, bad_b, 1.0)
    with pytest.raises(TypeError, match="recompute"):
        lora_op.shared_grouped_lora_delta(x, a, b, 1.0, recompute="yes")


def test_projection_rejects_missing_adapter_and_early_cast():
    x, a, b = operands()
    with pytest.raises(ValueError, match="both"):
        lora_projection(x, a, None, 1.0)
    with pytest.raises(ValueError, match="requires"):
        lora_projection(x, None, None, 1.0)
    with pytest.raises(TypeError, match="FP32"):
        lora_projection(x, a, b, 1.0, base_output=torch.zeros(x.shape[0], b.shape[0]).bfloat16())
    with pytest.raises(ValueError, match="width"):
        lora_projection(x, a, b, 1.0, base_output=torch.zeros(x.shape[0], b.shape[0] + 1))


@pytest.mark.parametrize("m", [0, 3])
def test_production_has_no_cpu_fallback(m):
    with pytest.raises(RuntimeError, match="ROCm GPU tensors; no fallback"):
        lora_op.shared_grouped_lora_delta(*operands(m=m), 1.0)
