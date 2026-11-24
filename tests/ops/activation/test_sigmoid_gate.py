# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""NativeSigmoidGateOp tests (Qwen3-Next hybrid-stack reference).

Covers:
- Native correctness (fp32 formula, dtype path, guards)
- Axis A batch invariance (slice + padding)
- Backward gradient flow
- Registry dispatch
"""

import pytest
import torch

from rl_engine.reference.activation.sigmoid_gate import NativeSigmoidGateOp
from rl_engine.runtime.registry import OpBackend, kernel_registry


@pytest.fixture
def op() -> NativeSigmoidGateOp:
    return NativeSigmoidGateOp()


def test_forward_matches_formula(op: NativeSigmoidGateOp) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 17, 64)
    gate = torch.randn(4, 17, 64)
    expected = x * torch.sigmoid(gate)
    torch.testing.assert_close(op(x, gate), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_paths(op: NativeSigmoidGateOp, dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 8, 32, dtype=dtype)
    gate = torch.randn(3, 8, 32, dtype=dtype)
    out = op(x, gate)
    assert out.dtype == dtype
    expected = (x.float() * torch.sigmoid(gate.float())).to(dtype)
    torch.testing.assert_close(out, expected)


def test_forward_fp32_forces_fp32(op: NativeSigmoidGateOp) -> None:
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    gate = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = op.forward_fp32(x, gate)
    assert out.dtype == torch.float32


def test_shape_guard(op: NativeSigmoidGateOp) -> None:
    with pytest.raises(ValueError):
        op(torch.randn(2, 8), torch.randn(2, 9))
    with pytest.raises(TypeError):
        op(torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64))
    with pytest.raises(TypeError):
        op(torch.randn(2, 8), torch.randn(2, 8, dtype=torch.bfloat16))


def test_batch_invariance_slice_and_padding(op: NativeSigmoidGateOp) -> None:
    """Axis A: slicing or padding the batch dim cannot change any row.

    The last dim is deliberately not a multiple of 8 so the CPU vectorized
    path has scalar-tail elements; a length-dependent implementation would
    diverge here.
    """
    torch.manual_seed(2)
    x = torch.randn(6, 5, 17)
    gate = torch.randn(6, 5, 17)
    full = op(x, gate)

    sliced = op(x[2:5], gate[2:5])
    torch.testing.assert_close(sliced, full[2:5], rtol=0, atol=0)

    padded_x = torch.cat([x, torch.zeros(3, 5, 17)], dim=0)
    padded_g = torch.cat([gate, torch.zeros(3, 5, 17)], dim=0)
    padded = op(padded_x, padded_g)
    torch.testing.assert_close(padded[:6], full, rtol=0, atol=0)


def test_backward_gradient_flow(op: NativeSigmoidGateOp) -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 6, 12, requires_grad=True)
    gate = torch.randn(2, 6, 12, requires_grad=True)
    out = op(x, gate)
    out.sum().backward()
    assert x.grad is not None and gate.grad is not None
    sig = torch.sigmoid(gate.detach())
    expected_x_grad = sig
    expected_g_grad = x.detach() * sig * (1 - sig)
    torch.testing.assert_close(x.grad, expected_x_grad)
    torch.testing.assert_close(gate.grad, expected_g_grad)


def test_registry_dispatch_cpu() -> None:
    resolved = kernel_registry.get_op("sigmoid_gate", device="cpu")
    assert isinstance(resolved, NativeSigmoidGateOp)
    candidates = kernel_registry._priority_map["cpu"]["sigmoid_gate"]
    assert candidates == [OpBackend.PYTORCH_SIGMOID_GATE]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_extreme_gate_gradients_are_finite(op, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.ones(7, device=device, requires_grad=True)
    gate = torch.tensor(
        [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0], device=device, requires_grad=True
    )
    out = op(x, gate)
    dx, dg = torch.autograd.grad(out.sum(), (x, gate), create_graph=True)
    assert torch.isfinite(out).all()
    assert torch.isfinite(dx).all() and torch.isfinite(dg).all()
    expected = torch.sigmoid(gate.detach())
    torch.testing.assert_close(dx, expected)
    torch.testing.assert_close(dg, expected * (1 - expected))
    assert torch.isfinite(torch.autograd.grad(dg.sum(), gate)[0]).all()
