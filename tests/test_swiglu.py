# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSiLUOp, NativeSwiGLUOp
from rl_engine.kernels.registry import kernel_registry

# Qwen3-8B SwiGLU intermediate dim (gate/up_proj output width).
_INTERMEDIATE = 12288


# Shared helper
def _rand(shape, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_silu_matches_fp32_reference(dtype: torch.dtype):
    x = torch.linspace(-6.0, 6.0, 33, dtype=dtype).reshape(3, 11)

    fp32_reference = x.float() * torch.sigmoid(x.float())
    result = NativeSiLUOp().forward(x)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSiLUOp().forward_fp32(x), fp32_reference)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_swiglu_matches_fp32_reference(dtype: torch.dtype):
    gate = torch.linspace(-4.0, 4.0, 48, dtype=dtype).reshape(2, 3, 8)
    up = torch.linspace(0.5, 2.0, 48, dtype=dtype).reshape(2, 3, 8)

    fp32_reference = gate.float() * torch.sigmoid(gate.float()) * up.float()
    result = NativeSwiGLUOp().forward(gate, up)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSwiGLUOp().forward_fp32(gate, up), fp32_reference)


def test_native_swiglu_rejects_mismatched_shape():
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)

    with pytest.raises(ValueError, match="share shape"):
        NativeSwiGLUOp().forward(gate, up)


# Axis A -- batch invariance, bitwise (the WS1 "aligned" property).
# A row's output must not depend on how many rows share the batch.
def test_silu_batch_invariance_slice():
    op = NativeSiLUOp()
    x = _rand((8, 32, _INTERMEDIATE), seed=2)
    full = op.forward_fp32(x)  # compute on full batch...
    assert torch.equal(op.forward_fp32(x[:1]), full[:1])  # ...then slice
    assert torch.equal(op.forward_fp32(x[3:5]), full[3:5])


def test_swiglu_batch_invariance_slice():
    op = NativeSwiGLUOp()
    gate = _rand((8, 32, _INTERMEDIATE), seed=3)
    up = _rand((8, 32, _INTERMEDIATE), seed=4)
    full = op.forward_fp32(gate, up)
    assert torch.equal(op.forward_fp32(gate[:1], up[:1]), full[:1])
    assert torch.equal(op.forward_fp32(gate[3:5], up[3:5]), full[3:5])


def test_silu_batch_invariance_with_padding():
    """Padding extra rows must not perturb the real rows (bitwise)."""
    op = NativeSiLUOp()
    x = _rand((4, _INTERMEDIATE), seed=5)
    padded = torch.cat([x, _rand((6, _INTERMEDIATE), seed=99)], dim=0)
    assert torch.equal(op.forward_fp32(padded)[:4], op.forward_fp32(x))


def test_swiglu_batch_invariance_with_padding():
    op = NativeSwiGLUOp()
    gate = _rand((4, _INTERMEDIATE), seed=6)
    up = _rand((4, _INTERMEDIATE), seed=7)
    pad_gate = torch.cat([gate, _rand((6, _INTERMEDIATE), seed=98)], dim=0)
    pad_up = torch.cat([up, _rand((6, _INTERMEDIATE), seed=97)], dim=0)
    assert torch.equal(op.forward_fp32(pad_gate, pad_up)[:4], op.forward_fp32(gate, up))


# Purity -- inputs not mutated in-place
def test_silu_inputs_not_mutated():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=8)
    xc = x.clone()
    op.forward(x)
    op.forward_fp32(x)
    assert torch.equal(x, xc)


def test_swiglu_inputs_not_mutated():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=9)
    up = _rand((2, _INTERMEDIATE), seed=10)
    gc, uc = gate.clone(), up.clone()
    op.forward(gate, up)
    op.forward_fp32(gate, up)
    assert torch.equal(gate, gc) and torch.equal(up, uc)


# Gradient flows (fp32 autograd = backward golden source)
def test_silu_gradient_flows():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=11).requires_grad_(True)
    op.forward_fp32(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_swiglu_gradient_flows():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=12).requires_grad_(True)
    up = _rand((2, _INTERMEDIATE), seed=13).requires_grad_(True)
    op.forward_fp32(gate, up).sum().backward()
    assert torch.isfinite(gate.grad).all() and torch.isfinite(up.grad).all()


def test_silu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSiLUOp()

    x_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    out_full = op.forward_fp32(x_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_full_sliced = x_full.grad[:1].clone()

    x_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(x_slice)
    out_slice.backward(dy_full[:1])

    assert torch.equal(x_slice.grad, grad_full_sliced)


def test_swiglu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSwiGLUOp()

    gate_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    up_full = _rand((8, 32, _INTERMEDIATE), seed=2).requires_grad_(True)
    out_full = op.forward_fp32(gate_full, up_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_gate_full_sliced = gate_full.grad[:1].clone()
    grad_up_full_sliced = up_full.grad[:1].clone()

    gate_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    up_slice = _rand((8, 32, _INTERMEDIATE), seed=2)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(gate_slice, up_slice)

    out_slice.backward(dy_full[:1])

    assert torch.equal(gate_slice.grad, grad_gate_full_sliced)
    assert torch.equal(up_slice.grad, grad_up_full_sliced)


def test_registry_dispatches_native_activation_ops():
    assert isinstance(kernel_registry.get_op("silu"), NativeSiLUOp)
    assert isinstance(kernel_registry.get_op("swiglu"), NativeSwiGLUOp)
