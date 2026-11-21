# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden semantics of the DSv4 shared expert's clamped SwiGLU (vLLM rollout).

On an H200 this golden matched a verbatim copy of vLLM's
``silu_and_mul_with_clamp`` CUDA kernel byte for byte over every BF16 gate
value against 16 up values (NaN, +-inf, +-0 included). CUDA ``torch.exp`` is
libdevice ``expf`` like vLLM's; CPU ``exp`` can differ in the last bit, so the
byte checks here run on CUDA.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp import (
    SWIGLU_LIMIT,
    silu_and_mul_clamp,
    silu_and_mul_clamp_backward,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _z(gate: list[float], up: list[float], device="cuda") -> torch.Tensor:
    return torch.tensor([gate + up], dtype=torch.float32).to(torch.bfloat16).to(device)


def _formula(gate: float, up: float) -> float:
    """The vLLM expression evaluated in FP32 with the same rounding points."""
    g = torch.tensor(min(gate, SWIGLU_LIMIT), dtype=torch.float32, device="cuda")
    u = torch.tensor(max(min(up, SWIGLU_LIMIT), -SWIGLU_LIMIT), dtype=torch.float32, device="cuda")
    act = (g / (torch.exp(-g) + torch.ones_like(g))).to(torch.bfloat16).float()
    return float((act * (u + 0.0)).to(torch.bfloat16))


@requires_cuda
def test_clamps_gate_above_and_up_on_both_sides():
    gates, ups = [0.5, 12.0, -30.0, 10.0], [20.0, -20.0, 3.0, 9.5]
    out = silu_and_mul_clamp(_z(gates, ups)).float()[0].tolist()
    assert out == [_formula(g, u) for g, u in zip(gates, ups)]
    # gate 12 -> 10 and up 20 -> 10: silu(10) * 10
    assert out[1] == _formula(10.0, -10.0)


@requires_cuda
def test_nan_gate_becomes_the_limit_and_negative_zero_up_becomes_positive():
    out = silu_and_mul_clamp(_z([float("nan"), 2.0], [1.0, -0.0]))
    assert out.float()[0, 0].item() == _formula(SWIGLU_LIMIT, 1.0)
    assert out.view(torch.int16)[0, 1].item() == 0  # +0, not -0 (u + 0.0)


@requires_cuda
def test_backward_is_zero_at_and_beyond_the_bounds():
    z = _z([10.0, 11.0, 1.0, 1.0, 1.0], [1.0, 1.0, 10.0, -10.0, 9.5])
    dh = torch.ones(1, 5, dtype=torch.bfloat16, device="cuda")
    dz = silu_and_mul_clamp_backward(dh, z).float()[0]
    dgate, dup = dz[:5], dz[5:]
    assert dgate[0].item() == 0.0 and dgate[1].item() == 0.0
    assert dup[2].item() == 0.0 and dup[3].item() == 0.0
    assert dgate[2].item() != 0.0 and dup[4].item() != 0.0


@requires_cuda
def test_backward_matches_autograd_inside_the_bounds():
    gen = torch.Generator().manual_seed(0)
    z = (torch.randn(8, 256, generator=gen) * 3).to(torch.bfloat16).cuda()
    dh = torch.randn(8, 128, generator=gen).to(torch.bfloat16).cuda()
    zf = z.float().requires_grad_(True)
    g = torch.clamp(zf[:, :128], max=SWIGLU_LIMIT)
    u = torch.clamp(zf[:, 128:], min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    (torch.nn.functional.silu(g) * u * dh.float()).sum().backward()
    got = silu_and_mul_clamp_backward(dh, z).float()
    rel = (got - zf.grad).norm() / zf.grad.norm()
    assert float(rel) < 1e-2  # BF16 output and the rounded activation in dup


@requires_cuda
def test_rows_are_independent():
    gen = torch.Generator().manual_seed(1)
    z = (torch.randn(64, 512, generator=gen) * 5).to(torch.bfloat16).cuda()
    out = silu_and_mul_clamp(z)
    assert torch.equal(
        silu_and_mul_clamp(z[10:11].contiguous()).view(torch.int16), out[10:11].view(torch.int16)
    )


def test_rejects_bad_input():
    with pytest.raises(TypeError, match="BF16"):
        silu_and_mul_clamp(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="even"):
        silu_and_mul_clamp(torch.zeros(1, 3, dtype=torch.bfloat16))
