# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""First-order autograd for the explicit ROCm shared LoRA delta."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.lora_delta import RocmLoRADelta
from rl_engine.contracts.operators.lora import validate_lora_inputs


class _LoRADelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, alpha, recompute):
        backend = RocmLoRADelta()
        y, u = backend.forward(x, a, b, alpha)
        ctx.backend = backend
        ctx.alpha = alpha
        ctx.recompute = recompute
        ctx.save_for_backward(x, a, b, *(() if recompute else (u,)))
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        # Accessing saved_tensors also checks that optimizer/load updates did
        # not mutate X/A/B between forward and backward, in either save mode.
        x, a, b, *saved_u = ctx.saved_tensors
        u = ctx.backend._project(x, a) if ctx.recompute else saved_u[0]
        grads = ctx.backend.backward(dy, x, a, b, ctx.alpha, u)
        # Low-level VJP is FP32. Autograd's BF16 input edges round exactly here.
        dx, da, db = (g if needed else None for g, needed in zip(grads, ctx.needs_input_grad[:3]))
        return dx, da, db, None, None


def shared_grouped_lora_delta(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    *,
    recompute: bool = False,
) -> torch.Tensor:
    """Return FP32 ``round_bf16(X @ A.T) @ B.T * alpha`` on ROCm.

    A and B are shared across already-dispatched rows, not per-expert weights.
    ``alpha`` is the effective multiplier (pass config.alpha/rank if needed).
    No base weights, routing, implicit quantization or automatic CPU fallback.
    ``recompute`` saves X/A/B and recreates U using the same forward path.
    """
    alpha = validate_lora_inputs(x, a, b, alpha)
    if not isinstance(recompute, bool):
        raise TypeError("LoRA recompute must be bool")
    return _LoRADelta.apply(x, a, b, alpha, recompute)
