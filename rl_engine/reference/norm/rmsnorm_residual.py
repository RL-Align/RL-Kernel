# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fixed-order FP32 reference for task 5 (``rmsnorm_residual``).

Every long reduction is a serial ascending left fold with one FP32
accumulator, and every multiply and add rounds separately. Nothing here calls
``torch.sum``, ``mean``, ``matmul`` or ``einsum``: those choose their own tree
and would silently un-freeze the bytes a kernel has to reproduce.

The reference runs on CPU and on the device under test. Transcendentals
(``rsqrt``) are evaluated by the device it runs on, so a byte comparison
against a kernel is made with both sides on the same device.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.contracts.operators.rmsnorm_residual import (
    EPS,
    PROFILE,
    RMSNormResidualSaved,
    validate_backward,
    validate_forward,
)


def fixed_sum(x: Tensor, dim: int) -> Tensor:
    """Left-fold ``x`` along ``dim`` in FP32, ascending index order, one add at a time."""
    moved = x.to(torch.float32).movedim(dim, 0)
    acc = torch.zeros(moved.shape[1:], dtype=torch.float32, device=moved.device)
    for index in range(moved.shape[0]):
        acc = acc + moved[index]
    return acc


def fixed_sumsq(x: Tensor, dim: int) -> Tensor:
    """``sum(x * x)`` along ``dim``; each square rounds before it is accumulated."""
    moved = x.to(torch.float32).movedim(dim, 0)
    acc = torch.zeros(moved.shape[1:], dtype=torch.float32, device=moved.device)
    for index in range(moved.shape[0]):
        acc = acc + moved[index] * moved[index]
    return acc


class RMSNormResidualReference:
    """Serial FP32 reference; slow on purpose, auditable line by line."""

    name = "reference"
    profile = PROFILE

    @staticmethod
    @torch.no_grad()
    def forward(
        x: Tensor, gamma: Tensor, eps: float = EPS
    ) -> tuple[Tensor, Tensor, RMSNormResidualSaved]:
        tokens, hidden = validate_forward(x, gamma, eps)
        x32 = x.to(torch.float32)
        s = fixed_sumsq(x32, dim=1)
        # A Python scalar lets PyTorch multiply by a rounded reciprocal.
        # A device tensor keeps this the contract's correctly rounded division.
        denominator = torch.full((), hidden, dtype=torch.float32, device=x.device)
        m = s / denominator
        r = torch.rsqrt(m + float(eps))
        y32 = (x32 * r.unsqueeze(1)) * gamma.to(torch.float32).unsqueeze(0)
        y = y32.to(torch.bfloat16)
        residual = x.clone(memory_format=torch.contiguous_format)
        return y, residual, RMSNormResidualSaved(r=r, hidden=hidden)

    @staticmethod
    @torch.no_grad()
    def backward(
        dy: Tensor, d_residual: Tensor, x: Tensor, gamma: Tensor, saved: RMSNormResidualSaved
    ) -> tuple[Tensor, Tensor]:
        tokens, hidden = validate_backward(dy, d_residual, x, gamma, saved)
        x32 = x.to(torch.float32)
        dy32 = dy.to(torch.float32)
        r = saved.r
        u = dy32 * gamma.to(torch.float32).unsqueeze(0)
        q = fixed_sum(u * x32, dim=1)
        r3 = (r * r) * r
        denominator = torch.full((), hidden, dtype=torch.float32, device=x.device)
        dx_norm = (r.unsqueeze(1) * u) - (
            ((x32 * r3.unsqueeze(1)) * q.unsqueeze(1)) / denominator
        )
        dx = dx_norm + d_residual.to(torch.float32)
        dgamma = fixed_sum((dy32 * x32) * r.unsqueeze(1), dim=0)
        return dx, dgamma


__all__ = ["RMSNormResidualReference", "fixed_sum", "fixed_sumsq"]
