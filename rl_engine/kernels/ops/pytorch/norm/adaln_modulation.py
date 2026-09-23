# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 AdaLN reference for one Qwen-Image attention or MLP modulation."""

from __future__ import annotations

import math

import torch

from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32


def _sum_hidden(values: torch.Tensor) -> torch.Tensor:
    """Fixed pairwise tree over H, independent of batch size and row count."""
    hidden = values.shape[-1]
    width = 1 << (hidden - 1).bit_length()
    if width != hidden:
        values = torch.nn.functional.pad(values, (0, width - hidden))
    while width > 1:
        width //= 2
        values = values[..., :width] + values[..., width:]
    return values[..., 0]


def _normalize_fp32(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1]
    x32 = x.float()
    mean = _sum_hidden(x32) / hidden
    centered = x32 - mean.unsqueeze(-1)
    variance_eps = _sum_hidden(centered * centered) / hidden + eps
    # CPU torch.sqrt(float32) can miss a correctly rounded result by one ULP.
    root = torch.sqrt(variance_eps.double()).float()
    rstd = torch.reciprocal(root)
    return centered * rstd.unsqueeze(-1), rstd


def _validate(x: torch.Tensor, modulation: torch.Tensor, eps: float) -> None:
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("x must have nonempty shape [B, S, H]")
    if modulation.shape != (x.shape[0], 3 * x.shape[-1]):
        raise ValueError("modulation must have shape [B, 3H]")
    if x.device != modulation.device or x.dtype != modulation.dtype:
        raise ValueError("x and modulation must share device and dtype")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("x and modulation must be FP32 or BF16")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")


class _AdaLNReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, modulation: torch.Tensor, eps: float):
        shift, scale, gate = modulation.float().chunk(3, dim=-1)
        norm, rstd = _normalize_fp32(x, eps)
        y = (norm * (1.0 + scale[:, None, :]) + shift[:, None, :]).to(x.dtype)
        ctx.save_for_backward(norm, rstd, scale)
        ctx.x_dtype = x.dtype
        ctx.mod_dtype = modulation.dtype
        return y, gate[:, None, :].to(modulation.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_gate: torch.Tensor):
        norm, rstd, scale = ctx.saved_tensors
        hidden = norm.shape[-1]
        dy = grad_y.float()
        dnorm = dy * (1.0 + scale[:, None, :])
        mean_dnorm = _sum_hidden(dnorm) / hidden
        mean_dnorm_norm = _sum_hidden(dnorm * norm) / hidden
        dx = (
            (dnorm - mean_dnorm.unsqueeze(-1) - norm * mean_dnorm_norm.unsqueeze(-1))
            * rstd.unsqueeze(-1)
        ).to(ctx.x_dtype)
        # S is folded in ascending logical-token order for every sample.
        dm = torch.cat(
            [
                torch.stack([reduce_rows_fp32(rows) for rows in dy]),
                torch.stack([reduce_rows_fp32(rows) for rows in dy * norm]),
                grad_gate.float().squeeze(1),
            ],
            dim=-1,
        ).to(ctx.mod_dtype)
        return dx, dm, None


def _select01_reference(
    x: torch.Tensor,
    modulation: torch.Tensor,
    modulate_index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("x must have nonempty shape [B, S, H]")
    batch, seq, hidden = x.shape
    if modulation.shape != (2 * batch, 3 * hidden):
        raise ValueError("indexed modulation must have shape [2B, 3H]")
    if modulate_index.shape not in ((batch, seq), (1, seq)):
        raise ValueError("modulate_index must have shape [B, S] or [1, S]")
    if x.device != modulation.device or x.device != modulate_index.device:
        raise ValueError("x, modulation, and modulate_index must share device")
    if x.dtype != modulation.dtype or x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("x and modulation must share FP32 or BF16 dtype")
    if modulate_index.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("modulate_index must have boolean or integer dtype")
    if not torch.all((modulate_index == 0) | (modulate_index == 1)).item():
        raise ValueError("modulate_index values must be 0 or 1")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    shift, scale, gate = modulation.float().chunk(3, dim=-1)
    index = modulate_index.bool().unsqueeze(-1)
    shift = torch.where(index, shift[batch:, None, :], shift[:batch, None, :])
    scale = torch.where(index, scale[batch:, None, :], scale[:batch, None, :])
    gate = torch.where(index, gate[batch:, None, :], gate[:batch, None, :])

    norm, _ = _normalize_fp32(x, eps)
    return (norm * (1.0 + scale) + shift).to(x.dtype), gate.to(modulation.dtype)


class NativeAdaLNModulationOp:
    def __call__(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
        *,
        modulate_index: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        return self.forward(x, modulation, modulate_index=modulate_index, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
        *,
        modulate_index: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        if modulate_index is not None:
            return _select01_reference(x, modulation, modulate_index, float(eps))
        _validate(x, modulation, eps)
        return _AdaLNReference.apply(x, modulation, float(eps))
