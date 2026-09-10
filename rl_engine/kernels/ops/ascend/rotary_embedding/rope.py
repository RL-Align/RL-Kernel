# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Ascend C RoPE backend (GPT-NeoX/Hugging Face rotate-half convention)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False


def _build_cos_sin(
    positions: Tensor,
    half: int,
    theta: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build fp32 [table_rows, half] caches with the reference RoPE formula."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    freqs = positions.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv_freq
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _rope_table(
    x: Tensor, positions: Tensor, theta: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten x so row modulo table length selects the correct position cache."""
    if x.dim() < 2:
        raise ValueError(
            f"x must have at least 2 dimensions, got shape {tuple(x.shape)}"
        )
    dim = x.shape[-1]
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be a positive even number, got {dim}")

    if positions.dim() == 1:
        table_len = int(positions.shape[0])
        x_2d = x.contiguous().reshape(-1, dim)
        if table_len == 0:
            if x_2d.shape[0] != 0:
                raise ValueError("positions cannot be empty when x contains rows")
        elif x_2d.shape[0] % table_len != 0:
            raise ValueError(
                f"row count {x_2d.shape[0]} not divisible by seq length {table_len}; "
                "expected a [..., S, D] contiguous layout."
            )
        cos, sin = _build_cos_sin(positions, dim // 2, float(theta), x.device)
        return x_2d, cos, sin

    if positions.dim() != 2:
        raise ValueError(
            f"positions must be [S] or [B, S], got shape {tuple(positions.shape)}"
        )
    batch, seq = positions.shape
    if x.shape[0] != batch or x.shape[-2] != seq:
        raise ValueError(
            f"positions {tuple(positions.shape)} is incompatible with x {tuple(x.shape)}; "
            "expected x [B, ..., S, D]"
        )
    if x.dim() == 4:
        x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(-1, dim)
    elif x.dim() == 3:
        x_2d = x.contiguous().reshape(-1, dim)
    else:
        raise ValueError(
            f"RoPE [B, S] positions require x [B, S, D] or [B, H, S, D], got {x.dim()}D"
        )

    table_len = batch * seq
    if table_len == 0:
        if x_2d.shape[0] != 0:
            raise ValueError("positions cannot be empty when x contains rows")
    elif x_2d.shape[0] % table_len != 0:
        raise ValueError(f"row count {x_2d.shape[0]} not divisible by B*S={table_len}")
    cos, sin = _build_cos_sin(positions, dim // 2, float(theta), x.device)
    return x_2d, cos, sin


def _restore_rope(out_2d: Tensor, x: Tensor, positions: Tensor) -> Tensor:
    if positions.dim() == 1 or x.dim() != 4:
        return out_2d.reshape(x.shape)
    heads, batch, seq, dim = x.shape[1], x.shape[0], x.shape[2], x.shape[3]
    return out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()


class _RoPEAscendFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin = _rope_table(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.x_shape = tuple(x.shape)
        ctx.pos_dim = positions.dim()
        out_2d = _C_npu.rope_apply_ascend(x_2d, cos, sin, 1.0)
        return _restore_rope(out_2d, x, positions)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.pos_dim == 2 and len(ctx.x_shape) == 4:
                grad_2d = (
                    grad_out.permute(1, 0, 2, 3)
                    .contiguous()
                    .reshape(-1, ctx.x_shape[-1])
                )
                out_2d = _C_npu.rope_apply_ascend(grad_2d, cos, sin, -1.0)
                heads, batch, seq, dim = (
                    ctx.x_shape[1],
                    ctx.x_shape[0],
                    ctx.x_shape[2],
                    ctx.x_shape[3],
                )
                grad_x = (
                    out_2d.reshape(heads, batch, seq, dim)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            else:
                grad_2d = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
                grad_x = _C_npu.rope_apply_ascend(grad_2d, cos, sin, -1.0).reshape(
                    grad_out.shape
                )
        return grad_x, None, None


class RoPEAscendOp:
    """Differentiable Ascend C RoPE backend for fp16, bf16, and fp32 inputs."""

    op_class = "elementwise"

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "rope_apply_ascend"):
            raise RuntimeError(
                "rope_apply_ascend is not compiled into _C_npu. Rebuild on an Ascend host with "
                "'KERNEL_ALIGN_FORCE_ASCEND=1 pip install --no-build-isolation -e .'."
            )
        logger.info(
            "Successfully linked to precompiled _C_npu.rope_apply_ascend kernel."
        )

    def __call__(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        theta: float = 1_000_000.0,
    ) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        theta: float = 1_000_000.0,
    ) -> Tensor:
        if x.device.type != "npu":
            raise RuntimeError(
                f"RoPEAscendOp requires an NPU tensor, got device '{x.device}'."
            )
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"RoPEAscendOp supports fp16, bf16, and fp32, got {x.dtype}."
            )
        return _RoPEAscendFunction.apply(x, positions, float(theta))
