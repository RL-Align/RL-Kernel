# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Ascend C Qwen-Image multi-axis RoPE backend (rotate-half convention).

Issue #386 `multi_axis_rope`: axes (16, 56, 56), text tokens on the grid
diagonal. The axis split and diagonal text placement live in the position
coordinates and the fp32 cos/sin tables built from them (see
``qwen_image_positions`` / ``build_multi_axis_cos_sin`` in the PyTorch
reference module); this backend applies the rotate-half rotation with the
Ascend C kernel, forward and backward.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.pytorch.rotary_embedding.multi_axis_rope import (
    QWEN_IMAGE_AXES_DIM,
    build_multi_axis_cos_sin,
)
from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False


def _fallback_op():
    """Portable op for inputs the Ascend forward cannot take.

    Triton rejects non-CUDA devices, so on NPU the only fallback is native.
    """
    from rl_engine.kernels.ops.pytorch.rotary_embedding.multi_axis_rope import (
        NativeMultiAxisRopeOp,
    )

    return NativeMultiAxisRopeOp()


def _rope_rows(x: Tensor, positions: Tensor) -> tuple[Tensor, int]:
    """Flatten x [..., S, D] to [n_rows, D] with row % S selecting the table row.

    Contiguous [..., S, D] flattening orders rows as consecutive S-blocks, so
    the modulo addressing of the Ascend C kernel lands every token on its own
    position row — the same contract as the single-sequence path of the
    generic RoPE wrapper.
    """
    if x.dim() < 2:
        raise ValueError(
            f"x must have at least 2 dimensions, got shape {tuple(x.shape)}"
        )
    dim = x.shape[-1]
    seq = positions.shape[-2] if positions.dim() >= 2 else positions.shape[-1]
    if dim != sum(QWEN_IMAGE_AXES_DIM):
        raise ValueError(
            f"x head_dim {dim} must equal sum(axes_dim)={sum(QWEN_IMAGE_AXES_DIM)}"
        )
    if seq == 0:
        if x.numel() != 0:
            raise ValueError("positions cannot be empty when x contains rows")
    elif x.numel() // dim % seq != 0:
        raise ValueError(
            f"row count {x.numel() // dim} not divisible by seq length {seq}; "
            "expected a [..., S, D] contiguous layout."
        )
    return x.contiguous().reshape(-1, dim), seq


class _MultiAxisRopeAscendFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        if positions.dim() == 3 and positions.shape[0] == 1:
            positions = positions[0]
        x_2d, seq = _rope_rows(x, positions)
        cos, sin = build_multi_axis_cos_sin(positions, QWEN_IMAGE_AXES_DIM, theta=theta)
        ctx.save_for_backward(cos, sin)
        ctx.x_shape = tuple(x.shape)
        out_2d = _C_npu.multi_axis_rope_ascend_forward(x_2d, cos, sin)
        return out_2d.reshape(ctx.x_shape)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_2d = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
            grad_x = _C_npu.multi_axis_rope_ascend_backward(grad_2d, cos, sin).reshape(
                ctx.x_shape
            )
        return grad_x, None, None


class MultiAxisRopeAscendOp:
    """Differentiable Ascend C Qwen-Image multi-axis RoPE backend.

    Supports fp16, bf16, and fp32 inputs; non-NPU or unsupported inputs fall
    back to the PyTorch reference.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "multi_axis_rope_ascend_forward"):
            raise RuntimeError(
                "multi_axis_rope_ascend is not compiled into _C_npu. Rebuild on an Ascend host "
                "with 'KERNEL_ALIGN_FORCE_ASCEND=1 pip install --no-build-isolation -e .'."
            )
        logger.info(
            "Successfully linked to precompiled _C_npu.multi_axis_rope_ascend kernel."
        )

    def __call__(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        theta: float = 10_000.0,
    ) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        theta: float = 10_000.0,
    ) -> Tensor:
        if x.device.type != "npu" or x.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return _fallback_op()(x, positions, theta=theta)
        return _MultiAxisRopeAscendFunction.apply(x, positions, float(theta))
