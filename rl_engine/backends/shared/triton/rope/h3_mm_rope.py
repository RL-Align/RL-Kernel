# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Standalone Triton implementation of MiniMax-H3 partial MM-RoPE.

The caller builds FP32 ``[S, 96]`` cosine/sine tables from the packed tokens'
``(t, h, w)`` positions: 16 frequencies per axis, concatenated and duplicated.
This module applies those tables to ``[B, S, H, 128]`` Q or K tensors after
QK RMSNorm. Channels ``i`` and ``i + 48`` rotate together; channels 96:128
pass through unchanged. Q and K can reuse the same tables.

Tables are cast to the activation dtype before multiplication. Products and
outputs round at the same boundaries as the eager H3 expression. Backward
uses the corresponding two local gradient contributions, without atomics or
cross-token reductions. Only first derivatives with respect to Q/K are exposed;
the position tables are constants in this operator's contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_HEAD_DIM = 128
_ROTARY_DIM = 96
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _h3_mm_rope_fwd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    x_stride_batch: tl.constexpr,
    x_stride_sequence: tl.constexpr,
    x_stride_head: tl.constexpr,
    x_stride_channel: tl.constexpr,
    cos_stride_sequence: tl.constexpr,
    cos_stride_channel: tl.constexpr,
    sin_stride_sequence: tl.constexpr,
    sin_stride_channel: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    """Rotate the leading 96 channels of one token/head row and copy the tail.

    Grid axes are ``(head, token, batch)``. Input strides describe
    ``[B, S, H, D]``; output rows are contiguous. Both table halves are loaded
    explicitly. Products round to the activation dtype before the final
    addition/subtraction, matching H3's eager operations.

    ``sequence_length`` counts tokens; ``num_heads`` counts heads per token.
    Each ``*_stride_*`` is an element offset for a one-step move along that
    tensor's axis. Shape and stride metadata specialize this kernel at compile
    time; changing their values may compile another version.
    """
    head = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    row = (batch * sequence_length + token) * num_heads + head
    x_start = batch * x_stride_batch + token * x_stride_sequence + head * x_stride_head
    y_start = row * HEAD_DIM
    HALF: tl.constexpr = ROTARY_DIM // 2
    dtype: tl.constexpr = x_ptr.dtype.element_ty

    # 48 pairs occupy 64 lanes; padded lanes do not load or store any channels.
    pairs = tl.arange(0, BLOCK_PAIRS)
    valid = pairs < HALF
    x_first = tl.load(x_ptr + x_start + pairs * x_stride_channel, valid, other=0).to(tl.float32)
    x_second = tl.load(x_ptr + x_start + (pairs + HALF) * x_stride_channel, valid, other=0).to(
        tl.float32
    )
    cos_first = (
        tl.load(cos_ptr + token * cos_stride_sequence + pairs * cos_stride_channel, valid, other=0)
        .to(dtype)
        .to(tl.float32)
    )
    cos_second = (
        tl.load(
            cos_ptr + token * cos_stride_sequence + (pairs + HALF) * cos_stride_channel,
            valid,
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )
    sin_first = (
        tl.load(sin_ptr + token * sin_stride_sequence + pairs * sin_stride_channel, valid, other=0)
        .to(dtype)
        .to(tl.float32)
    )
    sin_second = (
        tl.load(
            sin_ptr + token * sin_stride_sequence + (pairs + HALF) * sin_stride_channel,
            valid,
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )

    # Round each product to the activation dtype before the final add/subtract,
    # matching H3's eager operations. The launch also disables FP fusion.
    direct_first = (x_first * cos_first).to(dtype).to(tl.float32)
    direct_second = (x_second * cos_second).to(dtype).to(tl.float32)
    cross_first = (x_second * sin_first).to(dtype).to(tl.float32)
    cross_second = (x_first * sin_second).to(dtype).to(tl.float32)
    y_first = direct_first - cross_first
    y_second = direct_second + cross_second

    tl.store(y_ptr + y_start + pairs, y_first.to(dtype), valid)
    tl.store(y_ptr + y_start + pairs + HALF, y_second.to(dtype), valid)

    # The unrotated tail passes through without arithmetic or dtype conversion.
    tail = ROTARY_DIM + tl.arange(0, HEAD_DIM - ROTARY_DIM)
    passthrough = tl.load(x_ptr + x_start + tail * x_stride_channel)
    tl.store(y_ptr + y_start + tail, passthrough)


@triton.jit
def _h3_mm_rope_bwd_kernel(
    grad_y_ptr,
    cos_ptr,
    sin_ptr,
    grad_x_ptr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    grad_y_stride_batch: tl.constexpr,
    grad_y_stride_sequence: tl.constexpr,
    grad_y_stride_head: tl.constexpr,
    grad_y_stride_channel: tl.constexpr,
    cos_stride_sequence: tl.constexpr,
    cos_stride_channel: tl.constexpr,
    sin_stride_sequence: tl.constexpr,
    sin_stride_channel: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    """Apply the transpose rotation to one row of upstream gradients.

    Grid axes are ``(head, token, batch)``. Input strides belong to ``grad_y``;
    ``grad_x`` rows are contiguous. The saved table halves determine the
    coefficients, including when the halves differ. The unrotated tail has an
    identity derivative.

    Shape counts and element strides are compile-time metadata, as in forward.
    ``grad_y_stride_*`` describes the actual upstream gradient layout, including
    zero strides for broadcast gradients; it need not match x's forward layout.
    """
    head = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    row = (batch * sequence_length + token) * num_heads + head
    grad_y_start = (
        batch * grad_y_stride_batch + token * grad_y_stride_sequence + head * grad_y_stride_head
    )
    grad_x_start = row * HEAD_DIM
    HALF: tl.constexpr = ROTARY_DIM // 2
    dtype: tl.constexpr = grad_y_ptr.dtype.element_ty

    pairs = tl.arange(0, BLOCK_PAIRS)
    valid = pairs < HALF
    grad_y_first = tl.load(
        grad_y_ptr + grad_y_start + pairs * grad_y_stride_channel, valid, other=0
    ).to(tl.float32)
    grad_y_second = tl.load(
        grad_y_ptr + grad_y_start + (pairs + HALF) * grad_y_stride_channel, valid, other=0
    ).to(tl.float32)
    cos_first = (
        tl.load(cos_ptr + token * cos_stride_sequence + pairs * cos_stride_channel, valid, other=0)
        .to(dtype)
        .to(tl.float32)
    )
    cos_second = (
        tl.load(
            cos_ptr + token * cos_stride_sequence + (pairs + HALF) * cos_stride_channel,
            valid,
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )
    sin_first = (
        tl.load(sin_ptr + token * sin_stride_sequence + pairs * sin_stride_channel, valid, other=0)
        .to(dtype)
        .to(tl.float32)
    )
    sin_second = (
        tl.load(
            sin_ptr + token * sin_stride_sequence + (pairs + HALF) * sin_stride_channel,
            valid,
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )

    # Each input receives two local gradient contributions, rounded before addition.
    direct_first = (grad_y_first * cos_first).to(dtype).to(tl.float32)
    direct_second = (grad_y_second * cos_second).to(dtype).to(tl.float32)
    cross_first = (grad_y_second * sin_second).to(dtype).to(tl.float32)
    cross_second = (grad_y_first * sin_first).to(dtype).to(tl.float32)
    grad_x_first = direct_first + cross_first
    grad_x_second = direct_second - cross_second

    tl.store(grad_x_ptr + grad_x_start + pairs, grad_x_first.to(dtype), valid)
    tl.store(grad_x_ptr + grad_x_start + pairs + HALF, grad_x_second.to(dtype), valid)

    tail = ROTARY_DIM + tl.arange(0, HEAD_DIM - ROTARY_DIM)
    grad_tail = tl.load(grad_y_ptr + grad_y_start + tail * grad_y_stride_channel)
    tl.store(grad_x_ptr + grad_x_start + tail, grad_tail)


def _validate_inputs(x: Tensor, cos: Tensor, sin: Tensor) -> None:
    """Validate H3's shape/device contract without reading GPU tensor values.

    Head count may vary for head-sharded execution. Strided input/table views
    are accepted; empty inputs and trainable position tables are not supported.
    """
    if x.ndim != 4 or x.shape[-1] != _HEAD_DIM:
        raise ValueError("x must have shape [B, S, H, 128].")
    if x.numel() == 0:
        raise ValueError("x must have nonempty batch, sequence, and head dimensions.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must have dtype float16, bfloat16, or float32.")
    if x.device.type != "cuda":
        raise ValueError("TritonH3MMRoPEOp requires CUDA or ROCm tensors.")
    for name, table in (("cos", cos), ("sin", sin)):
        if table.shape != (x.shape[1], _ROTARY_DIM):
            raise ValueError(f"{name} must have shape [S, 96], matching x's sequence length.")
        if table.device != x.device:
            raise ValueError(f"{name} must be on the same device as x.")
        if table.dtype != torch.float32:
            raise TypeError(f"{name} must be an FP32 H3 position table.")
        if table.requires_grad:
            raise ValueError(f"{name} must not require gradients; position tables are constants.")


def _launch_h3_mm_rope_fwd(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Launch forward on x's device/current stream and return a contiguous output.

    Training and inference use this same kernel and fixed arithmetic. Input and
    table strides support noncontiguous views without copying.
    """
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    batch_size, sequence_length, num_heads, _ = x.shape
    grid = (num_heads, sequence_length, batch_size)
    # ROCm PyTorch uses the same torch.cuda device/stream API as CUDA PyTorch.
    with torch.cuda.device(x.device):
        _h3_mm_rope_fwd_kernel[grid](
            x,
            cos,
            sin,
            output,
            sequence_length=sequence_length,
            num_heads=num_heads,
            x_stride_batch=x.stride(0),
            x_stride_sequence=x.stride(1),
            x_stride_head=x.stride(2),
            x_stride_channel=x.stride(3),
            cos_stride_sequence=cos.stride(0),
            cos_stride_channel=cos.stride(1),
            sin_stride_sequence=sin.stride(0),
            sin_stride_channel=sin.stride(1),
            HEAD_DIM=_HEAD_DIM,
            ROTARY_DIM=_ROTARY_DIM,
            BLOCK_PAIRS=triton.next_power_of_2(_ROTARY_DIM // 2),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def _launch_h3_mm_rope_bwd(grad_y: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Launch backward on grad_y's device/current stream with saved position tables.

    Upstream strides support noncontiguous and broadcast gradients without
    copying. The returned input gradient is contiguous and has grad_y's dtype.
    """
    grad_x = torch.empty(grad_y.shape, dtype=grad_y.dtype, device=grad_y.device)
    batch_size, sequence_length, num_heads, _ = grad_y.shape
    grid = (num_heads, sequence_length, batch_size)
    with torch.cuda.device(grad_y.device):
        _h3_mm_rope_bwd_kernel[grid](
            grad_y,
            cos,
            sin,
            grad_x,
            sequence_length=sequence_length,
            num_heads=num_heads,
            grad_y_stride_batch=grad_y.stride(0),
            grad_y_stride_sequence=grad_y.stride(1),
            grad_y_stride_head=grad_y.stride(2),
            grad_y_stride_channel=grad_y.stride(3),
            cos_stride_sequence=cos.stride(0),
            cos_stride_channel=cos.stride(1),
            sin_stride_sequence=sin.stride(0),
            sin_stride_channel=sin.stride(1),
            HEAD_DIM=_HEAD_DIM,
            ROTARY_DIM=_ROTARY_DIM,
            BLOCK_PAIRS=triton.next_power_of_2(_ROTARY_DIM // 2),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return grad_x


class _H3MMRoPETritonFunction(torch.autograd.Function):
    """Save the position tables for a first-order input-gradient calculation."""

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        ctx.save_for_backward(cos, sin)
        return _launch_h3_mm_rope_fwd(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = _launch_h3_mm_rope_bwd(grad_y, cos, sin)
        return grad_x, None, None


class TritonH3MMRoPEOp:
    """Apply H3 rotate-half to 96 of 128 channels of a Q or K tensor.

    Args:
        x: FP16/BF16/FP32 activation of shape ``[B, S, H, 128]``. H3 normally
            uses 56 heads; smaller head shards use the same per-head arithmetic.
        cos: Constant FP32 ``[S, 96]`` table for the packed token positions.
        sin: Constant FP32 ``[S, 96]`` table on the same device as ``x``.

    The caller constructs the H3 three-axis tables once and reuses them for Q
    and K. Tables are shared across the batch and heads. Output has the input
    shape/dtype and is contiguous; channels 96:128 pass through bit-for-bit.
    This backend exposes first-order gradients for ``x`` only. GPU numerical
    and performance qualification is required before registry integration.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(x, cos, sin)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Validate the public inputs and apply partial rotation with autograd."""
        _validate_inputs(x, cos, sin)
        return _H3MMRoPETritonFunction.apply(x, cos, sin)
