# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen3.6-35B-A3B text RoPE for one Q or K tensor, with FP32 intermediates.

Inputs use ``[T, H, 256]``: token count and head count can differ between calls,
including packed training, prefill, decode, and Q/K head shards. The caller
supplies absolute ``positions[T]`` and FP32 ``cos/sin[P, 32]`` tables. Table
generation and engine layout conversion belong to the caller.

Channels ``i`` and ``i + 32`` rotate together for ``i in [0, 32)``. Channels
64:256 pass through unchanged. Both forward and backward keep coefficients
and arithmetic in FP32, with separate multiply/add operations; only the final
store converts to the input dtype. This deliberately differs from paths that
round coefficients or intermediate products to BF16.

Position tables are constants, and only first derivatives with respect to the
input are exposed. Position IDs must be in ``[0, P)``. Device assertions check
that precondition with ``TRITON_DEBUG=1``; masked table loads prevent invalid
IDs from reading outside the tables even when assertions are disabled.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

_HEAD_DIM = 256
_ROTARY_DIM = 64
_HALF_ROTARY_DIM = _ROTARY_DIM // 2
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_POSITION_DTYPES = (torch.int32, torch.int64)
_TABLE_DTYPE = torch.float32
# PyTorch uses the CUDA device type for both CUDA and ROCm tensors.
_SUPPORTED_DEVICE_TYPES = ("cuda",)
_MAX_GRID_HEADS = 65535


@triton.jit
def _qwen3_6_partial_rope_fwd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    y_ptr,
    num_positions,
    num_heads: tl.constexpr,
    x_stride_token,
    x_stride_head,
    x_stride_channel,
    cos_stride_position,
    cos_stride_pair,
    sin_stride_position,
    sin_stride_pair,
    positions_stride,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    """Rotate one token/head pair and copy its unrotated channels.

    ``x_ptr`` addresses a possibly strided [T, H, HEAD_DIM] input; ``y_ptr``
    addresses contiguous output. Each *_stride is in elements, not bytes.
    ``positions_ptr`` maps tokens to rows of the independently strided cosine
    and sine tables. ``num_positions`` is the table height, not token count.
    The grid owns (token, head); there is no cross-token/head reduction.
    """
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    x_start = token * x_stride_token + head * x_stride_head
    y_start = (token * num_heads + head) * HEAD_DIM

    position = tl.load(positions_ptr + token * positions_stride).to(tl.int64)
    valid_position = (position >= 0) & (position < num_positions)
    tl.device_assert(valid_position, "RoPE position is outside the cos/sin tables")

    # The rotated prefix has exactly 32 pairs, so no per-pair mask is needed.
    HALF: tl.constexpr = ROTARY_DIM // 2
    pairs = tl.arange(0, HALF)
    cos_offsets = position * cos_stride_position + pairs * cos_stride_pair
    sin_offsets = position * sin_stride_position + pairs * sin_stride_pair
    first_offsets = x_start + pairs * x_stride_channel
    second_offsets = first_offsets + HALF * x_stride_channel
    cos = tl.load(cos_ptr + cos_offsets, valid_position, other=float("nan"))
    sin = tl.load(sin_ptr + sin_offsets, valid_position, other=float("nan"))
    x_first = tl.load(x_ptr + first_offsets).to(tl.float32)
    x_second = tl.load(x_ptr + second_offsets).to(tl.float32)

    # No BF16 coefficient or product rounding. Launch disables implicit FMA.
    y_first = x_first * cos - x_second * sin
    y_second = x_second * cos + x_first * sin
    dtype: tl.constexpr = x_ptr.dtype.element_ty
    tl.store(y_ptr + y_start + pairs, y_first.to(dtype))
    tl.store(y_ptr + y_start + pairs + HALF, y_second.to(dtype))

    # 192 passthrough channels use a 256-lane arange with the extra lanes masked.
    # Load/store in the source dtype: even NaN payloads and signed zeros survive.
    tail = ROTARY_DIM + tl.arange(0, BLOCK_TAIL)
    valid_tail = tail < HEAD_DIM
    passthrough = tl.load(x_ptr + x_start + tail * x_stride_channel, mask=valid_tail, other=0)
    tl.store(y_ptr + y_start + tail, passthrough, mask=valid_tail)


@triton.jit
def _qwen3_6_partial_rope_bwd_kernel(
    grad_y_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    grad_x_ptr,
    num_positions,
    num_heads: tl.constexpr,
    grad_y_stride_token,
    grad_y_stride_head,
    grad_y_stride_channel,
    cos_stride_position,
    cos_stride_pair,
    sin_stride_position,
    sin_stride_pair,
    positions_stride,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    """Apply the transpose rotation; the unrotated tail has identity gradient.

    Pointer/stride arguments mirror the forward kernel, but source strides
    belong to grad_y (which may be broadcast or noncontiguous). grad_x is new
    contiguous storage. With y1=a*c-b*s and y2=b*c+a*s, the VJP is
    da=dy1*c+dy2*s and db=dy2*c-dy1*s. No x values or reductions are needed.
    """
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    grad_y_start = token * grad_y_stride_token + head * grad_y_stride_head
    grad_x_start = (token * num_heads + head) * HEAD_DIM

    position = tl.load(positions_ptr + token * positions_stride).to(tl.int64)
    valid_position = (position >= 0) & (position < num_positions)
    tl.device_assert(valid_position, "RoPE position is outside the cos/sin tables")

    HALF: tl.constexpr = ROTARY_DIM // 2
    pairs = tl.arange(0, HALF)
    cos_offsets = position * cos_stride_position + pairs * cos_stride_pair
    sin_offsets = position * sin_stride_position + pairs * sin_stride_pair
    first_offsets = grad_y_start + pairs * grad_y_stride_channel
    second_offsets = first_offsets + HALF * grad_y_stride_channel
    cos = tl.load(cos_ptr + cos_offsets, valid_position, other=float("nan"))
    sin = tl.load(sin_ptr + sin_offsets, valid_position, other=float("nan"))
    grad_y_first = tl.load(grad_y_ptr + first_offsets).to(tl.float32)
    grad_y_second = tl.load(grad_y_ptr + second_offsets).to(tl.float32)

    grad_x_first = grad_y_first * cos + grad_y_second * sin
    grad_x_second = grad_y_second * cos - grad_y_first * sin
    dtype: tl.constexpr = grad_y_ptr.dtype.element_ty
    tl.store(grad_x_ptr + grad_x_start + pairs, grad_x_first.to(dtype))
    tl.store(grad_x_ptr + grad_x_start + pairs + HALF, grad_x_second.to(dtype))

    tail = ROTARY_DIM + tl.arange(0, BLOCK_TAIL)
    valid_tail = tail < HEAD_DIM
    grad_tail = tl.load(
        grad_y_ptr + grad_y_start + tail * grad_y_stride_channel, mask=valid_tail, other=0
    )
    tl.store(grad_x_ptr + grad_x_start + tail, grad_tail, mask=valid_tail)


def _validate_inputs(x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> None:
    """Check shapes, dtypes, devices and constant tables without reading GPU data.

    The caller guarantees 0 <= positions[t] < P; a host max/min check would
    synchronize and prevent ordinary CUDA Graph capture. The kernel masks table
    loads and also checks this precondition when TRITON_DEBUG=1 is enabled.
    """
    if x.ndim != 3 or x.shape[-1] != _HEAD_DIM:
        raise ValueError(f"x must have shape [T, H, {_HEAD_DIM}], got {tuple(x.shape)}.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}.")
    if x.device.type not in _SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"x must have device type in {_SUPPORTED_DEVICE_TYPES}, got {x.device}.")
    if x.shape[1] > _MAX_GRID_HEADS:
        raise ValueError(f"head count must not exceed {_MAX_GRID_HEADS}, got {x.shape[1]}.")

    for name, table in (("cos", cos), ("sin", sin)):
        if table.ndim != 2 or table.shape[1] != _HALF_ROTARY_DIM:
            raise ValueError(
                f"{name} must have shape [P, {_HALF_ROTARY_DIM}], got {tuple(table.shape)}."
            )
        if table.dtype != _TABLE_DTYPE:
            raise TypeError(f"{name} must have dtype {_TABLE_DTYPE}, got {table.dtype}.")
        if table.device != x.device:
            raise ValueError(f"{name} must be on x.device ({x.device}), got {table.device}.")
        if table.requires_grad:
            raise ValueError(f"{name} is a constant table and must not require gradients.")
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must share shape, got {tuple(cos.shape)} and {tuple(sin.shape)}."
        )
    if x.numel() > 0 and cos.shape[0] == 0:
        raise ValueError(f"nonempty x requires nonempty tables, got shape {tuple(cos.shape)}.")

    expected_positions_shape = (x.shape[0],)
    if positions.shape != expected_positions_shape:
        raise ValueError(
            f"positions must have shape {expected_positions_shape}, got {tuple(positions.shape)}."
        )
    if positions.dtype not in _POSITION_DTYPES:
        raise TypeError(f"positions must have dtype in {_POSITION_DTYPES}, got {positions.dtype}.")
    if positions.device != x.device:
        raise ValueError(f"positions must be on x.device ({x.device}), got {positions.device}.")


def _launch_qwen3_6_partial_rope_fwd(
    x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor
) -> Tensor:
    """Launch validated [T, H, 256] inputs and return a new contiguous output.

    Metadata is checked by the public entry point. Empty inputs need no launch.
    The device guard selects x's accelerator and restores the caller's current
    device on exit. Disabling FP fusion preserves separate FP32 multiply/add
    rounding even if the compiler could otherwise emit an FMA instruction.
    """
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return y
    num_tokens, num_heads, _ = x.shape
    num_positions, _ = cos.shape
    grid = (num_tokens, num_heads)
    device_module = torch.get_device_module(x.device)
    with device_module.device(x.device):
        _qwen3_6_partial_rope_fwd_kernel[grid](
            x,
            cos,
            sin,
            positions,
            y,
            num_positions=num_positions,
            num_heads=num_heads,
            x_stride_token=x.stride(0),
            x_stride_head=x.stride(1),
            x_stride_channel=x.stride(2),
            cos_stride_position=cos.stride(0),
            cos_stride_pair=cos.stride(1),
            sin_stride_position=sin.stride(0),
            sin_stride_pair=sin.stride(1),
            positions_stride=positions.stride(0),
            HEAD_DIM=_HEAD_DIM,
            ROTARY_DIM=_ROTARY_DIM,
            BLOCK_TAIL=triton.next_power_of_2(_HEAD_DIM - _ROTARY_DIM),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y


def _launch_qwen3_6_partial_rope_bwd(
    grad_y: Tensor, cos: Tensor, sin: Tensor, positions: Tensor
) -> Tensor:
    """Return contiguous dx using grad_y's own strides, including zero strides.

    Autograd supplies grad_y with the forward output's shape/dtype/device; the
    saved tables and positions were already validated by forward. Use the same
    device guard and no-FMA arithmetic as forward, with one final dtype cast.
    """
    grad_x = torch.empty(grad_y.shape, dtype=grad_y.dtype, device=grad_y.device)
    if grad_y.numel() == 0:
        return grad_x
    num_tokens, num_heads, _ = grad_y.shape
    grid = (num_tokens, num_heads)
    num_positions, _ = cos.shape
    device_module = torch.get_device_module(grad_y.device)
    with device_module.device(grad_y.device):
        _qwen3_6_partial_rope_bwd_kernel[grid](
            grad_y,
            cos,
            sin,
            positions,
            grad_x,
            num_positions=num_positions,
            num_heads=num_heads,
            grad_y_stride_token=grad_y.stride(0),
            grad_y_stride_head=grad_y.stride(1),
            grad_y_stride_channel=grad_y.stride(2),
            cos_stride_position=cos.stride(0),
            cos_stride_pair=cos.stride(1),
            sin_stride_position=sin.stride(0),
            sin_stride_pair=sin.stride(1),
            positions_stride=positions.stride(0),
            HEAD_DIM=_HEAD_DIM,
            ROTARY_DIM=_ROTARY_DIM,
            BLOCK_TAIL=triton.next_power_of_2(_HEAD_DIM - _ROTARY_DIM),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return grad_x


class _Qwen3_6PartialRoPETritonFunction(torch.autograd.Function):
    """Save position IDs and coefficients; the linear backward does not need x."""

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Save the constant rotation data; x is not needed by the linear VJP."""
        ctx.save_for_backward(cos, sin, positions)
        return _launch_qwen3_6_partial_rope_fwd(x, cos, sin, positions)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor, None, None, None]:
        """Return dx and no table/position gradients; double backward is unsupported."""
        cos, sin, positions = ctx.saved_tensors
        grad_x = _launch_qwen3_6_partial_rope_bwd(grad_y, cos, sin, positions)
        return grad_x, None, None, None


class TritonQwen3_6PartialRoPEOp:
    """Rotate Qwen3.6's leading channels using caller-supplied FP32 tables.

    Args:
        x: BF16 or FP32 Q/K features, shape ``[T, H, 256]``.
        cos: FP32 cosine table, shape ``[P, 32]``; one coefficient per pair.
        sin: FP32 sine table with the same shape as cos.
        positions: Int32/int64 absolute table row IDs, shape ``[T]``.

    All inputs must share a device; strided views are supported. The returned
    tensor is contiguous with x's shape and dtype, and x is never modified.
    Q and K are separate calls so their head counts need not match. Table
    generation must use the model's position/frequency rules on both engines.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Delegate to :meth:`forward` without changing its tensor contract."""
        return self.forward(x, cos, sin, positions)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Apply FP32 rotation and round once to the source dtype on output.

        Args:
            x: BF16/FP32 Q or K features, shape [T, H, 256], on a Triton GPU.
            cos: Constant FP32 cosine table [P, 32] on x.device.
            sin: Constant FP32 sine table with cos's shape and device.
            positions: Int32/int64 row IDs [T] on x.device, each in [0, P).

        Returns:
            New contiguous tensor matching x's shape/dtype/device. Channels
            0:64 rotate in split-half pairs; channels 64:256 are copied exactly.

        Raises:
            TypeError: An input, table or index dtype is unsupported.
            ValueError: Shapes/devices are incompatible, tables require gradients,
                or the head count exceeds the second launch-grid axis limit.

        Only first-order x gradients are supported. Position bounds are a caller
        precondition: with debug disabled, invalid IDs yield NaN in the rotated
        prefix and preserve the tail instead of reading out of bounds. Identical
        inputs/tables/positions use the same arithmetic when tokens are chunked;
        engine-level alignment also requires matching upstream computations.
        """
        _validate_inputs(x, cos, sin, positions)
        return _Qwen3_6PartialRoPETritonFunction.apply(x, cos, sin, positions)
