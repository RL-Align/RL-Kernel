# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure PyTorch reference for MiniMax-H3 partial MM-RoPE.

The caller supplies FP32 ``[S, 96]`` cosine/sine tables for the packed tokens'
three-axis positions. Channels ``i`` and ``i + 48`` rotate together; channels
96:128 pass through unchanged. Tables are cast to the activation dtype before
rotation, and each eager multiplication/addition rounds in that dtype.

The explicit backward keeps products and their sum in FP32 and casts only the
final input gradient. Its coefficients have the same quantization as forward.
Position tables are constants; this reference does not import another RoPE
operator or a GPU backend.
"""

from __future__ import annotations

import torch
from torch import Tensor

_HEAD_DIM = 128
_ROTARY_DIM = 96
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_TABLE_DTYPE = torch.float32


def _validate_inputs(x: Tensor, cos: Tensor, sin: Tensor) -> None:
    """Check H3 shapes, dtypes, and constant tables on the input's device.

    CPU tensors are accepted so the reference also works without a GPU.
    Noncontiguous activations and tables follow the same public contract.
    """
    if x.ndim != 4 or x.shape[-1] != _HEAD_DIM:
        raise ValueError(f"x must have shape [B, S, H, {_HEAD_DIM}], got {tuple(x.shape)}.")
    if x.numel() == 0:
        raise ValueError(
            f"x must have nonempty batch, sequence, and head dimensions, got {tuple(x.shape)}."
        )
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}.")
    expected_table_shape = (x.shape[1], _ROTARY_DIM)
    for name, table in (("cos", cos), ("sin", sin)):
        if table.shape != expected_table_shape:
            raise ValueError(
                f"{name} must have shape {expected_table_shape}, got {tuple(table.shape)}."
            )
        if table.device != x.device:
            raise ValueError(f"{name} must be on x.device ({x.device}), got {table.device}.")
        if table.dtype != _TABLE_DTYPE:
            raise TypeError(f"{name} must have dtype {_TABLE_DTYPE}, got {table.dtype}.")
        if table.requires_grad:
            raise ValueError(
                f"{name} must not require gradients, got requires_grad={table.requires_grad}."
            )


class _H3MMRoPENativeFunction(torch.autograd.Function):
    """Preserve H3 forward rounding and use FP32 intermediate input gradients.

    Eager autograd would round each BF16 product gradient before adding its
    contributions. The explicit backward avoids that extra rounding while
    retaining the coefficients used by forward. Only x has a gradient.
    """

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        ctx.save_for_backward(cos, sin)
        # Broadcast each token's position coefficients across batch and heads.
        cos = cos.to(x.dtype)[None, :, None, :]
        sin = sin.to(x.dtype)[None, :, None, :]
        x_rotary = x[..., :_ROTARY_DIM]
        x_tail = x[..., _ROTARY_DIM:]

        # Rotate-half pairs channels 0:48 with 48:96, not adjacent channels.
        x_first, x_second = x_rotary.chunk(2, dim=-1)
        rotated = torch.cat((-x_second, x_first), dim=-1)
        y_rotary = x_rotary * cos + rotated * sin
        return torch.cat((y_rotary, x_tail), dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        cos, sin = ctx.saved_tensors
        # Match forward's coefficient quantization, then do gradient math in FP32.
        cos = cos.to(grad_y.dtype).float()[None, :, None, :]
        sin = sin.to(grad_y.dtype).float()[None, :, None, :]
        cos_first, cos_second = cos.chunk(2, dim=-1)
        sin_first, sin_second = sin.chunk(2, dim=-1)
        grad_first, grad_second = grad_y[..., :_ROTARY_DIM].float().chunk(2, dim=-1)

        # Transpose rotation: the cross terms use the opposite output's sine.
        grad_x_first = grad_first * cos_first + grad_second * sin_second
        grad_x_second = grad_second * cos_second - grad_first * sin_first
        grad_rotary = torch.cat((grad_x_first, grad_x_second), dim=-1).to(grad_y.dtype)
        grad_x = torch.cat((grad_rotary, grad_y[..., _ROTARY_DIM:]), dim=-1)
        return grad_x, None, None


class NativeH3MMRoPEOp:
    """Rotate the leading 96 channels of an H3 Q or K tensor using eager PyTorch.

    Args:
        x: BF16/FP32 activation of shape ``[B, S, H, 128]``.
        cos: Constant FP32 ``[S, 96]`` table on the same device as ``x``.
        sin: Constant FP32 ``[S, 96]`` table on the same device as ``x``.

    Tables are shared across the batch and heads. Output is contiguous, with
    x's shape and dtype; channels 96:128 are copied without arithmetic. Forward
    preserves H3's table/product rounding. The explicit backward uses FP32
    products and addition with forward-quantized coefficients, then rounds to
    x's dtype. First-order input gradients are supported; FP16 is not supported.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(x, cos, sin)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply H3's forward rounding with the explicit FP32 backward."""
        _validate_inputs(x, cos, sin)
        return _H3MMRoPENativeFunction.apply(x, cos, sin)

    def forward_fp32(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Independent accuracy reference with original FP32 tables and FP32 math.

        This deliberately omits the model's BF16 coefficient/product rounding.
        Autograd differentiates this expression for the unified operator checker;
        it does not reuse the candidate's handwritten backward. Model-path parity
        is tested separately against ``forward`` and the pinned H3 provider.
        """
        _validate_inputs(x, cos, sin)
        # No matmul/TF32 path is used; disable an enclosing autocast context too.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            first, second = x[..., :_ROTARY_DIM].chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            result = x[..., :_ROTARY_DIM] * cos[None, :, None, :]
            result = result + rotated * sin[None, :, None, :]
            return torch.cat((result, x[..., _ROTARY_DIM:]), dim=-1)
