# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure PyTorch reference for MiniMax-H3 partial MM-RoPE.

The caller supplies FP32 ``[S, 96]`` cosine/sine tables for the packed tokens'
three-axis positions. Channels ``i`` and ``i + 48`` rotate together; channels
96:128 pass through unchanged. Tables are cast to the activation dtype before
rotation, and each eager multiplication/addition rounds in that dtype.

PyTorch autograd computes the input gradient. Position tables are constants;
this reference does not import another RoPE operator or a GPU backend.
"""

from __future__ import annotations

import torch
from torch import Tensor

_HEAD_DIM = 128
_ROTARY_DIM = 96
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(x: Tensor, cos: Tensor, sin: Tensor) -> None:
    """Check H3 shapes, dtypes, and constant tables on the input's device.

    CPU tensors are accepted so the reference also works without a GPU.
    Noncontiguous activations and tables follow the same public contract.
    """
    if x.ndim != 4 or x.shape[-1] != _HEAD_DIM:
        raise ValueError("x must have shape [B, S, H, 128].")
    if x.numel() == 0:
        raise ValueError("x must have nonempty batch, sequence, and head dimensions.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must have dtype float16, bfloat16, or float32.")
    for name, table in (("cos", cos), ("sin", sin)):
        if table.shape != (x.shape[1], _ROTARY_DIM):
            raise ValueError(f"{name} must have shape [S, 96], matching x's sequence length.")
        if table.device != x.device:
            raise ValueError(f"{name} must be on the same device as x.")
        if table.dtype != torch.float32:
            raise TypeError(f"{name} must be an FP32 H3 position table.")
        if table.requires_grad:
            raise ValueError(f"{name} must not require gradients; position tables are constants.")


class NativeH3MMRoPEOp:
    """Rotate the leading 96 channels of an H3 Q or K tensor using eager PyTorch.

    Args:
        x: FP16/BF16/FP32 activation of shape ``[B, S, H, 128]``.
        cos: Constant FP32 ``[S, 96]`` table on the same device as ``x``.
        sin: Constant FP32 ``[S, 96]`` table on the same device as ``x``.

    Tables are shared across the batch and heads. Output is contiguous, with
    x's shape and dtype; channels 96:128 are copied without arithmetic. Normal
    PyTorch autograd provides gradients for x, including the unchanged tail.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(x, cos, sin)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply H3's table/product rounding and retain the autograd graph."""
        _validate_inputs(x, cos, sin)

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
