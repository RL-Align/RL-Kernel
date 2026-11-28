# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure PyTorch MiniMax-H3 rotate-half for channels 0:96 of each 128-channel head."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

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


def _resolve_storage_dtype(x_dtype: torch.dtype, storage_dtype: torch.dtype | None) -> torch.dtype:
    """Use the activation dtype for intermediate rounding unless an override is supplied."""
    if storage_dtype is None:
        return x_dtype
    if storage_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"storage_dtype must be in {_SUPPORTED_DTYPES}, got {storage_dtype}.")
    return storage_dtype


class _H3MMRoPENativeFunction(torch.autograd.Function):
    """Compute in FP32 and round each forward product and the output to the activation dtype."""

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        ctx.save_for_backward(cos, sin)
        # Round each token's coefficients to the model dtype before using them in FP32.
        cos = cos.to(x.dtype).to(torch.float32)[None, :, None, :]
        sin = sin.to(x.dtype).to(torch.float32)[None, :, None, :]
        rotary = x[..., :_ROTARY_DIM].to(torch.float32)
        tail = x[..., _ROTARY_DIM:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        # H3 rounds the two products independently before their FP32 sum.
        direct = (rotary * cos).to(x.dtype).to(torch.float32)
        cross = (rotated * sin).to(x.dtype).to(torch.float32)
        y_rotary = (direct + cross).to(x.dtype)
        return torch.cat((y_rotary, tail), dim=-1).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor):
        cos, sin = ctx.saved_tensors
        # Use the same quantized coefficients as forward, then do the gradient math in FP32.
        cos = cos.to(grad_y.dtype).to(torch.float32)[None, :, None, :]
        sin = sin.to(grad_y.dtype).to(torch.float32)[None, :, None, :]
        grad = grad_y[..., :_ROTARY_DIM].to(torch.float32)
        direct_first, direct_second = (grad * cos).chunk(2, dim=-1)
        cross_first, cross_second = (grad * sin).chunk(2, dim=-1)
        # Each sine term contributes to the other input half, with the sign from forward.
        grad_rotary = torch.cat((direct_first + cross_second, direct_second - cross_first), dim=-1)
        grad_x = torch.cat((grad_rotary.to(grad_y.dtype), grad_y[..., _ROTARY_DIM:]), dim=-1)
        return grad_x, None, None


class _H3MMRoPEReferenceFunction(torch.autograd.Function):
    """Apply H3's forward product rounding and compute its input gradients in FP32."""

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor, storage_dtype: torch.dtype) -> Tensor:
        ctx.input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            cos = cos.to(storage_dtype).to(torch.float32)[None, :, None, :]
            sin = sin.to(storage_dtype).to(torch.float32)[None, :, None, :]
            ctx.save_for_backward(cos, sin)

            values = x.to(torch.float32)
            rotary, tail = values[..., :_ROTARY_DIM], values[..., _ROTARY_DIM:]
            first, second = rotary.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            # H3 stores each product in the activation dtype. The reference adds them in FP32.
            direct = (rotary * cos).to(storage_dtype).to(torch.float32)
            cross = (rotated * sin).to(storage_dtype).to(torch.float32)
            return torch.cat((direct + cross, tail), dim=-1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor):
        cos, sin = ctx.saved_tensors
        grad = grad_y[..., :_ROTARY_DIM]
        # The transpose sends each sine contribution back to the opposite input half.
        first, second = (grad * sin).chunk(2, dim=-1)
        grad_rotary = grad * cos + torch.cat((second, -first), dim=-1)
        grad_x = torch.cat((grad_rotary, grad_y[..., _ROTARY_DIM:]), dim=-1)
        # A promoted FP32 input exposes the gradient before the model's final BF16 cast.
        return grad_x.to(ctx.input_dtype), None, None, None


class NativeH3MMRoPEOp:
    """Rotate channels ``i`` and ``i + 48`` in a BF16/FP32 ``[B, S, H, 128]`` Q or K.

    The FP32 ``[S, 96]`` tables are shared across heads and batches. The last 32 channels
    pass through to the contiguous output. Only ``x`` has a first-order gradient.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(x, cos, sin)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply the H3 rotation with the model's forward rounding."""
        _validate_inputs(x, cos, sin)
        return _H3MMRoPENativeFunction.apply(x, cos, sin)

    def forward_fp32(
        self, x: Tensor, cos: Tensor, sin: Tensor, *, storage_dtype: torch.dtype | None = None
    ) -> Tensor:
        """Check H3's storage rounding with an FP32 output.

        For an FP32 input promoted from BF16, pass ``storage_dtype=torch.bfloat16``
        to reproduce the model's forward casts while keeping the input gradient in FP32.
        """
        _validate_inputs(x, cos, sin)
        storage_dtype = _resolve_storage_dtype(x.dtype, storage_dtype)
        return _H3MMRoPEReferenceFunction.apply(x, cos, sin, storage_dtype)

    def forward_unrounded_fp32(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Measure the rotation with the original FP32 coefficients and FP32 products."""
        _validate_inputs(x, cos, sin)
        return _H3MMRoPEReferenceFunction.apply(x, cos, sin, torch.float32)
