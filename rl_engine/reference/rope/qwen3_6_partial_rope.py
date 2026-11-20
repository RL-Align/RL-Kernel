# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen3.6-35B-A3B text partial RoPE with explicit FP32 rotation arithmetic.

The model's full-attention layers rotate the first 64 of 256 channels. Position
and frequency generation belong to the caller: this operator only selects from
the supplied FP32 tables and applies the rotation to one Q or K tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

_HEAD_DIM = 256
_ROTARY_DIM = 64
_HALF_ROTARY_DIM = _ROTARY_DIM // 2
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_POSITION_DTYPES = (torch.int32, torch.int64)
_TABLE_DTYPE = torch.float32


def _validate_inputs(x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> None:
    """Validate tensor metadata; table-index bounds remain a caller precondition.

    The reference also runs on CPU. Unlike the Triton implementation, it has no
    launch-grid head limit. Neither path reads position values back to the host.
    """
    if x.ndim != 3 or x.shape[-1] != _HEAD_DIM:
        raise ValueError(f"x must have shape [T, H, {_HEAD_DIM}], got {tuple(x.shape)}.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}.")
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


def _rotate(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    *,
    transpose: bool = False,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute a rotation (or transpose) in FP32, then cast the rotated outputs.

    Cast before splitting so autograd's FP32 reference accumulates both paths in
    FP32 before converting an input gradient to BF16. Eager PyTorch executes the
    multiplies and addition/subtraction separately, matching disabled FMA fusion
    in the Triton launch. The tail bypasses arithmetic and, in the production
    path, conversion. Empty inputs must not index an unused, empty table.
    """
    if x.numel() == 0:
        return x.to(output_dtype).clone(memory_format=torch.contiguous_format)
    # Only the rotated prefix needs promotion; copying the tail should not
    # require a full-size FP32 temporary in the eager baseline.
    xf = x[..., :_ROTARY_DIM].float()
    c = cos.index_select(0, positions).unsqueeze(1)
    s = sin.index_select(0, positions).unsqueeze(1)
    first, second = xf[..., :_HALF_ROTARY_DIM], xf[..., _HALF_ROTARY_DIM:_ROTARY_DIM]
    if transpose:
        rotated_first = first * c + second * s
        rotated_second = second * c - first * s
    else:
        rotated_first = first * c - second * s
        rotated_second = second * c + first * s
    return torch.cat(
        (
            rotated_first.to(output_dtype),
            rotated_second.to(output_dtype),
            x[..., _ROTARY_DIM:].to(output_dtype),
        ),
        dim=-1,
    )


class _Qwen3_6PartialRoPENativeFunction(torch.autograd.Function):
    """Explicit first-order VJP with one final cast, matching the Triton contract."""

    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Save only the constants needed by the linear transpose rotation."""
        ctx.save_for_backward(cos, sin, positions)
        return _rotate(x, cos, sin, positions, output_dtype=x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor, None, None, None]:
        """Apply R^T in FP32 and return dx; table/position gradients are absent."""
        cos, sin, positions = ctx.saved_tensors
        grad_x = _rotate(grad_y, cos, sin, positions, transpose=True, output_dtype=grad_y.dtype)
        return grad_x, None, None, None


class NativeQwen3_6PartialRoPEOp:
    """PyTorch reference for Qwen3.6-35B-A3B's text full-attention RoPE.

    ``forward`` and ``__call__`` return the input dtype and expose an explicit
    first-order backward. ``forward_fp32`` omits the final cast and lets PyTorch
    differentiate the FP32 expression for the numerical accuracy reference.
    This does not generate frequencies or implement a complete attention layer.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Delegate to :meth:`forward` with the same tensor and gradient contract."""
        return self.forward(x, cos, sin, positions)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Rotate the leading channels and copy the remaining channels unchanged.

        Args:
            x: BF16/FP32 features ``[T, H, 256]``; Q and K are separate calls.
            cos: Constant FP32 table ``[P, 32]``; one coefficient per channel pair.
            sin: Constant FP32 table with the same shape and device as ``cos``.
            positions: Int32/int64 table row IDs ``[T]``, each in ``[0, P)``.

        Returns:
            New contiguous tensor with ``x``'s shape, dtype and device. All
            inputs must share a device; strided views and empty inputs work.

        Raises:
            TypeError: An input or coefficient/index dtype is unsupported.
            ValueError: Shapes/devices disagree or a table requires gradients.

        Index values are a caller precondition; no host-side min/max check is
        performed. PyTorch's index_select handles invalid indices in this path.
        """
        _validate_inputs(x, cos, sin, positions)
        return _Qwen3_6PartialRoPENativeFunction.apply(x, cos, sin, positions)

    def forward_fp32(self, x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
        """Evaluate the same rotation in FP32 without final output quantization.

        Arguments and metadata errors match :meth:`forward`. The result is a
        new FP32 tensor. Autograd differentiates this mathematical expression;
        a BF16 leaf still receives a BF16 gradient, as required by PyTorch.
        Use an FP32 leaf when inspecting unrounded input-gradient error.
        """
        _validate_inputs(x, cos, sin, positions)
        return _rotate(x, cos, sin, positions)
