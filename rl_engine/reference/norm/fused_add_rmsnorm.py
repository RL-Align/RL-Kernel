# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable PyTorch counterpart to the Triton FP32 fused add RMSNorm.

Both outputs stay in FP32, and normalization uses the unrounded FP32 residual
sum. Backward arithmetic uses FP32; each input gradient returns in that input's
dtype. Model integration must preserve these declared cast points.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from rl_engine.reference.reduction import sum_tile

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have shape [..., D] with D > 0.")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x.")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must have shape [D].")
    if residual.device != x.device or weight.device != x.device:
        raise ValueError("x, residual, and weight must be on the same device.")
    if any(t.dtype not in _SUPPORTED_DTYPES for t in (x, residual, weight)):
        raise TypeError(f"x, residual, and weight must have dtype in {_SUPPORTED_DTYPES}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")


class NativeFusedAddRMSNormOp(nn.Module):
    """FP32-output fused add RMSNorm; PyTorch supplies both output branches' gradients.

    Inputs may independently use FP16, BF16, or FP32 on the same device.
    Normalization uses the unrounded FP32 residual sum. Both outputs are FP32;
    each input gradient returns in that input's dtype.
    """

    op_class = "norm"

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (y, updated_residual), with gradients provided by PyTorch autograd.

        x and residual share shape [..., D] with D > 0; weight has shape [D].
        All inputs share a device. Both outputs have x's shape and dtype FP32.
        """
        _validate_inputs(x, residual, weight, eps)
        x_f = x.to(torch.float32)
        residual_f = residual.to(torch.float32)
        weight_f = weight.to(torch.float32)

        # 1. Add corresponding elements; updated_residual has shape [..., D].
        updated_residual = x_f + residual_f

        # 2. Square each element, then sum across the last dimension (one row).
        squared = updated_residual * updated_residual
        sum_squares = squared.sum(dim=-1, keepdim=True)

        # 3. Divide by the actual row width; mean_square has shape [..., 1].
        n_cols = x.shape[-1]
        mean_square = sum_squares / n_cols

        # 4. Compute 1 / sqrt(mean_square + eps), one value per row.
        inverse_rms = torch.rsqrt(mean_square + eps)

        # 5. Broadcast that value over the row, then apply each feature's weight.
        normalized = updated_residual * inverse_rms
        y = normalized * weight_f

        return y, updated_residual


@dataclass(frozen=True)
class OrderedForward:
    """Fixed-path FP32 outputs and row statistics; not an autograd graph.

    y and updated_residual retain the input shape. sum_squares has shape
    [..., 1], while inverse_rms is flattened to [M] like the production save.
    """

    y: torch.Tensor
    updated_residual: torch.Tensor
    sum_squares: torch.Tensor
    inverse_rms: torch.Tensor


@dataclass(frozen=True)
class OrderedBackward:
    """Explicit VJP and intermediate sums for arithmetic-path checks.

    Final gradients use their input dtypes. correction_sum is FP32 [..., 1];
    grad_weight_per_row is the unrounded FP32 [M, D] matrix before the row fold.
    """

    grad_x: torch.Tensor
    grad_residual: torch.Tensor
    grad_weight: torch.Tensor
    correction_sum: torch.Tensor
    grad_weight_per_row: torch.Tensor


@torch.no_grad()
def ordered_row_sum(values: torch.Tensor) -> torch.Tensor:
    """Pad D to its next power of two and apply the shared FP32 feature tree.

    Leading dimensions are independent rows. The result has shape [..., 1]
    for broadcasting. This mirrors sum_tile in the Triton row kernels; it
    deliberately does not use torch.sum or change the tree with batch size.
    """
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("values must have a nonempty feature dimension")
    n_cols = values.shape[-1]
    block_size = 1 << (n_cols - 1).bit_length()
    padded = torch.nn.functional.pad(values.float(), (0, block_size - n_cols))
    return sum_tile(padded).unsqueeze(-1)


@torch.no_grad()
def ordered_forward(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> OrderedForward:
    """Mirror the declared forward order, separately from the Native accuracy oracle.

    A device FP32 divisor requests tensor division rather than Python-scalar
    reciprocal multiplication. rsqrt remains PyTorch's implementation: matching
    the sum tree alone does not promise bitwise equality of elementary math
    across implementations. Inspect sum_squares separately from inverse_rms.
    """
    _validate_inputs(x, residual, weight, eps)
    updated_residual = x.float() + residual.float()
    sum_squares = ordered_row_sum(updated_residual * updated_residual)
    divisor = torch.tensor(x.shape[-1], device=x.device, dtype=torch.float32)
    inverse_rms = torch.rsqrt(sum_squares / divisor + eps)
    normalized = updated_residual * inverse_rms
    return OrderedForward(
        y=normalized * weight.float(),
        updated_residual=updated_residual,
        sum_squares=sum_squares,
        inverse_rms=inverse_rms.reshape(-1),
    )


@torch.no_grad()
def ordered_backward(
    updated_residual: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor | None,
    grad_updated_residual_output: torch.Tensor | None,
    *,
    x_dtype: torch.dtype,
    residual_dtype: torch.dtype,
) -> OrderedBackward:
    """Hand-written FP32 VJP with explicit feature and parameter reduction order.

    Pass either ordered_forward's saved values or the production saved values.
    The latter isolates backward arithmetic from any forward rsqrt difference.
    No PyTorch autograd reduction is used. The feature dot product uses the
    shared tree; weight contributions fold from +0 in ascending physical-row
    order, casting only the final gradient. Already-reduced microbatch gradients
    cannot be merged to reproduce this fold in general.
    """
    n_cols = updated_residual.shape[-1]
    inverse = inverse_rms.reshape(*updated_residual.shape[:-1], 1)
    upstream_y = torch.zeros_like(updated_residual) if grad_y is None else grad_y.float()
    upstream_residual = (
        torch.zeros_like(updated_residual)
        if grad_updated_residual_output is None
        else grad_updated_residual_output.float()
    )
    normalized = updated_residual * inverse
    grad_normalized = upstream_y * weight.float()
    contributions = (upstream_y * normalized).reshape(-1, n_cols)
    correction_sum = ordered_row_sum(grad_normalized * normalized)
    divisor = torch.tensor(n_cols, device=updated_residual.device, dtype=torch.float32)
    correction = correction_sum / divisor
    grad_updated = inverse * (grad_normalized - normalized * correction) + upstream_residual
    grad_weight = torch.zeros(n_cols, device=weight.device, dtype=torch.float32)
    for row in contributions:
        grad_weight = grad_weight + row
    return OrderedBackward(
        grad_x=grad_updated.to(x_dtype),
        grad_residual=grad_updated.to(residual_dtype),
        grad_weight=grad_weight.to(weight.dtype),
        correction_sum=correction_sum,
        grad_weight_per_row=contributions,
    )
