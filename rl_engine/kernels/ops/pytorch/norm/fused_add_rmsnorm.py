# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable PyTorch counterpart to the Triton FP32 arithmetic prototype.

Both outputs stay in FP32, and normalization uses the unrounded FP32 residual
sum. The public dtype contract and any BF16 rounding point are not finalized.
"""

import torch


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (y, updated_residual), with gradients provided by PyTorch autograd.

    Inputs: x and residual have the same shape [..., D], weight has shape [D],
    and all three tensors are on the same device. D must be positive.
    Outputs: both tensors have the same shape as x and dtype torch.float32.
    """
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


def fused_add_rmsnorm_backward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor,
    grad_updated_residual_output: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manually compute (grad_x, grad_residual, grad_weight) step by step.

    This is a standalone learning/reference helper, not a registered autograd
    backward. It receives the forward inputs explicitly to recompute the
    intermediates; a future autograd wrapper can provide saved values via ctx.

    Both upstream gradients have the same shape as x. Pass zeros for an unused
    output branch. Arithmetic uses FP32, and each returned gradient is cast to
    its input's dtype to match the casts in fused_add_rmsnorm.
    """
    # Recompute the forward intermediates needed by the derivative formulas.
    x_f = x.to(torch.float32)
    residual_f = residual.to(torch.float32)
    weight_f = weight.to(torch.float32)
    updated_residual = x_f + residual_f
    squared = updated_residual * updated_residual
    sum_squares = squared.sum(dim=-1, keepdim=True)
    n_cols = x.shape[-1]
    mean_square = sum_squares / n_cols
    inverse_rms = torch.rsqrt(mean_square + eps)
    normalized = updated_residual * inverse_rms

    grad_y_f = grad_y.to(torch.float32)
    grad_updated_residual_output_f = grad_updated_residual_output.to(torch.float32)

    # 1. y = normalized * weight. Weight is shared across all leading rows.
    grad_normalized = grad_y_f * weight_f
    grad_weight_per_row = grad_y_f * normalized
    grad_weight_f = grad_weight_per_row.reshape(-1, n_cols).sum(dim=0)

    # 2. normalized = updated_residual * inverse_rms.
    # Keep the direct contribution while following the inverse_rms branch.
    grad_updated_residual_direct = grad_normalized * inverse_rms
    grad_inverse_rms = (grad_normalized * updated_residual).sum(dim=-1, keepdim=True)

    # 3. inverse_rms = (mean_square + eps) ** (-0.5).
    grad_mean_square = grad_inverse_rms * (-0.5 * inverse_rms**3)

    # 4. mean_square = sum_squares / n_cols.
    grad_sum_squares = grad_mean_square / n_cols

    # 5. sum_squares = squared.sum(dim=-1, keepdim=True).
    grad_squared = grad_sum_squares.expand_as(squared)

    # 6. squared = updated_residual * updated_residual.
    grad_updated_residual_via_rms = grad_squared * (2.0 * updated_residual)

    # 7. Add both paths from y, then the separate residual-output branch.
    grad_updated_residual_from_y = grad_updated_residual_direct + grad_updated_residual_via_rms
    grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output_f

    # 8. updated_residual = x_f + residual_f: both local derivatives are 1.
    grad_x = grad_updated_residual_total.to(x.dtype)
    grad_residual = grad_updated_residual_total.to(residual.dtype)
    grad_weight = grad_weight_f.to(weight.dtype)

    return grad_x, grad_residual, grad_weight
