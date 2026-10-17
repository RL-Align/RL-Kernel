# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable PyTorch counterpart to the Triton FP32 fused add RMSNorm.

Both outputs stay in FP32, and normalization uses the unrounded FP32 residual
sum. Backward arithmetic uses FP32; each input gradient returns in that input's
dtype. Model integration must preserve these declared cast points.
"""

import math

import torch
from torch import nn

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
