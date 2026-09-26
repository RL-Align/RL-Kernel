# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch reference for final-logit softcapping with FP32 arithmetic and output."""

import torch
from torch import Tensor, nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class NativeFinalLogitSoftcapOp(nn.Module):
    """Compute ``30 * tanh(x / 30)``; autograd supplies input-dtype gradients."""

    op_class = "elementwise"

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"x must have dtype {_SUPPORTED_DTYPES}, got {x.dtype}.")
        x_f = x.to(torch.float32)
        return 30.0 * torch.tanh(x_f / 30.0)
