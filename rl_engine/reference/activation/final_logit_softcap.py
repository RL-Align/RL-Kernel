# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch reference for final-logit softcapping with FP32 arithmetic."""

import math

import torch
import torch.nn as nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

DEFAULT_SOFTCAP = 30.0


class NativeFinalLogitSoftcapOp(nn.Module):
    """Compute ``c * tanh(x / c)``; autograd supplies input-dtype gradients.

    ``forward`` computes in FP32 and casts back to the input dtype, matching
    the dtype-behavior path used by other WS1 elementwise references.
    ``forward_fp32`` is the ground-truth path: FP32 math and an FP32 result.
    """

    op_class = "elementwise"

    def __init__(self, softcap: float = DEFAULT_SOFTCAP) -> None:
        super().__init__()
        softcap = float(softcap)
        if not math.isfinite(softcap) or softcap <= 0.0:
            raise ValueError(f"softcap must be finite and positive, got {softcap}.")
        self.softcap = softcap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._softcap(x, output_dtype=x.dtype)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth: compute in fp32 and force fp32 output."""
        return self._softcap(x, output_dtype=torch.float32)

    def _softcap(self, x: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
        if x.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"x must have dtype fp16, bf16, or fp32, got {x.dtype}.")
        x_f = x.to(torch.float32)
        y = self.softcap * torch.tanh(x_f / self.softcap)
        return y.to(output_dtype)
