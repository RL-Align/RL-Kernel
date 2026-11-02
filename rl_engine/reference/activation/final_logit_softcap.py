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

    Both entry points compute in FP32 and return FP32. That departs from the
    other WS1 elementwise references, where ``forward`` casts back to the input
    dtype, and it is deliberate: these are logits on their way into a logsumexp,
    and the Triton wrapper already commits to an FP32 result. If ``forward``
    narrowed, ``registry.get_op`` would hand out BF16 logits on CPU or whenever
    Triton failed to import, with a max error around 0.1 against the FP32 path
    versus a few 1e-6 on CUDA, and callers could not tell which they had.

    ``forward_fp32`` is kept as the name ``OP_SPECS`` resolves for the gold path;
    it is the same computation as ``forward``.
    """

    op_class = "elementwise"

    def __init__(self, softcap: float = DEFAULT_SOFTCAP) -> None:
        super().__init__()
        softcap = float(softcap)
        if not math.isfinite(softcap) or softcap <= 0.0:
            raise ValueError(f"softcap must be finite and positive, got {softcap}.")
        self.softcap = softcap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._softcap(x, output_dtype=torch.float32)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth: compute in fp32 and force fp32 output."""
        return self._softcap(x, output_dtype=torch.float32)

    def _softcap(self, x: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
        if x.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"x must have dtype fp16, bf16, or fp32, got {x.dtype}.")
        x_f = x.to(torch.float32)
        y = self.softcap * torch.tanh(x_f / self.softcap)
        return y.to(output_dtype)
