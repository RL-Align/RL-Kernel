# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
from torch import Tensor


class NativeMatmulOp:
    """Pure PyTorch reference GEMM.

    It intentionally uses one `torch.matmul` call in fp32 for
    the gold path and does not implement split-K or manual blocked accumulation.
    """

    op_class = "reduction"

    def __init__(self) -> None:
        pass

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute `a @ b` and return the input dtype."""
        return self.forward_fp32(a, b).to(dtype=a.dtype)

    def forward_fp32(self, a: Tensor, b: Tensor) -> Tensor:
        """fp32 gold standard: cast inputs to fp32, then call `torch.matmul` once."""
        return torch.matmul(a.float(), b.float())
