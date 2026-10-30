# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""gtest operator adapters for P1-2 ``fp32_gemm_rms``.

The gold path is the pinned FP32 oracle; both candidates must match it with
``atol = rtol = 0`` (op class ``mhc_controller`` in the tolerance contract),
i.e. the gtest comparison is a raw byte check expressed as zero tolerance.
Every adapter routes through the shared autograd entry so ``--check-grad``
exercises the operator's own backward, not a torch-recorded graph.
"""

from __future__ import annotations

import torch

from rl_engine.mhc.fp32_gemm_rms import fp32_gemm_rms


class _GemmRMSOpBase:
    backend: str

    def forward(
        self, x_flat: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fp32_gemm_rms(x_flat, weight, eps, backend=self.backend)

    # gtest gold paths are looked up by method name.
    forward_fp32 = forward

    def __call__(
        self, x_flat: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x_flat, weight, eps)


class NativeGemmRMSOp(_GemmRMSOpBase):
    """Oracle-backed gold path (runs on CPU and CUDA)."""

    backend = "reference"


class CudaGemmRMSOp(_GemmRMSOpBase):
    backend = "cuda"


class TritonGemmRMSOp(_GemmRMSOpBase):
    backend = "triton"


__all__ = ["CudaGemmRMSOp", "NativeGemmRMSOp", "TritonGemmRMSOp"]
