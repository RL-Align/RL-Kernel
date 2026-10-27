# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Explicit ROCm LoRA composition; GPU qualification is required before registration.

Reuses the FP32-output det_gemm path for *every* GEMM. The generic BF16 tree
path rounds at intermediate K-tree nodes and is not this numerical profile.
Adapted from the LoRA composition in PR #416; no P5 development-branch imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rl_engine.contracts.operators.lora import (
    ROCM_LORA_PROFILE,
    validate_lora_backward,
    validate_lora_inputs,
)

if TYPE_CHECKING:
    from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp


class RocmLoRADelta:
    """Shared A/B over already-dispatched rows, including an empty local batch."""

    numeric_profile = ROCM_LORA_PROFILE

    def __init__(self) -> None:
        # Cache only the stateless GEMM wrapper, never weights or their transposes.
        self._gemm: TritonDetGemmOp | None = None

    def _check_device(self, device: torch.device) -> None:
        if torch.version.hip is None or device.type != "cuda":
            raise RuntimeError("LoRA ROCm backend requires ROCm GPU tensors; no fallback")
        arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
        if arch not in ("gfx942", "gfx950"):
            raise RuntimeError(f"LoRA ROCm candidate is limited to gfx942/gfx950, got {arch}")

    def _mm(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if lhs.shape[0] == 0 or rhs.shape[1] == 0 or lhs.shape[1] == 0:
            return torch.zeros((lhs.shape[0], rhs.shape[1]), dtype=torch.float32, device=lhs.device)
        if self._gemm is None:
            # Lazy import permits contract/CPU-reference tests without Triton.
            from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

            self._gemm = TritonDetGemmOp()
        with torch.cuda.device(lhs.device):
            return self._gemm.forward_fp32(lhs, rhs)

    @torch.no_grad()
    def _project(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._mm(x, a.t()).to(torch.bfloat16)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = validate_lora_inputs(x, a, b, alpha)
        self._check_device(x.device)
        u = self._project(x, a)
        y = self._mm(u, b.t()) * alpha
        return y, u

    @torch.no_grad()
    def backward(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = validate_lora_inputs(x, a, b, alpha)
        validate_lora_backward(dy, x, a, b, u)
        self._check_device(x.device)
        dv = (dy.float() * alpha).to(torch.bfloat16)
        du = self._mm(dv, b).to(torch.bfloat16)
        db = self._mm(dv.t(), u)
        da = self._mm(du.t(), x)
        dx = self._mm(du, a)
        return dx, da, db
