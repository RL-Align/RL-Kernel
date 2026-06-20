# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


class NativeRMSNormOp:
    """
    Pure Pytorch native RMSNorm reference
    out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self.forward(x, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Canonical entry: accumulate in fp32, cast the result back to x.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._rms_norm(x, weight, eps=eps, output_dtype=x.dtype)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Ground-truth: accumulate in fp32 and force fp32 output."""
        return self._rms_norm(x, weight, eps=eps, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
            raise ValueError(
                f"weight must be 1-D of size x.shape[-1]={x.shape[-1]}, "
                f"got tuple(weight.shape)={tuple(weight.shape)}"
            )
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + eps)
        out = normed * weight.float()
        return out.to(output_dtype)
