# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


class NativeQkRmsNormOp:
    """Pure PyTorch reference for Qwen-Image per-head QK RMSNorm (issue #386).

    Parameter-free RMSNorm applied to Q and K, per attention head, over
    head_dim (Qwen-Image uses 128): every row of the last dimension is
    normalized by its own RMS, with no weight and no bias.

    out = x * rsqrt(mean(x^2, dim=-1) + eps)

    Accumulates in fp32 and casts the result back to x.dtype, matching the
    dtype behavior of NativeRMSNormOp (the Axis-B accuracy candidate). With
    this formula the per-row scale rstd = rsqrt(mean(x_f32^2) + eps) is a
    function of the row alone, so the op is batch-invariant by construction.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        return self.forward(x, eps=eps)

    def forward(self, x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """Canonical entry: fp32 accumulation, output cast back to x.dtype."""
        return self._qk_rms_norm(x, eps=eps, output_dtype=x.dtype)

    def forward_fp32(self, x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """Ground-truth: accumulate in fp32 and force fp32 output."""
        return self._qk_rms_norm(x, eps=eps, output_dtype=torch.float32)

    @staticmethod
    def _qk_rms_norm(
        x: torch.Tensor,
        *,
        eps: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if x.dim() < 1 or x.shape[-1] == 0:
            raise ValueError(
                "x must be a non-empty tensor with a head_dim last dimension, "
                f"got shape {tuple(x.shape)}"
            )
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + eps)
        return normed.to(output_dtype)
