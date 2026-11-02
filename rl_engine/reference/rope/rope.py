# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
from torch import Tensor


class NativeRoPEOp:
    """Pure PyTorch reference RoPE — GPT-NeoX style (HF rotate-half).

    Qwen3-8B defaults: theta=1e6, head_dim=128, full-dimension rotation (half=64).
    Dimension pairing: (i, i+half) — NOT adjacent (i, i+1).
    cos/sin are computed internally in fp32 from positions and theta — no external
    cos/sin cache is accepted or returned.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        """Apply RoPE in input dtype. Cos/sin always computed in fp32."""
        cos, sin = self._compute_cos_sin(x, positions, theta=theta)
        xf = x
        x1, x2 = xf[..., : xf.shape[-1] // 2], xf[..., xf.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        out = xf * cos + rotated * sin
        return out.to(dtype=x.dtype)

    def forward_fp32(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        """fp32 gold standard: internal computation and output are fp32."""
        cos, sin = self._compute_cos_sin(x, positions, theta=theta)
        xf = x.float()
        x1, x2 = xf[..., : xf.shape[-1] // 2], xf[..., xf.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        out = xf * cos + rotated * sin
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_cos_sin(x: Tensor, positions: Tensor, *, theta: float) -> tuple[Tensor, Tensor]:
        """Compute cos/sin tables in fp32 from positions and theta.

        Args:
            x: [..., D] — only x.shape[-1] (head_dim) is used.
            positions: [S] or [B, S] int64 — absolute token positions.
            theta: RoPE base frequency (Qwen3 = 1e6).

        Returns:
            cos, sin: broadcastable to x shape, fp32.
        """
        D = x.shape[-1]
        half = D // 2

        # inv_freq[i] = 1 / (theta^(2i/D)) = 1 / (theta^(i/half))
        # shape: [half]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half)
        )

        # positions: [S] -> [S, 1] or [B, S] -> [B, S, 1]
        pos_float = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1)

        # freqs: [S, half] or [B, S, half]
        freqs = pos_float * inv_freq

        # Duplicate to full dim: [S, D] or [B, S, D]
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # Reshape for broadcasting with x: [B, H, S, D]
        if positions.dim() == 1:
            # positions [S] -> cos/sin [1, 1, S, D]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            # positions [B, S] -> cos/sin [B, 1, S, D]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return cos, sin
