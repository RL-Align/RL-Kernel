# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The diffusers MiniMax-H3 conditioning path, op for op, without diffusers.

Each function replays the tensor ops of the pinned reference
(``huggingface/diffusers@f53d552``, see ``h3_manifest.json``) in the same
order and dtypes, so it runs the same kernels the provider does on a given
device. This is the *provider* side of the RFC #420 comparisons; the FP32/FP64
goldens live in ``rl_engine.kernels.ops.pytorch.h3``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def provider_time_proj(timestep: torch.Tensor, num_channels: int = 256) -> torch.Tensor:
    """``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``."""

    half_dim = num_channels // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timestep.device
    )
    exponent = exponent / (half_dim - 0)
    emb = torch.exp(exponent)
    emb = timestep[:, None].float() * emb[None, :]
    emb = 1 * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    return emb


def provider_time_embedder(
    features: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """``TimestepEmbedding(256, 5376, out_dim=2688)``: linear_1 -> SiLU -> linear_2, FP32."""

    sample = F.linear(features.to(w1.dtype), w1, b1)
    sample = F.silu(sample)
    return F.linear(sample, w2, b2)
