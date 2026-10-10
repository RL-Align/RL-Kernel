# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The diffusers MiniMax-H3 conditioning path, op for op, without diffusers.

Each function replays the tensor ops of the pinned reference
(``huggingface/diffusers@f53d552``, see ``h3_manifest.json``) in the same
order and dtypes, so it runs the same kernels the provider does on a given
device. This is the *provider* side of the RFC #420 comparisons; the FP32/FP64
goldens live in ``rl_engine.reference.minimax_h3``.
"""

from __future__ import annotations

import math

import torch


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
