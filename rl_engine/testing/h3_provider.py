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
    """Replay provider FP32 cosine/sine features for ``(T,)`` timesteps.

    Matches ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``
    with the default channel count, returning shape ``(T, 256)``.
    """

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
    """Replay linear_1, SiLU, and linear_2 using the checkpoint weight dtype.

    H3's FP32 weights map features of shape ``(T, 256)`` to ``(T, 2688)``.
    """

    sample = F.linear(features.to(w1.dtype), w1, b1)
    sample = F.silu(sample)
    return F.linear(sample, w2, b2)


def provider_adaln_modulation(
    temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, hidden_size: int = 5376
) -> tuple[torch.Tensor, ...]:
    """Replay provider SiLU, the weight-dtype cast, and the AdaLN projection.

    Return six ``(3T, hidden_size)`` views in timestep-major modality order
    for checkpoint weights shaped ``(18 * hidden_size, 2688)``.
    """

    temb = F.linear(F.silu(temb).to(weight.dtype), weight, bias)
    temb = temb.view(-1, 6 * hidden_size)
    return temb.chunk(6, dim=-1)


def provider_adaln_row_gather(
    modulation: tuple[torch.Tensor, ...], timestep_indices: torch.Tensor, token_tags: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """``adaln_indices = timestep_indices * 3 + token_tags`` then six ``index_select``."""

    adaln_indices = timestep_indices * 3 + token_tags
    return tuple(tensor.index_select(0, adaln_indices) for tensor in modulation)


def provider_norm_modulate(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``norm(x) * (1.0 + scale[i]) + shift[i]`` with ``index_select``; block and norm_out."""

    norm_hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, eps)
    return norm_hidden_states * (1.0 + scale.index_select(0, indices)) + shift.index_select(
        0, indices
    )
