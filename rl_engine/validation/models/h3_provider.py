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


def provider_gate_residual(
    residual: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor, sublayer_output: torch.Tensor
) -> torch.Tensor:
    """``residual + gate.index_select(0, adaln_indices) * sublayer_output`` (block)."""

    return residual + gate.index_select(0, indices) * sublayer_output


def provider_final_adaln_out(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    timestep_indices: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``MiniMaxH3AdaLayerNormOut.forward``: shift first, indexed by timestep."""

    shift, scale = F.linear(F.silu(temb).to(weight.dtype), weight, bias).chunk(2, dim=-1)
    hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, eps)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
        0, timestep_indices
    )


# --- One transformer block (RFC #420 ``ws1_one_h3_block``) ------------------------------------


def provider_rope(
    position_ids: torch.Tensor, rope_freq_dim: int = 16, rope_theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``MiniMaxH3RotaryPosEmbed``: FP32 ``(S, 96)`` cos/sin over the ``(t, h, w)`` axes."""

    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=position_ids.device)
            / (2 * rope_freq_dim)
        )
    )
    position_ids = position_ids.to(torch.float32)
    freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def provider_apply_rotary(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """``_apply_rotary_emb``: rotate-half on the leading 96 of 128 channels of ``(B, S, H, D)``.

    The FP32 tables are cast to the activation dtype (BF16) before the rotation.
    """

    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


def provider_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """A bias-free ``nn.Linear`` of the block (Q/K/V, O and both FFN projections)."""

    return F.linear(x, weight)


def provider_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``norm_q``/``norm_k``: ``nn.RMSNorm(128)`` on ``(B, S, H, D)``."""

    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def provider_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """The native diffusers backend: SDPA on ``(B, H, S, D)`` views, non-causal, no mask.

    Returns ``(B, S, H*D)`` after ``flatten(2, 3).type_as(query)``.
    """

    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.permute(0, 2, 1, 3).flatten(2, 3).type_as(query)


def provider_swiglu(projected: torch.Tensor) -> torch.Tensor:
    """diffusers ``SwiGLU`` after its projection: ``hidden * silu(gate)`` on the two halves."""

    hidden_states, gate = projected.chunk(2, dim=-1)
    return hidden_states * F.silu(gate)
