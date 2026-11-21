# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 RoPE cache: FP32 ``[positions, rope_dim]`` = ``cos | sin`` (GPT-J pairs).

Layers with a compressor (C4 / C128) use YaRN with ``compress_rope_theta``: the cache
reproduces vLLM's ``DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache``
(``deepseek_yarn``, mscale disabled) op for op, so the rollout reads the same bytes.

Sliding-window-only layers (compress ratio 0) use plain RoPE with ``rope_theta``, as
DeepSeek's inference code (``precompute_freqs_cis`` with ``original_seq_len = 0``) and
Megatron-Core do: a 128-token window never reaches the distances YaRN rescales for.
vLLM (``ISEEKYAN/vllm@1a6542c``) applies YaRN to these layers as well; the rollout must
use this cache instead. The plain cache follows ``precompute_freqs_cis`` op for op.

Both are built on the given device, as vLLM and the reference do (CUDA): CPU ``pow`` /
``cos`` differ in the last bit. Only the first ``max_len`` positions are kept (each row
depends only on its position).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from rl_engine.models.deepseek_v4.config import Dsv4Config


def _correction_dim(rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(low_rot, high_rot, dim, base, max_pos) -> tuple[int, int]:
    low = math.floor(_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _ramp(low: float, high: float, dim: int, device) -> Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float, device=device) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def _plain(dim: int, base: float, max_len: int, dev: torch.device) -> Tensor:
    """DeepSeek's ``precompute_freqs_cis`` without YaRN, as ``cos | sin``."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=dev) / dim))
    t = torch.arange(max_len, device=dev)
    freqs = torch.outer(t, freqs)
    cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.cat((cis.real, cis.imag), dim=-1).contiguous()


def rope_cos_sin(config: Dsv4Config, compress_ratio: int, max_len: int, device="cuda") -> Tensor:
    """FP32 ``[max_len, qk_rope_head_dim]`` cache for a layer of ``compress_ratio``
    (1 = sliding window only: plain RoPE; 4 / 128: YaRN)."""
    rs = config.rope_scaling or {}
    dim = config.qk_rope_head_dim
    base = config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    factor = float(rs["factor"])
    original = int(rs["original_max_position_embeddings"])
    beta_fast, beta_slow = rs.get("beta_fast", 32), rs.get("beta_slow", 1)
    if max_len > original * factor:
        raise ValueError(f"max_len {max_len} exceeds the RoPE range {original * factor}")
    dev = torch.device(device)
    if compress_ratio == 1:
        return _plain(dim, base, max_len, dev)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)
    low, high = _correction_range(beta_fast, beta_slow, dim, base, original)
    mask = (1 - _ramp(low, high, dim // 2, dev)) * 1  # extrapolation_factor = 1
    inv_freq = inv_inter * (1 - mask) + inv_extra * mask

    t = torch.arange(max_len, device=dev, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    mscale = 1.0  # mscale = mscale_all_dim = 0 for DSv4: yarn_get_mscale ratio is 1
    return torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1)
