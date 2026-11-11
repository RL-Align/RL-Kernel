# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention query / output side (golden): q-norm + RoPE, KV entry, inverse RoPE + FP8.

Model math from vLLM's DSv4 attention (``ISEEKYAN/vllm@1a6542c``:
``fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`` and
``fused_inv_rope_fp8_quant.py``); the arithmetic order is our own fixed
contract (as for the compressor), so the CUDA kernels match byte for byte.
Head dim 512 = 448 nope + 64 RoPE (GPT-J pairs, ``cos_sin_cache[pos]`` with
cos first).

``q_norm_rope`` (each query head)::

    v = sum x^2: 16 values per lane in order, then a 32-lane butterfly
    r = 1 / sqrt(v / 512 + eps)            correctly rounded steps
    y = x * r;  RoPE: y0' = y0 c - y1 s,  y1' = y0 s + y1 c   (products rounded)
    BF16, heads padded with zeros up to ``padded_heads``

``kv_entry``: RoPE on the (already RMS-normed) KV row, then the fp8_ds_mla
entry (``compressor.quant_ds_mla``): 448 FP8 values in UE8M0 blocks of 64,
64 BF16 RoPE values.

``inv_rope_fp8`` (attention output, before ``wo_a``): inverse RoPE in FP32
(``x0' = x0 c + x1 s``, ``x1' = x1 c - x0 s``, products rounded), then each
128-value block quantized with ``s = max(amax, 1e-10) * fp32(1/448)`` rounded up
to a power of two from its bits; output per group of ``heads_per_group`` heads
as DeepGEMM's FP8 A operand ``[groups, T, heads_per_group * 512]`` with FP32
scales ``[groups, T, K / 128]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.pytorch.compressor import quant_ds_mla

HEAD_DIM, NOPE, ROPE = 512, 448, 64
LANE_VALUES = 16
FP8_MAX = 448.0


def _rope_pairs(x: Tensor, cos: Tensor, sin: Tensor, inverse: bool) -> Tensor:
    e, o = x[..., NOPE::2], x[..., NOPE + 1 :: 2]
    out = x.clone()
    if inverse:
        out[..., NOPE::2] = e * cos + o * sin
        out[..., NOPE + 1 :: 2] = o * cos - e * sin
    else:
        out[..., NOPE::2] = e * cos - o * sin
        out[..., NOPE + 1 :: 2] = e * sin + o * cos
    return out


def _cos_sin(cos_sin_cache: Tensor, positions: Tensor, extra_dims: int):
    cs = cos_sin_cache[positions.long()].float()
    cos, sin = cs[:, : ROPE // 2], cs[:, ROPE // 2 :]
    shape = (cs.shape[0],) + (1,) * extra_dims + (ROPE // 2,)
    return cos.reshape(shape), sin.reshape(shape)


def q_norm_rope(
    q: Tensor, positions: Tensor, cos_sin_cache: Tensor, eps: float, padded_heads: int
) -> Tensor:
    """BF16 ``[T, H, 512]`` -> BF16 ``[T, padded_heads, 512]`` (weightless RMSNorm + RoPE)."""
    tokens, heads, _ = q.shape
    x = q.float()
    sq = (x * x).reshape(tokens, heads, 32, LANE_VALUES)
    lane = sq[..., 0]
    for i in range(1, LANE_VALUES):
        lane = lane + sq[..., i]
    idx = torch.arange(32, device=q.device)
    for offset in (16, 8, 4, 2, 1):
        lane = lane + lane[..., idx ^ offset]
    total = lane[..., 0]
    eps_t = torch.tensor(eps, dtype=torch.float32, device=q.device)
    r = torch.ones_like(total) / torch.sqrt(total / torch.tensor(512.0, device=q.device) + eps_t)
    y = x * r[..., None]
    cos, sin = _cos_sin(cos_sin_cache, positions, 1)
    y = _rope_pairs(y, cos, sin, inverse=False).to(torch.bfloat16)
    out = torch.zeros(tokens, padded_heads, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    out[:, :heads] = y
    return out


def kv_entry(kv: Tensor, positions: Tensor, cos_sin_cache: Tensor):
    """RoPE'd fp8_ds_mla entry of each KV row: (FP8 [T, 448], scale bytes [T, 8], BF16 [T, 64])."""
    x = kv.float()
    cos, sin = _cos_sin(cos_sin_cache, positions, 0)
    return quant_ds_mla(_rope_pairs(x, cos, sin, inverse=False).to(torch.bfloat16))


def inv_rope_fp8(
    o: Tensor, positions: Tensor, cos_sin_cache: Tensor, groups: int
) -> tuple[Tensor, Tensor]:
    """Attention output BF16 ``[T, H, 512]`` -> FP8 ``[G, T, K]`` and FP32 ``[G, T, K/128]``."""
    tokens, heads, _ = o.shape
    x = o.float()
    cos, sin = _cos_sin(cos_sin_cache, positions, 1)
    x = _rope_pairs(x, cos, sin, inverse=True)
    k = heads // groups * HEAD_DIM
    blocks = x.reshape(tokens, groups, k // 128, 128)
    amax = torch.where(blocks.abs().isnan(), 0.0, blocks.abs()).amax(-1)
    floor = torch.tensor(1e-10, dtype=torch.float32, device=o.device)
    inv_max = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=o.device)
    s = torch.maximum(amax, floor) * inv_max
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    e = ((bits + 0x7FFFFF) >> 23) & 0xFF
    scale = (e << 23).to(torch.int32).view(torch.float32)
    inverse = ((254 - e) << 23).to(torch.int32).view(torch.float32)
    q = (blocks * inverse[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.reshape(tokens, groups, k).transpose(0, 1).contiguous()
    return q, scale.transpose(0, 1).contiguous()
