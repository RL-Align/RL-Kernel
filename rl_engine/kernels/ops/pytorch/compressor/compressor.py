# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 KV compressor (golden): state save and compress -> RMSNorm -> RoPE -> FP8.

The model math follows vLLM's DSv4 compressor (``ISEEKYAN/vllm@1a6542c``,
``compressor.py`` / ``save_partial_states.py`` /
``fused_compress_quant_cache.py``); the arithmetic order is our own and fixed
(design doc T02, decision (a)), so the CUDA kernel can match it byte for byte.
vLLM's Triton kernel agrees on almost every FP8 byte but is not the contract.

State save (every token)::

    state[slot] = [kv (coff*D) | score + ape[pos % ratio] (coff*D)]     FP32

Compress (each boundary token, ``(pos + 1) % ratio == 0``), per channel ``c``
of ``D``, over the window of ``W = coff * ratio`` tokens ``pos - W + 1 .. pos``
(tokens before position 0 are skipped; with overlap (C4, ``coff = 2``) the
older ``ratio`` tokens read channel half 0 and the newer half 1)::

    m     = max_j s_j
    e_j   = expf(s_j - m)                         (libdevice expf: run on CUDA)
    S     = (((e_0 + e_1) + e_2) + ...)            ascending j
    x_c   = ((((kv_0 * e_0) + kv_1 * e_1) + ...) / S       ascending j, no FMA
    v     = sum_c x_c^2: each run of 4 channels summed in order, 32 such partials
            combined by a butterfly (lane i adds lane i ^ 16, ^ 8, ^ 4, ^ 2, ^ 1),
            then the per-128-channel results in ascending order
    r     = 1 / sqrt(v / D + eps)                  each step correctly rounded
    n_c   = (x_c * r) * w_c
    RoPE on the last rope_dim channels, GPT-J pairs, position (pos // ratio) * ratio:
          n0' = n0 cos - n1 sin,  n1' = n1 cos + n0 sin   (products rounded, no FMA)
    BF16 round, then one of two cache entry formats. Scale rule (both): amax
    (NaN ignored), s = max(amax, 1e-4) * fp32(1/448) rounded up to 2^(e-127)
    from its bits, q = e4m3 satfinite(value * 2^-(e-127)).
      indexer (head 128): one scale over all 128 values, stored as FP32.
      fp8_ds_mla (head 512, main KV, FlashMLA's layout): the 448 nope values
      in 7 blocks of 64, each with its UE8M0 byte e; the 64 RoPE values stay
      BF16; per token 448 FP8 bytes + 64 BF16 (576 bytes), and 8 scale bytes
      (7 + a zero pad) after all token rows of the cache block.

Each entry reads only its own request's state, so the compressor is
batch-invariant and per-request block boundaries never cross requests.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
WARP = 32


def save_states(
    kv: Tensor,
    score: Tensor,
    ape: Tensor,
    positions: Tensor,
    state_cache: Tensor,
    slot_mapping: Tensor,
) -> None:
    """Write ``[kv | score + ape[pos % ratio]]`` rows into the paged FP32 state cache.

    ``state_cache`` is ``[blocks, block_size, 2 * width]``; slots ``< 0`` are skipped.
    """
    ratio, width = ape.shape
    if kv.shape[-1] != width or score.shape[-1] != width:
        raise ValueError(f"kv/score width must be {width}")
    keep = slot_mapping >= 0
    slots = slot_mapping[keep].long()
    rows = torch.cat(
        [kv[keep].float(), score[keep].float() + ape[positions[keep].long() % ratio]], -1
    )
    state_cache.view(-1, state_cache.shape[-1]).index_copy_(0, slots, rows)


def _butterfly_sum(squares: Tensor) -> Tensor:
    """Sum the last dim in the kernel's order.

    Each thread owns 4 consecutive channels and sums them in order; 32 thread
    partials form a butterfly (lane i adds lane i ^ 16, ^ 8, ^ 4, ^ 2, ^ 1);
    the per-warp results are added in ascending warp order.
    """
    quads = squares.reshape(*squares.shape[:-1], -1, WARP, 4)
    thread = ((quads[..., 0] + quads[..., 1]) + quads[..., 2]) + quads[..., 3]
    lane = torch.arange(WARP, device=squares.device)
    for offset in (16, 8, 4, 2, 1):
        thread = thread + thread[..., lane ^ offset]
    partial = thread[..., 0]
    total = partial[..., 0]
    for g in range(1, partial.shape[-1]):
        total = total + partial[..., g]
    return total


def compress_window(
    state_cache: Tensor,
    block_table: Tensor,
    token_to_req: Tensor,
    positions: Tensor,
    ratio: int,
    overlap: bool,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Compressed FP32 ``x [N, D]`` for the boundary tokens; returns ``(x, token_index)``."""
    boundary = ((positions + 1) % ratio == 0).nonzero().flatten()
    coff = 2 if overlap else 1
    window = coff * ratio
    block_size = state_cache.shape[1]
    width = coff * head_dim
    pos = positions[boundary].long()
    req = token_to_req[boundary].long()
    j = torch.arange(window, device=pos.device)
    rows = pos[:, None] - window + 1 + j[None]  # [N, W]
    valid = rows >= 0
    rows_c = rows.clamp(min=0)
    blocks = block_table[req[:, None], rows_c // block_size].long()
    flat = state_cache.view(-1, state_cache.shape[-1])
    states = flat[blocks * block_size + rows_c % block_size]  # [N, W, 2 * width]
    half = ((j >= ratio) & overlap).long() * head_dim  # [W]
    ch = torch.arange(head_dim, device=pos.device)
    kv = torch.gather(states, 2, (half[:, None] + ch[None]).expand(len(pos), window, head_dim))
    score = torch.gather(
        states, 2, (width + half[:, None] + ch[None]).expand(len(pos), window, head_dim)
    )

    neg_inf = torch.tensor(-float("inf"), device=pos.device)
    m = torch.where(valid[..., None], score, neg_inf).amax(dim=1)
    e = torch.exp(score - m[:, None])
    total = torch.zeros_like(m)
    for k in range(window):
        total = torch.where(valid[:, k, None], total + e[:, k], total)
    acc = torch.zeros_like(m)
    for k in range(window):
        acc = torch.where(valid[:, k, None], acc + kv[:, k] * e[:, k], acc)
    return acc / total, boundary


def norm_rope(
    x: Tensor,
    positions: Tensor,
    ratio: int,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
) -> Tensor:
    """RMSNorm then GPT-J RoPE at the compressed position; returns BF16 ``[N, D]``."""
    d = x.shape[-1]
    var = _butterfly_sum(x * x) / torch.tensor(float(d), device=x.device)
    eps_t = torch.tensor(eps, dtype=torch.float32, device=x.device)
    r = torch.ones_like(var) / torch.sqrt(var + eps_t)
    n = (x * r[:, None]) * rms_weight.float()[None]
    rope = cos_sin_cache.shape[1]
    half = rope // 2
    cp = (positions.long() // ratio) * ratio
    cs = cos_sin_cache[cp].float()
    cos, sin = cs[:, :half], cs[:, half:]
    even, odd = n[:, d - rope :: 2], n[:, d - rope + 1 :: 2]
    out = n.clone()
    out[:, d - rope :: 2] = even * cos - odd * sin
    out[:, d - rope + 1 :: 2] = odd * cos + even * sin
    return out.to(torch.bfloat16)


def quant_fp8_entry(values: Tensor) -> tuple[Tensor, Tensor]:
    """One power-of-two FP8 scale per row (the indexer cache entry format)."""
    full = values.float()
    mag = full.abs()
    exponent = _pow2_scale(torch.where(mag.isnan(), 0.0, mag).amax(dim=-1))
    scale = (exponent << 23).to(torch.int32).view(torch.float32)
    inverse = ((254 - exponent) << 23).to(torch.int32).view(torch.float32)
    q = (full * inverse[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _pow2_scale(amax: Tensor) -> Tensor:
    """Biased exponent e of 2^ceil(log2(max(amax, 1e-4) / 448)), from the bits."""
    floor = torch.tensor(AMAX_FLOOR, dtype=torch.float32, device=amax.device)
    inv_max = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=amax.device)
    s = torch.maximum(amax, floor) * inv_max
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((bits + 0x7FFFFF) >> 23) & 0xFF


def quant_ds_mla(values: Tensor, nope: int = 448, block: int = 64) -> tuple[Tensor, Tensor, Tensor]:
    """fp8_ds_mla entry of BF16 ``[N, 512]``.

    Returns (FP8 ``[N, 448]``, UE8M0 bytes ``[N, 8]``, BF16 ``[N, 64]``).
    """
    full = values.float()
    head = full[:, :nope].reshape(len(full), -1, block)
    mag = head.abs()
    exponent = _pow2_scale(torch.where(mag.isnan(), 0.0, mag).amax(dim=-1))  # [N, 7]
    inverse = ((254 - exponent) << 23).to(torch.int32).view(torch.float32)
    q = (head * inverse[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    scales = torch.zeros(len(full), 8, dtype=torch.uint8, device=values.device)
    scales[:, : exponent.shape[1]] = exponent.to(torch.uint8)
    return q.reshape(len(full), nope), scales, values[:, nope:].to(torch.bfloat16)


def dequant_ds_mla(q: Tensor, scales: Tensor, rope: Tensor, block: int = 64) -> Tensor:
    """BF16 ``[N, 512]`` from an fp8_ds_mla entry (exact: e4m3 * 2^k fits BF16)."""
    n, nope = q.shape
    factor = (scales[:, : nope // block].long() << 23).to(torch.int32).view(torch.float32)
    head = q.float().reshape(n, -1, block) * factor[..., None]
    return torch.cat([head.reshape(n, nope).to(torch.bfloat16), rope.to(torch.bfloat16)], -1)


def compress_main(
    state_cache: Tensor,
    block_table: Tensor,
    token_to_req: Tensor,
    positions: Tensor,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
    ratio: int,
    head_dim: int = 512,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Main-KV compressor entries (C4 with overlap, C128 without), fp8_ds_mla format.

    Returns ``(fp8 [N, 448], scale bytes [N, 8], rope bf16 [N, 64], token_index [N])``.
    """
    if not state_cache.is_cuda:
        raise RuntimeError("the compressor golden is defined on CUDA (libdevice expf)")
    x, tokens = compress_window(
        state_cache, block_table, token_to_req, positions, ratio, ratio == 4, head_dim
    )
    values = norm_rope(x, positions[tokens], ratio, rms_weight, eps, cos_sin_cache)
    q, scales, rope = quant_ds_mla(values, nope=head_dim - cos_sin_cache.shape[1])
    return q, scales, rope, tokens


def compress_indexer(
    state_cache: Tensor,
    block_table: Tensor,
    token_to_req: Tensor,
    positions: Tensor,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
    ratio: int = 4,
    overlap: bool = True,
    head_dim: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Indexer compressor entries: ``(q_fp8 [N, D], scale fp32 [N], token_index [N])``.

    Must run on CUDA (``expf`` is part of the contract).
    """
    if not state_cache.is_cuda:
        raise RuntimeError("the compressor golden is defined on CUDA (libdevice expf)")
    x, tokens = compress_window(
        state_cache, block_table, token_to_req, positions, ratio, overlap, head_dim
    )
    values = norm_rope(x, positions[tokens], ratio, rms_weight, eps, cos_sin_cache)
    q, scale = quant_fp8_entry(values)
    return q, scale, tokens
