# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 sparse attention core on FlashMLA (SM90, T05).

What vLLM runs under ``VLLM_BATCH_INVARIANT=1`` with
``VLLM_DS4_DECODE_KERNEL=sparse`` (``nvidia/flashmla.py``,
``common/ops/cache_utils.py`` at ``ISEEKYAN/vllm@1a6542c``), for prefill and
decode alike:

1. Each request ``b`` owns a fixed slab of ``M = N + G`` rows in a BF16 KV
   workspace: rows ``[0, N)`` hold its dequantized compressed entries (``N`` is
   fixed from the model config, ``max_model_len // ratio``), rows ``[N, N + G)``
   the dequantized sliding-window tokens ``gather_start .. seq_len - 1``.
2. Each query token gets one index list: its compressed ids (C4: the
   indexer's top-k; C128: every entry so far), **canonicalized** by a
   descending sort, then its window positions ``pos - swa_len + 1 .. pos``
   ascending, all offset to its request's slab, ``-1`` padded to a multiple
   of 128; the valid length is ``topk_len + swa_len`` with
   ``topk_len = min((pos + 1) // ratio, top_k)`` and
   ``swa_len = min(pos + 1, window)``.
3. ``flash_mla_sparse_fwd`` over the workspace with the per-head sink.

The probe (``scripts/probe_dsv4_flashmla_bi.py``) showed the kernel's bytes
do not depend on other queries, on where a slab sits in the workspace or on
padding, but do depend on index order; with the canonical order every step is
row-local, so the output is batch-invariant and decode equals prefill.
Gather / dequantize and index building are exact integer / power-of-two
operations done with PyTorch indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.flashmla import require_flashmla
from rl_engine.kernels.ops.pytorch.compressor import dequant_ds_mla

HEAD_DIM = 512
NOPE = 448
ROW_BYTES = 576  # 448 FP8 + 64 BF16
SCALE_BYTES = 8
INDEX_ALIGN = 128  # FlashMLA sparse prefill needs topk % 128 == 0 (h_q 64 and 128)


def read_ds_mla(cache: Tensor, slots: Tensor, block_size: int) -> Tensor:
    """Dequantize fp8_ds_mla entries at ``slots`` (int64) into BF16 ``[n, 512]``."""
    blocks = cache.view(cache.shape[0], -1)
    b, i = slots // block_size, slots % block_size
    row = (i * ROW_BYTES)[:, None] + torch.arange(ROW_BYTES, device=cache.device)
    sc = (block_size * ROW_BYTES + i * SCALE_BYTES)[:, None] + torch.arange(
        SCALE_BYTES, device=cache.device
    )
    data = blocks[b[:, None], row]
    scales = blocks[b[:, None], sc]
    q = data[:, :NOPE].contiguous().view(torch.float8_e4m3fn)
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16)
    return dequant_ds_mla(q, scales, rope)


@dataclass(frozen=True)
class SlabLayout:
    """Per-request workspace slabs: ``M = N + G`` rows each."""

    compressed_rows: int  # N, fixed from the model config
    window_rows: int  # G, at least the largest gather length in the batch

    @property
    def rows(self) -> int:
        return self.compressed_rows + self.window_rows


def build_workspace(
    layout: SlabLayout,
    seq_lens: Tensor,
    gather_lens: Tensor,
    swa_cache: Tensor,
    swa_block_table: Tensor,
    swa_block_size: int,
    compressed_cache: Tensor | None = None,
    compressed_block_table: Tensor | None = None,
    compressed_block_size: int = 0,
    ratio: int = 1,
) -> Tensor:
    """BF16 workspace ``[R * M, 1, 512]`` for ``R`` requests (unused rows are zero)."""
    requests = seq_lens.shape[0]
    device = seq_lens.device
    ws = torch.zeros(requests * layout.rows, HEAD_DIM, dtype=torch.bfloat16, device=device)
    base = torch.arange(requests, device=device, dtype=torch.int64) * layout.rows

    # Sliding-window tokens gather_start .. seq_len - 1 -> rows N + i.
    g = torch.arange(layout.window_rows, device=device)
    gl = gather_lens.long()[:, None]
    take = g[None] < gl
    pos = seq_lens.long()[:, None] - gl + g[None]
    req = torch.arange(requests, device=device)[:, None].expand_as(pos)
    r, p = req[take], pos[take]
    slots = swa_block_table[r, p // swa_block_size].long() * swa_block_size + p % swa_block_size
    ws[(base[:, None] + layout.compressed_rows + g[None])[take]] = read_ds_mla(
        swa_cache, slots, swa_block_size
    )

    if compressed_cache is not None:
        counts = seq_lens.long() // ratio
        c = torch.arange(layout.compressed_rows, device=device)
        take = c[None] < counts[:, None]
        req = torch.arange(requests, device=device)[:, None].expand(-1, layout.compressed_rows)
        r, e = req[take], c[None].expand(requests, -1)[take]
        slots = (
            compressed_block_table[r, e // compressed_block_size].long() * compressed_block_size
            + e % compressed_block_size
        )
        ws[(base[:, None] + c[None])[take]] = read_ds_mla(
            compressed_cache, slots, compressed_block_size
        )
    return ws.view(-1, 1, HEAD_DIM)


def build_indices(
    layout: SlabLayout,
    token_req: Tensor,
    token_pos: Tensor,
    gather_starts: Tensor,
    window: int,
    ratio: int,
    top_k: int,
    topk_local: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Canonical index lists ``int32 [T, 1, W]`` and valid lengths ``int32 [T]``.

    ``topk_local``: the C4 indexer's request-local compressed ids ``[T, top_k]``
    (any order, ``-1`` padded). ``None`` with ``ratio > 1`` means C128 (every
    compressed entry so far); ``ratio == 1`` is a sliding-window-only layer.
    """
    tokens = token_pos.shape[0]
    device = token_pos.device
    pos = token_pos.long()
    slab = token_req.long() * layout.rows
    if ratio > 1:
        topk_len = torch.clamp((pos + 1) // ratio, max=top_k)
        if topk_local is None:  # C128: every entry so far
            ids = torch.arange(top_k, device=device)[None].expand(tokens, -1)
            ids = torch.where(ids < topk_len[:, None], ids, -1)
        else:
            ids = topk_local.long()
        ids = torch.sort(ids, dim=-1, descending=True).values  # canonical order, -1 last
    else:
        top_k = 0
        topk_len = torch.zeros_like(pos)
        ids = torch.empty(tokens, 0, dtype=torch.long, device=device)
    swa_len = torch.clamp(pos + 1, max=window)
    width = -(-(top_k + window) // INDEX_ALIGN) * INDEX_ALIGN
    out = torch.full((tokens, width), -1, dtype=torch.long, device=device)
    k = torch.arange(top_k, device=device)[None]
    out[:, :top_k] = torch.where(k < topk_len[:, None], ids + slab[:, None], -1)
    # Window rows: N + (p - gather_start) for p in pos - swa_len + 1 .. pos, after topk_len.
    w = torch.arange(window, device=device)[None]
    first = pos - swa_len + 1 - gather_starts.long()[token_req.long()]
    win = slab[:, None] + layout.compressed_rows + first[:, None] + w
    cols = topk_len[:, None] + w
    keep = w < swa_len[:, None]
    rows = torch.arange(tokens, device=device)[:, None].expand_as(cols)
    out[rows[keep], cols[keep]] = win[keep]
    return out.to(torch.int32)[:, None, :], (topk_len + swa_len).to(torch.int32)


def sparse_attention(
    q: Tensor, workspace: Tensor, indices: Tensor, lens: Tensor, sink: Tensor, scale: float
) -> tuple[Tensor, Tensor]:
    """``flash_mla_sparse_fwd``: q BF16 ``[T, H, 512]`` -> (out BF16 ``[T, H, 512]``, lse)."""
    fm, _ = require_flashmla(q.device)
    out, lse = fm.flash_mla_sparse_fwd(q, workspace, indices, scale, HEAD_DIM, sink, lens)[:2]
    return out, lse
