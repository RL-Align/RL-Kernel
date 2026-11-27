# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 sparse attention core on FlashMLA (SM90)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.attention.flashmla import require_flashmla
from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.matmul.fp8_linear import Fp8BlockWeight, Fp8Linear

HEAD_DIM = 512
NOPE = 448
ROW_BYTES = 576  # 448 FP8 + 64 BF16
SCALE_BYTES = 8
INDEX_ALIGN = 128  # FlashMLA sparse prefill needs topk % 128 == 0 (h_q 64 and 128)


def ds_mla_gather(
    cache: Tensor, block_size: int, slots: Tensor, out: Tensor | None = None
) -> Tensor:
    """Dequantize fp8_ds_mla entries at ``slots`` into BF16 ``[n, 512]`` (CUDA kernel).

    ``slots == -1`` gives a zero row; rows with ``slots < -1`` are left untouched
    (so two gathers from different caches can fill one buffer). Exact: equal to
    the golden ``dequant_ds_mla``.
    """
    if out is None:
        out = torch.empty(slots.shape[0], HEAD_DIM, dtype=torch.bfloat16, device=slots.device)
    _ext().dsv4_ds_mla_gather(
        cache.view(cache.shape[0], -1), int(block_size), slots.long().contiguous(), out
    )
    return out


def read_ds_mla(cache: Tensor, slots: Tensor, block_size: int) -> Tensor:
    """Dequantize fp8_ds_mla entries at ``slots`` into BF16 ``[n, 512]``."""
    return ds_mla_gather(cache, block_size, slots)


def paged_slots(table: Tensor, req: Tensor, index: Tensor, block: int, valid: Tensor) -> Tensor:
    """Cache slot of logical row ``index`` of request ``req``; ``-1`` where not ``valid``."""
    safe = torch.where(valid, index, 0)
    slot = table[req, safe // block].long() * block + safe % block
    return torch.where(valid, slot, -1)


@dataclass(frozen=True)
class SlabLayout:
    """Per-request workspace slabs: ``M = N + G`` rows each."""

    compressed_rows: int  # N
    window_rows: int  # G

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
    requests = seq_lens.shape[0]
    device = seq_lens.device
    n, g_rows = layout.compressed_rows, layout.window_rows
    req = torch.arange(requests, device=device)[:, None]
    ws = torch.empty(requests * layout.rows, HEAD_DIM, dtype=torch.bfloat16, device=device)

    g = torch.arange(g_rows, device=device)[None]
    gl = gather_lens.long()[:, None]
    pos = seq_lens.long()[:, None] - gl + g
    window = paged_slots(swa_block_table, req, pos, swa_block_size, g < gl)
    slots = torch.full((requests, layout.rows), -2, dtype=torch.long, device=device)
    slots[:, n:] = window
    ds_mla_gather(swa_cache, swa_block_size, slots.view(-1), ws)

    slots[:, n:] = -2
    if compressed_cache is not None and n:
        c = torch.arange(n, device=device)[None]
        count = (seq_lens.long() // ratio)[:, None]
        slots[:, :n] = paged_slots(compressed_block_table, req, c, compressed_block_size, c < count)
    else:
        slots[:, :n] = -1
    ds_mla_gather(
        compressed_cache if compressed_cache is not None else swa_cache,
        compressed_block_size or swa_block_size,
        slots.view(-1),
        ws,
    )
    return ws.view(-1, 1, HEAD_DIM)


def canonical_ids(
    pos: Tensor,
    ratio: int,
    top_k: int,
    topk_local: Tensor | None,
    compressed: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Compressed ids per token in canonical (descending) order, ``-1`` last, and their count.

    ``compressed`` defaults to ``ratio > 1`` (DSv4, where 1 means window only); DSv4.1
    passes it explicitly, as its ratio 1 is a full-length compressed cache."""
    tokens = pos.shape[0]
    device = pos.device
    if not (ratio > 1 if compressed is None else compressed):
        return torch.empty(tokens, 0, dtype=torch.long, device=device), torch.zeros_like(pos)
    topk_len = torch.clamp((pos + 1) // ratio, max=top_k)
    if topk_local is None:  # C128: every entry so far
        ids = torch.arange(top_k, device=device)[None].expand(tokens, -1)
        ids = torch.where(ids < topk_len[:, None], ids, -1)
    else:
        ids = topk_local.long()
    return torch.sort(ids, dim=-1, descending=True).values, topk_len


def _index_lists(
    compressed_rows: Tensor,
    topk_len: Tensor,
    window_first: Tensor,
    swa_len: Tensor,
    top_k: int,
    window: int,
) -> tuple[Tensor, Tensor]:
    """``[compressed_rows[:topk_len] | window_first + 0 .. swa_len - 1 | -1 ...]``."""
    tokens = topk_len.shape[0]
    device = topk_len.device
    width = -(-(top_k + window) // INDEX_ALIGN) * INDEX_ALIGN
    out = torch.full((tokens, width + 1), -1, dtype=torch.long, device=device)
    k = torch.arange(top_k, device=device)[None]
    out[:, :top_k] = torch.where(k < topk_len[:, None], compressed_rows, -1)
    w = torch.arange(window, device=device)[None]
    keep = w < swa_len[:, None]
    cols = torch.where(keep, topk_len[:, None] + w, width)  # spare column for the rest
    out.scatter_(1, cols, torch.where(keep, window_first[:, None] + w, -1))
    return out[:, :width].to(torch.int32)[:, None, :], (topk_len + swa_len).to(torch.int32)


def build_indices(
    layout: SlabLayout,
    token_req: Tensor,
    token_pos: Tensor,
    gather_starts: Tensor,
    window: int,
    ratio: int,
    top_k: int,
    topk_local: Tensor | None = None,
    compressed: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Canonical index lists ``int32 [T, 1, W]`` into a slab workspace, and valid lengths.

    ``topk_local``: the C4 indexer's request-local compressed ids ``[T, top_k]``
    (any order, ``-1`` padded). ``None`` with ``ratio > 1`` means C128 (every
    compressed entry so far); ``ratio == 1`` is a sliding-window-only layer.
    """
    pos = token_pos.long()
    slab = token_req.long() * layout.rows
    ids, topk_len = canonical_ids(pos, ratio, top_k, topk_local, compressed)
    top_k = ids.shape[1]
    swa_len = torch.clamp(pos + 1, max=window)
    first = (
        slab + layout.compressed_rows + pos - swa_len + 1 - gather_starts.long()[token_req.long()]
    )
    return _index_lists(ids + slab[:, None], topk_len, first, swa_len, top_k, window)


def build_compact(
    token_req: Tensor,
    token_pos: Tensor,
    window: int,
    ratio: int,
    top_k: int,
    swa_cache: Tensor,
    swa_block_table: Tensor,
    swa_block_size: int,
    compressed_cache: Tensor | None = None,
    compressed_block_table: Tensor | None = None,
    compressed_block_size: int = 0,
    topk_local: Tensor | None = None,
    compressed: bool | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token workspace (top_k + window) with its index lists."""
    device = token_pos.device
    pos = token_pos.long()
    req = token_req.long()
    tokens = pos.shape[0]
    ids, topk_len = canonical_ids(pos, ratio, top_k, topk_local, compressed)
    top_k = ids.shape[1]
    rows = top_k + window
    swa_len = torch.clamp(pos + 1, max=window)
    ws = torch.empty(tokens * rows, HEAD_DIM, dtype=torch.bfloat16, device=device)
    slots = torch.full((tokens, rows), -2, dtype=torch.long, device=device)
    w = torch.arange(window, device=device)[None]
    win_pos = pos[:, None] - swa_len[:, None] + 1 + w
    slots[:, top_k:] = paged_slots(
        swa_block_table, req[:, None], win_pos, swa_block_size, w < swa_len[:, None]
    )
    ds_mla_gather(swa_cache, swa_block_size, slots.view(-1), ws)
    if top_k:
        slots[:, top_k:] = -2
        slots[:, :top_k] = paged_slots(
            compressed_block_table, req[:, None], ids, compressed_block_size, ids >= 0
        )
        ds_mla_gather(compressed_cache, compressed_block_size, slots.view(-1), ws)
    base = torch.arange(tokens, device=device, dtype=torch.long) * rows
    k = torch.arange(top_k, device=device)[None]
    idx, lens = _index_lists(base[:, None] + k, topk_len, base + top_k, swa_len, top_k, window)
    return ws.view(-1, 1, HEAD_DIM), idx, lens


def sparse_attention(
    q: Tensor, workspace: Tensor, indices: Tensor, lens: Tensor, sink: Tensor, scale: float
) -> tuple[Tensor, Tensor]:
    fm, _ = require_flashmla(q.device)
    out, _, lse = fm.flash_mla_sparse_fwd(q, workspace, indices, scale, HEAD_DIM, sink, lens)
    return out, lse


# Query / output side -------------------------------------------------------


def _ext():
    ext = importlib.import_module("rl_engine._C")
    names = ("dsv4_q_norm_rope_kv_insert", "dsv4_inv_rope_fp8", "dsv4_ds_mla_gather")
    missing = [n for n in names if not hasattr(ext, n)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def q_norm_rope_kv_insert(
    q: Tensor,
    kv: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    swa_cache: Tensor,
    swa_block_size: int,
    swa_slots: Tensor,
    padded_heads: int,
    eps: float,
) -> Tensor:
    """Query heads -> weightless RMSNorm + RoPE (BF16, padded); KV -> RoPE + fp8_ds_mla insert.

    Byte-equal to ``rl_engine.kernels.ops.pytorch.attention_dsv4`` (``q_norm_rope`` /
    ``kv_entry``). ``swa_slots[t] < 0`` skips token ``t``'s cache write.
    """
    return _ext().dsv4_q_norm_rope_kv_insert(
        q,
        kv,
        positions,
        cos_sin_cache,
        swa_cache.view(swa_cache.shape[0], -1),
        int(swa_block_size),
        swa_slots,
        int(padded_heads),
        float(eps),
    )


def q_rope_kv_insert(
    q: Tensor,
    kv: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    swa_cache: Tensor,
    swa_block_size: int,
    swa_slots: Tensor,
    padded_heads: int,
) -> Tensor:
    """DSv4.1: :func:`q_norm_rope_kv_insert` without the query heads' RMSNorm (DSv4.1 norms
    q before ``wq_b``). Byte-equal to ``attention_dsv4.q_rope`` / ``kv_entry``."""
    return _ext().dsv41_q_rope_kv_insert(
        q,
        kv,
        positions,
        cos_sin_cache,
        swa_cache.view(swa_cache.shape[0], -1),
        int(swa_block_size),
        swa_slots,
        int(padded_heads),
    )


def inv_rope_fp8(o: Tensor, positions: Tensor, cos_sin_cache: Tensor, groups: int):
    """Attention output -> (FP8 A ``[G, T, K]``, FP32 scales ``[G, T, K/128]`` MN-major)."""
    a, scales = _ext().dsv4_inv_rope_fp8(o, positions, cos_sin_cache, int(groups))
    return a, scales


class OutputProjection:
    """wo_b(wo_a_grouped(inv_rope_fp8(o))) on DeepGEMM FP8 (forward)."""

    EINSUM_MAX_TOKENS = 8192

    def __init__(self, wo_a: Fp8BlockWeight, wo_b: Fp8Linear, groups: int) -> None:
        self._dg, _ = require_deepgemm(wo_a.q.device)
        self.groups = groups
        rank = wo_a.shape[0] // groups
        self.o_lora_rank = rank
        self.wo_a = [
            Fp8BlockWeight(
                wo_a.q[g * rank : (g + 1) * rank].contiguous(),
                wo_a.scale[g * rank // 128 : (g + 1) * rank // 128].contiguous(),
            )
            for g in range(groups)
        ]
        q, sc = wo_a.q.contiguous(), wo_a.scale.contiguous()
        self._wo_a3 = (q.view(groups, rank, -1), sc.view(groups, rank // 128, -1))
        self.wo_b = wo_b

    def _wo_a(self, a: Tensor, scales: Tensor) -> Tensor:
        tokens = a.shape[1]
        if tokens < self.EINSUM_MAX_TOKENS:
            z = torch.empty(
                tokens, self.groups, self.o_lora_rank, dtype=torch.bfloat16, device=a.device
            )
            self._dg.fp8_einsum(
                "bhr,hdr->bhd", (a.permute(1, 0, 2), scales.permute(1, 0, 2)), self._wo_a3, z
            )
            return z.view(tokens, -1)
        z = torch.empty(
            self.groups, tokens, self.o_lora_rank, dtype=torch.bfloat16, device=a.device
        )
        for g, w in enumerate(self.wo_a):
            self._dg.fp8_gemm_nt((a[g], scales[g]), w.pair(), z[g], disable_ue8m0_cast=True)
        return z.transpose(0, 1).reshape(tokens, -1)

    def __call__(
        self, o: Tensor, positions: Tensor, cos_sin_cache: Tensor, lora=None, saved=None
    ) -> Tensor:
        tokens = o.shape[0]
        a, scales = inv_rope_fp8(o, positions, cos_sin_cache, self.groups)
        if tokens:
            z = self._wo_a(a, scales)
        else:
            z = torch.empty(
                0, self.groups * self.o_lora_rank, dtype=torch.bfloat16, device=o.device
            )
        y = self.wo_b(z)
        if lora is not None:
            from rl_engine.models.deepseek_v4.lora import lora_apply

            y = lora_apply(y, z, lora)
        if saved is not None:
            saved["z"] = z
        return y
