# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention layer forward on the batch-invariant SM90 kernel set (T07).

Follows vLLM's ``DeepseekV4Attention`` / ``DeepseekV4FlashMLAAttention``
(``ISEEKYAN/vllm@1a6542c``) for one layer of compress ratio 1 (sliding window
only), 4 (C4: compressor + indexer) or 128 (C128: compressor, all entries)::

    qr, kv  = split(fused_wqa_wkv(h))                 FP8 DeepGEMM
    qr, kv  = rmsnorm(qr), rmsnorm(kv)                rmsnorm_cuda (P1)
    q       = wq_b(qr)                                FP8 DeepGEMM
    q       = q-norm + RoPE; kv -> RoPE + fp8_ds_mla sliding-window cache
    ratio>1 compressor: bf16_gemm(h, wkv_wgate) -> state cache -> fp8_ds_mla KV cache
    ratio=4 indexer: bf16_gemm(h, weights_proj), wq_b_idx(qr) -> RoPE + FP8,
            indexer compressor -> FP8 K cache, MQA logits -> stable top-k
    o       = FlashMLA sparse over [compressed | window] with the sink
    y       = wo_b(wo_a_grouped(inverse RoPE + FP8(o)))

Every operator is batch-invariant (each was probed or tested on its own);
prefill and decode run the same path. Caches are paged per request; the
caller owns block tables and slot assignment (:class:`LayerCaches`).
This module is forward-only (rollout and actor recompute).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda import dsv4_attention as core
from rl_engine.kernels.ops.cuda.compressor import compress_indexer_cuda, compress_main_cuda
from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.fp8_linear import Fp8BlockWeight, Fp8Linear
from rl_engine.kernels.ops.cuda.indexer import indexer_q_rope_fp8_cuda, indexer_select_prefill
from rl_engine.kernels.ops.pytorch.compressor import save_states

STATE_BLOCK = 16
KV_BLOCK = 64


@dataclass(frozen=True)
class Dsv4AttentionConfig:
    hidden: int = 4096
    heads: int = 64
    padded_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window: int = 128
    compress_ratio: int = 4
    index_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    eps: float = 1e-6
    max_model_len: int = 65536

    @property
    def coff(self) -> int:
        return 2 if self.compress_ratio == 4 else 1


@dataclass
class Dsv4AttentionWeights:
    fused_wqa_wkv: Fp8Linear  # [q_lora + head_dim, hidden]
    q_norm: Tensor  # [q_lora]
    kv_norm: Tensor  # [head_dim]
    wq_b: Fp8Linear  # [heads * head_dim, q_lora]
    attn_sink: Tensor  # FP32 [padded_heads]
    o_proj: core.OutputProjection
    # compressor (ratio > 1)
    wkv_wgate: Tensor | None = None  # BF16 [2 * coff * head_dim, hidden]
    ape: Tensor | None = None  # FP32 [ratio, coff * head_dim]
    comp_norm: Tensor | None = None  # [head_dim]
    # indexer (ratio == 4)
    idx_wq_b: Fp8Linear | None = None  # [index_heads * index_head_dim, q_lora]
    idx_weights_proj: Tensor | None = None  # BF16 [index_heads, hidden]
    idx_wkv_wgate: Tensor | None = None  # BF16 [2 * 2 * index_head_dim, hidden]
    idx_ape: Tensor | None = None  # FP32 [4, 2 * index_head_dim]
    idx_norm: Tensor | None = None  # [index_head_dim]


@dataclass
class LayerCaches:
    """Paged per-request caches of one layer; block tables are ``[requests, blocks]`` int32."""

    swa: Tensor  # uint8 [blocks, 64 * 584], fp8_ds_mla, one row per token
    swa_table: Tensor
    state: Tensor | None = None  # FP32 [blocks, 16, 2 * coff * 512]
    state_table: Tensor | None = None
    kv: Tensor | None = None  # uint8 [blocks, 64 * 584], fp8_ds_mla, one row per entry
    kv_table: Tensor | None = None
    idx_state: Tensor | None = None  # FP32 [blocks, 16, 2 * 2 * 128]
    idx_state_table: Tensor | None = None
    idx_k: Tensor | None = None  # uint8 [blocks, 64 * 132], FP8 rows + FP32 scales
    idx_k_table: Tensor | None = None
    extra: dict = field(default_factory=dict)


def _slots(table: Tensor, req: Tensor, index: Tensor, block: int) -> Tensor:
    return table[req, index // block].long() * block + index % block


def _bf16_gemm(x: Tensor, w: Tensor, out_dtype: torch.dtype) -> Tensor:
    dg, _ = require_deepgemm(x.device)
    d = torch.empty(x.shape[0], w.shape[0], dtype=out_dtype, device=x.device)
    if x.shape[0]:
        dg.bf16_gemm_nt(x, w, d)
    return d


def _gather_indexer_k(caches: LayerCaches, seq_lens: Tensor, ratio: int):
    """Pack each request's indexer entries: (FP8 [sum n_r, 128], FP32 [sum n_r], offsets [R])."""
    counts = (seq_lens.long() // ratio).tolist()
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    device = seq_lens.device
    req = torch.cat([torch.full((n,), r, device=device) for r, n in enumerate(counts)]).long()
    e = torch.cat([torch.arange(n, device=device) for n in counts]).long()
    slots = _slots(caches.idx_k_table, req, e, KV_BLOCK)
    blocks = caches.idx_k.view(caches.idx_k.shape[0], -1)
    b, i = slots // KV_BLOCK, slots % KV_BLOCK
    cols = (i * 128)[:, None] + torch.arange(128, device=device)
    k = blocks[b[:, None], cols].contiguous().view(torch.float8_e4m3fn)
    sc_cols = (KV_BLOCK * 128 + i * 4)[:, None] + torch.arange(4, device=device)
    scale = blocks[b[:, None], sc_cols].contiguous().view(torch.float32).reshape(-1)
    return k, scale, torch.tensor(offsets[:-1], device=device, dtype=torch.int32)


class Dsv4Attention:
    def __init__(self, config: Dsv4AttentionConfig, weights: Dsv4AttentionWeights, cos_sin: Tensor):
        self.c = config
        self.w = weights
        self.cos_sin = cos_sin.float().contiguous()
        self.scale = config.head_dim**-0.5
        from rl_engine.backends.cuda.norm.rmsnorm import rmsnorm_cuda

        self._rmsnorm = rmsnorm_cuda

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        token_req: Tensor,
        seq_lens: Tensor,
        query_lens: Tensor,
        caches: LayerCaches,
    ) -> Tensor:
        """``hidden`` BF16 ``[T, hidden]`` for tokens grouped by request (``token_req`` in
        ``[0, R)``, each request's tokens contiguous and ascending in position);
        ``seq_lens[r]`` is the request's length after this step and ``query_lens[r]`` its
        token count in the batch. Returns BF16 ``[T, hidden]``."""
        c, w = self.c, self.w
        ratio = c.compress_ratio
        tokens = hidden.shape[0]
        positions = positions.long().contiguous()
        req = token_req.long()
        req32 = token_req.int().contiguous()

        qr_kv = w.fused_wqa_wkv(hidden)
        qr = self._rmsnorm(qr_kv[:, : c.q_lora_rank].contiguous(), w.q_norm, c.eps)
        kv = self._rmsnorm(qr_kv[:, c.q_lora_rank :].contiguous(), w.kv_norm, c.eps)
        q = w.wq_b(qr).view(tokens, c.heads, c.head_dim)
        swa_slots = _slots(caches.swa_table, req, positions, KV_BLOCK)
        q_pad = core.q_norm_rope_kv_insert(
            q, kv, positions, self.cos_sin, caches.swa, KV_BLOCK, swa_slots, c.padded_heads, c.eps
        )

        topk = None
        if ratio > 1:
            score = _bf16_gemm(hidden, w.wkv_wgate, torch.float32)
            width = c.coff * c.head_dim
            save_states(
                score[:, :width],
                score[:, width:],
                w.ape,
                positions,
                caches.state,
                _slots(caches.state_table, req, positions, STATE_BLOCK),
            )
            entry = positions // ratio
            kv_slots = torch.where(
                (positions + 1) % ratio == 0, _slots(caches.kv_table, req, entry, KV_BLOCK), -1
            )
            compress_main_cuda(
                caches.state,
                caches.state_table,
                req32,
                positions,
                w.comp_norm,
                c.eps,
                self.cos_sin,
                caches.kv,
                KV_BLOCK,
                kv_slots,
                ratio,
            )
        if ratio == 4:
            topk = self._indexer(hidden, qr, positions, req, req32, seq_lens, caches)

        gather = query_lens + torch.clamp(seq_lens - query_lens, max=c.window - 1)
        layout = core.SlabLayout(c.max_model_len // ratio if ratio > 1 else 0, int(gather.max()))
        ws = core.build_workspace(
            layout,
            seq_lens,
            gather,
            caches.swa,
            caches.swa_table,
            KV_BLOCK,
            caches.kv,
            caches.kv_table,
            KV_BLOCK,
            ratio,
        )
        top_k = c.index_topk if ratio == 4 else (c.max_model_len // ratio if ratio > 1 else 0)
        idx, lens = core.build_indices(
            layout, req, positions, seq_lens - gather, c.window, max(ratio, 1), top_k, topk
        )
        out, _ = core.sparse_attention(q_pad, ws, idx, lens, w.attn_sink, self.scale)
        return w.o_proj(out[:, : c.heads], positions, self.cos_sin)

    def _indexer(self, hidden, qr, positions, req, req32, seq_lens, caches) -> Tensor:
        c, w = self.c, self.w
        tokens = hidden.shape[0]
        weights = _bf16_gemm(hidden, w.idx_weights_proj, torch.bfloat16)
        iq = w.idx_wq_b(qr).view(tokens, c.index_heads, c.index_head_dim)
        q8, w_out = indexer_q_rope_fp8_cuda(
            iq, positions, self.cos_sin, weights, c.index_head_dim**-0.5, c.index_heads**-0.5
        )
        score = _bf16_gemm(hidden, w.idx_wkv_wgate, torch.float32)
        width = 2 * c.index_head_dim
        save_states(
            score[:, :width],
            score[:, width:],
            w.idx_ape,
            positions,
            caches.idx_state,
            _slots(caches.idx_state_table, req, positions, STATE_BLOCK),
        )
        k_slots = torch.where(
            (positions + 1) % 4 == 0, _slots(caches.idx_k_table, req, positions // 4, KV_BLOCK), -1
        )
        compress_indexer_cuda(
            caches.idx_state,
            caches.idx_state_table,
            req32,
            positions,
            w.idx_norm,
            c.eps,
            self.cos_sin,
            caches.idx_k,
            KV_BLOCK,
            k_slots,
        )
        k, k_scale, offsets = _gather_indexer_k(caches, seq_lens, 4)
        ks = offsets[req].contiguous()
        ke = (ks + ((positions + 1) // 4).int()).contiguous()
        if k.shape[0] == 0:
            return torch.full((tokens, c.index_topk), -1, dtype=torch.int32, device=hidden.device)
        return indexer_select_prefill(q8.contiguous(), w_out, k, k_scale, ks, ke, c.index_topk)


def random_layer(config: Dsv4AttentionConfig, seed: int = 0, device: str = "cuda"):
    """Random weights in checkpoint formats (FP8 block weights from BF16), for tests/benchmarks."""
    from rl_engine.kernels.ops.cuda.fp8_linear import quantize_weight

    gen = torch.Generator().manual_seed(seed)
    c = config

    def bf16(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).to(device)

    def fp8(n, k):
        return Fp8Linear.from_bf16(bf16(n, k, scale=k**-0.5))

    wo_a: Fp8BlockWeight = quantize_weight(
        bf16(c.o_groups * c.o_lora_rank, c.heads // c.o_groups * c.head_dim, scale=0.02)
    )
    weights = Dsv4AttentionWeights(
        fused_wqa_wkv=fp8(c.q_lora_rank + c.head_dim, c.hidden),
        q_norm=(1 + 0.1 * torch.randn(c.q_lora_rank, generator=gen)).to(torch.bfloat16).to(device),
        kv_norm=(1 + 0.1 * torch.randn(c.head_dim, generator=gen)).to(torch.bfloat16).to(device),
        wq_b=fp8(c.heads * c.head_dim, c.q_lora_rank),
        attn_sink=torch.randn(c.padded_heads, generator=gen).to(device),
        o_proj=core.OutputProjection(wo_a, fp8(c.hidden, c.o_groups * c.o_lora_rank), c.o_groups),
    )
    if c.compress_ratio > 1:
        weights.wkv_wgate = bf16(2 * c.coff * c.head_dim, c.hidden, scale=c.hidden**-0.5)
        weights.ape = torch.randn(c.compress_ratio, c.coff * c.head_dim, generator=gen).to(device)
        weights.comp_norm = (1 + 0.1 * torch.randn(c.head_dim, generator=gen)).to(device)
    if c.compress_ratio == 4:
        weights.idx_wq_b = fp8(c.index_heads * c.index_head_dim, c.q_lora_rank)
        weights.idx_weights_proj = bf16(c.index_heads, c.hidden, scale=c.hidden**-0.5)
        weights.idx_wkv_wgate = bf16(4 * c.index_head_dim, c.hidden, scale=c.hidden**-0.5)
        weights.idx_ape = torch.randn(4, 2 * c.index_head_dim, generator=gen).to(device)
        weights.idx_norm = (1 + 0.1 * torch.randn(c.index_head_dim, generator=gen)).to(device)
    return weights


def new_caches(config: Dsv4AttentionConfig, requests: int, max_len: int, device="cuda", seed=0):
    """Paged caches with shuffled physical blocks, sized for ``max_len`` tokens per request."""
    gen = torch.Generator().manual_seed(seed)
    c = config

    def paged(rows_per_req, block, row_shape, dtype, fill):
        blocks = -(-rows_per_req // block)
        total = requests * blocks + 1
        buf = torch.full((total,) + row_shape, fill, dtype=dtype, device=device)
        table = torch.randperm(total, generator=gen)[: requests * blocks].view(requests, blocks)
        return buf, table.int().to(device)

    swa, swa_t = paged(max_len, KV_BLOCK, (KV_BLOCK * 584,), torch.uint8, 0)
    caches = LayerCaches(swa=swa, swa_table=swa_t)
    if c.compress_ratio > 1:
        caches.state, caches.state_table = paged(
            max_len,
            STATE_BLOCK,
            (STATE_BLOCK, 2 * c.coff * c.head_dim),
            torch.float32,
            float("nan"),
        )
        caches.kv, caches.kv_table = paged(
            max_len // c.compress_ratio + 1, KV_BLOCK, (KV_BLOCK * 584,), torch.uint8, 0
        )
    if c.compress_ratio == 4:
        caches.idx_state, caches.idx_state_table = paged(
            max_len, STATE_BLOCK, (STATE_BLOCK, 4 * c.index_head_dim), torch.float32, float("nan")
        )
        caches.idx_k, caches.idx_k_table = paged(
            max_len // 4 + 1, KV_BLOCK, (KV_BLOCK * 132,), torch.uint8, 0
        )
    return caches
