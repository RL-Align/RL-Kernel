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

Every operator is batch-invariant (each was probed or tested on its own).
A decode step (one token per request) reads the paged indexer cache and builds
a per-token workspace; other steps pack indexer entries and build per-request
slabs. Both give the same bytes for the same token, and a decode step reads
nothing back to the host, so it can be captured in a CUDA graph. Caches are
paged per request; the caller owns block tables and slot assignment
(:class:`LayerCaches`) and builds each step's :class:`Dsv4Batch`.
This module is forward-only (rollout and actor recompute).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda import dsv4_attention as core
from rl_engine.kernels.ops.cuda.compressor import (
    compress_indexer_cuda,
    compress_main_cuda,
    save_states_cuda,
)
from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.fp8_linear import Fp8BlockWeight, Fp8Linear
from rl_engine.kernels.ops.cuda.indexer import (
    indexer_q_rope_fp8_cuda,
    indexer_select_decode,
    indexer_select_prefill,
)

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
    # LoRA adapters (lora.py, profile dsv4-lora-v1) by target: wq_a, wkv, wq_b, wo_b.
    lora: dict = field(default_factory=dict)


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


@dataclass
class Dsv4Batch:
    """One step's token layout, built on the host (like vLLM's attention metadata).

    Every size the forward needs is a host integer here, so the forward reads
    nothing back from the GPU and a decode step can be captured in a CUDA graph.
    """

    positions: Tensor  # int64 [T]
    token_req: Tensor  # int64 [T], request of each token (cache-table row)
    seq_lens: Tensor  # int32 [R], length after this step
    query_lens: Tensor  # int32 [R]
    gather_lens: Tensor  # int32 [R], window rows a request's slab holds
    decode: bool  # every request has exactly one token
    max_gather: int
    idx_offsets: Tensor  # int32 [R], start of each request's indexer entries (prefill)
    idx_entry_req: Tensor  # int64 [E]
    idx_entry_id: Tensor  # int64 [E]

    @classmethod
    def from_lens(cls, seq_lens: list[int], query_lens: list[int], window: int, device="cuda"):
        dev = torch.device(device)
        positions = torch.cat([torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)]).to(
            dev
        )
        token_req = torch.cat(
            [torch.full((q,), r, dtype=torch.long) for r, q in enumerate(query_lens)]
        ).to(dev)
        gather = [q + min(max(s - q, 0), window - 1) for s, q in zip(seq_lens, query_lens)]
        counts = [s // 4 for s in seq_lens]
        offsets = [0]
        for n in counts[:-1]:
            offsets.append(offsets[-1] + n)
        return cls(
            positions=positions,
            token_req=token_req,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=dev),
            gather_lens=torch.tensor(gather, dtype=torch.int32, device=dev),
            decode=all(q == 1 for q in query_lens),
            max_gather=max(gather),
            idx_offsets=torch.tensor(offsets, dtype=torch.int32, device=dev),
            idx_entry_req=torch.cat(
                [torch.full((n,), r, dtype=torch.long) for r, n in enumerate(counts)]
            ).to(dev),
            idx_entry_id=torch.cat([torch.arange(n) for n in counts]).long().to(dev),
        )


def _gather_indexer_k(caches: LayerCaches, batch: Dsv4Batch):
    """Pack each request's indexer entries (prefill): FP8 ``[E, 128]`` and FP32 scales ``[E]``."""
    slots = _slots(caches.idx_k_table, batch.idx_entry_req, batch.idx_entry_id, KV_BLOCK)
    rows = caches.idx_k.view(caches.idx_k.shape[0], -1)
    stride = rows.stride(0)
    b, i = slots // KV_BLOCK, slots % KV_BLOCK
    data = rows.as_strided((rows.shape[0], KV_BLOCK, 128), (stride, 128, 1))  # no copy
    scales = rows[:, KV_BLOCK * 128 : KV_BLOCK * 132].view(torch.float32)  # [blocks, 64]
    return data[b, i].contiguous().view(torch.float8_e4m3fn), scales[b, i].contiguous()


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
        """Convenience form: builds the :class:`Dsv4Batch` from device tensors (one host read)."""
        batch = Dsv4Batch.from_lens(
            seq_lens.tolist(), query_lens.tolist(), self.c.window, hidden.device
        )
        return self.forward_batch(hidden, batch, caches)

    def forward_batch(
        self, hidden: Tensor, batch: Dsv4Batch, caches: LayerCaches, saved: dict | None = None
    ) -> Tensor:
        """``hidden`` BF16 ``[T, hidden]``, tokens grouped by request in ascending position.

        Decode steps (one token per request) use the paged indexer and a per-token
        workspace; other steps use packed indexer entries and per-request slabs.
        Both give the same bytes for the same token. Returns BF16 ``[T, hidden]``.
        With ``saved`` (a dict), the intermediates the backward needs are stored in it
        (``attention_backward.py``); the output bytes do not change.
        """
        c, w = self.c, self.w
        ratio = c.compress_ratio
        tokens = hidden.shape[0]
        positions = batch.positions
        req = batch.token_req
        req32 = req.int()

        qr_kv = w.fused_wqa_wkv(hidden)
        if "wq_a" in w.lora or "wkv" in w.lora:
            from rl_engine.models.deepseek_v4.lora import lora_apply

            qa, kvp = qr_kv[:, : c.q_lora_rank], qr_kv[:, c.q_lora_rank :]
            if "wq_a" in w.lora:
                qa = lora_apply(qa, hidden, w.lora["wq_a"])
            if "wkv" in w.lora:
                kvp = lora_apply(kvp, hidden, w.lora["wkv"])
            qr_kv = torch.cat([qa, kvp], dim=1)
        qr = self._rmsnorm(qr_kv[:, : c.q_lora_rank].contiguous(), w.q_norm, c.eps)
        kv = self._rmsnorm(qr_kv[:, c.q_lora_rank :].contiguous(), w.kv_norm, c.eps)
        q = w.wq_b(qr)
        if "wq_b" in w.lora:
            from rl_engine.models.deepseek_v4.lora import lora_apply

            q = lora_apply(q, qr, w.lora["wq_b"])
        q = q.view(tokens, c.heads, c.head_dim)
        if saved is not None:
            saved.update(qr_kv=qr_kv, qr=qr, q=q)
        swa_slots = _slots(caches.swa_table, req, positions, KV_BLOCK)
        q_pad = core.q_norm_rope_kv_insert(
            q, kv, positions, self.cos_sin, caches.swa, KV_BLOCK, swa_slots, c.padded_heads, c.eps
        )

        topk = None
        if ratio > 1:
            score = _bf16_gemm(hidden, w.wkv_wgate, torch.float32)
            width = c.coff * c.head_dim
            save_states_cuda(
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
            topk = self._indexer(hidden, qr, batch, req, req32, caches)

        top_k = c.index_topk if ratio == 4 else (c.max_model_len // ratio if ratio > 1 else 0)
        if batch.decode:
            ws, idx, lens = core.build_compact(
                req,
                positions,
                c.window,
                max(ratio, 1),
                top_k,
                caches.swa,
                caches.swa_table,
                KV_BLOCK,
                caches.kv,
                caches.kv_table,
                KV_BLOCK,
                topk,
            )
        else:
            layout = core.SlabLayout(c.max_model_len // ratio if ratio > 1 else 0, batch.max_gather)
            ws = core.build_workspace(
                layout,
                batch.seq_lens,
                batch.gather_lens,
                caches.swa,
                caches.swa_table,
                KV_BLOCK,
                caches.kv,
                caches.kv_table,
                KV_BLOCK,
                ratio,
            )
            idx, lens = core.build_indices(
                layout,
                req,
                positions,
                batch.seq_lens - batch.gather_lens,
                c.window,
                max(ratio, 1),
                top_k,
                topk,
            )
        out, lse = core.sparse_attention(q_pad, ws, idx, lens, w.attn_sink, self.scale)
        if saved is not None:
            saved.update(q_pad=q_pad, ws=ws, idx=idx, lens=lens, out=out, lse=lse)
        return w.o_proj(
            out[:, : c.heads], positions, self.cos_sin, lora=w.lora.get("wo_b"), saved=saved
        )

    def _indexer(self, hidden, qr, batch: Dsv4Batch, req, req32, caches) -> Tensor:
        c, w = self.c, self.w
        tokens = hidden.shape[0]
        positions = batch.positions
        weights = _bf16_gemm(hidden, w.idx_weights_proj, torch.bfloat16)
        iq = w.idx_wq_b(qr).view(tokens, c.index_heads, c.index_head_dim)
        q8, w_out = indexer_q_rope_fp8_cuda(
            iq, positions, self.cos_sin, weights, c.index_head_dim**-0.5, c.index_heads**-0.5
        )
        score = _bf16_gemm(hidden, w.idx_wkv_wgate, torch.float32)
        width = 2 * c.index_head_dim
        save_states_cuda(
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
        if batch.decode:
            # Paged logits over each request's entries: same selection as prefill.
            context = (batch.seq_lens // 4).view(-1, 1).contiguous()
            return indexer_select_decode(
                q8.view(tokens, 1, c.index_heads, c.index_head_dim).contiguous(),
                w_out,
                caches.idx_k.view(caches.idx_k.shape[0], KV_BLOCK, 1, 132),
                context,
                caches.idx_k_table,
                c.max_model_len // 4,
                c.index_topk,
            )
        if batch.idx_entry_id.shape[0] == 0:
            return torch.full((tokens, c.index_topk), -1, dtype=torch.int32, device=hidden.device)
        k, k_scale = _gather_indexer_k(caches, batch)
        ks = batch.idx_offsets[req].contiguous()
        ke = (ks + ((positions + 1) // 4).int()).contiguous()
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
