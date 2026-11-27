# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 attention layer forward on the batch-invariant SM90 kernel set.

The DSv4 layer (``rl_engine.models.deepseek_v4.attention``) with DSv4.1's changes
(``docs/architecture/dsv41-design.md`` 4.4, DeepSeek's ``inference/model.py`` Attention):

- Dense linears are BF16 (D12): ``[wq_a; wkv]``, ``wq_b``, ``wo_a`` (per group), ``wo_b``.
- No per-head q-norm after ``wq_b`` (``q_rope_kv_insert``).
- KV / index sharing (design doc 2.3): a KV source compresses its input into the group's
  compressed cache (ratio 2: window-2 softmax pooling of the paged FP32 state; ratio 1:
  the ``wkv`` row itself) and writes the indexer K ``k_norm(wk(latent))`` from the BF16
  latent before RoPE; an index source scores its queries against the group's indexer K
  and publishes top-k ids; a reuse layer reads both. KV and indexer-K caches of a group
  are one set of tensors, owned by the source layer's :class:`LayerCaches` and aliased by
  the others; the ids travel in ``shared``.
- Two-level candidates: the candidate source publishes its top blocks, later indexers mask
  their logits to them (``kernels.ops.cuda.topk.candidates``).

Every row's bytes depend only on its own request: the indexer logits of a prefill are
gathered into request-local rows before the candidate mask and the top-k, so prefill and
decode run one selector (design doc rule 14).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.attention import dsv4_attention as core
from rl_engine.kernels.ops.cuda.compressor import (
    dsv41_compress_main_cuda,
    dsv41_store_rows_cuda,
    save_states_cuda,
)
from rl_engine.kernels.ops.cuda.indexer import indexer_q_rope_fp8_cuda
from rl_engine.kernels.ops.cuda.indexer.select import indexer_logits_decode, indexer_logits_prefill
from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear, mxfp8_fake_quant
from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda
from rl_engine.kernels.ops.cuda.topk.candidates import mask_to_candidates, select_candidate_blocks
from rl_engine.models.deepseek_v4.attention import KV_BLOCK, STATE_BLOCK, LayerCaches, _bf16_gemm

__all__ = [
    "Bf16OutputProjection",
    "Dsv41Attention",
    "Dsv41AttentionConfig",
    "Dsv41AttentionWeights",
    "Dsv41Batch",
    "LayerCaches",
    "block_sizes",
    "new_group_caches",
]

HEAD_DIM, ROPE, NOPE = 512, 64, 448


@dataclass(frozen=True)
class Dsv41AttentionConfig:
    hidden: int = 5120
    heads: int = 64
    padded_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window: int = 128
    layer_id: int = 0
    ratio: int = 0  # 0 window only, 1 or 2 compressed
    is_kv_source: bool = False
    is_index_source: bool = False
    index_source: int | None = None  # layer whose top-k ids this layer reads
    candidate_source: bool = False
    uses_candidates: bool = False
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    eps: float = 1e-20
    max_model_len: int = 65536


@dataclass
class Dsv41AttentionWeights:
    fused_wqa_wkv: Bf16Linear  # [q_lora + head_dim, hidden]
    q_norm: Tensor  # [q_lora]
    kv_norm: Tensor  # [head_dim]
    wq_b: Bf16Linear  # [heads * head_dim, q_lora]
    attn_sink: Tensor  # FP32 [padded_heads]
    o_proj: "Bf16OutputProjection"
    # KV source: compressor ([wkv; wgate] at ratio 2, wkv at ratio 1) and indexer K
    comp_wkv: Bf16Linear | None = None
    comp_norm: Tensor | None = None  # [head_dim]
    idx_wk: Bf16Linear | None = None  # [index_head_dim, head_dim]
    idx_k_norm: Tensor | None = None  # [index_head_dim]
    # index source: indexer query side
    idx_wq_b: Bf16Linear | None = None  # [index_heads * index_head_dim, q_lora]
    idx_weights_proj: Tensor | None = None  # BF16 [index_heads, hidden]
    # LoRA adapters (DSv4's lora.py) by target: wq_a, wkv, wq_b, wo_b.
    lora: dict = field(default_factory=dict)


class Bf16OutputProjection:
    """``wo_b(wo_a_grouped(inverse_rope(o)))`` (D12): the output's RoPE dims are rotated back
    in FP32 and rounded once, rounded to MXFP8 per 32 values (DeepSeek's in-place
    ``act_quant`` before its einsum; off with ``act_quant=False``), each group's heads go
    through its BF16 ``wo_a`` slice (einsum ``bsgd,grd->bsgr``), then ``wo_b``."""

    def __init__(self, wo_a: Tensor, wo_b: Bf16Linear, groups: int, act_quant: bool = True):
        rank = wo_a.shape[0] // groups
        self.groups, self.o_lora_rank = groups, rank
        self.act_quant = act_quant
        self.wo_a = [
            Bf16Linear(wo_a[g * rank : (g + 1) * rank].contiguous()) for g in range(groups)
        ]
        self.wo_b = wo_b

    @staticmethod
    def inverse_rope(o: Tensor, positions: Tensor, cos_sin: Tensor) -> Tensor:
        """BF16 ``[T, H, 512]``: the last 64 dims rotated back (GPT-J pairs), FP32, rounded once."""
        cs = cos_sin[positions.long()].float()
        cos, sin = cs[:, None, : ROPE // 2], cs[:, None, ROPE // 2 :]
        x = o.float()
        e, od = x[..., NOPE::2], x[..., NOPE + 1 :: 2]
        out = x.clone()
        out[..., NOPE::2] = e * cos + od * sin
        out[..., NOPE + 1 :: 2] = od * cos - e * sin
        return out.to(torch.bfloat16)

    def __call__(
        self, o: Tensor, positions: Tensor, cos_sin: Tensor, lora=None, saved=None
    ) -> Tensor:
        tokens, heads, _ = o.shape
        x = self.inverse_rope(o, positions, cos_sin).view(tokens, self.groups, -1)
        if self.act_quant:
            x = mxfp8_fake_quant(x)
        z = torch.cat([w(x[:, g].contiguous()) for g, w in enumerate(self.wo_a)], dim=1)
        y = self.wo_b(z)
        if lora is not None:
            from rl_engine.models.deepseek_v4.lora import lora_apply

            y = lora_apply(y, z, lora)
        if saved is not None:
            saved["z"] = z
        return y


@dataclass
class Dsv41Batch:
    """One step's token layout, built on the host; every size is a host integer.

    ``entries[ratio]``: the compressed entries of each request at that ratio for packed
    prefill logits: ``(offsets int32 [R], entry_req int64 [E], entry_id int64 [E], width)``
    with ``width`` the largest per-request count.
    """

    positions: Tensor  # int64 [T]
    token_req: Tensor  # int64 [T]
    seq_lens: Tensor  # int32 [R], length after this step
    query_lens: Tensor  # int32 [R]
    gather_lens: Tensor  # int32 [R], window rows a request's slab holds
    decode: bool
    max_gather: int
    entries: dict

    @classmethod
    def from_lens(
        cls, seq_lens: list[int], query_lens: list[int], window: int, ratios=(1, 2), device="cuda"
    ) -> "Dsv41Batch":
        dev = torch.device(device)
        positions = torch.cat([torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)])
        token_req = torch.cat(
            [torch.full((q,), r, dtype=torch.long) for r, q in enumerate(query_lens)]
        )
        gather = [q + min(max(s - q, 0), window - 1) for s, q in zip(seq_lens, query_lens)]
        entries = {}
        for ratio in ratios:
            counts = [s // ratio for s in seq_lens]
            offsets = [0]
            for n in counts[:-1]:
                offsets.append(offsets[-1] + n)
            entries[ratio] = (
                torch.tensor(offsets, dtype=torch.int32, device=dev),
                torch.cat([torch.full((n,), r, dtype=torch.long) for r, n in enumerate(counts)]).to(
                    dev
                ),
                torch.cat([torch.arange(n) for n in counts]).long().to(dev),
                max(counts),
            )
        return cls(
            positions=positions.to(dev),
            token_req=token_req.to(dev),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=dev),
            gather_lens=torch.tensor(gather, dtype=torch.int32, device=dev),
            decode=all(q == 1 for q in query_lens),
            max_gather=max(gather),
            entries=entries,
        )


def _slots(table: Tensor, req: Tensor, index: Tensor, block: int) -> Tensor:
    return table[req, index // block].long() * block + index % block


def block_sizes(caches: LayerCaches) -> tuple[int, int, int]:
    """Rows per block of the SWA, compressed KV and indexer-K caches (``caches.extra``
    ``"swa_block"`` / ``"kv_block"`` / ``"idx_block"``; :data:`KV_BLOCK` when absent). A
    serving engine's block sizes are its own (vLLM: 32 window rows, 64 / ratio entries)."""
    e = caches.extra
    return e.get("swa_block", KV_BLOCK), e.get("kv_block", KV_BLOCK), e.get("idx_block", KV_BLOCK)


def _gather_indexer_k(caches: LayerCaches, batch: Dsv41Batch, ratio: int):
    """Each request's indexer entries packed for the prefill logits: FP8 ``[E, 128]`` and
    FP32 scales ``[E]``."""
    _, entry_req, entry_id, _ = batch.entries[ratio]
    assert caches.idx_k is not None and caches.idx_k_table is not None
    blk = block_sizes(caches)[2]
    slots = _slots(caches.idx_k_table, entry_req, entry_id, blk)
    rows = caches.idx_k.view(caches.idx_k.shape[0], -1)
    b, i = slots // blk, slots % blk
    data = rows.as_strided((rows.shape[0], blk, 128), (rows.stride(0), 128, 1))
    scales = rows[:, blk * 128 : blk * 132].view(torch.float32)
    return data[b, i].contiguous().view(torch.float8_e4m3fn), scales[b, i].contiguous()


class Dsv41Attention:
    def __init__(
        self, config: Dsv41AttentionConfig, weights: Dsv41AttentionWeights, cos_sin: Tensor
    ):
        from rl_engine.backends.cuda.norm.rmsnorm import rmsnorm_cuda

        self.c, self.w = config, weights
        self.cos_sin = cos_sin.float().contiguous()
        self.scale = config.head_dim**-0.5
        self._rmsnorm = rmsnorm_cuda
        self._zero_ape: Tensor | None = None

    def forward_batch(
        self,
        hidden: Tensor,
        batch: Dsv41Batch,
        caches: LayerCaches,
        shared: dict,
        saved: dict | None = None,
    ) -> Tensor:
        """``hidden`` BF16 ``[T, hidden]`` (tokens grouped by request, ascending position)
        -> BF16 ``[T, hidden]``. ``shared`` carries the step's top-k ids
        (``shared["topk"][index source layer]``) and candidate blocks between layers."""
        c, w = self.c, self.w
        tokens = hidden.shape[0]
        positions, req = batch.positions, batch.token_req

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
        swa_block, kv_block, _ = block_sizes(caches)
        if not c.ratio:
            kv_block = swa_block  # window layers gather nothing compressed, from the SWA cache
        swa_slots = _slots(caches.swa_table, req, positions, swa_block)
        q_pad = core.q_rope_kv_insert(
            q, kv, positions, self.cos_sin, caches.swa, swa_block, swa_slots, c.padded_heads
        )

        topk = None
        if c.ratio:
            if c.is_kv_source:
                self._compress(hidden, batch, caches)
            if c.is_index_source:
                topk = self._index(hidden, qr, batch, caches, shared)
                shared.setdefault("topk", {})[c.layer_id] = topk
            else:
                topk = shared["topk"][c.index_source]

        ratio = max(c.ratio, 1)
        top_k = c.index_topk if c.ratio else 0
        compressed = c.ratio > 0
        if batch.decode:
            ws, idx, lens = core.build_compact(
                req, positions, c.window, ratio, top_k, caches.swa, caches.swa_table, swa_block,
                caches.kv, caches.kv_table, kv_block, topk, compressed=compressed,
            )  # fmt: skip
        else:
            layout = core.SlabLayout(
                c.max_model_len // ratio if compressed else 0, batch.max_gather
            )
            ws = core.build_workspace(
                layout, batch.seq_lens, batch.gather_lens, caches.swa, caches.swa_table, swa_block,
                caches.kv if compressed else None, caches.kv_table, kv_block, ratio,
            )  # fmt: skip
            idx, lens = core.build_indices(
                layout, req, positions, batch.seq_lens - batch.gather_lens, c.window, ratio,
                top_k, topk, compressed=compressed,
            )  # fmt: skip
        out, lse = core.sparse_attention(q_pad, ws, idx, lens, w.attn_sink, self.scale)
        if saved is not None:
            saved.update(q_pad=q_pad, ws=ws, idx=idx, lens=lens, out=out, lse=lse)
        return w.o_proj(
            out[:, : c.heads], positions, self.cos_sin, lora=w.lora.get("wo_b"), saved=saved
        )

    # ------------------------------------------------------------------ KV source
    def _compress(self, hidden: Tensor, batch: Dsv41Batch, caches: LayerCaches) -> None:
        """Compressed KV entries and indexer K of the group's caches at boundary tokens."""
        c, w = self.c, self.w
        positions, req = batch.positions, batch.token_req
        tokens = hidden.shape[0]
        assert w.comp_wkv is not None and w.idx_wk is not None
        _, kv_block, idx_block = block_sizes(caches)
        entry = positions // c.ratio
        boundary = (positions + 1) % c.ratio == 0
        kv_slots = torch.where(boundary, _slots(caches.kv_table, req, entry, kv_block), -1)
        latent = torch.zeros(tokens, c.head_dim, dtype=torch.bfloat16, device=hidden.device)
        if c.ratio == 2:
            score = w.comp_wkv(hidden, torch.float32)  # [T, 2 * head_dim]: kv | gate
            if self._zero_ape is None:
                self._zero_ape = torch.zeros(
                    2, c.head_dim, dtype=torch.float32, device=hidden.device
                )
            save_states_cuda(
                score[:, : c.head_dim],
                score[:, c.head_dim :],
                self._zero_ape,
                positions,
                caches.state,
                _slots(caches.state_table, req, positions, caches.state.shape[1]),
            )
            dsv41_compress_main_cuda(
                caches.state, caches.state_table, req.int(), positions, w.comp_norm, c.eps,
                self.cos_sin, caches.kv, kv_block, kv_slots, latent,
            )  # fmt: skip
        else:  # ratio 1: the BF16 wkv row (DeepSeek keeps it BF16), no pooling
            rows = w.comp_wkv(hidden).float()
            dsv41_store_rows_cuda(
                rows, positions, w.comp_norm, c.eps, self.cos_sin, caches.kv, kv_block,
                kv_slots, 1, latent,
            )  # fmt: skip
        k_pre = w.idx_wk(latent).float()  # BF16 wk output, as DeepSeek's and vLLM's
        k_slots = torch.where(boundary, _slots(caches.idx_k_table, req, entry, idx_block), -1)
        dsv41_store_rows_cuda(
            k_pre, positions, w.idx_k_norm, c.eps, self.cos_sin, caches.idx_k, idx_block,
            k_slots, c.ratio,
        )  # fmt: skip

    # ------------------------------------------------------------------ index source
    def _index(
        self, hidden: Tensor, qr: Tensor, batch: Dsv41Batch, caches: LayerCaches, shared: dict
    ) -> Tensor:
        """Request-local top-k compressed ids ``int32 [T, index_topk]`` (``-1`` padded)."""
        c, w = self.c, self.w
        tokens = hidden.shape[0]
        positions, req = batch.positions, batch.token_req
        assert w.idx_wq_b is not None and w.idx_weights_proj is not None
        weights = _bf16_gemm(hidden.contiguous(), w.idx_weights_proj, torch.bfloat16)
        iq = w.idx_wq_b(qr).view(tokens, c.index_heads, c.index_head_dim)
        q8, w_out = indexer_q_rope_fp8_cuda(
            iq, positions, self.cos_sin, weights, c.index_head_dim**-0.5, c.index_heads**-0.5
        )
        ratio = c.ratio
        ends = ((positions + 1) // ratio).int()
        # DeepGEMM's paged MQA logits (pinned build) take 64-entry blocks; other block sizes
        # gather the entries and run the prefill logits, which the gate holds byte-equal.
        if batch.decode and block_sizes(caches)[2] == KV_BLOCK:
            context = (batch.seq_lens // ratio).view(-1, 1).contiguous()
            local = indexer_logits_decode(
                q8.view(tokens, 1, c.index_heads, c.index_head_dim).contiguous(),
                w_out,
                caches.idx_k.view(caches.idx_k.shape[0], KV_BLOCK, 1, 132),
                context,
                caches.idx_k_table,
                c.max_model_len // ratio,
            )
        else:
            offsets, entry_req, _, width = batch.entries[ratio]
            local = torch.full(
                (tokens, max(width, 1)), -float("inf"), dtype=torch.float32, device=hidden.device
            )
            if entry_req.shape[0]:
                k, k_scale = _gather_indexer_k(caches, batch, ratio)
                ks = offsets[req].contiguous()
                ke = (ks + ends).contiguous()
                packed = indexer_logits_prefill(q8.contiguous(), w_out, k, k_scale, ks, ke)
                cols = torch.arange(local.shape[1], device=hidden.device)[None]
                src = (ks.long()[:, None] + cols).clamp(max=packed.shape[1] - 1)
                local = torch.where(
                    cols < ends.long()[:, None],
                    packed.float().gather(1, src),
                    torch.full_like(local, -float("inf")),
                )
        local = local.float()
        if c.candidate_source:
            shared["candidates"] = select_candidate_blocks(
                local, ends, c.candidate_topk_blocks, c.candidate_block_size
            )
        elif c.uses_candidates:
            local = mask_to_candidates(local, ends, shared["candidates"], c.candidate_block_size)
        return stable_topk_per_row_cuda(local, torch.zeros_like(ends), ends, c.index_topk)


def new_group_caches(
    config: Dsv41AttentionConfig,
    requests: int,
    max_len: int,
    device="cuda",
    seed: int = 0,
    group: LayerCaches | None = None,
) -> LayerCaches:
    """Paged caches of one layer (shuffled physical blocks, ``max_len`` tokens per request).

    The sliding-window cache is the layer's own. A KV source allocates its group's
    compressed KV and indexer-K caches (and the ratio-2 state); any other compressed layer
    passes its source's caches as ``group`` and aliases them.
    """
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
    if not c.ratio:
        return caches
    if c.is_kv_source:
        if c.ratio == 2:
            caches.state, caches.state_table = paged(
                max_len, STATE_BLOCK, (STATE_BLOCK, 2 * c.head_dim), torch.float32, float("nan")
            )
        entries = max_len // c.ratio + 1
        caches.kv, caches.kv_table = paged(entries, KV_BLOCK, (KV_BLOCK * 584,), torch.uint8, 0)
        caches.idx_k, caches.idx_k_table = paged(
            entries, KV_BLOCK, (KV_BLOCK * 132,), torch.uint8, 0
        )
        return caches
    if group is None:
        raise ValueError("a compressed layer that is not a KV source needs its group's caches")
    caches.kv, caches.kv_table = group.kv, group.kv_table
    caches.idx_k, caches.idx_k_table = group.idx_k, group.idx_k_table
    return caches


def attention_configs(cfg, max_model_len: int, **overrides) -> list[Dsv41AttentionConfig]:
    """Per-layer :class:`Dsv41AttentionConfig` of a :class:`Dsv41Config`."""
    out = []
    for i, role in enumerate(cfg.roles()):
        out.append(
            Dsv41AttentionConfig(
                hidden=cfg.hidden_size,
                heads=cfg.num_attention_heads,
                padded_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                rope_dim=cfg.qk_rope_head_dim,
                q_lora_rank=cfg.q_lora_rank,
                o_lora_rank=cfg.o_lora_rank,
                o_groups=cfg.o_groups,
                window=cfg.sliding_window,
                layer_id=i,
                ratio=role.ratio,
                is_kv_source=role.mode == "full",
                is_index_source=role.mode in ("full", "reindex"),
                index_source=role.index_source,
                candidate_source=role.candidate_source,
                uses_candidates=role.uses_candidates,
                candidate_topk_blocks=overrides.get(
                    "candidate_topk_blocks", cfg.candidate_topk_blocks
                ),
                candidate_block_size=cfg.candidate_block_size,
                index_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=overrides.get("index_topk", cfg.index_topk),
                eps=cfg.rms_norm_eps,
                max_model_len=max_model_len,
            )
        )
    return out


def random_layer(c: Dsv41AttentionConfig, seed: int = 0, device="cuda") -> Dsv41AttentionWeights:
    """Random BF16 weights (N(0, 1/sqrt(K)) linears, norms 1 + N(0, 0.1)) for tests."""
    gen = torch.Generator().manual_seed(seed)

    def bf16(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).to(device)

    def lin(n, k, act_quant=True):
        return Bf16Linear(bf16(n, k, scale=k**-0.5), act_quant)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=gen)).to(torch.bfloat16).to(device)

    kg = c.heads * c.head_dim // c.o_groups
    w = Dsv41AttentionWeights(
        fused_wqa_wkv=lin(c.q_lora_rank + c.head_dim, c.hidden),
        q_norm=norm(c.q_lora_rank),
        kv_norm=norm(c.head_dim),
        wq_b=lin(c.heads * c.head_dim, c.q_lora_rank),
        attn_sink=torch.randn(c.padded_heads, generator=gen).to(device),
        o_proj=Bf16OutputProjection(
            bf16(c.o_groups * c.o_lora_rank, kg, scale=kg**-0.5),
            lin(c.hidden, c.o_groups * c.o_lora_rank),
            c.o_groups,
        ),
    )
    if c.is_kv_source:
        w.comp_wkv = lin((2 if c.ratio == 2 else 1) * c.head_dim, c.hidden, False)
        w.comp_norm = norm(c.head_dim)
        w.idx_wk = lin(c.index_head_dim, c.head_dim, False)
        w.idx_k_norm = norm(c.index_head_dim)
    if c.is_index_source:
        w.idx_wq_b = lin(c.index_heads * c.index_head_dim, c.q_lora_rank)
        w.idx_weights_proj = bf16(c.index_heads, c.hidden, scale=c.hidden**-0.5)
    return w
