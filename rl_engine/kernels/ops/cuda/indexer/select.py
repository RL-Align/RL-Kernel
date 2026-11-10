# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer KV selection: DeepGEMM FP8 MQA logits + stable top-k (SM90).

The two steps the rollout runs after quantizing the indexer query
(``indexer_q_rope_fp8``), as vLLM's ``sparse_attn_indexer`` does under
``VLLM_BATCH_INVARIANT=1``:

* prefill: ``fp8_fp4_mqa_logits`` over a packed FP8 KV buffer with per-query
  ranges ``[ks, ke)``, then ``stable_topk_per_row`` over the same ranges;
* decode: ``fp8_fp4_paged_mqa_logits`` over the paged FP8 indexer cache, then
  ``stable_topk_per_row`` over ``[0, context_len)``. vLLM runs the prefill
  selector here too, so decode and chunked prefill pick the same KV entries.

Both DeepGEMM kernels were probed batch-invariant, and paged logits equal
prefill logits (``scripts/probe_dsv4_indexer_logits_bi.py``); the selector is
exact. Output ids are request-local compressed indices in the selector's key
order (not canonicalized: the FlashMLA wrapper sorts them), ``-1`` padded.
Fail-closed: DeepGEMM and the ``rl_engine._C`` selector are required.
"""

from __future__ import annotations

from torch import Tensor

from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda

SM90_PAGED_BLOCK_KV = 64


def indexer_logits_prefill(
    q_fp8: Tensor, weights: Tensor, k_fp8: Tensor, k_scale: Tensor, ks: Tensor, ke: Tensor
) -> Tensor:
    """FP32 logits ``[T, N]`` (row-strided view; only ``[ks, ke)`` of a row is defined)."""
    dg, _ = require_deepgemm(q_fp8.device)
    return dg.fp8_fp4_mqa_logits(
        (q_fp8, None),
        (k_fp8, k_scale),
        weights,
        ks,
        ke,
        clean_logits=False,
        max_seqlen_k=0,
        logits_dtype=weights.dtype,
    )


def indexer_select_prefill(
    q_fp8: Tensor,
    weights: Tensor,
    k_fp8: Tensor,
    k_scale: Tensor,
    ks: Tensor,
    ke: Tensor,
    top_k: int,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Top-k request-local KV ids ``int32 [T, top_k]`` for packed-KV prefill.

    ``q_fp8`` FP8 ``[T, H, D]`` and ``weights`` FP32 ``[T, H]`` come from
    ``indexer_q_rope_fp8``; ``k_fp8`` FP8 ``[N, D]`` / ``k_scale`` FP32 ``[N]``
    is the packed compressed KV; ``ks`` / ``ke`` int32 ``[T]`` bound each
    query's candidates.
    """
    logits = indexer_logits_prefill(q_fp8, weights, k_fp8, k_scale, ks, ke)
    return stable_topk_per_row_cuda(logits, ks, ke, top_k, out=out)


def indexer_logits_decode(
    q_fp8: Tensor,
    weights: Tensor,
    kv_cache: Tensor,
    context_lens: Tensor,
    block_table: Tensor,
    max_context_len: int,
) -> Tensor:
    """FP32 logits ``[B * next_n, max_context_len]`` from the paged FP8 cache.

    ``q_fp8`` FP8 ``[B, next_n, H, D]``; ``kv_cache`` uint8
    ``[blocks, 64, 1, D + 4]`` (64 FP8 rows then 64 FP32 scales per block);
    ``context_lens`` int32 ``[B, next_n]``; ``block_table`` int32 ``[B, max_blocks]``.
    """
    dg, capability = require_deepgemm(q_fp8.device)
    schedule = dg.get_paged_mqa_logits_metadata(
        context_lens, SM90_PAGED_BLOCK_KV, capability.num_sms
    )
    return dg.fp8_fp4_paged_mqa_logits(
        (q_fp8, None),
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule,
        max_context_len,
        clean_logits=False,
        logits_dtype=weights.dtype,
    )


def indexer_select_decode(
    q_fp8: Tensor,
    weights: Tensor,
    kv_cache: Tensor,
    context_lens: Tensor,
    block_table: Tensor,
    max_context_len: int,
    top_k: int,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Top-k ids ``int32 [B * next_n, top_k]`` for paged decode (same selector as prefill)."""
    logits = indexer_logits_decode(
        q_fp8, weights, kv_cache, context_lens, block_table, max_context_len
    )
    ends = context_lens.reshape(-1).contiguous()
    starts = ends.new_zeros(ends.shape)
    return stable_topk_per_row_cuda(logits, starts, ends, top_k, out=out)
