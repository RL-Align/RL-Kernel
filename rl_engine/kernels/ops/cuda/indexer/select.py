# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer KV selection: DeepGEMM FP8 MQA logits + stable top-k (SM90)."""

from __future__ import annotations

from torch import Tensor

from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda

SM90_PAGED_BLOCK_KV = 64


def indexer_logits_prefill(
    q_fp8: Tensor, weights: Tensor, k_fp8: Tensor, k_scale: Tensor, ks: Tensor, ke: Tensor
) -> Tensor:
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
    logits = indexer_logits_decode(
        q_fp8, weights, kv_cache, context_lens, block_table, max_context_len
    )
    ends = context_lens.reshape(-1).contiguous()
    starts = ends.new_zeros(ends.shape)
    return stable_topk_per_row_cuda(logits, starts, ends, top_k, out=out)
