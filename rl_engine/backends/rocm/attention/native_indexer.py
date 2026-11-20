# SPDX-License-Identifier: Apache-2.0
"""Pinned DSV4 native FP8 indexer Q/K writers, scoring and causal Top-K.

Actual original projections/parameters enter the loaded vLLM/AITER primitives.
This is the nondifferentiable selection boundary; native BF16 auxiliary-loss
execution is retained separately and only coefficient zero is admitted.
"""

import torch


def _keys(kv, score, ape, weight, cache, epsilon):
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )
    from vllm.models.deepseek_v4.common.ops.save_partial_states import (
        _SAVE_PARTIAL_STATES_KERNEL,
    )
    from vllm.models.deepseek_v4.compressor import CompressorMetadata
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
    )

    t, width = kv.shape
    if (
        kv.dtype != score.dtype
        or kv.dtype != torch.float32
        or width != 256
        or score.shape != kv.shape
    ):
        raise ValueError("native C4 indexer requires actual FP32 width256 projections")
    n, block = t // 4, 64
    pages, kpages = (t + 63) // 64, (n + 63) // 64
    state = torch.zeros((pages, block, 512), device=kv.device, dtype=torch.float32)
    positions = torch.arange(t, device=kv.device, dtype=torch.int64)
    owners = torch.zeros(t, device=kv.device, dtype=torch.int32)
    table = torch.arange(pages, device=kv.device, dtype=torch.int32)[None]
    # Native NORMAL layout is segregated within each page: block*128 code
    # bytes, then block*4 scale bytes. token_stride below is the code stride,
    # not 132; the writer uses packed.stride(0) for the complete page stride.
    packed = torch.zeros((kpages, block, 132), device=kv.device, dtype=torch.uint8)
    ktable = torch.arange(kpages, device=kv.device, dtype=torch.int32)[None]
    metadata = CompressorMetadata(
        block_table=ktable, slot_mapping=positions // 4, block_size=block
    )
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        state_cache=state,
        slot_mapping=positions,
        block_size=block,
        state_width=width,
        compress_ratio=4,
        pdl_kwargs={},
    )
    compress_norm_rope_store_triton(
        state_cache=state,
        num_actual=t,
        token_to_req_indices=owners,
        positions=positions,
        slot_mapping=positions,
        block_table=table,
        block_size=block,
        state_width=width,
        cos_sin_cache=cache,
        kv_cache=packed,
        k_cache_metadata=metadata,
        pdl_kwargs={},
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=weight,
        rms_norm_eps=epsilon,
        quant_block=128,
        token_stride=128,
        scale_dim=4,
    )
    from rl_engine.backends.rocm.fp8 import fp8_dtype

    codes = torch.empty((n, 128), device=kv.device, dtype=fp8_dtype(kv.device))
    scales = torch.empty((n, 4), device=kv.device, dtype=torch.uint8)
    lengths = torch.tensor([0, n], device=kv.device, dtype=torch.int32)
    cp_gather_indexer_k_quant_cache_triton(
        packed,
        codes,
        scales,
        ktable,
        lengths,
        token_to_seq=torch.zeros(n, device=kv.device, dtype=torch.int32),
        cache_layout="NORMAL",
    )
    return codes, scales.view(torch.float32)


@torch.no_grad()
def native_fp8_indexer(q, weights, kv, score, ape, norm_weight, cache, epsilon):
    """Full-prefix ordinary [T,B] input; causal selection per real query row."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _launch_aiter_top_k_per_row_prefill,
        rocm_fp8_mqa_logits,
    )

    t, batch, width = q.shape
    if width != 8192 or weights.shape != (t, batch, 64) or kv.shape != (t, batch, 256):
        raise ValueError("native FP8 indexer supports original H64/D128/C4 geometry")
    if (
        q.dtype != torch.bfloat16
        or weights.dtype != q.dtype
        or cache.dtype != torch.float32
    ):
        raise ValueError(
            "native FP8 indexer requires original BF16 Q/weights and FP32 RoPE"
        )
    n = t // 4
    if n <= 512:
        raise ValueError(
            "native FP8 scoring is only needed beyond the all-candidates shortcut"
        )
    positions = torch.arange(t, device=q.device, dtype=torch.int64)
    starts = torch.zeros(t, device=q.device, dtype=torch.int32)
    ends = ((positions + 1) // 4).to(torch.int32).clamp_max(n)
    scores, selected = [], []
    for b in range(batch):
        qc, wc = fused_indexer_q_rope_quant(
            positions,
            q[:, b].reshape(t, 64, 128).contiguous(),
            cache,
            weights[:, b].contiguous(),
            128**-0.5,
            64**-0.5,
            use_fp4=False,
        )
        kc, ks = _keys(
            kv[:, b].contiguous(),
            score[:, b].contiguous(),
            ape,
            norm_weight,
            cache,
            epsilon,
        )
        logits = rocm_fp8_mqa_logits(qc, (kc, ks), wc, starts, ends)
        indices = torch.full((t, 512), -1, device=q.device, dtype=torch.int32)
        if rocm_aiter_ops.is_indexer_top_k_supported(
            is_prefill=True,
            compress_ratio=4,
            num_rows=t,
            max_valid_seq_len=n,
            topk_tokens=512,
        ):
            _launch_aiter_top_k_per_row_prefill(logits, starts, ends, indices, 512)
        else:
            torch.ops._C.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                indices,
                t,
                logits.stride(0),
                logits.stride(1),
                512,
            )
        scores.append(logits)
        selected.append(indices.long())
    return torch.stack(scores), torch.stack(selected)
