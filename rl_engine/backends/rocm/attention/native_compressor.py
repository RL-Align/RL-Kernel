# SPDX-License-Identifier: Apache-2.0
"""Current production compressed-KV writer with an explicit full-prefix STE VJP."""

import torch


def _functional(kv, score, ape, weight, cache, ratio, eps):
    t, b, width = kv.shape
    n = t // ratio
    dim = 512
    values = kv[: n * ratio].reshape(n, ratio, b, width)
    scores = score[: n * ratio].reshape(n, ratio, b, width) + ape[None, :, None, :]
    if ratio == 4:
        previous = torch.cat((torch.zeros_like(values[:1, :, :, :dim]), values[:-1, :, :, :dim]), 0)
        old_scores = torch.cat(
            (torch.full_like(scores[:1, :, :, :dim], -float("inf")), scores[:-1, :, :, :dim]), 0
        )
        values = torch.cat((previous, values[:, :, :, dim:]), 1)
        scores = torch.cat((old_scores, scores[:, :, :, dim:]), 1)
    pooled = (scores.softmax(1) * values).sum(1)
    normalized = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps) * weight.float()
    pos = torch.arange(n, device=kv.device) * ratio
    cos, sin = cache.index_select(0, pos).chunk(2, -1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    tail = normalized[..., 448:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
    return torch.cat((normalized[..., :448], rotated), -1)


def _forward(kv, score, ape, weight, cache, ratio, eps):
    from vllm.models.deepseek_v4.compressor import CompressorMetadata
    from vllm.models.deepseek_v4.common.ops.save_partial_states import _SAVE_PARTIAL_STATES_KERNEL
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
        compress_norm_rope_store_two_stage_triton,
    )
    from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache

    t, b, width = kv.shape
    n = t // ratio
    if (
        kv.dtype != torch.float32
        or score.dtype != torch.float32
        or width != (1024 if ratio == 4 else 512)
    ):
        raise ValueError("native main compressor requires original FP32 projected operands")
    state_block = 64
    state_pages = (t + state_block - 1) // state_block
    state = torch.zeros(
        (b * state_pages, state_block, 2 * width), device=kv.device, dtype=torch.float32
    )
    positions = torch.arange(t, device=kv.device, dtype=torch.int64).repeat(b)
    owners = torch.arange(b, device=kv.device, dtype=torch.int32).repeat_interleave(t)
    slots = positions + owners.long() * state_pages * state_block
    table = torch.arange(b * state_pages, device=kv.device, dtype=torch.int32).reshape(
        b, state_pages
    )
    kblock = 64
    kpages = (n + kblock - 1) // kblock
    page_bytes = ((kblock * 584 + 575) // 576) * 576
    packed = torch.zeros((b * kpages, page_bytes), device=kv.device, dtype=torch.uint8)
    kcache = packed.as_strided((b * kpages, kblock, 576), (page_bytes, 576, 1))
    ktable = torch.arange(b * kpages, device=kv.device, dtype=torch.int32).reshape(b, kpages)
    kslots = positions // ratio + owners.long() * kpages * kblock
    metadata = CompressorMetadata(block_table=ktable, slot_mapping=kslots, block_size=kblock)
    flat_kv = kv.permute(1, 0, 2).reshape(-1, width).contiguous()
    flat_score = score.permute(1, 0, 2).reshape(-1, width).contiguous()
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=flat_kv,
        score=flat_score,
        ape=ape,
        positions=positions,
        state_cache=state,
        slot_mapping=slots,
        block_size=state_block,
        state_width=width,
        compress_ratio=ratio,
        pdl_kwargs={},
    )
    extra = {}
    function = compress_norm_rope_store_triton
    if ratio == 128:
        function = compress_norm_rope_store_two_stage_triton
        extra = dict(
            num_decode_tokens=0,
            compress_scratch=torch.empty((t * b, 512), device=kv.device, dtype=torch.float32),
        )
    function(
        state_cache=state,
        num_actual=t * b,
        token_to_req_indices=owners,
        positions=positions,
        slot_mapping=slots,
        block_table=table,
        block_size=state_block,
        state_width=width,
        cos_sin_cache=cache,
        kv_cache=kcache,
        k_cache_metadata=metadata,
        pdl_kwargs={},
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=ratio,
        overlap=ratio == 4,
        use_fp4_cache=False,
        rms_norm_weight=weight,
        rms_norm_eps=eps,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        **extra,
    )
    out = torch.empty((b, n, 512), device=kv.device, dtype=torch.bfloat16)
    lengths = torch.full((b,), n, device=kv.device, dtype=torch.int32)
    dequantize_and_gather_k_cache(out, packed, lengths, None, ktable, kblock, 0, use_fnuz=False)
    return out.permute(1, 0, 2).contiguous()


class _Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score, ape, weight, cache, ratio, eps):
        out = _forward(kv, score, ape, weight, cache, ratio, eps)
        ctx.save_for_backward(kv, score, ape, weight, cache)
        ctx.ratio, ctx.eps = ratio, eps
        return out

    @staticmethod
    def backward(ctx, dy):
        kv, score, ape, weight, cache = ctx.saved_tensors
        with torch.enable_grad():
            operands = [x.detach().requires_grad_(True) for x in (kv, score, ape, weight)]
            result = _functional(*operands, cache, ctx.ratio, ctx.eps)
            gradients = torch.autograd.grad(result, operands, dy.float())
        return *gradients, None, None, None


def compressed_kv(kv, score, ape, weight, cache, ratio, eps=1e-6):
    if ratio not in (4, 128) or kv.shape[0] < ratio:
        raise ValueError("main compressor codec requires at least one original C4/C128 group")
    return _Compress.apply(kv, score, ape, weight, cache, ratio, eps)
