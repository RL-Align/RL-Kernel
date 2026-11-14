# SPDX-License-Identifier: Apache-2.0
"""Original DSV4 Q norm/RoPE and FP8-DS-MLA KV codec, with explicit STE VJP."""

from contextlib import nullcontext

import torch

from rl_engine.backends.rocm.norm.qnorm_fp8 import rmsnorm_vjp


def inverse_rope_vjp(grad, positions, cache):
    """Transpose of GPT-J rotation, FP32 products before the final gradient cast."""
    cache_rows = cache.index_select(0, positions.reshape(-1)).reshape(*positions.shape, 64)
    cos, sin = cache_rows.chunk(2, -1)
    if grad.ndim == 4:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    dy = grad.float()
    tail = dy[..., 448:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    inverse = torch.stack((even * cos + odd * sin, odd * cos - even * sin), -1).flatten(-2)
    return torch.cat((dy[..., :448], inverse), -1)


class _QKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, kv, positions, cache, epsilon):
        # Register and execute the exact primitive used by the loaded ROCm vLLM.
        import vllm._custom_ops  # noqa: F401
        from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache

        t, b, h, d = query.shape
        if d != 512 or kv.shape != (t, b, 512) or positions.shape != (t, b):
            raise ValueError("initial QKV codec requires ordinary [T,B,H,512] geometry")
        if query.dtype != kv.dtype or query.dtype != torch.bfloat16 or cache.dtype != torch.float32:
            raise ValueError("QKV codec requires original BF16 operands and FP32 rotary cache")
        block = 64
        pages = (t + block - 1) // block
        page_bytes = ((block * 584 + 575) // 576) * 576
        packed = torch.zeros((b * pages, page_bytes), dtype=torch.uint8, device=query.device)
        offsets = torch.arange(t, device=query.device, dtype=torch.int64)[:, None]
        slots = offsets + torch.arange(b, device=query.device, dtype=torch.int64)[None, :] * (
            pages * block
        )
        flat_q = query.reshape(-1, h, d).contiguous()
        flat_kv = kv.reshape(-1, 512).contiguous()
        q = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            flat_q,
            flat_kv,
            packed,
            slots.reshape(-1),
            positions.reshape(-1).contiguous(),
            cache,
            h,
            epsilon,
            block,
        )
        gathered = torch.empty((b, t, 512), dtype=torch.bfloat16, device=query.device)
        lengths = torch.full((b,), t, dtype=torch.int32, device=query.device)
        table = torch.arange(b * pages, dtype=torch.int32, device=query.device).reshape(b, pages)
        dequantize_and_gather_k_cache(
            gathered, packed, lengths, None, table, block, 0, use_fnuz=True
        )
        ctx.save_for_backward(query, positions, cache)
        ctx.epsilon = epsilon
        return q.reshape(t, b, h, d), gathered.permute(1, 0, 2).contiguous()

    @staticmethod
    def backward(ctx, dq, dkv):
        query, positions, cache = ctx.saved_tensors
        unrotated = inverse_rope_vjp(dq, positions, cache)
        ones = torch.ones(512, dtype=query.dtype, device=query.device)
        dx, _ = rmsnorm_vjp(query, ones, unrotated, ctx.epsilon)
        # Quantized-cache derivative is explicitly identity STE, then inverse RoPE.
        dk = inverse_rope_vjp(dkv, positions, cache).to(query.dtype)
        return dx, dk, None, None, None


class DSV4QKVPreprocess(torch.nn.Module):
    profile = "dsv4-vllm-qnorm-rope-fp8dsmla-gfx942-v1"

    def __init__(self, config, compression_ratio):
        super().__init__()
        from vllm.config import VllmConfig, get_current_vllm_config_or_none, set_current_vllm_config
        from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

        context = (
            nullcontext()
            if get_current_vllm_config_or_none() is not None
            else set_current_vllm_config(VllmConfig())
        )
        with context, torch.device("cuda"):
            rope = build_deepseek_v4_rope(
                config,
                head_dim=512,
                rope_head_dim=64,
                max_position_embeddings=config.max_position_embeddings,
                compress_ratio=max(1, compression_ratio),
            )
        self.register_buffer("cos_sin_cache", rope.cos_sin_cache, persistent=False)
        self.epsilon = config.rms_norm_eps

    def _apply(self, fn, recurse=True):
        # Native Float16Module casts floating model buffers to BF16. Rotary
        # tables are FP32 operands of the production fused codec, not weights.
        cache = self._buffers.pop("cos_sin_cache")
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            self.register_buffer("cos_sin_cache", cache, persistent=False)
        return self

    def forward(self, query, kv, positions):
        if self.cos_sin_cache.device != query.device:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        out = _QKV.apply(query, kv, positions, self.cos_sin_cache, self.epsilon)
        if not hasattr(self, "_executed"):
            print(
                "RLK_QKV_PREPROCESS_EXECUTED",
                __file__,
                self.profile,
                "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
                flush=True,
            )
            self._executed = True
        return out
