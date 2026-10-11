# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Eager vLLM paged-KV adapter for the shared D=256 causal attention core.

The scheduler owns physical blocks and slot mappings. This backend neither
reconstructs request IDs nor retains a separate Python KV cache. Capability
registration remains conservative until the real scheduler matrix passes.
"""

from dataclasses import dataclass

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)

from rl_engine.models.qwen3_next.qwen3_next_forward import shared_attention


@dataclass
class SharedAttentionMetadata(AttentionMetadata):
    query_start_loc: tuple[int, ...]
    seq_lens: tuple[int, ...]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int


def validate_boundaries(starts, lengths, num_actual_tokens):
    if len(starts) != len(lengths) + 1 or not starts or starts[0] != 0:
        raise ValueError("Attention boundaries must start at zero and cover every request")
    if starts[-1] != num_actual_tokens or num_actual_tokens < 0:
        raise ValueError("Attention boundaries must exactly cover actual scheduled tokens")
    for start, end, length in zip(starts[:-1], starts[1:], lengths, strict=True):
        if end < start or length < end - start or length > 1040:
            raise ValueError("Invalid Qwen eager query/prefix length")


class SharedAttentionBuilder(AttentionMetadataBuilder[SharedAttentionMetadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        parallel = vllm_config.parallel_config
        topology = (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.prefill_context_parallel_size,
        )
        if topology != (4, 1, 1, 1):
            raise ValueError("Shared Qwen attention requires TP4/PP1/CP1")
        if not vllm_config.model_config.enforce_eager or vllm_config.speculative_config is not None:
            raise ValueError("Shared Qwen attention requires eager execution without speculation")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Prefix reuse is outside the fixed Qwen eager profile")

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len != 0:
            raise ValueError("Shared-prefix attention is outside the validated profile")
        common = common_attn_metadata
        if isinstance(common.causal, torch.Tensor):
            causal = bool(common.causal.all())
        else:
            causal = common.causal is True
        if not causal:
            raise ValueError("Only causal Qwen attention is supported")
        # Use exact device lengths, not optimistic CPU bounds from spec decode.
        starts = tuple(common.query_start_loc[: common.num_reqs + 1].cpu().tolist())
        lengths = tuple(common.seq_lens[: common.num_reqs].cpu().tolist())
        validate_boundaries(starts, lengths, common.num_actual_tokens)
        return SharedAttentionMetadata(
            starts,
            lengths,
            common.block_table_tensor[: common.num_reqs],
            common.slot_mapping[: common.num_actual_tokens],
            common.num_actual_tokens,
        )


class SharedAttentionImpl(AttentionImpl[SharedAttentionMetadata]):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        **kwargs,
    ):
        if (num_heads, head_size, num_kv_heads, scale) != (4, 256, 1, 1 / 16):
            raise ValueError("Shared Qwen attention requires four Q heads and one D256 KV head")
        unsupported = (
            alibi_slopes is not None,
            sliding_window is not None,
            logits_soft_cap not in (None, 0),
            attn_type != AttentionType.DECODER,
            kv_sharing_target_layer_name is not None,
            kv_cache_dtype not in ("auto", "bfloat16"),
        )
        if any(unsupported) or kwargs:
            raise ValueError("Unsupported extension to the fixed Qwen attention profile")
        self.num_heads, self.head_size, self.num_kv_heads = 4, 256, 1
        self.scale, self.kv_cache_dtype = scale, kv_cache_dtype

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Quantized output is outside the BF16 eager profile")
        if attn_metadata is None:
            # vLLM allocation profiling has no requests or valid cache to read.
            return output.zero_()
        metadata = attn_metadata
        n = metadata.num_actual_tokens
        validate_boundaries(metadata.query_start_loc, metadata.seq_lens, n)
        if any(
            t.dtype != torch.bfloat16 or t.device != query.device
            for t in (query, key, value, kv_cache, output)
        ):
            raise ValueError("Q/K/V/cache/output must be BF16 on the same CUDA device")
        if query.device.type != "cuda":
            raise ValueError("Shared Qwen attention executes on CUDA")
        if query.shape[1:] != (4, 256) or key.shape[1:] != (1, 256) or value.shape != key.shape:
            raise ValueError("Invalid projected Q/K/V shape")
        # vLLM 0.30 materializes logical cache [blocks, KV heads, tokens, K+V].
        if kv_cache.ndim != 4 or kv_cache.shape[1] != 1 or kv_cache.shape[-1] != 512:
            raise ValueError("Unexpected vLLM cache layout; no layout guessing is permitted")
        if min(query.shape[0], key.shape[0], output.shape[0]) < n:
            raise ValueError("Projected tensors do not cover every scheduled token")
        slots = metadata.slot_mapping.to(device=query.device, dtype=torch.long)
        if slots.shape != (n,) or slots.unique().numel() != n:
            raise ValueError("Active KV write slots must be unique and match scheduled tokens")
        block_size = kv_cache.shape[2]
        if bool(((slots < 0) | (slots >= kv_cache.shape[0] * block_size)).any()):
            raise ValueError("Active KV write slot is null or outside the allocated cache")
        plans = []
        for row, (start, end, length) in enumerate(
            zip(
                metadata.query_start_loc[:-1],
                metadata.query_start_loc[1:],
                metadata.seq_lens,
                strict=True,
            )
        ):
            if start == end:
                continue
            needed = (length + block_size - 1) // block_size
            physical = metadata.block_table[row, :needed].to(device=query.device, dtype=torch.long)
            if physical.numel() != needed or physical.unique().numel() != needed:
                raise ValueError("Request KV blocks are incomplete or repeated")
            if bool(((physical < 0) | (physical >= kv_cache.shape[0])).any()):
                raise ValueError("Request KV block is outside the allocated cache")
            logical = torch.arange(length, device=query.device)
            physical_slots = physical[logical // block_size] * block_size + logical % block_size
            if not torch.equal(physical_slots[-(end - start) :], slots[start:end]):
                raise ValueError("Block table and write slots disagree about absolute positions")
            plans.append((start, end, length, physical))
        if any(not torch.isfinite(tensor[:n]).all() for tensor in (query, key, value)):
            raise ValueError("Projected attention inputs must be finite")
        # Validate every request before mutating any scheduler-owned cache slot.
        blocks, offsets = slots // block_size, slots % block_size
        kv_cache[blocks, 0, offsets, :256] = key[:n, 0]
        kv_cache[blocks, 0, offsets, 256:] = value[:n, 0]
        output.zero_()
        for start, end, length, physical in plans:
            cached = kv_cache[physical, 0].reshape(-1, 512)[:length]
            if not torch.isfinite(cached).all():
                raise ValueError("Request KV prefix contains nonfinite values")
            k = cached[:, :256].reshape(1, 1, length, 256).contiguous()
            v = cached[:, 256:].reshape(1, 1, length, 256).contiguous()
            q = query[start:end].transpose(0, 1).unsqueeze(0).contiguous()
            attended = shared_attention(q, k, v)
            output[start:end].copy_(attended.squeeze(0).transpose(0, 1))
        return output


class SharedAttentionBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        # Attention.__init__ indexes AttentionBackendEnum with this value.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return SharedAttentionImpl

    @staticmethod
    def get_builder_cls():
        return SharedAttentionBuilder

    @classmethod
    def get_supported_head_sizes(cls):
        return [256]

    @classmethod
    def supports_kv_connector(cls):
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return False

    @classmethod
    def supports_batch_invariance(cls):
        from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.profile import (
            profile_is_certified,
        )

        return profile_is_certified()


def register_shared_attention_backend():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention"
        ".SharedAttentionBackend",
    )
