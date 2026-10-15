# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A real vLLM configuration for the Qwen3-Next TP4 profile.

Built with vLLM's own config classes -- ``ModelConfig`` reads a real Hugging
Face config written from the official model's dimensions, ``ParallelConfig``
validates the world size against the visible GPUs -- so nothing here is a stand-in
object. It needs no weights, but TP4 needs four visible GPUs, like the other
multi-GPU Qwen3-Next checks.
"""

from __future__ import annotations

import tempfile

import torch
from transformers import Qwen3NextConfig
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def official_hf_config() -> Qwen3NextConfig:
    """The dimensions of Qwen/Qwen3-Next-80B-A3B-Instruct."""
    return Qwen3NextConfig(
        hidden_size=2048,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=151936,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        layer_types=[
            "full_attention" if index % 4 == 3 else "linear_attention" for index in range(48)
        ],
        rope_parameters={
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )


def real_vllm_config(ep: int = 1) -> VllmConfig:
    """The certified profile as a real ``VllmConfig`` with expert parallel size ``ep`` (1 or 4)."""
    if torch.cuda.device_count() < 4:
        raise AssertionError("Four allocated GPUs are required")
    directory = tempfile.mkdtemp(prefix="qwen3-next-config-")
    official_hf_config().save_pretrained(directory)
    config = VllmConfig(
        model_config=ModelConfig(
            model=directory, skip_tokenizer_init=True, dtype="bfloat16", enforce_eager=True
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=4,
            enable_expert_parallel=ep == 4,
            expert_placement_strategy="linear",
        ),
        # The cache dtypes of the certified engine arguments (qwen3_next_native_rollout).
        cache_config=CacheConfig(
            enable_prefix_caching=False,
            mamba_cache_dtype="bfloat16",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode="none",
        ),
        kernel_config=KernelConfig(moe_backend="triton"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    config.additional_config = {
        "qwen3_next_shared_identity": {"topology": {"tp": 4, "pp": 1, "cp": 1, "ep": ep}}
    }
    return config
