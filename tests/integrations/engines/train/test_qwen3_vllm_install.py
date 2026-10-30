# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.ppo_qwen3 import (
    expected_shard,
    verify_acknowledgments,
)


def test_missing_duplicate_and_stale_rank_confirmations_fail():
    """Acks pass with one current entry per rank; missing, duplicate or stale ones raise."""
    acks = [dict(rank=r, tp=2, version=1, verified_tensors=12) for r in range(2)]
    verify_acknowledgments(acks, version=1, tp=2)
    for bad in (acks[:1], [acks[0], acks[0]], [acks[0], dict(acks[1], version=0)]):
        with pytest.raises(ValueError):
            verify_acknowledgments(bad, version=1, tp=2)


def test_qkv_shards_are_contiguous_q_k_v_not_megatron_grouped_order():
    """A vLLM fused QKV shard is the rank's Q, K and V row slices concatenated in that order."""
    weights = {
        f"model.layers.0.self_attn.{n}_proj.weight": torch.arange(rows * 8).reshape(rows, 8)
        + offset
        for n, rows, offset in [("q", 8, 0), ("k", 4, 100), ("v", 4, 200)]
    }
    actual = expected_shard(
        "model.layers.0.self_attn.qkv_proj.weight", weights, rank=1, tp=2, shape=(8, 8)
    )
    assert torch.equal(
        actual,
        torch.cat(
            [
                weights[f"model.layers.0.self_attn.{n}_proj.weight"][rows // 2 :]
                for n, rows in [("q", 8), ("k", 4), ("v", 4)]
            ]
        ),
    )


def test_parameter_without_checkpoint_source_names_the_parameter():
    """``expected_shard`` raises ``ValueError`` naming a vLLM parameter with no source tensor."""
    weights = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8)}
    for name in (
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
    ):
        with pytest.raises(ValueError, match="no checkpoint tensor maps to vLLM parameter"):
            expected_shard(name, weights, rank=0, tp=2, shape=(8, 8))
