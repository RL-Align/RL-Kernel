# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Segmented convolution must survive the engines' actual rank-major TP layout."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.models.qwen3_next.qwen3_next_engine_weights import (
    conv_hf_to_rank_major,
    conv_rank_major_to_hf,
    expert_hf_to_rank_major,
    expert_rank_major_to_hf,
    kv_hf_to_rank_major,
    kv_rank_major_to_hf,
    validate_training_topology,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal


def test_conv_rank_major_shards_preserve_q_k_v_segments():
    generator = torch.Generator().manual_seed(27)
    weight = torch.randn((8192, 1, 4), generator=generator).bfloat16()
    rank_major = conv_hf_to_rank_major(weight)
    segments = weight.split((2048, 2048, 4096))
    for rank, shard in enumerate(rank_major.chunk(4)):
        assert_tensor_bitwise_equal(shard, torch.cat([part.chunk(4)[rank] for part in segments]))
    assert_tensor_bitwise_equal(conv_rank_major_to_hf(rank_major), weight)


@pytest.mark.parametrize(
    "field",
    [
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "sequence_parallel",
        "fp32_residual_connection",
        "hidden_dropout",
        "mtp_num_layers",
    ],
)
def test_training_adapter_rejects_unsupported_topology(field):
    values = dict(
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        fp32_residual_connection=False,
        hidden_dropout=0,
        mtp_num_layers=0,
    )
    values[field] = 2 if field.endswith("size") else 1
    with pytest.raises(ValueError):
        validate_training_topology(
            SimpleNamespace(**values), SimpleNamespace(sequence_parallel=False)
        )


def test_kv_export_rejects_diverged_replica():
    weight = torch.randn(512, 2048).bfloat16()
    packed = kv_hf_to_rank_major(weight)
    assert_tensor_bitwise_equal(kv_rank_major_to_hf(packed), weight)
    packed[256, 0] = packed[256, 0] + 1
    with pytest.raises(AssertionError, match="KV replica pair 0"):
        kv_rank_major_to_hf(packed)


def test_expert_gather_restores_separate_hf_gate_and_up():
    gate, up = (torch.randn(512, 2048).bfloat16() for _ in range(2))
    packed = expert_hf_to_rank_major(gate, up)
    for rank, shard in enumerate(packed.chunk(4)):
        assert_tensor_bitwise_equal(shard[:128], gate.chunk(4)[rank])
        assert_tensor_bitwise_equal(shard[128:], up.chunk(4)[rank])
    restored = expert_rank_major_to_hf(packed)
    assert_tensor_bitwise_equal(restored[0], gate)
    assert_tensor_bitwise_equal(restored[1], up)
