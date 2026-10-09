# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Official HF interleaving and TP shard round trips, without a GPU dependency."""

import pytest
import torch

from rl_engine.integrations.qwen3_next_tp_gdn import (
    _GLOBAL_SHAPES,
    assemble_gdn_weights,
    shard_gdn_weights,
)
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal


@pytest.fixture(scope="module")
def weights():
    return {
        name: torch.arange(torch.Size(shape).numel(), dtype=torch.int32)
        .remainder(251)
        .to(torch.bfloat16)
        .reshape(shape)
        for name, shape in _GLOBAL_SHAPES.items()
    }


def test_official_interleaving_survives_tp4_round_trip(weights):
    shards = [shard_gdn_weights(weights, rank) for rank in range(4)]
    restored = assemble_gdn_weights(shards)
    for name, tensor in weights.items():
        assert_tensor_bitwise_equal(restored[name], tensor, name=name)
    for rank, shard in enumerate(shards):
        # Conv is Q/K/V segmented, unlike the GQA-interleaved qkvz projection.
        assert_tensor_bitwise_equal(
            shard["conv1d.weight"][:512], weights["conv1d.weight"][rank * 512 : (rank + 1) * 512]
        )
        assert_tensor_bitwise_equal(
            shard["conv1d.weight"][512:1024],
            weights["conv1d.weight"][2048 + rank * 512 : 2048 + (rank + 1) * 512],
        )
        assert_tensor_bitwise_equal(
            shard["in_proj_qkvz.weight"],
            weights["in_proj_qkvz.weight"][rank * 3072 : (rank + 1) * 3072],
        )


@pytest.mark.parametrize("rank", [-1, 4, True, 1.0])
def test_invalid_tp_rank_is_rejected(weights, rank):
    with pytest.raises(ValueError, match="TP4 rank"):
        shard_gdn_weights(weights, rank)


def test_divergent_replicated_norm_is_not_silently_exported(weights):
    shards = [shard_gdn_weights(weights, rank) for rank in range(4)]
    shards[1]["norm.weight"] = shards[1]["norm.weight"].clone()
    shards[1]["norm.weight"][0] = -1
    with pytest.raises(ValueError, match="Replicated"):
        assemble_gdn_weights(shards)


def test_missing_hf_weight_is_rejected(weights):
    incomplete = {name: value for name, value in weights.items() if name != "dt_bias"}
    with pytest.raises(ValueError, match="weight names"):
        shard_gdn_weights(incomplete, 0)
