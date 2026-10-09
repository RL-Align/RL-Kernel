# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""HF mappings, replica checks and official Qwen3-Next MoE dimensions."""

import pytest
import torch

from rl_engine.integrations import qwen3_next_tp
from rl_engine.integrations import qwen3_next_tp_blocks as blocks
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def pattern(shape):
    return (
        torch.arange(torch.Size(shape).numel(), dtype=torch.int32)
        .remainder(251)
        .to(torch.bfloat16)
        .reshape(shape)
    )


@pytest.mark.parametrize(
    "name",
    [
        "gate.weight",
        "shared_expert_gate.weight",
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "experts.0.gate_proj.weight",
        "experts.0.up_proj.weight",
        "experts.511.down_proj.weight",
    ],
)
def test_moe_hf_weight_roundtrip(name):
    value = pattern(blocks.moe_hf_shapes()[name])
    shards = [blocks.shard_moe_weight(name, value, rank) for rank in range(4)]
    exact(blocks.assemble_moe_weight(name, shards), value, name=name)
    if name.endswith("down_proj.weight"):
        assert shards[0].shape == (2048, 128)
    elif name.endswith(("gate_proj.weight", "up_proj.weight")):
        assert shards[0].shape == (128, 2048)


def test_moe_export_rejects_router_replica_drift():
    values = [pattern((512, 2048)) for _ in range(4)]
    values[2][0, 0] = -1
    with pytest.raises(ValueError, match="Replicated MoE"):
        blocks.assemble_moe_weight("gate.weight", values)


@pytest.mark.parametrize("rank", [-1, 4, True, 0.5])
def test_invalid_shard_rank_rejected(rank):
    with pytest.raises(ValueError, match="TP4 rank"):
        blocks.shard_moe_weight("gate.weight", torch.empty(512, 2048, dtype=torch.bfloat16), rank)


def test_official_shared_expert_width_is_512_and_each_expert_has_three_weights():
    shapes = blocks.moe_hf_shapes()
    assert len(shapes) == 512 * 3 + 5
    assert shapes["shared_expert.gate_proj.weight"] == (512, 2048)
    assert shapes["shared_expert.down_proj.weight"] == (2048, 512)


def test_tp4_moe_rejects_a_missing_or_wrong_size_tp_group(monkeypatch):
    monkeypatch.setattr(qwen3_next_tp.dist, "is_initialized", lambda: False)
    with pytest.raises(ValueError, match="four-rank TP group"):
        blocks.TP4MoE(group=None, device="meta")
    monkeypatch.setattr(qwen3_next_tp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen3_next_tp.dist, "get_world_size", lambda group: 8)
    with pytest.raises(ValueError, match="four-rank TP group"):
        blocks.TP4MoE(group=None, device="meta")


def test_tp4_moe_parameter_ownership(monkeypatch):
    group = object()
    monkeypatch.setattr(qwen3_next_tp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen3_next_tp.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(blocks.dist, "get_rank", lambda group: 2)
    module = blocks.TP4MoE(group=group, device="meta")
    assert module.rank == 2
    assert module.experts.gate_up.shape == (512, 256, 2048)
    assert module.experts.down.shape == (512, 2048, 128)
    assert (
        module.experts.gate_up.partition_dim == 1 and module.experts.gate_up.partition_stride == 2
    )
    assert module.experts.down.partition_dim == 2
    assert module.shared_expert.down_proj.weight.partition_dim == 1
    # Router and shared-expert gate are replicated: no TP attributes, full shapes.
    assert not getattr(module.gate.weight, "tensor_model_parallel", False)
    assert not getattr(module.shared_expert_gate.weight, "tensor_model_parallel", False)
    assert module.gate.weight.shape == (512, 2048)
