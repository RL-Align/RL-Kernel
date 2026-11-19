# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""HF mappings, KV replica ownership and official Qwen3-Next attention/MoE dimensions."""

import pytest
import torch

from rl_engine.models.qwen3_next import qwen3_next_tp
from rl_engine.models.qwen3_next import qwen3_next_tp_blocks as blocks
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def pattern(shape):
    return (
        torch.arange(torch.Size(shape).numel(), dtype=torch.int32)
        .remainder(251)
        .to(torch.bfloat16)
        .reshape(shape)
    )


@pytest.fixture(scope="module")
def attention_weights():
    return {name: pattern(shape) for name, shape in blocks.ATTENTION_HF_SHAPES.items()}


def test_attention_hf_roundtrip_and_kv_pair_layout(attention_weights):
    shards = [blocks.shard_attention_weights(attention_weights, rank) for rank in range(4)]
    for name, value in blocks.assemble_attention_weights(shards).items():
        exact(value, attention_weights[name], name=name)
    for rank, shard in enumerate(shards):
        exact(
            shard["q_proj.weight"],
            attention_weights["q_proj.weight"][rank * 2048 : (rank + 1) * 2048],
        )
        begin = rank // 2 * 256
        exact(shard["k_proj.weight"], attention_weights["k_proj.weight"][begin : begin + 256])


def test_attention_export_rejects_disagreeing_kv_replica(attention_weights):
    shards = [blocks.shard_attention_weights(attention_weights, rank) for rank in range(4)]
    shards[1]["k_proj.weight"] = shards[1]["k_proj.weight"].clone()
    shards[1]["k_proj.weight"][0, 0] = -1
    with pytest.raises(ValueError, match="Replicated KV"):
        blocks.assemble_attention_weights(shards)


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


@pytest.fixture
def fake_tp(monkeypatch):
    group = object()
    monkeypatch.setattr(blocks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(blocks.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(blocks, "_kv_replica_groups", lambda group: (object(), object()))
    return group


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_attention_parameter_ownership_and_actual_hf_storage(
    fake_tp, monkeypatch, rank, attention_weights
):
    monkeypatch.setattr(blocks.dist, "get_rank", lambda group: rank)
    module = blocks.TP4FullAttention(group=fake_tp, device="cpu")
    module.load_hf_weights(attention_weights)
    expected = blocks.shard_attention_weights(attention_weights, rank)
    for name, value in module.export_local_hf_weights().items():
        exact(value, expected[name], name=name)
    assert module.q_proj.weight.tensor_model_parallel
    assert module.o_proj.weight.partition_dim == 1
    assert module.k_proj.weight.shared == (rank % 2 == 1)
    assert module.v_proj.weight.shared == (rank % 2 == 1)
    assert not getattr(module.q_norm.weight, "tensor_model_parallel", False)
    state = module.initial_state(3)
    assert len(state.keys) == len(state.values) == 3
    assert all(value.shape == (1, 1, 0, 256) for value in state.keys)


def test_rope_uses_absolute_positions_and_preserves_unrotated_tail(fake_tp, monkeypatch):
    monkeypatch.setattr(blocks.dist, "get_rank", lambda group: 0)
    module = blocks.TP4FullAttention(group=fake_tp, device="cpu")
    x = torch.randn(8, 4, 256, dtype=torch.bfloat16)
    positions = torch.arange(61, 69)
    whole = module._rotate(x, positions)
    exact(whole[..., 64:], x[..., 64:])
    exact(
        torch.cat([module._rotate(x[:3], positions[:3]), module._rotate(x[3:], positions[3:])]),
        whole,
    )
    assert not torch.equal(module._rotate(x, torch.arange(8)), whole)
