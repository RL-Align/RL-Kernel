# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU layout checks for the native distributed Qwen3 capture adapter."""

from types import SimpleNamespace as NS

import pytest
import torch

from rl_engine.alignment.debug.qwen3_layout import layer_parameters
from rl_engine.alignment.debug import qwen3_runtime
from examples.debug.qwen3_parallel import PARAMETER_NAMES, load_megatron, matrix_axes, pair_lanes


def fixture_layer(rank, world):
    hidden, heads, groups, head, ffn = 64, 8, 4, 64, 512

    def weight(rows, cols):
        return torch.arange(rows * cols).reshape(rows, cols).float()

    full = {
        "q": weight(heads * head, hidden),
        "k": weight(groups * head, hidden) + 1,
        "v": weight(groups * head, hidden) + 2,
        "o": weight(hidden, heads * head),
        "gate": weight(ffn, hidden),
        "up": weight(ffn, hidden) + 3,
        "down": weight(hidden, ffn),
        "attention_norm": torch.arange(hidden).float(),
        "mlp_norm": torch.arange(hidden).float() + 1,
        "q_norm": torch.arange(head).float(),
        "k_norm": torch.arange(head).float() + 1,
    }
    local = {
        name: (
            value
            if value.ndim == 1
            else value.chunk(world, dim=1 if name in ("o", "down") else 0)[rank]
        )
        for name, value in full.items()
    }
    packed = torch.cat(
        [local[name].reshape(groups // world, -1, head, hidden) for name in ("q", "k", "v")],
        dim=1,
    ).flatten(0, 2)
    gate_up = torch.cat([local["gate"], local["up"]])
    training = NS(
        config=NS(kv_channels=head, num_query_groups=groups, num_attention_heads=heads),
        self_attention=NS(
            linear_qkv=NS(weight=packed, layer_norm_weight=local["attention_norm"]),
            linear_proj=NS(weight=local["o"]),
            q_layernorm=NS(weight=local["q_norm"]),
            k_layernorm=NS(weight=local["k_norm"]),
        ),
        mlp=NS(
            linear_fc1=NS(weight=gate_up, layer_norm_weight=local["mlp_norm"]),
            linear_fc2=NS(weight=local["down"]),
        ),
    )
    rollout = NS(
        self_attn=NS(
            head_dim=head,
            total_num_kv_heads=groups,
            total_num_heads=heads,
            qkv_proj=NS(weight=torch.cat([local[name] for name in ("q", "k", "v")])),
            o_proj=NS(weight=local["o"]),
            q_norm=NS(weight=local["q_norm"]),
            k_norm=NS(weight=local["k_norm"]),
        ),
        mlp=NS(gate_up_proj=NS(weight=gate_up), down_proj=NS(weight=local["down"])),
        input_layernorm=NS(weight=local["attention_norm"]),
        post_attention_layernorm=NS(weight=local["mlp_norm"]),
    )
    return full, training, rollout


@pytest.mark.parametrize("world", [1, 2, 4])
def test_shared_qwen3_layout_and_legacy_tiles_agree(world, monkeypatch):
    train_tiles, rollout_tiles = {}, {}
    pieces = {}
    for rank in range(world):
        full, training, rollout = fixture_layer(rank, world)
        left = layer_parameters(training, "training", rank, world)
        right = layer_parameters(rollout, "rollout", rank, world)
        for name in full:
            assert (left[name].row, left[name].col, left[name].axis) == (
                right[name].row,
                right[name].col,
                right[name].axis,
            )
            assert torch.equal(left[name].value, right[name].value)
            pieces.setdefault(name, []).append(left[name])
        monkeypatch.setattr(
            qwen3_runtime, "topology_coordinates", lambda side, r=rank: (r, world, 0, 1)
        )
        train_tiles.update(qwen3_runtime.weight_tiles(training, "training"))
        rollout_tiles.update(qwen3_runtime.weight_tiles(rollout, "rollout"))
    assert train_tiles == rollout_tiles
    for name, shards in pieces.items():
        reconstructed = (
            shards[0].value
            if shards[0].axis is None
            else torch.cat([shard.value for shard in shards], dim=shards[0].axis)
        )
        assert torch.equal(reconstructed, full[name])


@pytest.mark.parametrize("world", [1, 2, 4])
def test_checkpoint_loader_is_inverse_of_runtime_layout(world):
    full, _, _ = fixture_layer(0, world)
    values = {f"model.layers.0.{PARAMETER_NAMES[name]}": value for name, value in full.items()}
    values.update(
        {
            "model.embed_tokens.weight": torch.arange(128 * 64).float().reshape(128, 64),
            "lm_head.weight": torch.ones(128, 64),
            "model.norm.weight": torch.ones(64),
        }
    )

    class Checkpoint(dict):
        config = {"head_dim": 64, "num_key_value_heads": 4, "hidden_size": 64}

    for rank in range(world):
        _, layer, _ = fixture_layer(rank, world)
        layer.self_attention.linear_qkv.weight.zero_()
        layer.mlp.linear_fc1.weight.zero_()
        model = NS(
            embedding=NS(word_embeddings=NS(weight=torch.empty(128 // world, 64))),
            output_layer=NS(weight=torch.empty(128 // world, 64)),
            decoder=NS(layers=[layer], final_layernorm=NS(weight=torch.empty(64))),
        )
        load_megatron(model, Checkpoint(values), rank, world)
        for name, shard in layer_parameters(layer, "training", rank, world).items():
            expected = (
                full[name] if shard.axis is None else full[name].chunk(world, shard.axis)[rank]
            )
            assert torch.equal(shard.value, expected)


def test_pair_rejects_incompatible_semantic_plans(tmp_path):
    import json

    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for folder, stage in ((left, "before_rope"), (right, "after_rope")):
        manifest = {"schema_version": "rlkernel.events.v1", "boundaries": [{"id": stage}]}
        (folder / "capture-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="boundaries"):
        pair_lanes(left, right, tmp_path / "paired")
    assert not (tmp_path / "paired").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"devices": "0,0"},
        {"devices": ""},
        {"training": "2x0"},
        {"training": "2x3"},
        {"training": "1x1,1x1"},
        {"rollout_tps": "1,-1"},
        {"rollout_tps": "1,1"},
    ],
)
def test_invalid_parallel_matrix_fails_before_launch(changes):
    args = NS(**{"devices": "0,1,2,5", "training": "1x1,2x2", "rollout_tps": "1,2,4", **changes})
    with pytest.raises(ValueError):
        matrix_axes(args)
