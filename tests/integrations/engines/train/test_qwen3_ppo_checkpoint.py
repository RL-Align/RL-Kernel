# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.integrations.engines.train.megatron.qwen3_checkpoint import (
    hf_to_megatron,
    megatron_to_hf,
)
from tests.distributed.tp.check_tp_ppo_frameworks import snapshot
from tests.integrations.common.test_tp_scalar import _init


def fixture():
    c = dict(
        model_type="qwen3",
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        head_dim=2,
        intermediate_size=12,
        vocab_size=17,
        num_hidden_layers=2,
    )
    shapes = {
        "model.embed_tokens.weight": (17, 8),
        "lm_head.weight": (17, 8),
        "model.norm.weight": (8,),
    }
    for i in range(2):
        for n, shape in {
            "input_layernorm.weight": (8,),
            "post_attention_layernorm.weight": (8,),
            "self_attn.q_norm.weight": (2,),
            "self_attn.k_norm.weight": (2,),
            "self_attn.q_proj.weight": (8, 8),
            "self_attn.k_proj.weight": (4, 8),
            "self_attn.v_proj.weight": (4, 8),
            "self_attn.o_proj.weight": (8, 8),
            "mlp.gate_proj.weight": (12, 8),
            "mlp.up_proj.weight": (12, 8),
            "mlp.down_proj.weight": (8, 12),
        }.items():
            shapes[f"model.layers.{i}." + n] = shape
    gen = torch.Generator().manual_seed(33)
    return c, {n: torch.randn(shape, generator=gen) for n, shape in shapes.items()}


@pytest.mark.parametrize("tp", [1, 2])
def test_checkpoint_roundtrip_preserves_all_real_tensor_slots(tp):
    c, state = fixture()
    shards = [hf_to_megatron(state, c, tp=tp, rank=r) for r in range(tp)]
    restored = megatron_to_hf(shards, c)
    assert restored.keys() == state.keys()
    for name in state:
        assert torch.equal(restored[name], state[name]), name


def test_gqa_group_layout_and_gate_up_stride_are_independently_correct():
    c, state = fixture()
    s = hf_to_megatron(state, c, tp=2, rank=1)
    prefix = "model.layers.0."
    expect = torch.cat(
        [
            state[prefix + "self_attn.q_proj.weight"][4:],
            state[prefix + "self_attn.k_proj.weight"][2:],
            state[prefix + "self_attn.v_proj.weight"][2:],
        ]
    )
    assert torch.equal(s["decoder.layers.0.self_attention.linear_qkv.weight"], expect)
    assert torch.equal(
        s["decoder.layers.0.mlp.linear_fc1.weight"][:6],
        state[prefix + "mlp.gate_proj.weight"][6:],
    )
    assert torch.equal(
        s["decoder.layers.0.mlp.linear_fc1.weight"][6:],
        state[prefix + "mlp.up_proj.weight"][6:],
    )
    assert torch.count_nonzero(s["output_layer.weight"][-1]) == 0


def test_replicated_norm_disagreement_is_rejected():
    c, state = fixture()
    shards = [hf_to_megatron(state, c, tp=2, rank=r) for r in range(2)]
    shards[1]["decoder.layers.1.self_attention.q_layernorm.weight"].add_(1)
    with pytest.raises(ValueError, match="replicated weights"):
        megatron_to_hf(shards, c)


def _snapshot_worker(rank, size, rendezvous):
    """Exercise real Gloo gathering with Core 0.15.3's stride-one FC1 annotations."""
    _init(rank, size, rendezvous, "gloo")
    try:
        config, state = fixture()
        for dtype in (torch.float32, torch.bfloat16):
            expected = {name: value.to(dtype) for name, value in state.items()}
            shard = hf_to_megatron(expected, config, tp=size, rank=rank)
            model = torch.nn.Module()
            for name, value in shard.items():
                parts = name.split(".")
                module = model
                for part in parts[:-1]:
                    if part not in module._modules:
                        module.add_module(part, torch.nn.Module())
                    module = module._modules[part]
                parameter = torch.nn.Parameter(value.clone())
                axis = None
                if name in (
                    "embedding.word_embeddings.weight",
                    "output_layer.weight",
                ) or (name.endswith(("linear_qkv.weight", "linear_fc1.weight"))):
                    axis = 0
                elif name.endswith(("linear_proj.weight", "linear_fc2.weight")):
                    axis = 1
                if axis is not None:
                    parameter.tensor_model_parallel = True
                    parameter.partition_dim = axis
                    parameter.partition_stride = 1
                parameter.grad = value * 2
                module.register_parameter(parts[-1], parameter)
            for layer in model.decoder.layers.children():
                layer.mlp.config = SimpleNamespace(gated_linear_unit=True)

            for gradients in (False, True):
                gathered = snapshot(model, dist.group.WORLD, gradients=gradients)
                # This is the exact mapping consumed by VllmPPOClient.install.
                restored = megatron_to_hf([gathered], config)
                for name, value in expected.items():
                    assert torch.equal(restored[name], value * (2 if gradients else 1)), name

            for layer in model.decoder.layers.children():
                layer.mlp.config.gated_linear_unit = False
            gathered = snapshot(model, dist.group.WORLD)
            name = "decoder.layers.0.mlp.linear_fc1.weight"
            plain_shards = [
                hf_to_megatron(expected, config, tp=size, rank=r)[name] for r in range(size)
            ]
            assert torch.equal(gathered[name], torch.cat(plain_shards, dim=0))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2])
def test_actor_snapshot_preserves_gate_up_weights_and_gradients(tmp_path, tp):
    """TP exports retain every HF tensor in FP32/BF16 and leave ungated FC1 unchanged."""
    mp.start_processes(
        _snapshot_worker,
        args=(tp, str(tmp_path / "snapshot")),
        nprocs=tp,
        start_method="spawn",
    )


def test_reference_report_preserves_loading_diagnostics():
    from examples.post_training.tp_ppo.calibrate_qwen3 import report_json_default

    report = {
        "status": "failed",
        "error": "incomplete checkpoint",
        "loading": {"missing_keys": {"b", "a"}, "unexpected_keys": set()},
    }
    restored = json.loads(json.dumps(report, default=report_json_default))
    assert restored["loading"] == {"missing_keys": ["a", "b"], "unexpected_keys": []}
    assert restored["status"] == "failed"
    assert restored["error"] == "incomplete checkpoint"
    with pytest.raises(TypeError):
        json.dumps({"invalid": object()}, default=report_json_default)


def test_independent_hf_targets_and_losses_match_handled_termination():
    from examples.post_training.tp_ppo.fixed_contract import CONFIG
    from examples.post_training.tp_ppo.hf_ppo_reference import independent_loss, independent_targets
    from rl_engine.integrations.common.ppo import build_ppo_batch, ppo_policy_loss, ppo_value_loss
    from tests.integrations.common.test_ppo import rollout_fixture

    rollout = rollout_fixture()
    values = (
        torch.arange(rollout.input_ids.numel(), dtype=torch.float32).reshape_as(rollout.input_ids)
        * 0.03
    )
    reference = torch.full_like(rollout.behavior_logps, -0.4)
    actual = independent_targets(rollout.packet(), values, reference)
    expected = build_ppo_batch(
        rollout,
        state_values=values,
        rewards=actual["rewards"],
        reference_logps=reference,
        config=CONFIG,
    )
    for name, value in expected.targets().items():
        torch.testing.assert_close(actual[name], value)
    torch.testing.assert_close(
        independent_loss("actor", reference, rollout.packet(), actual),
        ppo_policy_loss(reference, expected, CONFIG.policy_clip),
    )
    torch.testing.assert_close(
        independent_loss("critic", values, rollout.packet(), actual),
        ppo_value_loss(values[:, :-1], expected, CONFIG.value_clip),
    )


def test_independent_hf_targets_reject_a_row_without_action_tokens():
    from examples.post_training.tp_ppo.hf_ppo_reference import independent_targets
    from tests.integrations.common.test_ppo import rollout_fixture

    rollout = rollout_fixture()
    packet = dict(rollout.packet())
    packet["action_mask"] = packet["action_mask"].clone()
    packet["action_mask"][0] = False
    values = torch.zeros_like(rollout.input_ids, dtype=torch.float32)
    with pytest.raises(ValueError, match="at least one action token"):
        independent_targets(packet, values, torch.zeros_like(rollout.behavior_logps))


def test_fixed_evidence_rejects_corruption_and_inventory_shape(tmp_path):
    from examples.post_training.tp_ppo.fixed_contract import read_tensor, write_tensor

    row = write_tensor(tmp_path, "gradient", torch.arange(4.0))
    torch.testing.assert_close(read_tensor(tmp_path, row), torch.arange(4.0))
    row["shape"] = [2, 2]
    with pytest.raises(ValueError, match="invalid tensor"):
        read_tensor(tmp_path, row)
    (tmp_path / row["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity changed"):
        read_tensor(tmp_path, row)


def test_scalar_checkpoint_mapping_never_requires_an_lm_head():
    config, state = fixture()
    shard = hf_to_megatron(state, config, tp=1, rank=0)
    del shard["output_layer.weight"]
    restored = megatron_to_hf([shard], config, policy=False)
    assert "lm_head.weight" not in restored
    for name, value in restored.items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
