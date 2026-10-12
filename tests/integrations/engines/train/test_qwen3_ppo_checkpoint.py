# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

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
        s["decoder.layers.0.mlp.linear_fc1.weight"][:6], state[prefix + "mlp.gate_proj.weight"][6:]
    )
    assert torch.equal(
        s["decoder.layers.0.mlp.linear_fc1.weight"][6:], state[prefix + "mlp.up_proj.weight"][6:]
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
                if name in ("embedding.word_embeddings.weight", "output_layer.weight") or (
                    name.endswith(("linear_qkv.weight", "linear_fc1.weight"))
                ):
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
