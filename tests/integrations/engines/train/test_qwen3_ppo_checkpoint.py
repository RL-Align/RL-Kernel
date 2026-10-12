# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.qwen3_checkpoint import (
    hf_to_megatron,
    megatron_to_hf,
)


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
