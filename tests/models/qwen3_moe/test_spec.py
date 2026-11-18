# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe Qwen3.6-35B-A3B identity tests. No expert-tensor allocation."""

from __future__ import annotations

import json

import pytest

from rl_engine.models.qwen3_moe.spec import OFFICIAL_LAYER_TYPES, Qwen3MoeSpec, official_config


@pytest.fixture()
def spec():
    return Qwen3MoeSpec.from_config(official_config())


def test_spec_matches_official_fingerprint(spec):
    assert spec.model_id == "Qwen3.6-35B-A3B"
    assert spec.num_hidden_layers == 40
    assert spec.hidden_size == 2048
    assert spec.vocab_size == 248320
    assert spec.head_dim == 256
    assert spec.num_attention_heads == 16
    assert spec.num_key_value_heads == 2
    assert spec.attn_output_gate is True
    assert spec.linear_num_key_heads == 16
    assert spec.linear_num_value_heads == 32
    assert spec.linear_key_head_dim == 128
    assert spec.linear_value_head_dim == 128
    assert spec.linear_conv_kernel_dim == 4
    assert spec.partial_rotary_factor == 0.25
    assert spec.rope_theta == 10000000
    assert spec.mrope_interleaved is True
    assert spec.mrope_section == (11, 11, 10)
    assert spec.num_experts == 256
    assert spec.num_experts_per_tok == 8
    assert spec.moe_intermediate_size == 512
    assert spec.shared_expert_intermediate_size == 512
    assert spec.hidden_act == "silu"
    assert spec.tie_word_embeddings is False
    assert spec.rms_norm_eps == 1e-06
    assert spec.dtype == "bfloat16"
    assert spec.max_position_embeddings == 262144
    assert spec.layer_types == OFFICIAL_LAYER_TYPES
    assert spec.layer_types.count("linear_attention") == 30
    assert spec.layer_types.count("full_attention") == 10
    assert spec.text_forward_includes_vision is False
    assert spec.text_forward_includes_mtp is False


def test_node_names_cover_full_topology(spec):
    names = spec.node_names()
    assert names[0] == "embedding"
    assert names[-4:] == ("final_layernorm", "lm_head", "logprob", "loss")
    assert spec.layer_type(0) == "linear_attention"
    assert spec.layer_type(3) == "full_attention"
    assert spec.layer_type(39) == "full_attention"
    assert "layers.0.gdn" in names
    assert "layers.0.router" in names
    assert "layers.3.attn_output_gate" in names
    assert "layers.39.expert_down" in names
    assert spec.node_kind("layers.0.gdn") == "gdn"
    assert spec.node_kind("layers.0.expert_gate_up") == "grouped_gemm"
    assert spec.node_kind("layers.0.router") == "router"
    assert spec.node_kind("layers.3.attn_output_gate") == "attn_output_gate"
    # 1 embed + 30 linear * 24 + 10 full * 26 + 4 tail
    assert len(names) == 1 + 30 * 24 + 10 * 26 + 4


def test_forbids_dense_config():
    dense = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
    }
    with pytest.raises(ValueError, match="Qwen3ForCausalLM"):
        Qwen3MoeSpec.from_config(dense)


def test_forbids_shrinking_layers():
    config = official_config()
    config["text_config"]["num_hidden_layers"] = 2
    with pytest.raises(ValueError, match="num_hidden_layers"):
        Qwen3MoeSpec.from_config(config)


def test_forbids_shrinking_experts():
    config = official_config()
    config["text_config"]["num_experts"] = 128
    with pytest.raises(ValueError, match="num_experts"):
        Qwen3MoeSpec.from_config(config)


def test_parameter_shapes_match_pinned_layout(spec):
    gate_up = spec.parameter_shape("model.language_model.layers.0.mlp.experts.gate_up_proj")
    q_proj = spec.parameter_shape("model.language_model.layers.3.self_attn.q_proj.weight")
    assert gate_up == (256, 2048, 1024)
    assert q_proj == (8192, 2048)
    missing = spec.missing_parameters([])
    assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in missing
    assert "model.language_model.layers.0.linear_attn.conv1d.weight" in missing
    assert "model.visual.pos_embed.weight" not in spec.parameter_names()
    assert "mtp.norm.weight" not in spec.parameter_names()


def test_missing_operators_names_gdn(spec):
    assert "gdn" in spec.missing_operators([])
    assert "gdn" not in spec.missing_operators(["gdn"])


def test_parameter_names_match_index_weight_map(spec, tmp_path):
    names = spec.parameter_names()
    weight_map = {name: "model-00001-of-00026.safetensors" for name in names}
    weight_map["model.visual.pos_embed.weight"] = "model-00002-of-00026.safetensors"
    weight_map["mtp.norm.weight"] = "model-00026-of-00026.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    loaded = json.loads(index.read_text(encoding="utf-8"))["weight_map"]

    assert spec.missing_parameters(loaded) == ()

    del loaded["lm_head.weight"]
    assert "lm_head.weight" in spec.missing_parameters(loaded)
