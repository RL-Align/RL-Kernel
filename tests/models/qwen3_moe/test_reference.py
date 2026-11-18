# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe Qwen3.6-35B-A3B layer assembly tests. No official expert allocation."""

from __future__ import annotations

import pytest
import torch

from rl_engine.layers.moe import LayerDims, forward_layer, planned_layer_nodes
from rl_engine.models.qwen3_moe.reference import check_weight_shapes
from rl_engine.models.qwen3_moe.spec import Qwen3MoeSpec, official_config


def _tiny_dims() -> LayerDims:
    return LayerDims(
        hidden_size=16,
        rms_norm_eps=1e-6,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.25,
        rope_theta=10000,
        mrope_section=(1, 0, 0),
    )


def _tiny_weights(dims: LayerDims, layer_type: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    hidden = dims.hidden_size

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    weights = {
        "input_layernorm.weight": torch.zeros(hidden),
        "post_attention_layernorm.weight": torch.zeros(hidden),
        "mlp.gate.weight": randn(dims.num_experts, hidden),
        "mlp.experts.gate_up_proj": randn(dims.num_experts, hidden, 2 * dims.moe_intermediate_size),
        "mlp.experts.down_proj": randn(dims.num_experts, dims.moe_intermediate_size, hidden),
        "mlp.shared_expert.gate_proj.weight": randn(dims.shared_expert_intermediate_size, hidden),
        "mlp.shared_expert.up_proj.weight": randn(dims.shared_expert_intermediate_size, hidden),
        "mlp.shared_expert.down_proj.weight": randn(hidden, dims.shared_expert_intermediate_size),
        "mlp.shared_expert_gate.weight": randn(1, hidden),
    }
    if layer_type == "linear_attention":
        conv = dims.linear_num_key_heads * dims.linear_key_head_dim * 2
        conv += dims.linear_num_value_heads * dims.linear_value_head_dim
        value = dims.linear_num_value_heads * dims.linear_value_head_dim
        weights.update(
            {
                "linear_attn.in_proj_qkv.weight": randn(conv, hidden),
                "linear_attn.in_proj_z.weight": randn(value, hidden),
                "linear_attn.in_proj_a.weight": randn(dims.linear_num_value_heads, hidden),
                "linear_attn.in_proj_b.weight": randn(dims.linear_num_value_heads, hidden),
                "linear_attn.conv1d.weight": randn(conv, 1, dims.linear_conv_kernel_dim),
                "linear_attn.dt_bias": randn(dims.linear_num_value_heads),
                "linear_attn.A_log": randn(dims.linear_num_value_heads),
                "linear_attn.norm.weight": torch.zeros(dims.linear_value_head_dim),
                "linear_attn.out_proj.weight": randn(hidden, value),
            }
        )
    else:
        query = dims.num_attention_heads * dims.head_dim
        key = dims.num_key_value_heads * dims.head_dim
        weights.update(
            {
                "self_attn.q_proj.weight": randn(query * 2, hidden),
                "self_attn.k_proj.weight": randn(key, hidden),
                "self_attn.v_proj.weight": randn(key, hidden),
                "self_attn.o_proj.weight": randn(hidden, query),
                "self_attn.q_norm.weight": torch.zeros(dims.head_dim),
                "self_attn.k_norm.weight": torch.zeros(dims.head_dim),
            }
        )
    return weights


def test_layer_nodes_cover_spec_topology():
    spec = Qwen3MoeSpec.from_config(official_config())
    for index in (0, 3):
        prefix = f"layers.{index}."
        suffixes = tuple(name[len(prefix) :] for name in spec.node_names() if name.startswith(prefix))
        assert planned_layer_nodes(spec.layer_type(index)) == suffixes


def test_linear_layer_stops_before_deferred_operators():
    dims = _tiny_dims()
    weights = _tiny_weights(dims, "linear_attention")
    nodes: dict[str, torch.Tensor] = {}
    result = forward_layer(
        torch.randn(2, 3, dims.hidden_size),
        weights,
        dims,
        "linear_attention",
        nodes=nodes,
    )
    assert result.complete is False
    assert result.executed == (
        "input_layernorm",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    )
    assert result.pending_tasks == (17, 18, 19, 20, 21, 22)
    assert ("conv1d", 21) in result.skipped
    assert ("gdn", 21) in result.skipped
    assert ("expert_gate_up", 17) in result.skipped
    assert ("shared_expert_gate", 20) in result.skipped
    assert "out_proj" in result.blocked
    assert "router" in result.blocked
    assert "gdn" not in nodes
    assert "expert_down" not in nodes
    assert set(nodes) == set(result.executed)


def test_full_attention_reaches_router_and_keeps_gate_gradient():
    dims = _tiny_dims()
    weights = _tiny_weights(dims, "full_attention")
    weights["mlp.gate.weight"].requires_grad_(True)
    nodes: dict[str, torch.Tensor] = {}
    result = forward_layer(
        torch.randn(1, 4, dims.hidden_size),
        weights,
        dims,
        "full_attention",
        nodes=nodes,
    )
    assert result.complete is False
    assert result.executed[-5:] == (
        "residual_attn",
        "post_attention_layernorm",
        "router",
        "topk",
        "dispatch",
    )
    assert "attn_output_gate" in result.executed
    assert result.pending_tasks == (17, 18, 19, 20)
    assert ("expert_swiglu", 17) in result.skipped
    assert ("shared_expert_gate", 20) in result.skipped
    assert "combine" in result.blocked
    assert "router" in nodes
    nodes["router"].sum().backward()
    assert weights["mlp.gate.weight"].grad is not None


def test_mismatched_expert_shape_is_rejected():
    spec = Qwen3MoeSpec.from_config(official_config())
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with pytest.raises(ValueError, match="gate_up_proj"):
        check_weight_shapes(spec, {name: torch.empty(1)})
