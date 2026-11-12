# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen3 logical parameter views shared by live and portable diagnostics.

Only storage/layout mapping lives here. There are no forward implementations or
framework imports, and none of these views changes numerical execution.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterShard:
    value: torch.Tensor
    row: int = 0
    col: int = 0
    axis: int | None = None


def layer_parameters(instance, side: str, rank: int, world: int) -> dict[str, ParameterShard]:
    if side not in {"training", "rollout"}:
        raise ValueError(f"unknown Qwen3 runtime side: {side}")
    training = side == "training"
    attn = instance.self_attention if training else instance.self_attn
    config = getattr(instance, "config", None)
    head = int(config.kv_channels if training else attn.head_dim)
    groups = int(config.num_query_groups if training else attn.total_num_kv_heads)
    heads = int(config.num_attention_heads if training else attn.total_num_heads)
    local_groups, local_heads = max(1, groups // world), heads // world
    group_rank = rank if groups >= world else rank // (world // groups)
    if training:
        qkv = attn.linear_qkv.weight
        packed = qkv.reshape(local_groups, -1, head, qkv.shape[-1])
        q = packed[:, :-2].reshape(-1, qkv.shape[-1])
        k = packed[:, -2].reshape(-1, qkv.shape[-1])
        v = packed[:, -1].reshape(-1, qkv.shape[-1])
        gate, up = instance.mlp.linear_fc1.weight.chunk(2, dim=0)
        proj, down = attn.linear_proj.weight, instance.mlp.linear_fc2.weight
        norms = {
            "attention_norm": attn.linear_qkv.layer_norm_weight,
            "mlp_norm": instance.mlp.linear_fc1.layer_norm_weight,
            "q_norm": attn.q_layernorm.weight,
            "k_norm": attn.k_layernorm.weight,
        }
    else:
        q, k, v = attn.qkv_proj.weight.split(
            (local_heads * head, local_groups * head, local_groups * head), dim=0
        )
        gate, up = instance.mlp.gate_up_proj.weight.chunk(2, dim=0)
        proj, down = attn.o_proj.weight, instance.mlp.down_proj.weight
        norms = {
            "attention_norm": instance.input_layernorm.weight,
            "mlp_norm": instance.post_attention_layernorm.weight,
            "q_norm": attn.q_norm.weight,
            "k_norm": attn.k_norm.weight,
        }
    return {
        **{name: ParameterShard(value) for name, value in norms.items()},
        "q": ParameterShard(q, row=rank * q.shape[0], axis=0),
        "k": ParameterShard(k, row=group_rank * k.shape[0], axis=0),
        "v": ParameterShard(v, row=group_rank * v.shape[0], axis=0),
        "o": ParameterShard(proj, col=rank * proj.shape[1], axis=1),
        "gate": ParameterShard(gate, row=rank * gate.shape[0], axis=0),
        "up": ParameterShard(up, row=rank * up.shape[0], axis=0),
        "down": ParameterShard(down, col=rank * down.shape[1], axis=1),
    }
