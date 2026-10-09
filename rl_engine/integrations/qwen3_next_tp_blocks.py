# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Official Qwen3-Next TP4 MoE block for shared adapters.

The block owns the local arithmetic and differentiable TP boundaries. It does
not register an engine backend or advertise full-model acceptance.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rl_engine.integrations.qwen3_next_forward import shared_linear, shared_moe
from rl_engine.integrations.qwen3_next_tp import (
    _CopyToTP,
    _parallel_parameter,
    _rank,
    _ReduceFromTP,
    _tp_group,
    _weight,
)
from rl_engine.testing.tensor_identity import tensor_bitwise_equal


def moe_hf_shapes():
    result = {
        "gate.weight": (512, 2048),
        "shared_expert_gate.weight": (1, 2048),
    }
    for prefix in ("shared_expert", *(f"experts.{expert}" for expert in range(512))):
        result[f"{prefix}.gate_proj.weight"] = (512, 2048)
        result[f"{prefix}.up_proj.weight"] = (512, 2048)
        result[f"{prefix}.down_proj.weight"] = (2048, 512)
    return result


def shard_moe_weight(name, value, rank):
    """Shard one official HF tensor without materializing all expert weights."""
    _rank(rank)
    shapes = moe_hf_shapes()
    if name not in shapes:
        raise ValueError(f"Unknown Qwen3-Next MoE weight: {name}")
    _weight(value, shapes[name], name)
    if name in ("gate.weight", "shared_expert_gate.weight"):
        return value
    return value.chunk(4, dim=1 if name.endswith("down_proj.weight") else 0)[rank].contiguous()


def assemble_moe_weight(name, values):
    if len(values) != 4:
        raise ValueError("Exactly four MoE weight shards are required")
    shapes = moe_hf_shapes()
    if name not in shapes:
        raise ValueError(f"Unknown Qwen3-Next MoE weight: {name}")
    if name in ("gate.weight", "shared_expert_gate.weight"):
        for value in values:
            _weight(value, shapes[name], name)
        if any(not tensor_bitwise_equal(value, values[0]) for value in values[1:]):
            raise ValueError(f"Replicated MoE weight differs across TP: {name}")
        return values[0]
    axis = 1 if name.endswith("down_proj.weight") else 0
    shape = list(shapes[name])
    shape[axis] //= 4
    for value in values:
        _weight(value, tuple(shape), name)
    return torch.cat(tuple(values), dim=axis)


class TP4MoE(nn.Module):
    """All 512 experts at EP1 with intermediate width 128 per TP rank."""

    def __init__(self, *, group, device):
        super().__init__()
        self.group = _tp_group(group)
        self.rank = dist.get_rank(self.group)
        factory = {"device": device, "dtype": torch.bfloat16}
        self.gate = nn.Linear(2048, 512, bias=False, **factory)
        self.experts = nn.Module()
        self.experts.register_parameter(
            "gate_up", nn.Parameter(torch.empty(512, 256, 2048, **factory))
        )
        self.experts.register_parameter(
            "down", nn.Parameter(torch.empty(512, 2048, 128, **factory))
        )
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(2048, 128, bias=False, **factory)
        self.shared_expert.up_proj = nn.Linear(2048, 128, bias=False, **factory)
        self.shared_expert.down_proj = nn.Linear(128, 2048, bias=False, **factory)
        self.shared_expert_gate = nn.Linear(2048, 1, bias=False, **factory)
        _parallel_parameter(self.experts.gate_up, 1, stride=2)
        _parallel_parameter(self.experts.down, 2)
        _parallel_parameter(self.shared_expert.gate_proj.weight, 0)
        _parallel_parameter(self.shared_expert.up_proj.weight, 0)
        _parallel_parameter(self.shared_expert.down_proj.weight, 1)

    def _hf_tensor(self, name):
        if name.startswith("experts."):
            _, expert, projection, _ = name.split(".")
            expert = int(expert)
            if projection == "down_proj":
                return self.experts.down[expert]
            offset = 0 if projection == "gate_proj" else 128
            return self.experts.gate_up[expert, offset : offset + 128]
        obj = self
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def load_hf_weights(self, weights):
        if set(weights) != set(moe_hf_shapes()):
            raise ValueError("MoE weight names do not match the official HF contract")
        with torch.no_grad():
            for name in moe_hf_shapes():
                self._hf_tensor(name).copy_(shard_moe_weight(name, weights[name], self.rank))

    def export_local_hf_weights(self):
        return {name: self._hf_tensor(name).detach() for name in moe_hf_shapes()}

    def forward(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != 2048 or hidden.dtype != torch.bfloat16:
            raise ValueError("MoE hidden must be BF16 [tokens, 2048]")
        copied = _CopyToTP.apply(hidden, self.group)
        routed, _ = shared_moe(
            copied,
            _CopyToTP.apply(self.gate.weight, self.group),
            self.experts.gate_up,
            self.experts.down,
        )
        gate = shared_linear(copied, self.shared_expert.gate_proj.weight)
        up = shared_linear(copied, self.shared_expert.up_proj.weight)
        activated = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
        shared = shared_linear(activated, self.shared_expert.down_proj.weight)
        shared_gate = shared_linear(
            copied, _CopyToTP.apply(self.shared_expert_gate.weight, self.group)
        ).sigmoid()
        return _ReduceFromTP.apply(routed + shared * shared_gate, self.group)
