# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Official Qwen3-Next TP4 full-attention and MoE blocks for shared adapters.

These blocks own the local arithmetic and differentiable TP boundaries. They
do not register an engine backend or advertise full-model acceptance.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rl_engine.integrations.qwen3_next_forward import shared_attention, shared_linear, shared_moe
from rl_engine.integrations.qwen3_next_tp import (
    _CopyToTP,
    _parallel_parameter,
    _rank,
    _ReduceFromTP,
    _tp_group,
    _weight,
)
from rl_engine.testing.tensor_identity import tensor_bitwise_equal

ATTENTION_HF_SHAPES = {
    "q_proj.weight": (8192, 2048),
    "k_proj.weight": (512, 2048),
    "v_proj.weight": (512, 2048),
    "o_proj.weight": (2048, 4096),
    "q_norm.weight": (256,),
    "k_norm.weight": (256,),
}


def shard_attention_weights(weights: Mapping[str, torch.Tensor], rank: int):
    """Preserve Q/gate interleaving and replicate each KV head on a rank pair."""
    _rank(rank)
    if set(weights) != set(ATTENTION_HF_SHAPES):
        raise ValueError("Attention weight names do not match the official HF contract")
    result = {}
    for name, shape in ATTENTION_HF_SHAPES.items():
        value = weights[name]
        _weight(value, shape, name)
        if name.startswith(("q_norm.", "k_norm.")):
            part = value
        elif name in ("k_proj.weight", "v_proj.weight"):
            part = value.chunk(2, dim=0)[rank // 2]
        else:
            part = value.chunk(4, dim=1 if name == "o_proj.weight" else 0)[rank]
        result[name] = part.contiguous()
    return result


def assemble_attention_weights(shards: Sequence[Mapping[str, torch.Tensor]]):
    """Invert four local HF mappings and reject divergent replicated storage."""
    if len(shards) != 4 or any(set(shard) != set(ATTENTION_HF_SHAPES) for shard in shards):
        raise ValueError("Exactly four complete attention HF shards are required")
    result = {}
    for name, shape in ATTENTION_HF_SHAPES.items():
        values = [shard[name] for shard in shards]
        if name.startswith(("q_norm.", "k_norm.")):
            for value in values:
                _weight(value, shape, name)
            if any(not tensor_bitwise_equal(value, values[0]) for value in values[1:]):
                raise ValueError(f"Replicated attention norm differs: {name}")
            result[name] = values[0]
        elif name in ("k_proj.weight", "v_proj.weight"):
            for value in values:
                _weight(value, (256, 2048), name)
            if not tensor_bitwise_equal(values[0], values[1]) or not tensor_bitwise_equal(
                values[2], values[3]
            ):
                raise ValueError(f"Replicated KV head differs within its rank pair: {name}")
            result[name] = torch.cat((values[0], values[2]), dim=0)
        else:
            axis = 1 if name == "o_proj.weight" else 0
            local_shape = list(shape)
            local_shape[axis] //= 4
            for value in values:
                _weight(value, tuple(local_shape), name)
            result[name] = torch.cat(values, dim=axis)
    return result


@lru_cache(maxsize=4)
def _kv_replica_groups(group):
    ranks = dist.get_process_group_ranks(group)
    # Every TP rank creates both groups in the same order. Only members use a
    # group's collective, and the rank pairs follow the official KV layout.
    return tuple(
        dist.new_group(ranks[start : start + 2], use_local_synchronization=True) for start in (0, 2)
    )


def _norm_module(device):
    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(torch.zeros(256, device=device, dtype=torch.bfloat16))
    )
    return module


@dataclass(frozen=True)
class KVState:
    """Per-slot [1,1,length,256] KV tensors; length is the next absolute position."""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]


def _packed_sequences(hidden, state, indices, cu_seqlens):
    if (
        hidden.ndim != 2
        or hidden.shape[1] != 2048
        or hidden.dtype != torch.bfloat16
        or not hidden.is_cuda
    ):
        raise ValueError("Attention hidden must be CUDA BF16 [tokens, 2048]")
    if hidden.shape[0] < 1:
        raise ValueError("Attention requires at least one active token")
    if (
        not isinstance(state, KVState)
        or len(state.keys) < 2
        or len(state.keys) != len(state.values)
    ):
        raise ValueError("KVState requires a reserved slot and matching key/value slots")
    if (
        indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != hidden.device
    ):
        raise ValueError("State indices must be an integer vector on the input device")
    if (
        cu_seqlens.device.type != "cpu"
        or cu_seqlens.ndim != 1
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("cu_seqlens must be a CPU integer vector")
    slots, ends = indices.tolist(), cu_seqlens.tolist()
    if len(ends) != len(slots) + 1 or ends[0] != 0 or ends[-1] != hidden.shape[0]:
        raise ValueError("cu_seqlens must describe all packed tokens")
    if any(left > right for left, right in zip(ends, ends[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if len(set(slots)) != len(slots) or any(slot <= 0 or slot >= len(state.keys) for slot in slots):
        raise ValueError("Active state slots must be unique and exclude reserved slot zero")
    for key, value in zip(state.keys, state.values):
        if (
            key.ndim != 4
            or key.shape[:2] != (1, 1)
            or key.shape[-1] != 256
            or key.shape != value.shape
        ):
            raise ValueError("KV slots must have matching [1,1,length,256] tensors")
        if (
            key.dtype != hidden.dtype
            or value.dtype != hidden.dtype
            or key.device != hidden.device
            or value.device != hidden.device
        ):
            raise ValueError("KV slots must have the input device and BF16 dtype")
    return slots, ends


class TP4FullAttention(nn.Module):
    """Four local query heads and one pair-replicated KV head, with explicit state."""

    def __init__(self, *, group, device, eps=1e-6, rope_theta=10000000):
        super().__init__()
        self.group = _tp_group(group)
        self.rank = dist.get_rank(self.group)
        self.kv_group = _kv_replica_groups(self.group)[self.rank // 2]
        if eps != 1e-6 or rope_theta != 10000000:
            raise ValueError("Full attention requires official epsilon and RoPE theta")
        self.eps = eps
        factory = {"device": device, "dtype": torch.bfloat16}
        self.q_proj = nn.Linear(2048, 2048, bias=False, **factory)
        self.k_proj = nn.Linear(2048, 256, bias=False, **factory)
        self.v_proj = nn.Linear(2048, 256, bias=False, **factory)
        self.o_proj = nn.Linear(1024, 2048, bias=False, **factory)
        self.q_norm, self.k_norm = _norm_module(device), _norm_module(device)
        _parallel_parameter(self.q_proj.weight, 0)
        _parallel_parameter(self.o_proj.weight, 1)
        _parallel_parameter(self.k_proj.weight, 0, duplicate=self.rank % 2 == 1)
        _parallel_parameter(self.v_proj.weight, 0, duplicate=self.rank % 2 == 1)
        # Kept as a Python float, not a buffer: Megatron's Float16Module casts every
        # floating-point buffer of the actor to BF16, and BF16-rounded inverse
        # frequencies rotate keys differently from the rollout engine's FP32 ones.
        self.rope_theta = float(rope_theta)

    def load_hf_weights(self, weights):
        self.load_state_dict(shard_attention_weights(weights, self.rank), strict=True)

    def export_local_hf_weights(self):
        return self.state_dict()

    def initial_state(self, blocks):
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2:
            raise ValueError("KV state requires a reserved slot and an active slot")
        return KVState(
            tuple(self.q_proj.weight.new_empty((1, 1, 0, 256)) for _ in range(blocks)),
            tuple(self.q_proj.weight.new_empty((1, 1, 0, 256)) for _ in range(blocks)),
        )

    @staticmethod
    def inverse_frequencies(rope_theta, device):
        """FP32 inverse frequencies of the 64-wide rotary quarter, computed on demand."""
        exponents = torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64
        return 1.0 / (float(rope_theta) ** exponents)

    def _rotate(self, value, positions):
        inv_freq = TP4FullAttention.inverse_frequencies(self.rope_theta, value.device)
        angles = positions.float()[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(value.dtype)[:, None], angles.sin().to(value.dtype)[:, None]
        rotary, tail = value[..., :64], value[..., 64:]
        rotated = torch.cat((-rotary[..., 32:], rotary[..., :32]), dim=-1)
        return torch.cat((rotary * cos + rotated * sin, tail), dim=-1)

    def forward(self, hidden, state, indices, cu_seqlens):
        from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

        slots, ends = _packed_sequences(hidden, state, indices, cu_seqlens)
        copied = _CopyToTP.apply(hidden, self.group)
        query, gate = shared_linear(copied, self.q_proj.weight).reshape(-1, 4, 512).chunk(2, dim=-1)
        key = shared_linear(copied, _CopyToTP.apply(self.k_proj.weight, self.kv_group)).reshape(
            -1, 1, 256
        )
        value = shared_linear(copied, _CopyToTP.apply(self.v_proj.weight, self.kv_group)).reshape(
            -1, 1, 256
        )
        norm = Qwen3NextRMSNormCudaOp()
        query = norm(query, _CopyToTP.apply(self.q_norm.weight, self.group), eps=self.eps)
        key = norm(key, _CopyToTP.apply(self.k_norm.weight, self.group), eps=self.eps)
        next_keys, next_values = list(state.keys), list(state.values)
        outputs = []
        for slot, begin, end in zip(slots, ends[:-1], ends[1:]):
            if begin == end:
                continue
            offset = state.keys[slot].shape[2]
            if offset + end - begin > 262144:
                raise ValueError("KV continuation exceeds the official maximum position")
            positions = torch.arange(offset, offset + end - begin, device=hidden.device)
            q = self._rotate(query[begin:end], positions).transpose(0, 1).unsqueeze(0)
            k = self._rotate(key[begin:end], positions).transpose(0, 1).unsqueeze(0)
            v = value[begin:end].transpose(0, 1).unsqueeze(0)
            next_keys[slot] = torch.cat((state.keys[slot], k), dim=2)
            next_values[slot] = torch.cat((state.values[slot], v), dim=2)
            out = shared_attention(q, next_keys[slot], next_values[slot])
            outputs.append(out.squeeze(0).transpose(0, 1))
        attended = torch.cat(outputs, dim=0) * gate.sigmoid()
        local = shared_linear(attended.reshape(-1, 1024), self.o_proj.weight)
        return _ReduceFromTP.apply(local, self.group), KVState(tuple(next_keys), tuple(next_values))


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
