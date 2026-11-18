# SPDX-License-Identifier: Apache-2.0
"""One Qwen3.6-35B-A3B decoder layer.

Tasks 17–22 own the expert MLP, grouped GEMM, shared expert, and GDN/conv
operators. This assembly records those nodes and does not compute them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

NATIVE_OPS_WIRED = False
REFERENCE_BACKEND = "assembly"

_LINEAR = "linear_attention"
_FULL = "full_attention"

# Node suffix -> issue #563 task that must land before the node can run.
# Tasks 18 and 19 follow the task-17 expert MLP. Task 22 follows task-21 GDN.
OPERATOR_TASKS = {
    "conv1d": 21,
    "gdn": 21,
    "gdn_norm": 21,
    "expert_gate_up": 17,
    "expert_swiglu": 17,
    "expert_down": 17,
    "shared_gate_proj": 20,
    "shared_up_proj": 20,
    "shared_swiglu": 20,
    "shared_down_proj": 20,
    "shared_expert_gate": 20,
}
_FOLLOWUP_TASKS = {17: (18, 19), 21: (22,)}


@dataclass(frozen=True)
class LayerDims:
    """Shapes for one layer. Official models take these from ``Qwen3MoeSpec``."""

    hidden_size: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: int
    mrope_section: tuple[int, int, int]

    @classmethod
    def from_spec(cls, spec) -> LayerDims:
        return cls(
            hidden_size=spec.hidden_size,
            rms_norm_eps=spec.rms_norm_eps,
            num_experts=spec.num_experts,
            num_experts_per_tok=spec.num_experts_per_tok,
            moe_intermediate_size=spec.moe_intermediate_size,
            shared_expert_intermediate_size=spec.shared_expert_intermediate_size,
            linear_num_key_heads=spec.linear_num_key_heads,
            linear_num_value_heads=spec.linear_num_value_heads,
            linear_key_head_dim=spec.linear_key_head_dim,
            linear_value_head_dim=spec.linear_value_head_dim,
            linear_conv_kernel_dim=spec.linear_conv_kernel_dim,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            partial_rotary_factor=spec.partial_rotary_factor,
            rope_theta=spec.rope_theta,
            mrope_section=spec.mrope_section,
        )


@dataclass(frozen=True)
class NodeStep:
    name: str
    status: str
    task: int | None = None


@dataclass(frozen=True)
class LayerResult:
    steps: tuple[NodeStep, ...]

    @property
    def executed(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps if step.status == "executed")

    @property
    def skipped(self) -> tuple[tuple[str, int], ...]:
        return tuple((step.name, step.task) for step in self.steps if step.status == "skipped")

    @property
    def blocked(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps if step.status == "blocked")

    @property
    def complete(self) -> bool:
        return not self.skipped and not self.blocked

    @property
    def pending_tasks(self) -> tuple[int, ...]:
        tasks = {task for _, task in self.skipped}
        for task, followups in _FOLLOWUP_TASKS.items():
            if task in tasks:
                tasks.update(followups)
        return tuple(sorted(tasks))


class OperatorDeferred(RuntimeError):
    """Raised when a forward still depends on a task 17–22 operator."""

    def __init__(self, pending_tasks: tuple[int, ...]):
        self.pending_tasks = pending_tasks
        tasks = ", ".join(str(task) for task in pending_tasks)
        super().__init__(f"MoE layer is waiting on operator tasks {tasks}")


def planned_layer_nodes(layer_type: str) -> tuple[str, ...]:
    """Node suffixes for one layer, in execution order."""

    if layer_type == _LINEAR:
        attention = (
            "input_layernorm",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "conv1d",
            "gdn",
            "gdn_norm",
            "out_proj",
            "residual_attn",
        )
    elif layer_type == _FULL:
        attention = (
            "input_layernorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_norm",
            "k_norm",
            "rope_q",
            "rope_k",
            "attn",
            "attn_output_gate",
            "o_proj",
            "residual_attn",
        )
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    moe = (
        "post_attention_layernorm",
        "router",
        "topk",
        "dispatch",
        "expert_gate_up",
        "expert_swiglu",
        "expert_down",
        "shared_gate_proj",
        "shared_up_proj",
        "shared_swiglu",
        "shared_down_proj",
        "shared_expert_gate",
        "combine",
        "residual_mlp",
    )
    return attention + moe


def layer_steps(layer_type: str) -> tuple[NodeStep, ...]:
    """Mark task 17–22 nodes as skipped. Later nodes stay blocked."""

    seen_deferred = False
    steps: list[NodeStep] = []
    for name in planned_layer_nodes(layer_type):
        task = OPERATOR_TASKS.get(name)
        if task is not None:
            steps.append(NodeStep(name, "skipped", task))
            seen_deferred = True
        elif seen_deferred:
            steps.append(NodeStep(name, "blocked"))
        else:
            steps.append(NodeStep(name, "executed"))
    return tuple(steps)


def forward_layer(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    dims: LayerDims,
    layer_type: str,
    *,
    position_ids: torch.Tensor | None = None,
    nodes: dict[str, torch.Tensor] | None = None,
) -> LayerResult:
    """Run the prefix that does not depend on tasks 17–22."""

    steps = layer_steps(layer_type)
    prefix = tuple(step.name for step in steps if step.status == "executed")
    _run_prefix(hidden, weights, dims, layer_type, position_ids, nodes, prefix)
    return LayerResult(steps)


def _run_prefix(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    dims: LayerDims,
    layer_type: str,
    position_ids: torch.Tensor | None,
    nodes: dict[str, torch.Tensor] | None,
    prefix: tuple[str, ...],
) -> None:
    cursor = 0

    def take(name: str) -> bool:
        nonlocal cursor
        if cursor < len(prefix) and prefix[cursor] == name:
            cursor += 1
            return True
        return False

    if take("input_layernorm"):
        normed = _record(
            nodes,
            "input_layernorm",
            _rms_norm(hidden, _weight(weights, "input_layernorm.weight"), dims.rms_norm_eps),
        )
    else:
        return
    if layer_type == _LINEAR:
        _linear_projections(normed, weights, nodes, take)
    elif layer_type == _FULL:
        mixed = _full_attention(normed, weights, dims, position_ids, nodes, take)
        if take("residual_attn"):
            hidden = _record(nodes, "residual_attn", hidden + mixed)
        else:
            return
        _route(hidden, weights, dims, nodes, take)
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    if cursor != len(prefix):
        raise RuntimeError(f"layer prefix stopped at {prefix[cursor]!r}")


def _linear_projections(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    nodes: dict[str, torch.Tensor] | None,
    take,
) -> None:
    for name, suffix in (
        ("in_proj_qkv", "linear_attn.in_proj_qkv.weight"),
        ("in_proj_z", "linear_attn.in_proj_z.weight"),
        ("in_proj_a", "linear_attn.in_proj_a.weight"),
        ("in_proj_b", "linear_attn.in_proj_b.weight"),
    ):
        if not take(name):
            return
        _record(nodes, name, _linear(hidden, _weight(weights, suffix)))


def _full_attention(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    dims: LayerDims,
    position_ids: torch.Tensor | None,
    nodes: dict[str, torch.Tensor] | None,
    take,
) -> torch.Tensor:
    if position_ids is None:
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
    query_dim = dims.num_attention_heads * dims.head_dim
    packed = _linear(hidden, _weight(weights, "self_attn.q_proj.weight"))
    query, gate = packed.split((query_dim, query_dim), dim=-1)
    if not take("q_proj"):
        return hidden
    _record(nodes, "q_proj", query)
    if not take("k_proj"):
        return query
    key = _record(nodes, "k_proj", _linear(hidden, _weight(weights, "self_attn.k_proj.weight")))
    if not take("v_proj"):
        return query
    value = _record(nodes, "v_proj", _linear(hidden, _weight(weights, "self_attn.v_proj.weight")))
    if not take("q_norm"):
        return query
    query = _record(nodes, "q_norm", _qk_norm(query, _weight(weights, "self_attn.q_norm.weight"), dims))
    if not take("k_norm"):
        return query
    key = _record(nodes, "k_norm", _qk_norm(key, _weight(weights, "self_attn.k_norm.weight"), dims))
    if not take("rope_q"):
        return query
    query = _record(nodes, "rope_q", _partial_mrope(query, position_ids, dims))
    if not take("rope_k"):
        return query
    key = _record(nodes, "rope_k", _partial_mrope(key, position_ids, dims))
    if not take("attn"):
        return query
    attn = _record(nodes, "attn", _causal_gqa(query, key, value, dims))
    if not take("attn_output_gate"):
        return attn
    gated = _record(nodes, "attn_output_gate", attn * torch.sigmoid(gate))
    if not take("o_proj"):
        return gated
    return _record(nodes, "o_proj", _linear(gated, _weight(weights, "self_attn.o_proj.weight")))


def _route(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    dims: LayerDims,
    nodes: dict[str, torch.Tensor] | None,
    take,
) -> None:
    if not take("post_attention_layernorm"):
        return
    routed = _record(
        nodes,
        "post_attention_layernorm",
        _rms_norm(hidden, _weight(weights, "post_attention_layernorm.weight"), dims.rms_norm_eps),
    )
    flat = routed.reshape(-1, dims.hidden_size)
    logits = flat.float() @ _weight(weights, "mlp.gate.weight").float().t()
    if not take("router"):
        return
    _record(nodes, "router", logits.reshape(*routed.shape[:-1], dims.num_experts).to(dtype=routed.dtype))
    scores, expert_ids = _topk(logits, dims.num_experts_per_tok)
    if not take("topk"):
        return
    _record(nodes, "topk", scores.to(dtype=routed.dtype).reshape(*routed.shape[:-1], dims.num_experts_per_tok))
    if not take("dispatch"):
        return
    _record(nodes, "dispatch", flat.unsqueeze(1).expand(-1, expert_ids.shape[-1], -1))


def _topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    scores, ids = torch.topk(probs, k, dim=-1)
    scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    return scores, ids


def _causal_gqa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dims: LayerDims) -> torch.Tensor:
    batch, seq, _ = query.shape
    q = query.view(batch, seq, dims.num_attention_heads, dims.head_dim).transpose(1, 2)
    k = key.view(batch, seq, dims.num_key_value_heads, dims.head_dim).transpose(1, 2)
    v = value.view(batch, seq, dims.num_key_value_heads, dims.head_dim).transpose(1, 2)
    groups = dims.num_attention_heads // dims.num_key_value_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scale = dims.head_dim ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    causal = torch.ones(seq, seq, device=query.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float()).transpose(1, 2).reshape(batch, seq, -1)
    return out.to(dtype=query.dtype)


def _partial_mrope(states: torch.Tensor, position_ids: torch.Tensor, dims: LayerDims) -> torch.Tensor:
    rotary = int(dims.head_dim * dims.partial_rotary_factor)
    if sum(dims.mrope_section) * 2 != rotary:
        raise ValueError("mrope sections must cover the partial rotary width")
    batch, seq, hidden = states.shape
    heads = hidden // dims.head_dim
    values = states.view(batch, seq, heads, dims.head_dim)
    rotated = _rope_rotate(values[..., :rotary], position_ids, dims.rope_theta, dims.mrope_section)
    return torch.cat((rotated, values[..., rotary:]), dim=-1).reshape(batch, seq, hidden)


def _rope_rotate(
    values: torch.Tensor,
    position_ids: torch.Tensor,
    theta: int,
    sections: tuple[int, int, int],
) -> torch.Tensor:
    half = values.shape[-1] // 2
    freqs = torch.arange(half, device=values.device, dtype=torch.float32)
    inv = theta ** (-freqs / half)
    angles = position_ids.to(dtype=torch.float32).unsqueeze(-1) * inv
    cos = torch.cos(angles).to(dtype=values.dtype).unsqueeze(2)
    sin = torch.sin(angles).to(dtype=values.dtype).unsqueeze(2)
    first, second = values[..., :half], values[..., half:]
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    if sum(sections) != half:
        raise ValueError("mrope section width does not match the rotary half")
    return rotated


def _rms_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = values.float().pow(2).mean(dim=-1, keepdim=True)
    normed = values.float() * torch.rsqrt(variance + eps)
    return (normed * (1.0 + weight.float())).to(dtype=values.dtype)


def _qk_norm(states: torch.Tensor, weight: torch.Tensor, dims: LayerDims) -> torch.Tensor:
    batch, seq, hidden = states.shape
    heads = hidden // dims.head_dim
    values = states.view(batch, seq, heads, dims.head_dim)
    normed = _rms_norm(values, weight, dims.rms_norm_eps)
    return normed.reshape(batch, seq, hidden)


def _linear(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(values, weight)


def _weight(weights: Mapping[str, torch.Tensor], suffix: str) -> torch.Tensor:
    try:
        return weights[suffix]
    except KeyError as exc:
        raise KeyError(f"layer is missing weight {suffix!r}") from exc


def _record(nodes: dict[str, torch.Tensor] | None, name: str, value: torch.Tensor) -> torch.Tensor:
    if nodes is not None:
        nodes[name] = value
    return value
