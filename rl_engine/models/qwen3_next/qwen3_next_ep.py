# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3-Next routed experts at EP4 over the TP4 group (expert tensor parallel 1).

Rank ``r`` owns experts ``[128 r, 128 r + 128)`` at full intermediate width.
Attention, GDN, the embedding, the LM head and the shared expert stay TP4.

The MoE input is already replicated on every TP rank, so dispatch is a local
selection: every rank computes the same routes (``shared_router`` and
``stable_top10_routes``), evaluates the routes whose expert it owns with the
fixed-config grouped GEMM of ``shared_moe``, and the ranks exchange those rows
with an all-gather (bytes only). Each rank then holds the same ``[T, 10, H]``
route outputs, every row written by exactly one owner, and applies
``combine_routes`` (ten ordered FP32 multiply-adds) locally.

Backward mirrors it: each rank computes weight gradients only for its own
experts (no cross-rank sum) and the per-route input-gradient rows of its routes;
those rows are all-gathered and every token's ten contributions are summed in
ascending expert order, the order the EP1 path uses.

The bits differ from the TP4 (expert tensor parallel 4) profile, whose down
projection is four K = 128 partials reduced by a tree; EP4 is certified as its
own profile: replay and rollout must both run it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from rl_engine.models.qwen3_next.qwen3_next_forward import (
    _cuda_tensors,
    _expert_activation,
    _vllm_linear,
    combine_routes,
    grouped_route_linear,
    shared_linear,
    shared_router,
    stable_top10_routes,
)
from rl_engine.models.qwen3_next.qwen3_next_tp import (
    _CopyToTP,
    _parallel_parameter,
    _rank,
    _ReduceFromTP,
    _tp_group,
    _weight,
)
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import moe_hf_shapes, shard_moe_weight

EXPERTS, LOCAL_EXPERTS, EP = 512, 128, 4
HIDDEN, INTERMEDIATE = 2048, 512
EP_PROFILE_ID = "qwen3-next-moe-ep4-v1"


def supported_topology(topology) -> bool:
    """TP4/PP1/CP1 with expert parallelism 1 (ETP4) or 4 (ETP1) over the TP group."""
    return (
        isinstance(topology, dict)
        and {k: v for k, v in topology.items() if k != "ep"} == {"tp": 4, "pp": 1, "cp": 1}
        and topology.get("ep") in (1, 4)
    )


def expert_owner(indices: torch.Tensor) -> torch.Tensor:
    return torch.div(indices, LOCAL_EXPERTS, rounding_mode="floor")


def expert_map(rank: int, device) -> torch.Tensor:
    """int32 [512]: local expert index on ``rank`` or -1 (vLLM's EP convention)."""
    if rank not in range(EP):
        raise ValueError("EP4 rank must be 0..3")
    mapping = torch.full((EXPERTS,), -1, dtype=torch.int32, device=device)
    mapping[rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, dtype=torch.int32, device=device
    )
    return mapping


def _slots_by_owner(route_indices: torch.Tensor):
    """Flat route-slot ids per owning rank, ascending; identical on every rank."""
    owners = expert_owner(route_indices.reshape(-1))
    return [torch.nonzero(owners == r).flatten() for r in range(EP)]


def _exchange(local_rows: torch.Tensor, slots, total: int, group) -> torch.Tensor:
    """All-gather each rank's rows for its own slots; return all ``total`` slots.

    Ranks pad to the largest slot count (known everywhere, since routes are
    replicated), so the gather has one shape on every rank. Every slot is
    owned by exactly one rank; a slot owned by none or by two fails closed.
    """
    from rl_engine.distributed.algorithms.collectives import collective_for_group

    width = local_rows.shape[-1]
    counts = [int(s.numel()) for s in slots]
    if sum(counts) != total:
        raise RuntimeError("EP route slots are missing or duplicated")
    padded = max(max(counts), 1)
    rank = dist.get_rank(group)
    send = local_rows.new_zeros((padded, width))
    send[: counts[rank]] = local_rows
    collective = collective_for_group(group, min_size_bytes=send.numel() * send.element_size() * EP)
    gathered = collective.all_gather(send).reshape(EP, padded, width)
    out = local_rows.new_empty((total, width))
    for owner, owned in enumerate(slots):
        out.index_copy_(0, owned, gathered[owner, : counts[owner]])
    return out


class _EPRoutedExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_up, down, route_indices, rank, group):
        ctx.save_for_backward(x, gate_up, down, route_indices)
        ctx.rank, ctx.group = rank, group
        tokens, top_k = route_indices.shape
        slots = _slots_by_owner(route_indices)
        mapping = expert_map(rank, x.device)
        with torch.cuda.device(x.device):
            projected = grouped_route_linear(x, gate_up, route_indices, mapping)
            flat = projected.reshape(-1, projected.shape[-1]).index_select(0, slots[rank])
            activated = _expert_activation(flat)[0].to(x.dtype)
            # Second projection on the owned rows only: one row per route.
            owned_indices = route_indices.reshape(-1).index_select(0, slots[rank])
            local = grouped_route_linear(activated, down, owned_indices[:, None], mapping).reshape(
                -1, x.shape[1]
            )
        rows = _exchange(local, slots, tokens * top_k, group)
        return rows.reshape(tokens, top_k, x.shape[1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, gate_up, down, route_indices = ctx.saved_tensors
        rank, group = ctx.rank, ctx.group
        tokens, top_k = route_indices.shape
        slots = _slots_by_owner(route_indices)
        owned = slots[rank]
        dgate_up = torch.zeros_like(gate_up) if ctx.needs_input_grad[1] else None
        ddown = torch.zeros_like(down) if ctx.needs_input_grad[2] else None
        flat_indices = route_indices.reshape(-1)
        grad_flat = grad_output.reshape(-1, x.shape[1])
        contributions = x.new_zeros((owned.numel(), x.shape[1]))
        with torch.cuda.device(x.device):
            linear = _vllm_linear()
            # Recompute the first projection with the forward's grouped kernel.
            projected_all = grouped_route_linear(
                x, gate_up, route_indices, expert_map(rank, x.device)
            )
            projected_all = projected_all.reshape(-1, projected_all.shape[-1])
            for local_expert in range(LOCAL_EXPERTS):
                expert = rank * LOCAL_EXPERTS + local_expert
                position = torch.nonzero(flat_indices.index_select(0, owned) == expert).flatten()
                if position.numel() == 0:
                    continue
                slot = owned.index_select(0, position)
                rows = x.index_select(0, torch.div(slot, top_k, rounding_mode="floor"))
                activated, silu_gate, gate32, up32 = _expert_activation(
                    projected_all.index_select(0, slot)
                )
                grad = grad_flat.index_select(0, slot)
                if ddown is not None:
                    ddown[local_expert].copy_(linear(grad.t(), activated.to(x.dtype).t()))
                dactivated = linear(grad, down[local_expert].t()).float()
                sigmoid = gate32.sigmoid()
                dgate = dactivated * up32 * (sigmoid + gate32 * sigmoid * (1 - sigmoid))
                dup = dactivated * silu_gate
                dprojected = torch.cat((dgate, dup), dim=-1).to(x.dtype)
                if dgate_up is not None:
                    dgate_up[local_expert].copy_(linear(dprojected.t(), rows.t()))
                contributions.index_copy_(
                    0, position, linear(dprojected, gate_up[local_expert].t())
                )
        dx = None
        if ctx.needs_input_grad[0]:
            per_slot = _exchange(contributions, slots, tokens * top_k, group)
            per_slot = per_slot.reshape(tokens, top_k, x.shape[1])
            # Ascending expert order per token, as the EP1 backward accumulates.
            order = torch.argsort(route_indices, dim=1)
            dx = torch.zeros_like(x)
            for k in range(top_k):
                dx = dx + per_slot.gather(
                    1, order[:, k, None, None].expand(-1, 1, x.shape[1])
                ).squeeze(1)
        return dx, dgate_up, ddown, None, None, None


def shared_moe_ep(x, router_weight, gate_up_local, down_local, *, group):
    """EP4 routed experts; returns the combined [T, H] output (complete, replicated)."""
    _cuda_tensors(x, router_weight, gate_up_local, down_local)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != HIDDEN:
        raise ValueError("Qwen3-Next MoE input must be BF16 [T, 2048]")
    if gate_up_local.shape != (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN):
        raise ValueError("EP4 gate_up must be [128, 1024, 2048] (full width)")
    if down_local.shape != (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE):
        raise ValueError("EP4 down must be [128, 2048, 512] (full width)")
    if gate_up_local.dtype != x.dtype or down_local.dtype != x.dtype:
        raise ValueError("Qwen3-Next expert weights must remain BF16")
    if not dist.is_initialized() or group is None or dist.get_world_size(group) != EP:
        raise ValueError("EP4 experts require an explicit four-rank group")
    routes = stable_top10_routes(shared_router(x, router_weight))
    outputs = _EPRoutedExperts.apply(
        x, gate_up_local, down_local, routes.indices, dist.get_rank(group), group
    )
    return combine_routes(outputs, routes.weights), routes


def owns_expert(name: str, rank: int) -> bool:
    """Whether the official HF tensor ``name`` is held by ``rank`` at EP4."""
    if not name.startswith("experts."):
        return True
    return int(name.split(".")[1]) // LOCAL_EXPERTS == rank


def ep_local_hf_names(rank: int):
    _rank(rank)
    return [name for name in moe_hf_shapes() if owns_expert(name, rank)]


class EP4MoE(nn.Module):
    """Routed experts at EP4 (full width, 128 per rank); shared expert at TP4."""

    # Set by a trace owner (see qwen3_next_subnode); receives the route indices.
    subnode_hook = None

    def __init__(self, *, group, device):
        super().__init__()
        self.group = _tp_group(group)
        self.rank = dist.get_rank(self.group)
        factory = {"device": device, "dtype": torch.bfloat16}
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias=False, **factory)
        self.experts = nn.Module()
        self.experts.register_parameter(
            "gate_up",
            nn.Parameter(torch.empty(LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN, **factory)),
        )
        self.experts.register_parameter(
            "down", nn.Parameter(torch.empty(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, **factory))
        )
        # Expert parameters are owned by one rank each: not tensor-parallel.
        for parameter in (self.experts.gate_up, self.experts.down):
            parameter.allreduce = False
            parameter.expert_parallel = True
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(HIDDEN, 128, bias=False, **factory)
        self.shared_expert.up_proj = nn.Linear(HIDDEN, 128, bias=False, **factory)
        self.shared_expert.down_proj = nn.Linear(128, HIDDEN, bias=False, **factory)
        self.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False, **factory)
        _parallel_parameter(self.shared_expert.gate_proj.weight, 0)
        _parallel_parameter(self.shared_expert.up_proj.weight, 0)
        _parallel_parameter(self.shared_expert.down_proj.weight, 1)

    def _hf_tensor(self, name):
        if name.startswith("experts."):
            _, expert, projection, _ = name.split(".")
            local = int(expert) - self.rank * LOCAL_EXPERTS
            if local not in range(LOCAL_EXPERTS):
                raise ValueError(f"{name} is not owned by EP rank {self.rank}")
            if projection == "down_proj":
                return self.experts.down[local]
            offset = 0 if projection == "gate_proj" else INTERMEDIATE
            return self.experts.gate_up[local, offset : offset + INTERMEDIATE]
        obj = self
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def load_hf_weights(self, weights):
        """``weights`` maps official names to tensors; only owned experts are read."""
        if not set(ep_local_hf_names(self.rank)) <= set(weights):
            raise ValueError("MoE weight names do not cover this EP rank's experts")
        with torch.no_grad():
            for name in ep_local_hf_names(self.rank):
                value = weights[name]
                _weight(value, moe_hf_shapes()[name], name)
                if name.startswith("experts."):
                    self._hf_tensor(name).copy_(value)
                else:
                    self._hf_tensor(name).copy_(shard_moe_weight(name, value, self.rank))

    def export_local_hf_weights(self):
        """Owned experts at full width; replicated and TP-sharded tensors as held."""
        return {name: self._hf_tensor(name).detach() for name in ep_local_hf_names(self.rank)}

    def forward(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN or hidden.dtype != torch.bfloat16:
            raise ValueError("MoE hidden must be BF16 [tokens, 2048]")
        # The routed path consumes the replicated input directly: its outputs
        # and input gradient are complete on every rank, so nothing is reduced.
        routed, routes = shared_moe_ep(
            hidden, self.gate.weight, self.experts.gate_up, self.experts.down, group=self.group
        )
        if self.subnode_hook is not None:
            self.subnode_hook("router", routes.indices)
        copied = _CopyToTP.apply(hidden, self.group)
        gate = shared_linear(copied, self.shared_expert.gate_proj.weight)
        up = shared_linear(copied, self.shared_expert.up_proj.weight)
        activated = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
        shared = shared_linear(activated, self.shared_expert.down_proj.weight)
        shared_gate = shared_linear(
            copied, _CopyToTP.apply(self.shared_expert_gate.weight, self.group)
        ).sigmoid()
        return routed + _ReduceFromTP.apply(shared * shared_gate, self.group)
