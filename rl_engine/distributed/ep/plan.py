# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Packing plan and inverse mapping for DSV4 expert parallelism.

Everything here is integer arithmetic on int64 tensors: a plan is a pure function
of its inputs, so it never depends on arrival timing. Three row orders exist:

* send: ``(destination, token, slot)``, the order the backward receipt requires;
* arrival: ``(source, token, slot)``, what ``all_to_all_single`` delivers;
* expert: ``(local expert, source, token, slot)``, what the expert kernels consume.

The return trip sends rows in arrival order and needs no metadata, because they
land on the source in send order. ``EpSendPlan.unpack_return`` then restores
``(token, slot)`` order, which does not depend on the EP size.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

EP_MAPPING_SCHEMA = "p4-token-slot.v1"

WIRE_RUN_TAG, WIRE_SOURCE, WIRE_TOKEN, WIRE_SLOT, WIRE_EXPERT = range(5)
WIRE_COLUMNS = 5


class EpPlanError(ValueError):
    """Raised when routing input or received metadata breaks the plan contract."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise EpPlanError(message)


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    return torch.cat((counts.new_zeros(1), counts.cumsum(0)))


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse


def _select(rows: torch.Tensor, index: torch.Tensor, name: str) -> torch.Tensor:
    _require(rows.ndim >= 1, f"{name} must have a row dimension")
    _require(
        rows.shape[0] == index.numel(),
        f"{name} has {rows.shape[0]} rows, plan has {index.numel()}",
    )
    return rows.index_select(0, index.to(rows.device))


@dataclass(frozen=True, eq=False)
class EpPlacement:
    """Global expert id -> owning EP rank and local expert id."""

    num_ranks: int
    expert_owner: torch.Tensor
    expert_local: torch.Tensor
    local_counts: torch.Tensor
    fingerprint: str

    @classmethod
    def from_owners(cls, owners, num_ranks: int) -> EpPlacement:
        """Local ids follow ascending global expert id within each owner."""
        _require(
            type(num_ranks) is int and num_ranks > 0,
            "num_ranks must be a positive integer",
        )
        owner = torch.as_tensor(owners, device="cpu")
        _require(
            owner.ndim == 1 and owner.numel() > 0 and not owner.is_floating_point(),
            "owners must be a non-empty 1-D integer table",
        )
        owner = owner.to(torch.int64).clone()
        _require(
            bool(((owner >= 0) & (owner < num_ranks)).all()),
            "expert owner outside the EP group",
        )
        counts = torch.bincount(owner, minlength=num_ranks)
        order = torch.argsort(owner, stable=True)
        local = torch.empty_like(owner)
        local[order] = torch.arange(owner.numel()) - _offsets(counts)[owner[order]]
        digest = hashlib.sha256(owner.numpy().tobytes()).hexdigest()
        return cls(num_ranks, owner, local, counts, f"{num_ranks}:{digest}")

    def local_expert_ids(self, rank: int) -> torch.Tensor:
        """Global ids owned by ``rank``, indexed by local expert id."""
        _require(type(rank) is int and 0 <= rank < self.num_ranks, "rank outside the EP group")
        return torch.nonzero(self.expert_owner == rank).reshape(-1)


@dataclass(frozen=True)
class EpIdentity:
    """Names the forward invocation whose rows may be mixed together."""

    run_id: str
    model_version: str
    placement_version: str

    def __post_init__(self) -> None:
        _require(
            all(
                isinstance(v, str) and v.strip()
                for v in (self.run_id, self.model_version, self.placement_version)
            ),
            "run_id, model_version and placement_version must be non-empty strings",
        )

    def run_tag(self, placement: EpPlacement, top_k: int) -> int:
        """Non-negative 63-bit tag stamped on every wire row."""
        payload = json.dumps(
            [self.run_id, self.model_version, self.placement_version, placement.fingerprint, top_k]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") & (2**63 - 1)


@dataclass(frozen=True, eq=False)
class EpCombineMapping:
    """Rows and identity in ``(token, slot)`` order, for ``MoEOutputCombineInputs``."""

    rows: torch.Tensor
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    slot_indices: torch.Tensor
    mapping_run_id: str
    mapping_schema_version: str = EP_MAPPING_SCHEMA


@dataclass(frozen=True, eq=False)
class EpSendPlan:
    """Source-side view. All per-row tensors are in send order."""

    identity: EpIdentity
    run_tag: int
    source_rank: int
    num_ranks: int
    num_tokens: int
    top_k: int
    tokens: torch.Tensor
    slots: torch.Tensor
    experts: torch.Tensor
    destinations: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    to_canonical: torch.Tensor

    @property
    def num_rows(self) -> int:
        return self.tokens.numel()

    @property
    def slot_ids(self) -> torch.Tensor:
        return self.tokens * self.top_k + self.slots

    def wire_metadata(self) -> torch.Tensor:
        """``[S, WIRE_COLUMNS]`` int64, one identity row per routed slot."""
        return torch.stack(
            (
                torch.full_like(self.tokens, self.run_tag),
                torch.full_like(self.tokens, self.source_rank),
                self.tokens,
                self.slots,
                self.experts,
            ),
            dim=1,
        )

    def pack_token_rows(self, x: torch.Tensor) -> torch.Tensor:
        """``[T, ...]`` per-token values, such as the expert input, to send order."""
        _require(x.ndim >= 1 and x.shape[0] == self.num_tokens, "x must have one row per token")
        return x.index_select(0, self.tokens.to(x.device))

    def pack_slot_values(self, values: torch.Tensor) -> torch.Tensor:
        """``[T, top_k, ...]`` per-slot values, such as the routing weight, to send order."""
        _require(
            values.ndim >= 2 and values.shape[:2] == (self.num_tokens, self.top_k),
            "values must have shape [num_tokens, top_k, ...]",
        )
        flat = values.reshape(self.num_tokens * self.top_k, *values.shape[2:])
        return flat.index_select(0, self.slot_ids.to(values.device))

    def unpack_return(self, rows: torch.Tensor) -> EpCombineMapping:
        """Rows returned in send order, restored to ``(token, slot)`` order.

        Invalid slots stay absent rather than being filled, so the P6 boundary
        rejects an incomplete Top-K when ``num_tokens`` is declared.
        """
        perm = self.to_canonical
        return EpCombineMapping(
            _select(rows, perm, "returned rows"),
            self.tokens[perm],
            self.experts[perm],
            self.slots[perm],
            self.identity.run_id,
        )

    def verify_echo(self, echoed: torch.Tensor) -> None:
        """Check metadata echoed back in send order: missing, foreign or duplicated rows."""
        _require(
            echoed.ndim == 2 and echoed.shape[1] == WIRE_COLUMNS and echoed.dtype == torch.int64,
            f"echoed metadata must be int64 [S, {WIRE_COLUMNS}]",
        )
        _require(
            echoed.shape[0] == self.num_rows,
            f"missing or extra rows: sent {self.num_rows}, got {echoed.shape[0]}",
        )
        expected = self.wire_metadata().to(echoed.device)
        _require(
            bool((echoed[:, WIRE_RUN_TAG] == self.run_tag).all()),
            "echoed row belongs to another run",
        )
        ids = echoed[:, WIRE_TOKEN] * self.top_k + echoed[:, WIRE_SLOT]
        _require(
            torch.unique(ids).numel() == self.num_rows,
            "duplicate token/slot in echoed metadata",
        )
        _require(torch.equal(echoed, expected), "echoed metadata differs from the send plan")

    def saved_route_slots(self) -> list[tuple[int, int, int, int]]:
        """``(token, slot, expert, destination)`` in the backward receipt's order.

        Destinations are EP-group ranks; slow by design, so training only.
        """
        return list(
            zip(
                self.tokens.tolist(),
                self.slots.tolist(),
                self.experts.tolist(),
                self.destinations.tolist(),
                strict=True,
            )
        )


@dataclass(frozen=True, eq=False)
class EpRecvPlan:
    """Expert-side view. Per-row tensors are in arrival order."""

    run_tag: int
    rank: int
    counts: torch.Tensor
    offsets: torch.Tensor
    sources: torch.Tensor
    tokens: torch.Tensor
    slots: torch.Tensor
    experts: torch.Tensor
    local_experts: torch.Tensor
    to_expert: torch.Tensor
    to_arrival: torch.Tensor
    expert_offsets: torch.Tensor

    @property
    def num_rows(self) -> int:
        return self.tokens.numel()

    def to_expert_order(self, rows: torch.Tensor) -> torch.Tensor:
        return _select(rows, self.to_expert, "arrival rows")

    def to_arrival_order(self, rows: torch.Tensor) -> torch.Tensor:
        return _select(rows, self.to_arrival, "expert rows")


def build_send_plan(
    topk_experts: torch.Tensor,
    placement: EpPlacement,
    identity: EpIdentity,
    source_rank: int,
    valid: torch.Tensor | None = None,
) -> EpSendPlan:
    """Build the source-side plan from ``[T, top_k]`` global expert ids."""
    _require(
        topk_experts.ndim == 2
        and topk_experts.shape[1] > 0
        and not topk_experts.is_floating_point()
        and topk_experts.dtype != torch.bool,
        "topk_experts must be an integer [num_tokens, top_k] tensor",
    )
    _require(
        type(source_rank) is int and 0 <= source_rank < placement.num_ranks,
        "source_rank outside the EP group",
    )
    device = topk_experts.device
    num_tokens, top_k = topk_experts.shape
    experts = topk_experts.reshape(-1).to(torch.int64)
    slot_ids = torch.arange(experts.numel(), dtype=torch.int64, device=device)
    if valid is not None:
        _require(
            valid.dtype == torch.bool and valid.shape == topk_experts.shape,
            "valid must be a bool tensor shaped like topk_experts",
        )
        keep = valid.to(device).reshape(-1)
        experts, slot_ids = experts[keep], slot_ids[keep]
    num_experts = placement.expert_owner.numel()
    _require(
        bool(((experts >= 0) & (experts < num_experts)).all()),
        "expert id outside the placement",
    )

    # slot_ids ascend, so a stable sort on destination gives (destination, token, slot).
    destinations = placement.expert_owner.to(device)[experts]
    perm = torch.argsort(destinations, stable=True)
    slot_ids = slot_ids[perm]
    counts = torch.bincount(destinations, minlength=placement.num_ranks)
    return EpSendPlan(
        identity=identity,
        run_tag=identity.run_tag(placement, top_k),
        source_rank=source_rank,
        num_ranks=placement.num_ranks,
        num_tokens=num_tokens,
        top_k=top_k,
        tokens=torch.div(slot_ids, top_k, rounding_mode="floor"),
        slots=slot_ids % top_k,
        experts=experts[perm],
        destinations=destinations[perm],
        counts=counts,
        offsets=_offsets(counts),
        to_canonical=torch.argsort(slot_ids),
    )


def build_recv_plan(
    metadata: torch.Tensor,
    recv_counts: torch.Tensor,
    placement: EpPlacement,
    identity: EpIdentity,
    rank: int,
    top_k: int,
) -> EpRecvPlan:
    """Validate received ``[R, WIRE_COLUMNS]`` metadata and derive the expert order.

    A row lost together with its count cannot be seen from this side; the source
    detects that case through ``EpSendPlan.verify_echo``.
    """
    _require(type(rank) is int and 0 <= rank < placement.num_ranks, "rank outside the EP group")
    _require(type(top_k) is int and top_k > 0, "top_k must be a positive integer")
    _require(
        metadata.ndim == 2 and metadata.shape[1] == WIRE_COLUMNS and metadata.dtype == torch.int64,
        f"metadata must be int64 [R, {WIRE_COLUMNS}]",
    )
    _require(
        recv_counts.dtype == torch.int64 and recv_counts.shape == (placement.num_ranks,),
        "recv_counts must be int64 with one entry per EP rank",
    )
    device = metadata.device
    recv_counts = recv_counts.to(device)
    _require(bool((recv_counts >= 0).all()), "negative receive count")
    _require(
        int(recv_counts.sum()) == metadata.shape[0],
        f"missing or extra rows: counts say {int(recv_counts.sum())}, got {metadata.shape[0]}",
    )

    run_tag = identity.run_tag(placement, top_k)
    sources = metadata[:, WIRE_SOURCE]
    tokens = metadata[:, WIRE_TOKEN]
    slots = metadata[:, WIRE_SLOT]
    experts = metadata[:, WIRE_EXPERT]
    _require(bool((metadata[:, WIRE_RUN_TAG] == run_tag).all()), "row belongs to another run")
    expected_sources = torch.repeat_interleave(
        torch.arange(placement.num_ranks, device=device), recv_counts
    )
    _require(
        torch.equal(sources, expected_sources),
        "row source disagrees with its arrival position",
    )
    _require(bool((tokens >= 0).all()), "negative token index")
    _require(bool(((slots >= 0) & (slots < top_k)).all()), "slot index outside top_k")
    _require(
        bool(((experts >= 0) & (experts < placement.expert_owner.numel())).all()),
        "expert id outside the placement",
    )
    _require(
        bool((placement.expert_owner.to(device)[experts] == rank).all()),
        "row routed to a rank that does not own its expert",
    )

    # Sources emit strictly ascending slot ids, so a repeat cannot hide non-adjacently
    # without also breaking the order check.
    ids = tokens * top_k + slots
    same_source = sources[1:] == sources[:-1]
    _require(
        not bool((same_source & (ids[1:] == ids[:-1])).any()),
        "duplicate token/slot from one source",
    )
    _require(
        not bool((same_source & (ids[1:] < ids[:-1])).any()),
        "rows out of token/slot order",
    )

    # Stable on arrival order, so the total key is (expert, source, token, slot).
    local_experts = placement.expert_local.to(device)[experts]
    to_expert = torch.argsort(local_experts, stable=True)
    per_expert = torch.bincount(local_experts, minlength=int(placement.local_counts[rank]))
    return EpRecvPlan(
        run_tag=run_tag,
        rank=rank,
        counts=recv_counts,
        offsets=_offsets(recv_counts),
        sources=sources,
        tokens=tokens,
        slots=slots,
        experts=experts,
        local_experts=local_experts,
        to_expert=to_expert,
        to_arrival=_inverse(to_expert),
        expert_offsets=_offsets(per_expert),
    )


__all__ = [
    "EP_MAPPING_SCHEMA",
    "WIRE_COLUMNS",
    "EpCombineMapping",
    "EpIdentity",
    "EpPlacement",
    "EpPlanError",
    "EpRecvPlan",
    "EpSendPlan",
    "build_recv_plan",
    "build_send_plan",
]
