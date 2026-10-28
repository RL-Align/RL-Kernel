# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Audit independent expert slots at the P4/P6 boundary without float reduction.

Expected routes and values must come from the pre-transport routing plan and an
independent oracle, not be reconstructed from received rows. Rows are identified
by (source EP-group rank, source-local token, slot); expert and destination rank
are checked against that plan. Rank numbers are local to the declared EP group.
This host-side diagnostic accepts device tensors, synchronizing them to CPU.
It is not a dispatch implementation and does not verify a transport's identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

MAPPING_SCHEMA = "p6-ep-integrity.v1"


@dataclass(frozen=True)
class EPRankResults:
    """One rank's returned rows in arbitrary arrival order (tensors are mutable).

    ``routes`` is int64 [rows, 4]: source_rank, token, slot, expert.
    ``values`` is [rows, hidden]. Empty expert ranks must still submit a batch.
    Backend/build fields describe the producer; the auditor does not attest them.
    """

    rank: int
    mapping_run_id: str
    producer_backend: str
    producer_build_fingerprint: str
    routes: torch.Tensor
    values: torch.Tensor
    mapping_schema: str = MAPPING_SCHEMA


@dataclass(frozen=True)
class EPIntegrityIssue:
    code: str
    rank: int | None = None
    source_rank: int | None = None
    token: int | None = None
    slot: int | None = None
    expert: int | None = None
    row: int | None = None


@dataclass(frozen=True)
class EPIntegrityReport:
    passed: bool
    world_size: int
    expected_slots: int
    received_slots: int
    issue_count: int
    issues: tuple[EPIntegrityIssue, ...]
    mapping_run_id: str
    producers: tuple[tuple[int, str, str], ...]
    schema_version: str = "p6.ep_integrity.v1"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible bounded diagnostics; payload tensors are not logged."""

        return asdict(self)


def _count(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer")


def _routes(value: torch.Tensor, columns: int) -> torch.Tensor:
    if value.dtype != torch.int64 or value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"routes must be int64 [rows, {columns}]")
    return value.detach().cpu()


def audit_ep_results(
    *,
    expected_routes: torch.Tensor,
    expected_values: torch.Tensor,
    rank_results: Sequence[EPRankResults],
    source_num_tokens: Sequence[int],
    top_k: int,
    expert_to_rank: Sequence[int],
    mapping_run_id: str,
    max_issues: int = 32,
) -> EPIntegrityReport:
    """Check full EP coverage, ownership, IDs, and original payload bytes.

    Expected int64 routes have five columns: source_rank, token, slot, expert,
    expert_rank. Every declared source-local (token, slot) must occur exactly
    once. Malformed expectations raise ValueError; faulty received results yield
    a failing report. Arrival order is irrelevant. Equal numeric values with
    different bytes (including signed zeros) are payload mismatches.
    Empty source/expert ranks are valid; an entirely empty run is not evidence.
    """

    world = len(source_num_tokens)
    _count("world_size", world)
    _count("top_k", top_k)
    _count("max_issues", max_issues)
    for count in source_num_tokens:
        _count("source_num_tokens", count, allow_zero=True)
    if not isinstance(mapping_run_id, str) or not mapping_run_id.strip():
        raise ValueError("mapping_run_id must be explicit and non-empty")
    if not expert_to_rank:
        raise ValueError("expert_to_rank must not be empty")
    for owner in expert_to_rank:
        _count("expert owner rank", owner, allow_zero=True)
        if owner >= world:
            raise ValueError("expert owner rank is outside the EP group")
    expected = _routes(expected_routes, 5)
    total = sum(source_num_tokens) * top_k
    if total == 0 or expected.shape[0] != total:
        raise ValueError("expected routes must cover a non-empty full token-slot domain")
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if (
        expected_values.ndim != 2
        or expected_values.shape[0] != total
        or expected_values.shape[1] == 0
        or expected_values.dtype not in supported
    ):
        raise ValueError("expected_values must be fp16/bf16/fp32 [expected_slots, hidden>0]")
    oracle = expected_values.detach().cpu().contiguous()
    oracle_bytes = oracle.view(torch.uint8).reshape(total, -1)
    identity: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for row, (source, token, slot, expert, owner) in enumerate(expected.tolist()):
        if not (
            0 <= source < world
            and 0 <= token < source_num_tokens[source]
            and 0 <= slot < top_k
            and 0 <= expert < len(expert_to_rank)
            and owner == expert_to_rank[expert]
        ):
            raise ValueError("expected route has invalid domain or expert ownership")
        key = (source, token, slot)
        if key in identity:
            raise ValueError("expected routes contain duplicate token-slot identities")
        identity[key] = (expert, owner, row)

    issues: list[EPIntegrityIssue] = []
    issue_count = 0

    def fail(code: str, **coordinates: int) -> None:
        nonlocal issue_count
        issue_count += 1
        if len(issues) < max_issues:
            issues.append(EPIntegrityIssue(code, **coordinates))

    ranks_seen: set[int] = set()
    pairs_seen: set[tuple[int, int, int]] = set()
    producers: list[tuple[int, str, str]] = []
    received = 0
    # Rank order stabilizes the first reported error; per-rank row order is logged.
    for batch in sorted(
        rank_results,
        key=lambda batch: batch.rank if isinstance(batch.rank, int) else world,
    ):
        rank = batch.rank
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
            fail("invalid_rank")
            continue
        if rank in ranks_seen:
            fail("duplicate_rank", rank=rank)
        ranks_seen.add(rank)
        producers.append((rank, batch.producer_backend, batch.producer_build_fingerprint))
        if not (
            isinstance(batch.producer_backend, str)
            and batch.producer_backend.strip()
            and isinstance(batch.producer_build_fingerprint, str)
            and batch.producer_build_fingerprint.strip()
        ):
            fail("missing_producer_identity", rank=rank)
        if batch.mapping_run_id != mapping_run_id:
            fail("mapping_run_id_mismatch", rank=rank)
        if batch.mapping_schema != MAPPING_SCHEMA:
            fail("mapping_schema_mismatch", rank=rank)
        try:
            routes = _routes(batch.routes, 4)
        except ValueError:
            fail("route_layout_mismatch", rank=rank)
            continue
        rows = routes.shape[0]
        received += rows
        values_valid = (
            batch.values.dtype == oracle.dtype
            and batch.values.ndim == 2
            and batch.values.shape == (rows, oracle.shape[1])
        )
        if not values_valid:
            fail("payload_layout_mismatch", rank=rank)
        else:
            value_bytes = batch.values.detach().cpu().contiguous().view(torch.uint8)
        for row, (source, token, slot, expert) in enumerate(routes.tolist()):
            coordinates = dict(
                rank=rank, source_rank=source, token=token, slot=slot, expert=expert, row=row
            )
            key = (source, token, slot)
            if key not in identity:
                fail("unexpected_token_slot", **coordinates)
                continue
            if key in pairs_seen:
                fail("duplicate_token_slot", **coordinates)
            pairs_seen.add(key)
            expected_expert, owner, oracle_row = identity[key]
            if expert != expected_expert:
                fail("expert_mismatch", **coordinates)
            if rank != owner:
                fail("expert_rank_mismatch", **coordinates)
            if values_valid and not torch.equal(value_bytes[row], oracle_bytes[oracle_row]):
                fail("payload_bytes_mismatch", **coordinates)
    for rank in sorted(set(range(world)) - ranks_seen):
        fail("missing_rank", rank=rank)
    for source, token, slot in sorted(set(identity) - pairs_seen):
        fail("missing_token_slot", source_rank=source, token=token, slot=slot)
    return EPIntegrityReport(
        passed=issue_count == 0,
        world_size=world,
        expected_slots=total,
        received_slots=received,
        issue_count=issue_count,
        issues=tuple(issues),
        mapping_run_id=mapping_run_id,
        producers=tuple(producers),
    )
