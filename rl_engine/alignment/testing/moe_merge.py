# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate MoE merge source identities and collect reference evidence.

This module is for alignment tests. Model code calls the tensor operator directly.
Application counts describe validated source history and the reference schedule;
they are not runtime dispatch measurements or proof of truthful producer data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.registry import kernel_registry
from rl_engine.kernels.semantic_registry import OperatorRequirements, OperatorSession

RECEIPT_VERSION = "moe_merge.receipt.v1"
BACKEND_ID = "rlkernel.moe.shared_residual_merge.reference.v1"
ORDER = ("shared", "residual", "cast_bf16")
BOUNDARIES = (
    "after_shared",
    "after_residual",
    "final_bf16",
)


class MergeContractError(ValueError):
    def __init__(self, status: str, field: str):
        self.status = status
        self.field = field
        super().__init__(f"{status}: {field}")


def _require(condition: bool, status: str, field: str) -> None:
    if not condition:
        raise MergeContractError(status, field)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape and raw logical bytes, including the sign bit of zero."""
    snapshot = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_hash_json([str(snapshot.dtype), list(snapshot.shape)]).encode())
    digest.update(snapshot.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MergeIdentity:
    case_id: str
    run_id: str
    pass_id: str
    checkpoint_fingerprint: str
    weight_fingerprint: str
    route_plan_fingerprint: str
    exchange_plan_fingerprint: str
    combine_plan_fingerprint: str
    fixture_checksum: str
    global_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class MergeSource:
    """Producer evidence supplied with a tensor, never synthesized by the merge."""

    role: str
    source_id: str
    identity: MergeIdentity
    tensor_checksum: str


@dataclass(frozen=True)
class MergeContext:
    identity: MergeIdentity
    # Exact source ids expected by the caller, in routed/shared/residual order.
    expected_source_ids: tuple[str, str, str]
    sources: tuple[MergeSource, ...]
    # Validated upstream application events for this token batch. All rows must
    # share this history; heterogeneous histories require separate calls.
    upstream_events: tuple[str, ...]
    rank: int
    token_owner_ranks: tuple[int, ...]
    shared_replica_ranks: tuple[int, ...]
    merge_order: tuple[str, ...] = ORDER
    input_row_weighted: bool = True


@dataclass
class MergeResult:
    token_moe_row_bf16: Tensor
    receipt: dict[str, Any]
    debug_boundaries: dict[str, Tensor]


def _validate_context(context: MergeContext) -> dict[str, MergeSource]:
    _require(isinstance(context, MergeContext), "INVALID_COMBINE_PLAN", "context")
    identity = context.identity
    _require(isinstance(identity, MergeIdentity), "IDENTITY_DRIFT", "identity")
    for field, value in asdict(identity).items():
        if field != "global_token_ids":
            _require(isinstance(value, str) and bool(value.strip()), "IDENTITY_DRIFT", field)
    tokens = identity.global_token_ids
    _require(
        isinstance(tokens, tuple)
        and bool(tokens)
        and all(type(t) is int and t >= 0 for t in tokens)
        and len(set(tokens)) == len(tokens),
        "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
        "global_token_ids",
    )
    expected_ids = context.expected_source_ids
    _require(
        isinstance(expected_ids, tuple)
        and len(expected_ids) == 3
        and all(isinstance(s, str) and s.strip() for s in expected_ids)
        and len(set(expected_ids)) == 3,
        "IDENTITY_DRIFT",
        "expected_source_ids",
    )
    _require(
        isinstance(context.sources, tuple)
        and all(isinstance(s, MergeSource) for s in context.sources),
        "IDENTITY_DRIFT",
        "sources",
    )
    # Compare identities before looking at payload values or arithmetic policy.
    for source in context.sources:
        _require(
            isinstance(source.role, str)
            and isinstance(source.source_id, str)
            and isinstance(source.tensor_checksum, str),
            "IDENTITY_DRIFT",
            "source fields",
        )
        _require(source.identity == identity, "IDENTITY_DRIFT", f"{source.role}.identity")
    roles = Counter(s.role for s in context.sources)
    _require(
        not (set(roles) - {"routed", "shared", "residual"}),
        "INVALID_COMBINE_PLAN",
        "sources.role (mHC post is not a merge input)",
    )
    for role, status in (
        ("shared", "SHARED_APPLIED_NOT_ONCE"),
        ("residual", "RESIDUAL_APPLIED_NOT_ONCE"),
        ("routed", "INVALID_COMBINE_PLAN"),
    ):
        _require(roles[role] == 1, status, f"sources.{role}")
    sources = {s.role: s for s in context.sources}
    for role, source_id in zip(("routed", "shared", "residual"), expected_ids, strict=True):
        _require(sources[role].source_id == source_id, "IDENTITY_DRIFT", f"{role}.source_id")

    events = context.upstream_events
    _require(
        context.input_row_weighted is True,
        "INVALID_COMBINE_PLAN",
        "weighted routed input",
    )
    _require(
        isinstance(events, tuple) and all(isinstance(e, str) for e in events),
        "INVALID_COMBINE_PLAN",
        "upstream_events",
    )
    counts = Counter(events)
    _require(
        counts["route_weight"] == 1,
        "INVALID_COMBINE_PLAN" if counts["route_weight"] == 0 else "ROUTE_WEIGHT_APPLIED_TWICE",
        "route_weight",
    )
    for name, status in (
        ("shared", "SHARED_APPLIED_NOT_ONCE"),
        ("residual", "RESIDUAL_APPLIED_NOT_ONCE"),
        ("cast_bf16", "EARLY_OR_MULTIPLE_DOWNCAST"),
    ):
        _require(counts[name] == 0, status, f"upstream_events.{name}")
    _require(set(events) == {"route_weight"}, "INVALID_COMBINE_PLAN", "upstream_events")
    _require(context.merge_order == ORDER, "ADDITION_ORDER_MISMATCH", "merge_order")
    _require(type(context.rank) is int and context.rank >= 0, "MISSING_PROVENANCE", "rank")
    _require(
        isinstance(context.token_owner_ranks, tuple)
        and len(context.token_owner_ranks) == len(tokens)
        and all(type(r) is int and r == context.rank for r in context.token_owner_ranks),
        "SHARED_APPLIED_NOT_ONCE",
        "token_owner_ranks",
    )
    replicas = context.shared_replica_ranks
    _require(
        isinstance(replicas, tuple)
        and bool(replicas)
        and all(type(r) is int and r >= 0 for r in replicas)
        and len(set(replicas)) == len(replicas),
        "INVALID_COMBINE_PLAN",
        "shared_replica_ranks",
    )
    return sources


def _finite(value: Tensor, boundary: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), "NON_FINITE", boundary)


def check_moe_merge(
    routed: Tensor,
    shared: Tensor,
    residual: Tensor,
    *,
    context: MergeContext,
    debug: bool = False,
) -> MergeResult:
    """Validate producer evidence and collect reference boundaries for acceptance.

    Identity/discrete failures precede numeric checks. Checksums and finite checks
    synchronize GPU tensors; this helper belongs to validation, not model forward.
    """
    sources = _validate_context(context)
    inputs = {"routed": routed, "shared": shared, "residual": residual}
    _require(
        routed.ndim == 2 and routed.shape[0] == len(context.identity.global_token_ids),
        "INVALID_COMBINE_PLAN",
        "routed.shape",
    )
    _require(routed.dtype == torch.float32, "EARLY_OR_MULTIPLE_DOWNCAST", "routed.dtype")
    for role, value in inputs.items():
        _require(
            tensor_sha256(value) == sources[role].tensor_checksum,
            "IDENTITY_DRIFT",
            f"{role}.tensor_checksum",
        )
    for role, value in inputs.items():
        _finite(value, role)

    session = OperatorSession(kernel_registry.semantic)
    resolution = session.resolve(
        semantic_op="shared_residual_merge",
        requested_backend=BACKEND_ID,
        target="rollout",
        requirements=OperatorRequirements(
            device="rocm" if torch.version.hip and routed.is_cuda else routed.device.type,
            dtype="float32",
        ),
    )
    op = session.instantiate(resolution)
    values = op.forward_with_intermediates(routed, shared, residual)
    boundary_tensors = dict(zip(BOUNDARIES, values, strict=True))
    for key, value in boundary_tensors.items():
        _finite(value, key)
    instance = session.instance_provenance(resolution, op)
    provenance = {
        "operator_instance": instance.to_dict(),
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": str(routed.device),
        "device_name": (torch.cuda.get_device_name(routed.device) if routed.is_cuda else "cpu"),
        "build_runtime": torch.version.hip or torch.version.cuda,
        "launch_parameters": None,
        "launch_provenance_status": "NOT_OBSERVED_AT_REFERENCE_LEVEL",
    }
    boundaries = [
        {
            "key": key,
            "event_index": index,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "layout": "contiguous",
            "checksum": tensor_sha256(value),
            "rank": context.rank,
            "identity": asdict(context.identity),
            "backend": instance.backend_id,
            "implementation_fingerprint": instance.implementation_fingerprint,
        }
        for index, (key, value) in enumerate(boundary_tensors.items())
    ]
    applied = Counter(context.upstream_events + ORDER)
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "identity": asdict(context.identity),
        "sources": [asdict(sources[role]) for role in inputs],
        "upstream_events": list(context.upstream_events),
        "input_row_weighted": context.input_row_weighted,
        "reference_events": list(ORDER),
        "count_basis": "validated_upstream_history_and_reference_schedule",
        "route_weight_applied_count": applied["route_weight"],
        "shared_applied_count": applied["shared"],
        "residual_applied_count": applied["residual"],
        "local_downcast_count": applied["cast_bf16"],
        "merge_order_hash": _hash_json({"version": RECEIPT_VERSION, "order": ORDER}),
        "rank": context.rank,
        "token_owner_ranks": list(context.token_owner_ranks),
        "shared_replica_ranks": list(context.shared_replica_ranks),
        "identity_discrete_gate": "PASS",
        "scope": "local_forward_reference",
        "actual_provenance": provenance,
        "boundaries": boundaries,
    }
    return MergeResult(values[-1], receipt, boundary_tensors if debug else {})
