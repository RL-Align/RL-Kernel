# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P6/T04 eager forward reference; no routing, collective, or mHC arithmetic.

The receipt/context interface is a T01 review proposal, not a second CombinePlan.
Upstream validation binds its opaque plan fingerprint and source receipts. This
module validates the local merge boundary; it cannot authenticate dishonest
producer receipts or certify unobserved GPU launch parameters.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

CONTRACT_VERSION = "p6-task-contract.v1"
FOUNDATION_ABI = "foundation-moe-return.v1"
RECEIPT_VERSION = "p6.t04.merge-receipt.v1-proposal"
BACKEND_ID = "rlkernel.moe.shared_residual_merge.reference.v1"
ORDER = ("shared", "residual", "cast_bf16")
BOUNDARIES = (
    "P6.merge.after_shared",
    "P6.merge.after_residual",
    "P6.merge.final_bf16",
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
    contract_version: str = CONTRACT_VERSION
    foundation_abi: str = FOUNDATION_ABI
    receipt_version: str = RECEIPT_VERSION
    merge_order: tuple[str, ...] = ORDER
    input_row_weighted: bool = True
    route_weight_owner: str = "P5"


@dataclass
class MergeResult:
    token_moe_row_bf16: Tensor
    receipt: dict[str, Any]
    debug_boundaries: dict[str, Tensor]


def _validate_context(context: MergeContext) -> dict[str, MergeSource]:
    _require(isinstance(context, MergeContext), "INVALID_COMBINE_PLAN", "context")
    for field, expected in (
        ("contract_version", CONTRACT_VERSION),
        ("foundation_abi", FOUNDATION_ABI),
        ("receipt_version", RECEIPT_VERSION),
    ):
        _require(getattr(context, field) == expected, "SCHEMA_MISMATCH", field)
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
        context.input_row_weighted is True and context.route_weight_owner == "P5",
        "INVALID_COMBINE_PLAN",
        "weighted row owner",
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


class _ArithmeticTrace(TorchDispatchMode):
    """Observe actual ATen adds/casts without replacing the underlying operations."""

    def __init__(self):
        super().__init__()
        self.operations: list[dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.add.Tensor, torch.ops.aten._to_copy.default):
            self.operations.append(
                {
                    "operator": str(func),
                    "output_dtype": str(result.dtype),
                    "shape": list(result.shape),
                }
            )
        return result


@torch.no_grad()
def shared_residual_merge_fwd(
    routed: Tensor,
    shared: Tensor,
    residual: Tensor,
    *,
    context: MergeContext,
    debug: bool = False,
) -> MergeResult:
    """Merge an owner-local batch; return BF16 output and hash-only evidence.

    CPU and CUDA eager reference only. Hashing synchronizes/copies to CPU; this
    is deliberately not a production kernel, CUDA Graph path, or backward op.
    ``context`` is required: three bare tensors cannot prove exactly-once.
    """
    _require(not torch.compiler.is_compiling(), "UNSUPPORTED_CAPABILITY", "compiled reference")
    sources = _validate_context(context)
    inputs = {"routed": routed, "shared": shared, "residual": residual}
    for role, value in inputs.items():
        _require(type(value) is Tensor, "UNSUPPORTED_CAPABILITY", f"{role}.tensor")
        _require(
            value.layout == torch.strided and value.ndim == 2 and value.is_contiguous(),
            "UNSUPPORTED_CAPABILITY",
            f"{role}.layout",
        )
        _require(value.device.type in {"cpu", "cuda"}, "UNSUPPORTED_CAPABILITY", f"{role}.device")
        _require(
            value.shape[0] == len(context.identity.global_token_ids) and value.shape[1] > 0,
            "INVALID_COMBINE_PLAN",
            f"{role}.shape",
        )
        _require(
            value.shape == routed.shape and value.device == routed.device,
            "INVALID_COMBINE_PLAN",
            f"{role}.shape/device",
        )
        if role == "routed":
            _require(value.dtype == torch.float32, "EARLY_OR_MULTIPLE_DOWNCAST", "routed.dtype")
        else:
            _require(
                value.dtype in {torch.float32, torch.bfloat16},
                "UNSUPPORTED_CAPABILITY",
                f"{role}.dtype",
            )
    if routed.device.type == "cuda":
        with torch.cuda.device(routed.device):
            _require(
                not torch.cuda.is_current_stream_capturing(),
                "UNSUPPORTED_CAPABILITY",
                "CUDA Graph reference",
            )
    for role, value in inputs.items():
        _require(
            tensor_sha256(value) == sources[role].tensor_checksum,
            "IDENTITY_DRIFT",
            f"{role}.tensor_checksum",
        )
    for role, value in inputs.items():
        _finite(value, role)

    events: list[str] = []
    # Disable an enclosing autocast context; each eager add materializes FP32.
    trace = _ArithmeticTrace()
    with torch.autocast(device_type=routed.device.type, enabled=False), trace:
        after_shared = torch.add(routed, shared.float())
        events.append("shared")
        _finite(after_shared, BOUNDARIES[0])
        after_residual = torch.add(after_shared, residual.float())
        events.append("residual")
        _finite(after_residual, BOUNDARIES[1])
        output = after_residual.to(torch.bfloat16)
        events.append("cast_bf16")
        _finite(output, BOUNDARIES[2])

    actual_adds = [op for op in trace.operations if op["operator"] == "aten.add.Tensor"]
    actual_downcasts = [
        op
        for op in trace.operations
        if op["operator"] == "aten._to_copy.default" and op["output_dtype"] == "torch.bfloat16"
    ]
    _require(
        len(actual_adds) == 2 and all(op["output_dtype"] == "torch.float32" for op in actual_adds),
        "ADDITION_ORDER_MISMATCH",
        "observed ATen adds",
    )
    _require(len(actual_downcasts) == 1, "EARLY_OR_MULTIPLE_DOWNCAST", "observed ATen casts")

    boundary_tensors = dict(zip(BOUNDARIES, (after_shared, after_residual, output), strict=True))
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {
        "backend": BACKEND_ID,
        "implementation_sha256": implementation_hash,
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": str(routed.device),
        "device_name": (
            torch.cuda.get_device_name(routed.device) if routed.device.type == "cuda" else "cpu"
        ),
        "build_runtime": torch.version.hip or torch.version.cuda,
        "observed_aten_operations": trace.operations,
        "accumulator_dtype": "torch.float32",
        "output_dtype": "torch.bfloat16",
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
            "backend": BACKEND_ID,
            "implementation_sha256": implementation_hash,
        }
        for index, (key, value) in enumerate(boundary_tensors.items())
    ]
    applied = Counter(context.upstream_events + tuple(events))
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "foundation_abi": FOUNDATION_ABI,
        "identity": asdict(context.identity),
        "sources": [asdict(sources[role]) for role in inputs],
        "upstream_events": list(context.upstream_events),
        "input_row_weighted": context.input_row_weighted,
        "route_weight_owner": context.route_weight_owner,
        "executed_events": events,
        "route_weight_applied_count": applied["route_weight"],
        "shared_applied_count": applied["shared"],
        "residual_applied_count": applied["residual"],
        "local_downcast_count": len(actual_downcasts),
        "merge_order_hash": _hash_json({"version": RECEIPT_VERSION, "order": events}),
        "rank": context.rank,
        "token_owner_ranks": list(context.token_owner_ranks),
        "shared_replica_ranks": list(context.shared_replica_ranks),
        "identity_discrete_gate": "PASS",
        "scope": "local_forward_reference",
        "certification": "INCOMPLETE: T01 ABI approval and GPU/integration evidence required",
        "actual_provenance": provenance,
        "boundaries": boundaries,
    }
    return MergeResult(output, receipt, boundary_tensors if debug else {})


class SharedResidualMergeReferenceOp:
    """Explicit reference backend factory for the existing semantic registry."""

    __call__ = staticmethod(shared_residual_merge_fwd)
