# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ TEMPORARY: Stand-in until the T01 start kit publishes router_contract.py. │
# │ Replace this file in place so consumers keep the same import path.         │
# └─────────────────────────────────────────────────────────────────────────┘

"""MoE router unified fail-closed verdict codes (frozen).

Rules frozen with this table:
- ``RouterVerdict`` is INT32. Device may write only 1-2 (3-9 reserved), provider
  10-22, runner 50-72, certification 90-99 reserved. New codes can only be
  appended, never re-ordered.
- ``PASS(0)`` is a single-operator verdict; ``CASE_PASS(50)`` is case-level.
  The two must not be conflated.
- Operator APIs never return runner/certification codes; only the case runner
  may produce ``CASE_PASS``.
- Multi-error cases pick the primary verdict by fixed priority:
  infrastructure integrity / identity / schema -> upstream evidence ->
  discrete plan -> numeric bytes -> fingerprint -> diagnostics. The runner
  must not rewrite this priority.
"""

from __future__ import annotations

from enum import IntEnum


class RouterVerdict(IntEnum):
    """Unified fail-closed verdict codes; append-only once published."""

    # --- provider layer (operator verdicts) ---
    PASS = 0  # launch/readback/echo/status all clean
    # device-writable (kernel may only atomicMin these two)
    NON_FINITE = 1  # the router's own computation produced non-finite
    HASH_TABLE_INDEX_OUT_OF_RANGE = 2
    # provider layer
    IDENTITY_DRIFT = 10
    SCHEMA_MISMATCH = 11
    CORRUPT_ARTIFACT = 12
    INCOMPLETE_ARTIFACT = 13
    LOGIT_ROUND_POINT_MISMATCH = 14
    HASH_TABLE_MISMATCH = 15
    UNSUPPORTED_CAPABILITY = 16
    ZERO_ACTIVE_TOKENS = 17
    UPSTREAM_NON_FINITE = 18
    GATE_SHARDING_MISMATCH = 19
    MISSING_RANK = 20
    PRE_UPDATE_WEIGHT_DRIFT = 21
    STALE_RUN_METADATA = 22

    # --- runner layer (case verdicts) ---
    CASE_PASS = 50
    ROUTE_WEIGHT_BYTES_MISMATCH = 51
    SCORE_BYTES_MISMATCH = 52
    GRADIENT_BYTES_MISMATCH = 53
    BYTE_MISMATCH = 54
    TOPK_ORDER_MISMATCH = 55
    TIE_BREAK_POLICY_MISMATCH = 56
    INVALID_DISCRETE_PLAN = 57
    INVALID_PROFILE = 58
    ROUTE_SEMANTIC_FINGERPRINT_MISMATCH = 59
    ROUTE_ARTIFACT_FINGERPRINT_MISMATCH = 60
    SELECTION_GRADIENT_PRESENT = 61
    FORBIDDEN_LOCAL_SHARD_TOPK = 62
    AMBIGUOUS_GLOBAL_TOKEN_MAPPING = 63
    INVALID_PLACEMENT_MAP = 64
    PLACEMENT_MAP_VERSION_MISMATCH = 65
    SILENT_FALLBACK = 66
    MISSING_PROVENANCE = 67
    MISSING_BOUNDARY_TRACE = 68
    UPSTREAM_CONTRACT_MISMATCH = 69
    UPSTREAM_VERDICT_MISSING = 70
    UPSTREAM_EVIDENCE_MISSING = 71
    NATURAL_ROUTE_MISMATCH = 72


#: Device-writable status values (kernel may only write these via atomicMin).
DEVICE_WRITABLE = frozenset({RouterVerdict.NON_FINITE, RouterVerdict.HASH_TABLE_INDEX_OUT_OF_RANGE})

#: Values reserved for future device use (3-9); anything else written by a
#: kernel is CORRUPT_ARTIFACT.
DEVICE_RESERVED_RANGE = range(3, 10)

#: Provider-writable band.
PROVIDER_RANGE = range(10, 23)

#: Runner-writable band.
RUNNER_RANGE = range(50, 73)


def is_valid_device_status(value: int) -> bool:
    """True iff a kernel-written device status value is legal (1, 2)."""
    try:
        return RouterVerdict(value) in DEVICE_WRITABLE
    except ValueError:
        return False


def classify_writable_band(value: int) -> str:
    """Return which layer owns ``value``; used to police layer violations."""
    if value == 0 or value in DEVICE_WRITABLE:
        return "device-or-provider"
    if value in PROVIDER_RANGE:
        return "provider"
    if value in RUNNER_RANGE:
        return "runner"
    return "reserved"


def primary_verdict(verdicts: list[RouterVerdict]) -> RouterVerdict | None:
    """Pick the primary verdict among multiple failures (fixed priority).

    Fixed priority: infrastructure integrity / identity / schema ->
    upstream evidence -> discrete plan -> numeric bytes -> fingerprint ->
    diagnostics. Implementation: explicit rank map, stable for unknown codes
    (they rank last; ties keep input order — ``min`` returns the first
    minimal element).
    """
    if not verdicts:
        return None

    rank: dict[RouterVerdict, int] = {}
    order = [
        # infrastructure integrity / identity / schema (audit-path verdicts:
        # a non-auditable substitute path is provenance-level, not numeric)
        [
            RouterVerdict.CORRUPT_ARTIFACT,
            RouterVerdict.IDENTITY_DRIFT,
            RouterVerdict.SCHEMA_MISMATCH,
            RouterVerdict.STALE_RUN_METADATA,
            RouterVerdict.INCOMPLETE_ARTIFACT,
            RouterVerdict.SILENT_FALLBACK,
        ],
        # upstream evidence
        [
            RouterVerdict.UPSTREAM_VERDICT_MISSING,
            RouterVerdict.UPSTREAM_EVIDENCE_MISSING,
            RouterVerdict.UPSTREAM_CONTRACT_MISMATCH,
            RouterVerdict.UPSTREAM_NON_FINITE,
            RouterVerdict.MISSING_PROVENANCE,
            RouterVerdict.MISSING_BOUNDARY_TRACE,
            RouterVerdict.MISSING_RANK,
        ],
        # discrete plan
        [
            RouterVerdict.INVALID_DISCRETE_PLAN,
            RouterVerdict.TOPK_ORDER_MISMATCH,
            RouterVerdict.TIE_BREAK_POLICY_MISMATCH,
            RouterVerdict.FORBIDDEN_LOCAL_SHARD_TOPK,
            RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
            RouterVerdict.INVALID_PLACEMENT_MAP,
            RouterVerdict.PLACEMENT_MAP_VERSION_MISMATCH,
            RouterVerdict.HASH_TABLE_MISMATCH,
            RouterVerdict.LOGIT_ROUND_POINT_MISMATCH,
            RouterVerdict.GATE_SHARDING_MISMATCH,
            RouterVerdict.INVALID_PROFILE,
        ],
        # numeric bytes
        [
            RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH,
            RouterVerdict.SCORE_BYTES_MISMATCH,
            RouterVerdict.GRADIENT_BYTES_MISMATCH,
            RouterVerdict.BYTE_MISMATCH,
            RouterVerdict.SELECTION_GRADIENT_PRESENT,
            RouterVerdict.NON_FINITE,
            RouterVerdict.HASH_TABLE_INDEX_OUT_OF_RANGE,
        ],
        # fingerprint
        [
            RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
            RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
            RouterVerdict.PRE_UPDATE_WEIGHT_DRIFT,
        ],
        # diagnostics
        [RouterVerdict.NATURAL_ROUTE_MISMATCH],
    ]
    for group_rank, group in enumerate(order):
        for v in group:
            rank[v] = group_rank

    return min(verdicts, key=lambda v: rank.get(v, len(order)))
