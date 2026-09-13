# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P3 unified fail-closed verdict codes (contract §6, frozen).

Rules frozen by the contract:
- ``P3Verdict`` is INT32. Device may write only 1-2 (3-9 reserved), provider
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


class P3Verdict(IntEnum):
    """Unified fail-closed verdict codes (contract §6)."""

    # --- provider layer (operator verdicts) ---
    PASS = 0                     # launch/readback/echo/status all clean
    # device-writable (kernel may only atomicMin these two)
    NON_FINITE = 1               # P3's own computation produced non-finite
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
DEVICE_WRITABLE = frozenset({P3Verdict.NON_FINITE, P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE})

#: Values reserved for future device use (3-9); anything else written by a
#: kernel is CORRUPT_ARTIFACT per contract §6.
DEVICE_RESERVED_RANGE = range(3, 10)

#: Provider-writable band.
PROVIDER_RANGE = range(10, 23)

#: Runner-writable band.
RUNNER_RANGE = range(50, 73)


def is_valid_device_status(value: int) -> bool:
    """True iff a kernel-written device status value is legal (1, 2)."""
    try:
        return P3Verdict(value) in DEVICE_WRITABLE
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


def primary_verdict(verdicts: list[P3Verdict]) -> P3Verdict | None:
    """Pick the primary verdict among multiple failures (contract §6 order).

    Fixed priority: infrastructure integrity / identity / schema ->
    upstream evidence -> discrete plan -> numeric bytes -> fingerprint ->
    diagnostics. Implementation: explicit rank map, stable for unknown codes
    (they rank last, preserving input order via ``sorted`` stability).
    """
    if not verdicts:
        return None

    rank: dict[P3Verdict, int] = {}
    order = [
        # infrastructure integrity / identity / schema
        [P3Verdict.CORRUPT_ARTIFACT, P3Verdict.IDENTITY_DRIFT, P3Verdict.SCHEMA_MISMATCH,
         P3Verdict.STALE_RUN_METADATA, P3Verdict.INCOMPLETE_ARTIFACT],
        # upstream evidence
        [P3Verdict.UPSTREAM_VERDICT_MISSING, P3Verdict.UPSTREAM_EVIDENCE_MISSING,
         P3Verdict.UPSTREAM_CONTRACT_MISMATCH, P3Verdict.UPSTREAM_NON_FINITE,
         P3Verdict.MISSING_PROVENANCE, P3Verdict.MISSING_BOUNDARY_TRACE,
         P3Verdict.MISSING_RANK],
        # discrete plan
        [P3Verdict.INVALID_DISCRETE_PLAN, P3Verdict.TOPK_ORDER_MISMATCH,
         P3Verdict.TIE_BREAK_POLICY_MISMATCH, P3Verdict.FORBIDDEN_LOCAL_SHARD_TOPK,
         P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING, P3Verdict.INVALID_PLACEMENT_MAP,
         P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH, P3Verdict.HASH_TABLE_MISMATCH,
         P3Verdict.LOGIT_ROUND_POINT_MISMATCH, P3Verdict.GATE_SHARDING_MISMATCH,
         P3Verdict.INVALID_PROFILE],
        # numeric bytes
        [P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH, P3Verdict.SCORE_BYTES_MISMATCH,
         P3Verdict.GRADIENT_BYTES_MISMATCH, P3Verdict.BYTE_MISMATCH,
         P3Verdict.SELECTION_GRADIENT_PRESENT, P3Verdict.SILENT_FALLBACK,
         P3Verdict.NON_FINITE, P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE],
        # fingerprint
        [P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
         P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
         P3Verdict.PRE_UPDATE_WEIGHT_DRIFT],
        # diagnostics
        [P3Verdict.NATURAL_ROUTE_MISMATCH],
    ]
    for group_rank, group in enumerate(order):
        for v in group:
            rank[v] = group_rank

    return sorted(verdicts, key=lambda v: rank.get(v, len(order)))[0]
