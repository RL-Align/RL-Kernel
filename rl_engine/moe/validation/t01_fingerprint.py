# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ TEMPORARY: Stand-in until T01 publishes the canonical schema.             │
# │ Replace in place so consumers retain the same import path.                 │
# └─────────────────────────────────────────────────────────────────────────┘

"""Minimal canonical serialization and fingerprints for router validation.

Scope note: the *unique* assembler and the frozen schema are separate
deliverables (the start kit will publish the official one). Until then this
module is the minimal, explicitly-versioned stand-in: single-sourced and
replaceable in-place when the start kit publishes the real schema. It
implements L1 (same-config repeat -> artifact hash) and L2 (cross
batch/pack/padding -> semantic hash only), and must not grow assembler
logic beyond hashing.

Rules implemented:
- semantic hash canonical order: schema/model identity, layer/mode/token,
  policy/round/tie, table/bias (absent -> 32 zero bytes + present=false),
  selection/weight source, then per slot 0..k-1 (k = number of selected
  experts; DSV4-Flash uses k=6, the form is k-agnostic):
  topk_index/logical_expert_id/valid/invalid_reason/route_weight/
  weight_score/selection_score/capacity.
- ints little-endian, floats raw bytes, type/length-prefixed.
- semantic hash EXCLUDES fingerprint itself, run/engine/attempt/case/rank/
  backend, and padding rows.
- artifact hash: per (case,config,rank), over all rows by source_row
  INCLUDING padding, plus Envelope canonical fields. L1 compares artifact
  hash; cross-config compares semantic hash only.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from dataclasses import dataclass, field

CANONICAL_SCHEMA_VERSION = "router-canonical.v1"


# --- primitive encoders (type/length-prefixed) ------------------------------


def enc_int(value: int) -> bytes:
    return b"i" + struct.pack("<q", int(value))


def enc_float32(value: float) -> bytes:
    return b"f" + struct.pack("<f", float(value))


def enc_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return b"s" + struct.pack("<Q", len(raw)) + raw


def enc_bool(value: bool) -> bytes:
    return b"b" + (b"\x01" if value else b"\x00")


def enc_optional_str(value: str | None) -> bytes:
    if value is None:
        return b"o\x00"
    raw = value.encode("utf-8")
    return b"o\x01" + struct.pack("<Q", len(raw)) + raw


ABSENT_FIELD_32Z = b"\x00" * 32  # absent -> 32 zero bytes + present=false


def enc_absent() -> bytes:
    return enc_bool(False) + ABSENT_FIELD_32Z


def enc_fingerprint(value: str | None) -> bytes:
    if value is None:
        return enc_absent()
    try:
        return enc_bool(True) + bytes.fromhex(value)
    except ValueError:
        # fail-closed with a locatable message: a fabricated non-hex value
        # (e.g. the string "None") must not pass silently as a fingerprint
        raise ValueError(f"fingerprint must be a hex string, got {value[:16]!r}") from None


# --- route record model ------------------------------------------------------


@dataclass(frozen=True)
class RouteRow:
    """One (token, slot) Core record, minimal canonical field set."""

    global_token_id: int
    input_token_id: int
    absolute_layer: int
    router_mode: str  # hash | learned
    topk_index: int  # slot ordinal, 0..k-1
    logical_expert_id: int
    valid: bool
    invalid_reason: str | None
    route_weight: float
    weight_score: float
    selection_score: float


@dataclass(frozen=True)
class RouteIdentity:
    """Per-case identity fields entering the semantic hash header."""

    schema_version: str = CANONICAL_SCHEMA_VERSION
    checkpoint_id: str = "ckpt"
    weight_id: str = "w"
    logit_round_point: str = "fp32_direct"
    tie_break_policy: str = "q_desc_id_asc"
    capacity_policy: str = "dropless_v1"
    table_fingerprint: str | None = None  # hash layers
    bias_fingerprint: str | None = None  # learned layers
    selection_source: str = "pre_bias_score"
    weight_source: str = "pre_bias_score"


# --- semantic hash -----------------------------------------------------------


def _row_bytes(row: RouteRow) -> bytes:
    return b"".join(
        [
            enc_int(row.global_token_id),
            enc_int(row.input_token_id),
            enc_int(row.absolute_layer),
            enc_str(row.router_mode),
            enc_int(row.topk_index),
            enc_int(row.logical_expert_id),
            enc_bool(row.valid),
            enc_optional_str(row.invalid_reason),
            enc_float32(row.route_weight),
            enc_float32(row.weight_score),
            enc_float32(row.selection_score),
        ]
    )


def _header_bytes(identity: RouteIdentity, absolute_layer: int, router_mode: str) -> bytes:
    return b"".join(
        [
            enc_str(identity.schema_version),
            enc_str(identity.checkpoint_id),
            enc_str(identity.weight_id),
            enc_int(absolute_layer),
            enc_str(router_mode),
            enc_str(identity.logit_round_point),
            enc_str(identity.tie_break_policy),
            enc_str(identity.capacity_policy),
            enc_fingerprint(identity.table_fingerprint),
            enc_fingerprint(identity.bias_fingerprint),
            enc_str(identity.selection_source),
            enc_str(identity.weight_source),
        ]
    )


def route_semantic_hash(
    rows: list[RouteRow],
    identity: RouteIdentity,
) -> str:
    """Case-level hash over routing-decision units in ascending order.

    A routing decision is identified by ``(absolute_layer,
    global_token_id)`` — the same token routed at different layers is two
    separate decisions and must never be merged into one hash. Padding rows
    (``global_token_id == -1``) never enter the semantic hash. Per-unit
    hashes are available via :func:`per_token_semantic_hashes` so callers
    can report unit-level diffs instead of only a case hash.
    """
    per_unit = per_token_semantic_hashes(rows, identity)
    h = hashlib.sha256()
    h.update(enc_str("router.semantic.v1"))
    for layer, token in sorted(per_unit):
        h.update(enc_int(layer))
        h.update(enc_int(token))
        h.update(bytes.fromhex(per_unit[layer, token]))
    return h.hexdigest()


def per_token_semantic_hashes(
    rows: list[RouteRow],
    identity: RouteIdentity,
) -> dict[tuple[int, int], str]:
    """Semantic hash per routing-decision unit ``(absolute_layer, token)``.

    The semantic unit is the routing decision: ``(absolute_layer,
    global_token_id)``. Hashing is invariant under any physical row
    permutation — within a unit, rows enter the hash sorted by
    ``topk_index``; the unit header names the layer explicitly. Padding
    rows (``global_token_id == -1``) never enter the semantic hash.
    """
    by_unit: dict[tuple[int, int], list[RouteRow]] = defaultdict(list)
    for r in rows:
        if r.global_token_id == -1:  # padding excluded from semantic hash
            continue
        by_unit[r.absolute_layer, r.global_token_id].append(r)

    out: dict[tuple[int, int], str] = {}
    for (layer, token), unit_rows in by_unit.items():
        mode = unit_rows[0].router_mode
        h = hashlib.sha256()
        h.update(_header_bytes(identity, layer, mode))
        h.update(enc_int(token))
        for r in sorted(unit_rows, key=lambda x: x.topk_index):
            h.update(_row_bytes(r))
        out[layer, token] = h.hexdigest()
    return out


# --- artifact hash (L1) ------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeFields:
    """Minimal Envelope canonical fields entering the artifact hash."""

    run_id: str
    engine_id: str
    attempt_id: int
    source_row: int
    physical_expert_id: int
    rank: int


@dataclass
class Artifact:
    """One (case, config, rank) artifact: all rows incl. padding + envelope."""

    identity: RouteIdentity
    rows: list[RouteRow]  # ALL rows by source_row, incl. padding
    envelopes: list[EnvelopeFields]  # aligned with rows by source_row
    extra_envelope: dict[str, str] = field(default_factory=dict)


def route_artifact_hash(artifact: Artifact) -> str:
    """Artifact hash over all rows (incl. padding) + Envelope."""
    h = hashlib.sha256()
    h.update(enc_str("router.artifact.v1"))
    h.update(
        _header_bytes(
            artifact.identity,
            artifact.rows[0].absolute_layer if artifact.rows else -1,
            artifact.rows[0].router_mode if artifact.rows else "none",
        )
    )
    for row, env in zip(artifact.rows, artifact.envelopes, strict=True):
        h.update(_row_bytes(row))
        h.update(enc_int(env.source_row))
        h.update(enc_int(env.physical_expert_id))
        h.update(enc_int(env.rank))
        h.update(enc_str(env.run_id))
        h.update(enc_str(env.engine_id))
        h.update(enc_int(env.attempt_id))
    for k in sorted(artifact.extra_envelope):
        h.update(enc_str(k))
        h.update(enc_str(artifact.extra_envelope[k]))
    return h.hexdigest()
