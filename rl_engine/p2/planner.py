# SPDX-License-Identifier: Apache-2.0
"""Logical paged-cache and ordered-collective mocks; never production stores."""

from __future__ import annotations

import base64
import copy
import math
import struct
from dataclasses import asdict

import torch

from .contract import (
    COMPRESSION,
    MODES,
    Identity,
    Status,
    canonical,
    digest,
    integer,
    require,
    schema_guard,
)
from .oracle import fixed_sum


def _fp32_partial(raw: bytes) -> None:
    require(
        isinstance(raw, bytes) and len(raw) > 0 and len(raw) % 4 == 0,
        Status.SCHEMA_MISMATCH,
        "FP32 partial byte alignment",
    )
    require(
        all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)),
        Status.NON_FINITE,
        "FP32 partial state",
    )


class PlannerMock:
    """Opaque byte transport, ownership and lifecycle, deliberately no arithmetic.

    A caller supplies projected FP32 partial bytes and completed row bytes. The
    mock cannot certify the compressor that produced those bytes. Snapshot mode
    metadata is outside the state hash, so state equality really is cross-mode.
    """

    def __init__(self, identity: Identity, page_size: int = 16):
        identity.validate()
        integer(page_size, "page_size", 1)
        require(
            128 % page_size == 0,
            Status.INVALID_PAGE_OR_GENERATION,
            "page size must divide recent window",
        )
        self.identity, self.page_size = identity, page_size
        self.position = -1
        self.recent: list[dict] = []
        self.streams = {s: {"partial": [], "previous": [], "rows": []} for s in ("main", "index")}

    def _entry(self, stream: str, logical: int, data: bytes, *, recent: bool = False) -> dict:
        require(
            isinstance(data, bytes) and len(data) > 0,
            Status.SCHEMA_MISMATCH,
            "nonempty opaque bytes",
        )
        slot = logical % 128 if recent else logical
        return {
            "logical": logical,
            "slot": slot,
            "page": slot // self.page_size,
            "page_identity": self.identity.stream_id(stream, f"page:{slot // self.page_size}"),
            "generation": logical // 128 + 1 if recent else 1,
            "data": base64.b64encode(data).decode("ascii"),
        }

    def step(
        self,
        position: int,
        recent: bytes,
        main_partial: bytes | None = None,
        index_partial: bytes | None = None,
        main_row: bytes | None = None,
        index_row: bytes | None = None,
    ) -> dict:
        require(
            type(position) is int and position == self.position + 1,
            Status.INVALID_GLOBAL_POSITION,
            "tokens must be contiguous, exactly once",
        )
        plan = COMPRESSION[self.identity.layer]
        cr = plan["cr"]
        completed = bool(cr and (position + 1) % cr == 0)
        # Validate every input before mutating anything: failed commits are atomic.
        require(
            (main_row is not None) == completed
            and (index_row is not None) == (completed and plan["index"]),
            Status.EARLY_OR_DUPLICATE_COMMIT,
            "missing/early/extra completed row",
        )
        require(
            (main_partial is not None) == bool(cr) and (index_partial is not None) == plan["index"],
            Status.INVALID_COMPRESSION_PLAN,
            "layer/stream partials",
        )
        new_recent = self._entry("recent", position, recent, recent=True)
        new_parts, new_rows = {}, {}
        for stream, partial, row in (
            ("main", main_partial, main_row),
            ("index", index_partial, index_row),
        ):
            if partial is not None:
                _fp32_partial(partial)
                new_parts[stream] = base64.b64encode(partial).decode("ascii")
            if row is not None:
                new_rows[stream] = self._entry(stream, (position + 1) // cr - 1, row)
        self.recent = (self.recent + [new_recent])[-128:]
        for stream, part in new_parts.items():
            state = self.streams[stream]
            state["partial"].append(part)
            if stream in new_rows:
                state["rows"].append(new_rows[stream])
                state["previous"] = state["partial"][:] if plan["overlap"] else []
                state["partial"] = []
        self.position = position
        return self.snapshot()

    def snapshot(self) -> dict:
        result = {
            "schema": "p2-state.v1",
            "identity": asdict(self.identity),
            "page_size": self.page_size,
            "position": self.position,
            "generation": self.position + 1,
            "recent": self.recent,
            **self.streams,
        }
        return copy.deepcopy({**result, "state_hash": digest(result)})

    @classmethod
    def restore(cls, snapshot: dict) -> PlannerMock:
        validate_snapshot(snapshot)
        mock = cls(Identity(**snapshot["identity"]), snapshot["page_size"])
        mock.position = snapshot["position"]
        mock.recent = copy.deepcopy(snapshot["recent"])
        mock.streams = {s: copy.deepcopy(snapshot[s]) for s in ("main", "index")}
        return mock


@schema_guard
def validate_snapshot(s: dict) -> None:
    require(
        set(s)
        == {
            "schema",
            "identity",
            "page_size",
            "position",
            "generation",
            "recent",
            "main",
            "index",
            "state_hash",
        },
        Status.SCHEMA_MISMATCH,
        "state keys",
    )
    require(s["schema"] == "p2-state.v1", Status.SCHEMA_MISMATCH, "state schema")
    identity = Identity(**s["identity"])
    mock = PlannerMock(identity, s["page_size"])
    p = s["position"]
    integer(p, "position", -1)
    require(
        type(s["generation"]) is int and s["generation"] == p + 1,
        Status.INVALID_PAGE_OR_GENERATION,
        "state generation",
    )
    require(isinstance(s["recent"], list), Status.SCHEMA_MISMATCH, "recent")
    require(
        [e["logical"] for e in s["recent"]] == list(range(max(0, p - 127), p + 1)),
        Status.INVALID_PAGE_OR_GENERATION,
        "recent range",
    )
    for e in s["recent"]:
        require(
            all(type(e[key]) is int for key in ("logical", "slot", "page", "generation")),
            Status.SCHEMA_MISMATCH,
            "integer recent metadata",
        )
        raw = base64.b64decode(e["data"], validate=True)
        require(
            e == mock._entry("recent", e["logical"], raw, recent=True),
            Status.INVALID_PAGE_OR_GENERATION,
            "recent page/slot/generation",
        )
    cr = COMPRESSION[identity.layer]["cr"]
    for stream in ("main", "index"):
        state = s[stream]
        require(
            set(state) == {"partial", "previous", "rows"}, Status.SCHEMA_MISMATCH, "stream keys"
        )
        active = bool(cr) and (stream == "main" or identity.layer == "C4")
        count = (p + 1) // cr if active else 0
        partial = (p + 1) % cr if active else 0
        previous = 4 if active and identity.layer == "C4" and count else 0
        require(
            len(state["rows"]) == count
            and len(state["partial"]) == partial
            and len(state["previous"]) == previous,
            Status.EARLY_OR_DUPLICATE_COMMIT,
            "partial/row cardinality",
        )
        for data in state["partial"] + state["previous"]:
            raw = base64.b64decode(data, validate=True)
            _fp32_partial(raw)
        for i, row in enumerate(state["rows"]):
            require(
                all(type(row[key]) is int for key in ("logical", "slot", "page", "generation")),
                Status.SCHEMA_MISMATCH,
                "integer compressed metadata",
            )
            raw = base64.b64decode(row["data"], validate=True)
            require(
                row == mock._entry(stream, i, raw),
                Status.INVALID_PAGE_OR_GENERATION,
                "compressed page/row/generation",
            )
    require(
        s["state_hash"] == digest({k: v for k, v in s.items() if k != "state_hash"}),
        Status.STATE_BYTES_MISMATCH,
        "state checksum",
    )


@schema_guard
def validate_state_sequence(states: list[dict], identity: dict) -> list[str]:
    """Shared per-token gate for live envelopes and sealed recordings."""
    require(
        isinstance(states, list) and len(states) > 0, Status.INCOMPLETE_ARTIFACT, "per-token states"
    )
    hashes = []
    for position, state in enumerate(states):
        require(state["identity"] == identity, Status.IDENTITY_DRIFT, "state identity")
        require(
            state["position"] == position, Status.INVALID_GLOBAL_POSITION, "missing/reordered state"
        )
        validate_snapshot(state)
        hashes.append(state["state_hash"])
    return hashes


def compare_snapshots(expected: dict, actual: dict) -> Status:
    # Identity before state, and state before any downstream output comparison.
    require(expected["identity"] == actual["identity"], Status.IDENTITY_DRIFT, "state identity")
    validate_snapshot(expected)
    validate_snapshot(actual)
    require(canonical(expected) == canonical(actual), Status.STATE_BYTES_MISMATCH, "state bytes")
    return Status.PASS


def candidate_plan(layer: str, position: int, selected: list[int] | None = None) -> dict:
    Identity(layer).validate()
    integer(position, "position")
    cr = COMPRESSION[layer]["cr"]
    completed = (position + 1) // cr if cr else 0
    if layer == "C4":
        require(
            isinstance(selected, list) and len(selected) <= 512,
            Status.INVALID_TOPK_ORDER,
            "C4 requires explicit Top-K order",
        )
        ids = selected
        require(
            all(type(i) is int and 0 <= i < completed for i in ids) and len(ids) == len(set(ids)),
            Status.INVALID_TOPK_ORDER,
            "duplicate/future/invalid C4 id",
        )
    else:
        require(selected is None, Status.INVALID_CANDIDATE_ORDER, "no indexer on C0/C128")
        ids = list(range(completed))
    return {
        "layer": layer,
        "position": position,
        "compressed": ids,
        "recent": list(range(max(0, position - 127), position + 1)),
        "order": "compressed_then_recent",
        "softmax_denominators": 1,
        "sink_has_value": False,
    }


@schema_guard
def validate_candidates(plan: dict) -> None:
    require(plan["softmax_denominators"] == 1, Status.MULTIPLE_SOFTMAX_DENOMINATORS, "ONE softmax")
    require(plan["sink_has_value"] is False, Status.INVALID_SINK_SEMANTICS, "sink has no V")
    expected = candidate_plan(
        plan["layer"], plan["position"], plan["compressed"] if plan["layer"] == "C4" else None
    )
    require(plan == expected, Status.INVALID_CANDIDATE_ORDER, "canonical candidates")


def topology(tokens: int, cp: int, tp: int) -> dict:
    integer(tokens, "tokens", 1)
    integer(cp, "cp", 1)
    integer(tp, "tp", 1)
    require(
        cp in (1, 2, 4) and tp in (1, 2, 4, 8),
        Status.UNSUPPORTED_CAPABILITY,
        "CP/TP mock configurations",
    )
    # Balanced contiguous ranges; ownership belongs to the completing token.
    ranges = [(tokens * r // cp, tokens * (r + 1) // cp) for r in range(cp)]
    owners = [next(r for r, (a, b) in enumerate(ranges) if a <= t < b) for t in range(tokens)]
    return {
        "version": "p2-topology-mock.v1",
        "cp": cp,
        "tp": tp,
        "ranges": [list(r) for r in ranges],
        "token_owners": owners,
        "head_ranges": [[r * 64 // tp, (r + 1) * 64 // tp] for r in range(tp)],
        "global_index_visibility": list(range(tokens // 4)),
        "c4_owners": [owners[t] for t in range(3, tokens, 4)],
        "c128_owners": [owners[t] for t in range(127, tokens, 128)],
    }


@schema_guard
def validate_topology(plan: dict, tokens: int) -> None:
    expected = topology(tokens, plan["cp"], plan["tp"])
    require(
        plan["global_index_visibility"] == expected["global_index_visibility"],
        Status.MISSING_GLOBAL_VISIBILITY,
        "global Index-K",
    )
    require(
        plan["token_owners"] == expected["token_owners"]
        and plan["c4_owners"] == expected["c4_owners"]
        and plan["c128_owners"] == expected["c128_owners"],
        Status.DUPLICATE_LOGICAL_OWNER,
        "unique completion owner",
    )
    require(plan == expected, Status.SCHEMA_MISMATCH, "rank/topology metadata")


def ordered_collective(shards: dict[int, torch.Tensor], tp: int) -> torch.Tensor:
    """P4 mock: gather head contributions, then one GLOBAL tree (not local sums)."""
    require(tp in (1, 2, 4, 8), Status.UNSUPPORTED_CAPABILITY, "TP")
    require(set(shards) == set(range(tp)), Status.MISSING_RANK, "ordered collective ranks")
    shapes = [shards[r].shape for r in range(tp)]
    require(
        all(len(s) >= 1 and s[0] == 64 // tp and s[1:] == shapes[0][1:] for s in shapes),
        Status.SCHEMA_MISMATCH,
        "per-global-head contributions required, not partial sums",
    )
    return fixed_sum(torch.cat([shards[r].float() for r in range(tp)], 0), 0)


def validate_mode(mode: str) -> None:
    require(mode in MODES, Status.UNSUPPORTED_CAPABILITY, mode)
