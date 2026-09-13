# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""First-mismatch localization and stable attribution (contract §4-T09).

T09 must "按 (absolute_layer, site, pass, event_index, global_token_id, rank)
定位首差分；输出 owner/Issue/boundary/phase/artifact" — i.e. a failing
comparison points at exactly one first divergence and names the responsible
owner task, issue, boundary, phase and artifact, instead of reporting an
averaged error.

This module is pure bookkeeping: it never judges numeric equality (that is
``comparison.py``'s job); it only locates the first divergent event between
two already-normalized trace streams and maps it to ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from rl_engine.moe.p3_verdicts import P3Verdict


class MismatchKey(NamedTuple):
    """The contract's six-tuple localization key.

    Order is fixed by the contract: (absolute_layer, site, pass, event_index,
    global_token_id, rank). Sorting by this tuple yields the canonical event
    order in which the *first* mismatch is searched.
    """

    absolute_layer: int
    site: str            # score | selection | weight | handoff (§2.5)
    pass_direction: str  # forward | backward
    event_index: int
    global_token_id: int
    rank: int


# --- static owner / issue attribution table (contract §3.1 issue map) ------
# site -> (owner task, issue). Kept explicit and exhaustive on purpose: a
# mismatch with no entry must fail closed (UNKNOWN_OWNER), never guess.

_ATTRIBUTION: dict[str, tuple[str, str]] = {
    "score": ("T02", "#41"),        # router_sqrt_softplus fwd/bwd
    "hash_lookup": ("T03", "#42"),  # tid2eid lookup / slot order
    "topk": ("T01", "#43"),         # stable_topk6 (T01-owned golden)
    "selection": ("T04", "#44"),    # learned selection pre/post bias
    "weight": ("T05", "#43"),       # assembler / fingerprint owns weight rows
    "handoff": ("T05", "#46"),      # CombinePlanSeed / downstream handoff
    "bwd": ("T06", "#42/#44"),      # router backward
    "tp_sp": ("T07", "#45"),        # TP/SP global top-6 / hidden integrity
    "placement": ("T08", "#46"),    # CP/DP/PP/EP placement
}

VALID_SITES = frozenset({"score", "hash_lookup", "topk", "selection", "weight", "handoff"})
VALID_PASSES = frozenset({"forward", "backward"})


class UnknownSiteError(ValueError):
    """Raised when a trace event carries a site outside the contract set."""


@dataclass(frozen=True)
class TraceEvent:
    """One normalized comparison event from a recorded trace.

    ``payload`` holds the compared field(s) for this event; equality of
    payloads is decided by the comparator, not here.
    """

    key: MismatchKey
    payload: dict[str, Any]


@dataclass(frozen=True)
class FirstMismatch:
    """Result of a first-mismatch search."""

    found: bool
    key: MismatchKey | None
    owner: str | None          # e.g. "T02"; None when found=False
    issue: str | None          # e.g. "#41"
    boundary: str              # what boundary the event sits on
    phase: str                 # WS1 | WS2 | Integration
    artifact: str | None       # which sealed artifact diverged
    detail: str                # human-readable first-divergence description


_EMPTY = FirstMismatch(
    found=False, key=None, owner=None, issue=None,
    boundary="none", phase="WS1", artifact=None, detail="no mismatch",
)


def _site_owner(site: str) -> tuple[str, str]:
    try:
        return _ATTRIBUTION[site]
    except KeyError:
        raise UnknownSiteError(
            f"site {site!r} has no owner mapping; refusing to guess attribution"
        ) from None


def _validate_event(event: TraceEvent) -> None:
    if event.key.site not in VALID_SITES:
        raise UnknownSiteError(f"event site {event.key.site!r} not in {sorted(VALID_SITES)}")
    if event.key.pass_direction not in VALID_PASSES:
        raise UnknownSiteError(
            f"event pass {event.key.pass_direction!r} not in {sorted(VALID_PASSES)}"
        )
    # backward pass only makes sense on gradient-carrying sites
    if event.key.pass_direction == "backward" and event.key.site not in {"score", "bwd"}:
        raise UnknownSiteError(
            f"backward pass at site {event.key.site!r} violates §2.5 event order"
        )


def canonical_order(events: list[TraceEvent]) -> list[TraceEvent]:
    """Sort events into the canonical comparison order (by six-tuple key)."""
    return sorted(events, key=lambda e: e.key)


def first_mismatch(
    lhs: list[TraceEvent],
    rhs: list[TraceEvent],
    *,
    payload_equal: Any = None,
    phase: str = "WS1",
    artifact_name: str = "route_trace",
) -> FirstMismatch:
    """Locate the first divergent event between two trace streams.

    Args:
        lhs: events from the reference/oracle side (already normalized).
        rhs: events from the candidate side.
        payload_equal: callable ``(lhs_payload, rhs_payload) -> bool``; when
            omitted, plain ``==`` is used. The comparator passes its own
            stage-aware equality here (byte-exact / discrete-exact).
        phase: WS1 | WS2 | Integration, recorded in the result.
        artifact_name: which artifact the streams came from, for the report.

    Fail-closed behaviour: both streams are validated first; an unknown site
    or pass raises ``UnknownSiteError`` rather than producing a possibly
    wrong attribution. Length mismatch is reported as the mismatch at the
    first index where one stream runs out (missing evidence is never "equal").
    """
    if payload_equal is None:
        def payload_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:  # noqa: F811
            return a == b

    for stream in (lhs, rhs):
        for event in stream:
            _validate_event(event)

    ordered_lhs = canonical_order(lhs)
    ordered_rhs = canonical_order(rhs)

    for idx, (a, b) in enumerate(zip(ordered_lhs, ordered_rhs)):
        if a.key != b.key:
            owner, issue = _site_owner(a.key.site)
            return FirstMismatch(
                found=True, key=min(a.key, b.key), owner=owner, issue=issue,
                boundary=f"event[{idx}] key divergence", phase=phase,
                artifact=artifact_name,
                detail=f"keys differ at event {idx}: lhs={a.key} rhs={b.key}",
            )
        if not payload_equal(a.payload, b.payload):
            owner, issue = _site_owner(a.key.site)
            return FirstMismatch(
                found=True, key=a.key, owner=owner, issue=issue,
                boundary=f"event[{idx}] payload", phase=phase,
                artifact=artifact_name,
                detail=(
                    f"first payload divergence at {a.key}: "
                    f"lhs={_short(a.payload)} rhs={_short(b.payload)}"
                ),
            )

    if len(ordered_lhs) != len(ordered_rhs):
        shorter = "lhs" if len(ordered_lhs) < len(ordered_rhs) else "rhs"
        idx = min(len(ordered_lhs), len(ordered_rhs))
        survivor = ordered_rhs if shorter == "lhs" else ordered_lhs
        key = survivor[idx].key
        owner, issue = _site_owner(key.site)
        return FirstMismatch(
            found=True, key=key, owner=owner, issue=issue,
            boundary=f"event[{idx}] stream length", phase=phase,
            artifact=artifact_name,
            detail=f"{shorter} stream ended early; unmatched event {key}",
        )

    return _EMPTY


def _short(payload: dict[str, Any]) -> str:
    text = repr(payload)
    return text if len(text) <= 96 else text[:93] + "..."
