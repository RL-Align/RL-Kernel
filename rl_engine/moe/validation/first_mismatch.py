# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""First-mismatch localization and stable attribution.

The first divergence between two trace streams is localized by the six-tuple
(absolute_layer, site, pass, event_index, global_token_id, rank) and
mapped to the component that owns the failing stage — i.e. a failing
comparison points at exactly one first divergence and names the responsible
component, boundary, phase and artifact, instead of reporting an
averaged error.

This module is pure bookkeeping: it never judges numeric equality (that is
``comparison.py``'s job); it only locates the first divergent event between
two already-normalized trace streams and maps it to ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple


class MismatchKey(NamedTuple):
    """The six-tuple localization key.

    Order is fixed: (absolute_layer, site, pass, event_index, global_token_id,
    rank). Sorting by this tuple yields the canonical event order in which
    the *first* mismatch is searched.

    """

    absolute_layer: int
    # score | hash_lookup | topk | selection | weight | handoff | bwd | tp_sp | placement
    site: str
    pass_direction: str  # forward | backward
    event_index: int
    global_token_id: int
    rank: int


# --- static component attribution table ------------------------------------
# site -> owning component. Kept explicit and exhaustive on purpose: a
# mismatch with no entry must fail closed (UNKNOWN_OWNER), never guess.

_ATTRIBUTION: dict[str, tuple[str, str]] = {
    "score": ("router-score", "#41"),
    "hash_lookup": ("tid2eid-lookup", "#42"),
    "topk": ("stable-topk", "#43"),
    "selection": ("learned-selection", "#44"),
    "weight": ("route-plan", "#43/#44"),
    "handoff": ("combine-plan", "#43/#46"),
    "bwd": ("router-backward", "#42/#44"),
    "tp_sp": ("tp-sp-topk", "#45"),
    "placement": ("parallel-placement", "#46"),
}

# Valid sites are exactly the attribution table's keys: the table is the
# single source of truth, so a new site cannot be accepted without an owner.
VALID_SITES = frozenset(_ATTRIBUTION)
VALID_PASSES = frozenset({"forward", "backward"})


class UnknownSiteError(ValueError):
    """Raised when a trace event carries a site outside the known set."""


@dataclass(frozen=True)
class TraceEvent:
    """One normalized comparison event from a recorded trace.

    ``payload`` holds the compared field(s) for this event; equality of
    payloads is decided by the comparator, not here.

    ``key`` identifies where the event occurred; ``payload`` stores the
    value produced there.
    """

    key: MismatchKey
    payload: dict[str, Any]


@dataclass(frozen=True)
class FirstMismatch:
    """Result of a first-mismatch search with stable attribution metadata."""

    found: bool
    key: MismatchKey | None
    owner: str | None  # owning component, e.g. "router-score"; None when found=False
    boundary: str  # what boundary the event sits on
    phase: str  # WS1 | WS2 | Integration
    artifact: str | None  # which sealed artifact diverged
    detail: str  # human-readable first-divergence description

    @property
    def issue(self) -> str | None:
        """Contract issue owning this mismatch site."""
        if self.key is None or self.key.site not in _ATTRIBUTION:
            return None
        return _ATTRIBUTION[self.key.site][1]


def _site_owner(site: str) -> str:
    """Return the component responsible for a validated trace site."""
    try:
        return _ATTRIBUTION[site][0]
    except KeyError:
        raise UnknownSiteError(
            f"site {site!r} has no owner mapping; refusing to guess attribution"
        ) from None


def _validate_event(event: TraceEvent) -> None:
    """Reject events whose site or propagation direction cannot be audited."""
    if event.key.site not in VALID_SITES:
        raise UnknownSiteError(f"event site {event.key.site!r} not in {sorted(VALID_SITES)}")
    if event.key.pass_direction not in VALID_PASSES:
        raise UnknownSiteError(
            f"event pass {event.key.pass_direction!r} not in {sorted(VALID_PASSES)}"
        )
    # Only score and bwd sites may record backward events. Discrete paths are
    # non-differentiable by contract.
    if event.key.pass_direction == "backward" and event.key.site not in {"score", "bwd"}:
        raise UnknownSiteError(f"backward pass at site {event.key.site!r} violates the event order")


def canonical_order(events: list[TraceEvent]) -> list[TraceEvent]:
    """Sort events into canonical six-tuple order."""
    return sorted(events, key=lambda e: e.key)


def first_mismatch(
    lhs: list[TraceEvent],
    rhs: list[TraceEvent],
    *,
    payload_equal: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
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

        def payload_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
            return a == b

    for stream in (lhs, rhs):
        for event in stream:
            _validate_event(event)

    ordered_lhs = canonical_order(lhs)
    ordered_rhs = canonical_order(rhs)

    for idx, (a, b) in enumerate(zip(ordered_lhs, ordered_rhs, strict=False)):
        if a.key != b.key:
            # Derive both key and owner from the earlier event; taking them
            # from different sides would produce contradictory attribution.
            key = min(a.key, b.key)
            return FirstMismatch(
                found=True,
                key=key,
                owner=_site_owner(key.site),
                boundary=f"event[{idx}] key divergence",
                phase=phase,
                artifact=artifact_name,
                detail=f"keys differ at event {idx}: lhs={a.key} rhs={b.key}",
            )
        if not payload_equal(a.payload, b.payload):
            return FirstMismatch(
                found=True,
                key=a.key,
                owner=_site_owner(a.key.site),
                boundary=f"event[{idx}] payload",
                phase=phase,
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
        return FirstMismatch(
            found=True,
            key=key,
            owner=_site_owner(key.site),
            boundary=f"event[{idx}] stream length",
            phase=phase,
            artifact=artifact_name,
            detail=(
                f"{shorter} stream ended early; unmatched event {key} "
                f"(lhs={len(ordered_lhs)} events, rhs={len(ordered_rhs)} events)"
            ),
        )

    return FirstMismatch(
        found=False,
        key=None,
        owner=None,
        boundary="none",
        phase=phase,
        artifact=artifact_name,
        detail="no mismatch",
    )


def _short(payload: dict[str, Any]) -> str:
    """Bound payload text so one mismatch cannot flood a human-readable report."""
    text = repr(payload)
    return text if len(text) <= 96 else text[:93] + "..."
