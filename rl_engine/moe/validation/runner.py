# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Router validation runners: shared results, WS1 stages, and WS2 checks.

This module owns every validation-stage runner and the report vocabulary they
share (``ValidationReport`` / ``make_pass`` / ``make_fail``), so sibling modules
(paired_check) and tests import one module instead of reaching into each
other's internals.

WS1 stages:
- L1 repeat          : same config re-run -> ``route_artifact_fingerprint``
  must match byte-for-byte (recorded runs must pass).
- L2 invariance      : batch/pack/padding/launch perturbations change only
  Envelope/source_row; Core semantic hash per token must be IDENTICAL
  (batch/pack/padding/launch invariant).
- L3a oracle         : candidate vs bit-defined CPU oracle, default
  byte-exact on Core rows.
- L3b dual-engine    : recorded Megatron vs Miles traces compared through
  the ordered comparator (engine-vs-engine must be byte-exact).

WS2 extensions (rank dimension + cross-config):
- :func:`check_rank_completeness` — every expected rank must contribute an
  artifact (missing -> MISSING_RANK, provider band).
- :func:`run_ws2_cross_config` — compare two configurations' rank sets:
  partition ownership must cover the expected token set exactly once;
  replica ownership (TP) may repeat tokens but every carrier's semantic
  hash must be identical; any token-level divergence names the first
  offending (token, rank) pair via the six-tuple.

Each runner returns a :class:`ValidationReport`; strict mismatches surface the
first mismatch (six-tuple + owning component) and are never averaged away.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import struct
from typing import Any

import torch

from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.comparison import TraceComparator
from rl_engine.moe.validation.t01_fingerprint import (
    Artifact,
    RouteRow,
    per_token_semantic_hashes,
    route_artifact_hash,
)
from rl_engine.moe.validation.first_mismatch import (
    FirstMismatch,
    MismatchKey,
    TraceEvent,
)


# --- shared report type --------------------------------------------------------


@dataclass
class ValidationReport:
    """Shared result type for every validation stage.

    ``verdict`` is ``None`` on success. ``first_mismatch`` localizes the
    earliest divergence, while ``extra`` carries stage-specific data.
    """

    stage: str  # L1 | L2 | L3a | L3b | WS2-rank | WS2-cross | paired*
    case_id: str
    passed: bool
    verdict: RouterVerdict | None  # primary; None iff passed
    detail: str = ""
    first_mismatch: FirstMismatch | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        flag = "PASS" if self.passed else f"FAIL({self.verdict.name if self.verdict else '?'})"
        return f"[{self.stage}] case={self.case_id} {flag} {self.detail[:120]}"


def make_fail(
    stage: str,
    case_id: str,
    verdict: RouterVerdict,
    detail: str,
    fm: FirstMismatch | None = None,
    **extra: Any,
) -> ValidationReport:
    """Build a failed report with an explicit verdict."""
    return ValidationReport(
        stage=stage,
        case_id=case_id,
        passed=False,
        verdict=verdict,
        detail=detail,
        first_mismatch=fm,
        extra=extra,
    )


def make_pass(stage: str, case_id: str, detail: str = "", **extra: Any) -> ValidationReport:
    """Build a passed report; success has no error verdict."""
    return ValidationReport(
        stage=stage, case_id=case_id, passed=True, verdict=None, detail=detail, extra=extra
    )


# --- L1: repeat --------------------------------------------------------------


def run_l1_repeat(
    case_id: str,
    run_a: Artifact,
    run_b: Artifact,
) -> ValidationReport:
    """L1: require identical canonical artifact hashes across a repeat run.

    The hash includes Core data and stable identity/Envelope fields. Row-level
    localization is diagnostic only and never weakens the strict hash gate.
    """
    try:
        ha, hb = route_artifact_hash(run_a), route_artifact_hash(run_b)
    except ValueError as exc:
        return make_fail(
            "L1",
            case_id,
            RouterVerdict.CORRUPT_ARTIFACT,
            f"artifact structure is invalid: {exc}",
        )
    if ha == hb:
        return make_pass("L1", case_id, artifact_hash=ha)
    # Identity, length, or out-of-row metadata differences may remain at
    # artifact-level when no differing source-row block can be localized.
    idx = next(
        (
            i
            for i, (ra, rb) in enumerate(zip(run_a.rows, run_b.rows, strict=False))
            if route_artifact_hash(
                Artifact(identity=run_a.identity, rows=[ra], envelopes=[run_a.envelopes[i]])
            )
            != route_artifact_hash(
                Artifact(identity=run_b.identity, rows=[rb], envelopes=[run_b.envelopes[i]])
            )
        ),
        None,
    )
    where = (
        f"first differing source_row block at index {idx}"
        if idx is not None
        else "artifact hashes differ (row-level localization unavailable)"
    )
    return make_fail(
        "L1",
        case_id,
        RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
        f"artifact hash {ha[:12]}.. vs {hb[:12]}..: {where}",
        artifact_hash_a=ha,
        artifact_hash_b=hb,
    )


# --- L2: batch/pack/padding/launch invariance --------------------------------


def run_l2_invariance(
    case_id: str,
    base: Artifact,
    perturbed: Artifact,
    *,
    variant: str,
) -> ValidationReport:
    """Cross-variant: per-unit semantic hashes must be IDENTICAL.

    The semantic unit is the routing decision ``(absolute_layer,
    global_token_id)``. Only Core semantics are compared; Envelope/
    source_row/padding placement may legitimately differ. Missing units
    -> INCOMPLETE_ARTIFACT; extra or duplicated units ->
    AMBIGUOUS_GLOBAL_TOKEN_MAPPING.

    Perturbation axes: batch/pack/padding changes are exercised directly
    (repack + padding rows). Launch/fusion perturbations, in the synthetic
    stand-in, are modeled by Envelope-level differences (run_id / engine_id /
    attempt_id / placement) — Core semantics must be invariant to those by
    construction; a recorded producer exercising real launch geometry lands
    with the start kit.
    """
    sa = per_token_semantic_hashes(base.rows, base.identity)
    sb = per_token_semantic_hashes(perturbed.rows, perturbed.identity)

    missing = sorted(set(sa) - set(sb))
    extra_units = sorted(set(sb) - set(sa))
    if missing:
        pretty = [f"(layer={lay}, token={tok})" for lay, tok in missing[:8]]
        return make_fail(
            "L2",
            case_id,
            RouterVerdict.INCOMPLETE_ARTIFACT,
            f"variant={variant}: routing decisions missing in perturbed run: {pretty}",
            missing=[list(u) for u in missing],
            variant=variant,
        )
    if extra_units:
        pretty = [f"(layer={lay}, token={tok})" for lay, tok in extra_units[:8]]
        return make_fail(
            "L2",
            case_id,
            RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
            f"variant={variant}: unexpected routing decisions: {pretty}",
            extra=[list(u) for u in extra_units],
            variant=variant,
        )

    for unit in sorted(sa):
        if sa[unit] != sb[unit]:
            layer, token = unit
            return make_fail(
                "L2",
                case_id,
                RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
                f"variant={variant}: first differing unit=(layer={layer}, token={token}) "
                f"semantic {sa[unit][:12]}.. vs {sb[unit][:12]}..",
                first_layer=layer,
                first_token=token,
                variant=variant,
            )
    return make_pass(
        "L2", case_id, f"variant={variant}: semantic identical", units=len(sa), variant=variant
    )


# --- L3a: oracle byte-exact ----------------------------------------------------


def run_l3a_oracle(
    case_id: str,
    oracle_rows: list[RouteRow],
    candidate_rows: list[RouteRow],
) -> ValidationReport:
    """CUDA candidate vs bit-defined oracle: Core rows byte-exact.

    Padding rows (global_token_id == -1) are excluded here; they are audited
    by L1's artifact gate only (padding is audited solely by the
    same-config artifact gate).
    """

    def row_key(r: RouteRow) -> tuple[int, int, int]:
        return r.absolute_layer, r.global_token_id, r.topk_index

    def field_differs(field_name: str, lhs: Any, rhs: Any) -> bool:
        if field_name in {"route_weight", "weight_score", "selection_score"}:
            try:
                return struct.pack("<f", float(lhs)) != struct.pack("<f", float(rhs))
            except (OverflowError, struct.error, TypeError, ValueError):
                return True
        return lhs != rhs

    o = sorted([r for r in oracle_rows if r.global_token_id != -1], key=row_key)
    c = sorted([r for r in candidate_rows if r.global_token_id != -1], key=row_key)

    if len(o) != len(c):
        return make_fail(
            "L3a",
            case_id,
            RouterVerdict.INCOMPLETE_ARTIFACT,
            f"active row count oracle={len(o)} candidate={len(c)}",
        )

    for ro, rc in zip(o, c, strict=True):
        # Non-finite values fail before byte comparison so identical NaN
        # payloads cannot produce a false pass.
        for side, row in (("oracle", ro), ("candidate", rc)):
            for field_name in ("route_weight", "weight_score", "selection_score"):
                value = float(getattr(row, field_name))
                if not math.isfinite(value):
                    site = "weight" if field_name == "route_weight" else "score"
                    owner = "route-plan" if site == "weight" else "router-score"
                    fm = FirstMismatch(
                        found=True,
                        key=MismatchKey(
                            absolute_layer=row.absolute_layer,
                            site=site,
                            pass_direction="forward",
                            event_index=row.topk_index,
                            global_token_id=row.global_token_id,
                            rank=0,
                        ),
                        owner=owner,
                        boundary="L3a non-finite gate",
                        phase="WS1",
                        artifact="core_rows",
                        detail=f"{side} token={row.global_token_id} "
                        f"slot={row.topk_index} field={field_name} "
                        f"is non-finite: {value!r}",
                    )
                    return make_fail(
                        "L3a",
                        case_id,
                        RouterVerdict.NON_FINITE,
                        fm.detail,
                        fm,
                        nonfinite_field=field_name,
                        nonfinite_side=side,
                    )

        differing = [
            f
            for f in (
                "global_token_id",
                "input_token_id",
                "absolute_layer",
                "router_mode",
                "topk_index",
                "logical_expert_id",
                "valid",
                "invalid_reason",
                "route_weight",
                "weight_score",
                "selection_score",
            )
            if field_differs(f, getattr(ro, f), getattr(rc, f))
        ]
        if differing:
            if {"topk_index", "logical_expert_id"} & set(differing):
                site = "selection" if ro.router_mode == "learned" else "hash_lookup"
                owner = "learned-selection" if ro.router_mode == "learned" else "tid2eid-lookup"
            elif "route_weight" in differing:
                site, owner = "weight", "route-plan"
            elif {"weight_score", "selection_score"} & set(differing):
                site, owner = "score", "router-score"
            else:
                site, owner = "handoff", "combine-plan"
            fm = FirstMismatch(
                found=True,
                key=MismatchKey(
                    absolute_layer=ro.absolute_layer,
                    site=site,
                    pass_direction="forward",
                    event_index=ro.topk_index,
                    global_token_id=ro.global_token_id,
                    rank=0,
                ),
                owner=owner,
                boundary="L3a byte gate",
                phase="WS1",
                artifact="core_rows",
                detail=f"token={ro.global_token_id} slot={ro.topk_index} "
                f"fields differ: {differing}",
            )
            verdict = (
                RouterVerdict.TOPK_ORDER_MISMATCH
                if {"topk_index", "logical_expert_id"} & set(differing)
                else RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH
                if "route_weight" in differing
                else RouterVerdict.SCORE_BYTES_MISMATCH
                if {"weight_score", "selection_score"} & set(differing)
                else RouterVerdict.INVALID_DISCRETE_PLAN
                if {"valid", "invalid_reason", "router_mode"} & set(differing)
                else RouterVerdict.BYTE_MISMATCH
            )
            return make_fail("L3a", case_id, verdict, fm.detail, fm, differing_fields=differing)
    return make_pass("L3a", case_id, f"{len(o)} active rows byte-exact", active_rows=len(o))


# --- L3b: recorded dual-engine ---------------------------------------------------


def run_l3b_dual_engine(
    case_id: str,
    lhs_meta: dict[str, Any],
    rhs_meta: dict[str, Any],
    lhs_events: list[TraceEvent],
    rhs_events: list[TraceEvent],
    lhs_weights: torch.Tensor,
    rhs_weights: torch.Tensor,
    lhs_dz: torch.Tensor,
    rhs_dz: torch.Tensor,
    *,
    lhs_scores: torch.Tensor,
    rhs_scores: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    lhs_selection_grad: torch.Tensor | None = None,
    rhs_selection_grad: torch.Tensor | None = None,
) -> ValidationReport:
    """Recorded Megatron vs Miles through the ordered comparator.

    Engine-vs-engine must be byte-exact; the four-stage walk (identity ->
    discrete -> score/weight -> gradient) locates the first divergence with
    owner attribution. ``*_selection_grad`` carries explicit selection-path
    gradient evidence: non-zero values there are a 61 defect even when the
    engines byte-agree on dz (selection/Hash/Top-K/tie are non-differentiable).
    """
    cmp = TraceComparator(case_id, phase="WS1")
    cmp.check_identity(lhs_meta, rhs_meta)
    if not cmp.report().stopped_early:
        cmp.check_discrete(lhs_events, rhs_events)
        stage_ok = cmp.report().stages[-1].passed if len(cmp.report().stages) > 1 else False
        if stage_ok:
            cmp.check_score_weight(
                lhs_weights,
                rhs_weights,
                lhs_scores=lhs_scores,
                rhs_scores=rhs_scores,
                active_mask=active_mask,
            )
            if cmp.report().stages[-1].passed:
                cmp.check_gradient(
                    lhs_dz,
                    rhs_dz,
                    lhs_selection_grad=lhs_selection_grad,
                    rhs_selection_grad=rhs_selection_grad,
                )

    rep = cmp.report()
    if rep.passed:
        return make_pass("L3b", case_id, "dual-engine byte-exact across 4 stages")
    stage = next((s for s in rep.stages if not s.passed), None)
    return make_fail(
        "L3b",
        case_id,
        rep.primary,
        stage.notes[0]
        if stage and stage.notes
        else (stage.mismatch.detail if stage and stage.mismatch else "walk failed"),
        stage.mismatch if stage else None,
        walked=[s.stage for s in rep.stages],
    )


# --- WS2: rank completeness + cross-config ------------------------------------

TOKEN_PARTITION = "partition"  # CP/DP: token appears on exactly one rank
TOKEN_REPLICA = "replica"  # TP: token may repeat; hashes must agree


@dataclass(frozen=True)
class RankArtifact:
    """Artifact submitted by one rank in a parallel configuration."""

    rank: int
    group: str  # e.g. "tp2-sp0", "dp4-cp2"
    artifact: Artifact


def check_rank_completeness(
    case_id: str,
    artifacts: list[RankArtifact],
    expected_ranks: range | list[int],
) -> ValidationReport:
    """Require the actual rank set to exactly match one configuration.

    Duplicate or unexpected ranks and mixed groups indicate stale data and
    take precedence over missing-rank diagnostics.
    """
    expected = set(expected_ranks)
    seen = Counter(ra.rank for ra in artifacts)
    duplicated = sorted(r for r, n in seen.items() if n > 1)
    unexpected = sorted(seen.keys() - expected)
    groups = sorted({ra.group for ra in artifacts})
    if duplicated or unexpected or len(groups) > 1:
        problems: list[str] = []
        if duplicated:
            problems.append(f"duplicated ranks: {duplicated}")
        if unexpected:
            problems.append(f"unexpected ranks: {unexpected}")
        if len(groups) > 1:
            problems.append(f"mixed groups: {groups}")
        return make_fail(
            "WS2-rank",
            case_id,
            RouterVerdict.STALE_RUN_METADATA,
            "; ".join(problems),
            duplicated_ranks=duplicated,
            unexpected_ranks=unexpected,
            groups=groups,
            hint="expected one artifact per (case, config, rank); stale run mix-in?",
        )
    missing = sorted(expected - seen.keys())
    if missing:
        return make_fail(
            "WS2-rank",
            case_id,
            RouterVerdict.MISSING_RANK,
            f"missing ranks: {missing}",
            missing_ranks=missing,
        )
    return make_pass("WS2-rank", case_id, f"ranks {sorted(seen)} complete", ranks=sorted(seen))


def _tokens_by_rank(arts: list[RankArtifact]) -> dict[int, dict[tuple[int, int], str]]:
    """Map each rank to ``(layer, token) -> semantic_hash``."""
    out: dict[int, dict[tuple[int, int], str]] = {}
    for ra in arts:
        out[ra.rank] = per_token_semantic_hashes(ra.artifact.rows, ra.artifact.identity)
    return out


def run_ws2_cross_config(
    case_id: str,
    base: list[RankArtifact],
    other: list[RankArtifact],
    *,
    base_config: str,
    other_config: str,
    ownership: str = TOKEN_PARTITION,
) -> ValidationReport:
    """Cross-config: same token -> identical Core semantic hash.

    ``base`` defines the expected token set. ``other`` must cover it under
    its ownership mode:
    - partition: each base token present exactly once across ``other`` ranks;
      missing -> INCOMPLETE_ARTIFACT, duplicated -> AMBIGUOUS_GLOBAL_TOKEN_MAPPING.
    - replica: token may appear on several ranks; every occurrence must carry
      the base hash; a divergent carrier -> ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
      with (token, rank) named.
    Envelope differences between configs are expected and NOT judged here.
    """
    if ownership not in (TOKEN_PARTITION, TOKEN_REPLICA):
        raise ValueError(f"unknown ownership mode: {ownership}")

    bt = _tokens_by_rank(base)
    ot = _tokens_by_rank(other)
    expected = {t for per in bt.values() for t in per}

    carriers: dict[tuple[int, int], list[int]] = {}
    for rank, per in ot.items():
        for t in per:
            carriers.setdefault(t, []).append(rank)

    missing = sorted(expected - set(carriers))
    if missing:
        pretty = [f"(layer={lay}, token={tok})" for lay, tok in missing[:8]]
        return make_fail(
            "WS2-cross",
            case_id,
            RouterVerdict.INCOMPLETE_ARTIFACT,
            f"{base_config}->{other_config}: routing decisions uncovered: {pretty}",
            missing=[list(u) for u in missing],
            base_config=base_config,
            other_config=other_config,
        )

    extra = sorted(set(carriers) - expected)
    if extra:
        pretty = [f"(layer={lay}, token={tok})" for lay, tok in extra[:8]]
        return make_fail(
            "WS2-cross",
            case_id,
            RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
            f"{base_config}->{other_config}: routing decisions owned by no "
            f"base-config rank: {pretty}",
            extra=[list(u) for u in extra],
            base_config=base_config,
            other_config=other_config,
        )

    if ownership == TOKEN_PARTITION:
        dup = sorted(t for t, rs in carriers.items() if len(rs) > 1)
        if dup:
            pretty = [f"(layer={lay}, token={tok})" for lay, tok in dup[:4]]
            carriers_pretty = {f"(l={lay},t={tok})": carriers[(lay, tok)] for lay, tok in dup[:4]}
            return make_fail(
                "WS2-cross",
                case_id,
                RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
                f"{base_config}->{other_config}: routing decisions owned by >1 rank: "
                f"{carriers_pretty}",
                duplicated=[list(u) for u in dup],
                base_config=base_config,
                other_config=other_config,
            )

    base_hash = {t: next((per[t] for per in bt.values() if t in per), None) for t in expected}
    for u in sorted(expected):
        for rank in sorted(carriers[u]):
            if ot[rank][u] != base_hash[u]:
                layer, token = u
                return make_fail(
                    "WS2-cross",
                    case_id,
                    RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
                    f"{base_config}->{other_config}: unit=(layer={layer}, token={token}) "
                    f"rank={rank} semantic {ot[rank][u][:12]}.. vs base {base_hash[u][:12]}..",
                    first_layer=layer,
                    first_token=token,
                    first_rank=rank,
                    base_config=base_config,
                    other_config=other_config,
                )
    return make_pass(
        "WS2-cross",
        case_id,
        f"{base_config}->{other_config}: {len(expected)} routing decisions "
        f"semantic-exact ({ownership}, ranks={sorted(ot)})",
        units=len(expected),
        ownership=ownership,
        base_config=base_config,
        other_config=other_config,
    )
