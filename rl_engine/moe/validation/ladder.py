# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validation ladder L1-L3b runners (contract §4-T09 / §7 WS1 Gate).

Ladders:
- L1 repeat          : same config re-run -> ``route_artifact_fingerprint``
  must match byte-for-byte (§7 WS1 "L1/L2/L3a/L3b recorded 通过").
- L2 invariance      : batch/pack/padding/launch perturbations change only
  Envelope/source_row; Core semantic hash per token must be IDENTICAL
  (§1.3 WS1 "batch/pack/padding/launch invariant").
- L3a oracle         : candidate vs bit-defined CPU oracle, default
  byte-exact on Core rows (§2.5).
- L3b dual-engine    : recorded Megatron vs Miles traces compared through
  the ordered comparator (§2.5 engine-vs-engine must be byte-exact).

Each runner returns a :class:`LadderReport`; strict mismatches surface the
first mismatch (six-tuple + owner/issue) and are never averaged away.
"""

from __future__ import annotations

import torch

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.comparison import TraceComparator
from rl_engine.moe.validation.fingerprint import (
    Artifact,
    RouteIdentity,
    RouteRow,
    per_token_semantic_hashes,
    route_artifact_hash,
)
from rl_engine.moe.validation.first_mismatch import (
    FirstMismatch,
    MismatchKey,
    TraceEvent,
)
from rl_engine.moe.validation.report import (
    LadderReport,
    make_fail as _fail,
    make_pass as _pass,
)


# --- L1: repeat --------------------------------------------------------------

def run_l1_repeat(
    case_id: str,
    run_a: Artifact,
    run_b: Artifact,
) -> LadderReport:
    """Same-config repeat: artifact hash must be byte-identical (§2.4)."""
    ha, hb = route_artifact_hash(run_a), route_artifact_hash(run_b)
    if ha == hb:
        return _pass("L1", case_id, artifact_hash=ha)
    # locate first diverging row for the report (artifact-level first diff)
    idx = next(
        (i for i, (ra, rb) in enumerate(zip(run_a.rows, run_b.rows))
         if route_artifact_hash(Artifact(
             identity=run_a.identity, rows=[ra], envelopes=[run_a.envelopes[i]]))
         != route_artifact_hash(Artifact(
             identity=run_b.identity, rows=[rb], envelopes=[run_b.envelopes[i]]))),
        None,
    )
    where = f"first differing source_row block at index {idx}" if idx is not None \
        else "artifact hashes differ (row-level localization unavailable)"
    return _fail("L1", case_id, P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
                 f"artifact hash {ha[:12]}.. vs {hb[:12]}..: {where}",
                 artifact_hash_a=ha, artifact_hash_b=hb)


# --- L2: batch/pack/padding/launch invariance --------------------------------

def run_l2_invariance(
    case_id: str,
    base: Artifact,
    perturbed: Artifact,
    *,
    variant: str,
) -> LadderReport:
    """Cross-variant: per-token semantic hashes must be IDENTICAL (§2.4).

    Only Core semantics are compared; Envelope/source_row/padding placement
    may legitimately differ. Missing tokens -> INCOMPLETE_ARTIFACT; extra or
    duplicated tokens -> AMBIGUOUS_GLOBAL_TOKEN_MAPPING.
    """
    sa = per_token_semantic_hashes(base.rows, base.identity)
    sb = per_token_semantic_hashes(perturbed.rows, perturbed.identity)

    missing = sorted(set(sa) - set(sb))
    extra_tokens = sorted(set(sb) - set(sa))
    if missing:
        return _fail("L2", case_id, P3Verdict.INCOMPLETE_ARTIFACT,
                     f"variant={variant}: tokens missing in perturbed run: {missing[:8]}",
                     missing=missing, variant=variant)
    if extra_tokens:
        return _fail("L2", case_id, P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
                     f"variant={variant}: unexpected tokens: {extra_tokens[:8]}",
                     extra_tokens=extra_tokens, variant=variant)

    for token in sorted(sa):
        if sa[token] != sb[token]:
            return _fail(
                "L2", case_id, P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
                f"variant={variant}: first differing token={token} "
                f"semantic {sa[token][:12]}.. vs {sb[token][:12]}..",
                first_token=int(token), variant=variant,
            )
    return _pass("L2", case_id, f"variant={variant}: semantic identical",
                 tokens=len(sa), variant=variant)


# --- L3a: oracle byte-exact ----------------------------------------------------

def run_l3a_oracle(
    case_id: str,
    identity: RouteIdentity,
    oracle_rows: list[RouteRow],
    candidate_rows: list[RouteRow],
) -> LadderReport:
    """CUDA candidate vs bit-defined oracle: Core rows byte-exact (§2.5).

    Padding rows (global_token_id == -1) are excluded here; they are audited
    by L1's artifact gate only (§2.5 "padding 只由同配置 artifact gate 审计").
    """
    key = lambda r: (r.global_token_id, r.topk_index)  # noqa: E731
    o = sorted([r for r in oracle_rows if r.global_token_id != -1], key=key)
    c = sorted([r for r in candidate_rows if r.global_token_id != -1], key=key)

    if len(o) != len(c):
        return _fail("L3a", case_id, P3Verdict.INCOMPLETE_ARTIFACT,
                     f"active row count oracle={len(o)} candidate={len(c)}")

    for ro, rc in zip(o, c):
        differing = [
            f for f in (
                "global_token_id", "input_token_id", "absolute_layer",
                "router_mode", "topk_index", "logical_expert_id", "valid",
                "invalid_reason", "route_weight", "weight_score",
                "selection_score",
            )
            if getattr(ro, f) != getattr(rc, f)
        ]
        if differing:
            fm = FirstMismatch(
                found=True,
                key=MismatchKey(absolute_layer=ro.absolute_layer,
                                site="selection" if ro.router_mode == "learned" else "hash_lookup",
                                pass_direction="forward",
                                event_index=ro.topk_index,
                                global_token_id=ro.global_token_id, rank=0),
                owner="T04" if ro.router_mode == "learned" else "T03",
                issue="#44" if ro.router_mode == "learned" else "#42",
                boundary="L3a byte gate", phase="WS1", artifact="core_rows",
                detail=f"token={ro.global_token_id} slot={ro.topk_index} "
                       f"fields differ: {differing}",
            )
            verdict = (P3Verdict.TOPK_ORDER_MISMATCH
                       if {"topk_index", "logical_expert_id"} & set(differing)
                       else P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH
                       if "route_weight" in differing
                       else P3Verdict.BYTE_MISMATCH)
            return _fail("L3a", case_id, verdict, fm.detail, fm,
                         differing_fields=differing)
    return _pass("L3a", case_id, f"{len(o)} active rows byte-exact",
                 active_rows=len(o))


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
    active_mask: torch.Tensor | None = None,
) -> LadderReport:
    """Recorded Megatron vs Miles through the ordered comparator (§2.5).

    Engine-vs-engine must be byte-exact; the four-stage walk (identity ->
    discrete -> score/weight -> gradient) locates the first divergence with
    owner/issue attribution.
    """
    cmp = TraceComparator(case_id, phase="WS1")
    cmp.check_identity(lhs_meta, rhs_meta)
    if not cmp.report().stopped_early:
        cmp.check_discrete(lhs_events, rhs_events)
        stage_ok = cmp.report().stages[-1].passed if len(cmp.report().stages) > 1 else False
        if stage_ok:
            cmp.check_score_weight(lhs_weights, rhs_weights, active_mask=active_mask)
            if cmp.report().stages[-1].passed:
                cmp.check_gradient(lhs_dz, rhs_dz)

    rep = cmp.report()
    if rep.passed:
        return _pass("L3b", case_id, "dual-engine byte-exact across 4 stages")
    stage = next((s for s in rep.stages if not s.passed), None)
    return _fail("L3b", case_id, rep.primary, stage.notes[0] if stage and stage.notes
                 else (stage.mismatch.detail if stage and stage.mismatch else "walk failed"),
                 stage.mismatch if stage else None,
                 walked=[s.stage for s in rep.stages])
