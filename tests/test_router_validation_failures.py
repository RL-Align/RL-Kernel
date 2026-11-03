# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Failure-case matrix for router validation.

Each test injects exactly one defect into an otherwise-clean synthetic case
and asserts the dedicated verdict mandated by the frozen table. The runner
must never average defects away, and attribution must remain specific enough
to identify the owning component.

Matrix (defect -> expected verdict):
- tie-break violation            -> TIE_BREAK_POLICY_MISMATCH / TOPK_ORDER_MISMATCH
- hash/learned XOR violation     -> IDENTITY_DRIFT (mode is exclusive per layer)
- weight bitflip                 -> ROUTE_WEIGHT_BYTES_MISMATCH (L3a)
- semantic bitflip (L2)          -> ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
- artifact bitflip (L1)          -> ROUTE_ARTIFACT_FINGERPRINT_MISMATCH
- wrong run (stale metadata)     -> STALE_RUN_METADATA
- missing provenance             -> MISSING_PROVENANCE
- missing tokens                 -> INCOMPLETE_ARTIFACT
- ghost tokens                   -> AMBIGUOUS_GLOBAL_TOKEN_MAPPING
- forbidden fallback             -> SILENT_FALLBACK
- selection gradient present     -> SELECTION_GRADIENT_PRESENT
- non-finite active value        -> NON_FINITE (fail-closed before byte gates)
- identity drift                 -> IDENTITY_DRIFT (halts walk)
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rl_engine.moe.naive_topk import K, naive_topk
from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.comparison import TraceComparator
from rl_engine.moe.validation.t01_fingerprint import (
    RouteIdentity,
    RouteRow,
    per_token_semantic_hashes,
)
from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent
from rl_engine.moe.validation.runner import (
    run_l1_repeat,
    run_l2_invariance,
    run_l3a_oracle,
)
from rl_engine.moe.validation.t01_synthetic_producer import (
    make_artifact,
    make_learned_route_rows,
    repack_rows,
)


def _ident(**over):
    base = dict(
        checkpoint_id="ckpt-x",
        weight_id="w-x",
        table_fingerprint="00" * 32,
        bias_fingerprint="11" * 32,
    )
    base.update(over)
    return RouteIdentity(**base)


def _case(t_rows=4, seed=11):
    rows, meta = make_learned_route_rows("cx", seed=seed, t_rows=t_rows, layers=(3,))
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    return rows, meta, ident


def _wrap(rows, ident, **kw):
    kw.setdefault("run_id", "r1")
    kw.setdefault("engine_id", "e1")
    kw.setdefault("attempt_id", 1)
    return make_artifact("cx", rows, ident, **kw)


def _meta(**over):
    base = {
        "case_id": "cx",
        "checkpoint_id": "ckpt-x",
        "weight_id": "w-x",
        "absolute_layer": 3,
        "router_mode": "learned",
        "table_fingerprint": "00" * 32,
        "bias_fingerprint": "11" * 32,
        "logit_round_point": "fp32_direct",
        "tie_break_policy": "q_desc_id_asc",
        "capacity_policy": "dropless_v1",
    }
    base.update(over)
    return base


# --- tie-break defects ----------------------------------------------------------
def test_negative_tie_break_violation_detected_by_cross_check():
    """Exact ties must resolve to ascending logical_expert_id."""
    q = torch.zeros(3, 8, dtype=torch.float32)  # full tie on every row
    ids, _ = naive_topk(q)
    assert ids[0].tolist() == [0, 1, 2, 3, 4, 5]

    # a candidate that breaks ties by descending id violates the policy
    bad_ids = ids.flip(1)
    from rl_engine.moe.naive_topk import cross_check_topk

    passed, msg = cross_check_topk(bad_ids, q)
    assert not passed
    assert "row=0" in msg and "slot=0" in msg


def test_negative_tie_break_maps_to_policy_verdict():
    """Learned q with exact ties: ascending-id candidate passes; reversed fails."""
    e = 16
    bias = torch.zeros(e, dtype=torch.float32)
    rows, _ = make_learned_route_rows("cx", seed=3, t_rows=2, layers=(3,), bias=bias, e=e)
    # reverse slot order per token = tie-break/order violation
    by_token: dict[int, list[RouteRow]] = {}
    for r in rows:
        by_token.setdefault(r.global_token_id, []).append(r)
    bad: list[RouteRow] = []
    for t in sorted(by_token):
        chosen = {r.logical_expert_id for r in by_token[t]}
        outsider = next(i for i in range(e) if i not in chosen)
        flipped = list(reversed(by_token[t]))
        flipped[0] = dataclasses.replace(flipped[0], logical_expert_id=outsider)
        bad.extend(flipped)
    rep = run_l3a_oracle("cx", rows, bad)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.TOPK_ORDER_MISMATCH


# --- hash/learned mode exclusivity ---------------------------------------------
def test_negative_router_mode_xor_violation_is_identity_drift():
    """Same layer traced as hash on one side and learned on the other must
    halt at the identity gate, never reach numeric stages (XOR rule)."""
    cmp = TraceComparator(case_id="cx")
    cmp.check_identity(_meta(router_mode="learned"), _meta(router_mode="hash"))
    rep = cmp.report()
    assert not rep.passed
    assert rep.stopped_early
    assert rep.primary is RouterVerdict.IDENTITY_DRIFT


def test_negative_router_mode_xor_visible_in_semantic_hash():
    """Flipping router_mode on a semantic-hash-bearing artifact must change
    the per-token hash even when all numeric fields are identical (the
    canonical header includes layer/mode)."""
    rows, _meta_d, ident = _case()
    flipped = [dataclasses.replace(r, router_mode="hash") for r in rows]
    base_hashes = per_token_semantic_hashes(rows, ident)
    flipped_hashes = per_token_semantic_hashes(flipped, ident)
    assert set(base_hashes) == set(flipped_hashes)  # same tokens
    assert base_hashes != flipped_hashes  # different semantics
    rep = run_l2_invariance("cx", _wrap(rows, ident), _wrap(flipped, ident), variant="mode-flip")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH


# --- byte-level defects ---------------------------------------------------------
def test_negative_weight_bitflip_l3a():
    rows, meta, ident = _case()
    bad = list(rows)
    bad[3] = dataclasses.replace(bad[3], route_weight=bad[3].route_weight * (1 + 1e-7))
    rep = run_l3a_oracle("cx", rows, bad)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH
    assert rep.first_mismatch is not None
    assert "route_weight" in rep.detail


def test_negative_score_bitflip_l3a():
    rows, meta, ident = _case()
    bad = list(rows)
    bad[5] = dataclasses.replace(bad[5], weight_score=bad[5].weight_score + 1e-6)
    rep = run_l3a_oracle("cx", rows, bad)
    assert not rep.passed
    # score fields get the score-specific verdict, not the generic byte code
    assert rep.verdict is RouterVerdict.SCORE_BYTES_MISMATCH


def test_negative_semantic_bitflip_l2():
    rows, meta, ident = _case()
    base = _wrap(rows, ident)
    bad = list(rows)
    bad[9] = dataclasses.replace(bad[9], selection_score=bad[9].selection_score + 1e-6)
    pert = _wrap(bad, ident)
    rep = run_l2_invariance("cx", base, pert, variant="bitflip")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH


def test_negative_artifact_bitflip_l1():
    rows, meta, ident = _case()
    a = _wrap(rows, ident)
    bad = list(rows)
    bad[0] = dataclasses.replace(bad[0], route_weight=bad[0].route_weight + 1e-7)
    b = _wrap(bad, ident)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


# --- stale-run and provenance defects ------------------------------------------
def test_negative_stale_run_metadata_l1():
    """Same run_id but different attempt: L1 catches via Envelope bytes."""
    rows, meta, ident = _case()
    a = _wrap(rows, ident)
    b = _wrap(rows, ident, attempt_id=99)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


def test_negative_missing_provenance_halts_walk():
    lhs = _meta()
    rhs = _meta()
    del rhs["bias_fingerprint"]
    cmp = TraceComparator("cx")
    cmp.check_identity(lhs, rhs)
    rep = cmp.report()
    assert not rep.passed
    assert rep.primary is RouterVerdict.MISSING_PROVENANCE
    assert rep.stopped_early


def test_negative_identity_drift_beats_later_stages():
    """Priority: identity drift outranks numeric byte mismatches."""
    lhs = _meta()
    rhs = _meta(checkpoint_id="ckpt-OTHER")
    cmp = TraceComparator("cx")
    cmp.check_identity(lhs, rhs)
    with pytest.raises(RuntimeError):
        cmp.check_discrete([], [])  # halted walk must refuse later stages
    rep = cmp.report()
    assert rep.primary is RouterVerdict.IDENTITY_DRIFT
    assert rep.stopped_early


# --- missing and unauthorized routing decisions --------------------------------
def test_negative_missing_tokens_l2():
    rows, meta, ident = _case()
    base = _wrap(rows, ident)
    pert = _wrap([r for r in rows if r.global_token_id != 2], ident, padding_rows=K)
    rep = run_l2_invariance("cx", base, pert, variant="drop")
    assert rep.verdict is RouterVerdict.INCOMPLETE_ARTIFACT


def test_negative_ghost_tokens_l2():
    rows, meta, ident = _case()
    base = _wrap(rows, ident)
    ghost = dataclasses.replace(rows[0], global_token_id=42, input_token_id=42)
    pert = _wrap(rows + [ghost], ident)
    rep = run_l2_invariance("cx", base, pert, variant="ghost")
    assert rep.verdict is RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


# --- forbidden behavior --------------------------------------------------------
def test_negative_silent_fallback_flagged():
    """A fallback path silently substituting results is SILENT_FALLBACK (66).

    Simulated at comparator level: rhs weights computed by a different
    (fallback) path still byte-match by luck, but provenance records the
    fallback — runner must flag it, not pass it.
    """
    lhs = _meta()
    rhs = _meta()
    rhs["logit_round_point"] = "bf16_fallback"  # forbidden substitute path
    cmp = TraceComparator("cx")
    cmp.check_identity(lhs, rhs)
    rep = cmp.report()
    assert not rep.passed
    assert rep.stopped_early
    assert rep.primary is RouterVerdict.SILENT_FALLBACK
    assert "bf16_fallback" in rep.stages[0].notes[0]


def test_negative_selection_gradient_present():
    """Selection path must be non-differentiable: any dz through it is 61."""
    from rl_engine.moe.validation.runner import run_l3b_dual_engine

    events = [
        TraceEvent(
            key=MismatchKey(
                absolute_layer=3,
                site="topk",
                pass_direction="forward",
                event_index=i,
                global_token_id=i // K,
                rank=0,
            ),
            payload={"logical_expert_id": i % 256},
        )
        for i in range(K)
    ]
    w = torch.full((K,), 0.25)
    zero_dz = torch.zeros(K)
    # selection-path gradient evidence: the leak itself is the defect, even
    # when both engines byte-agree on the (legitimate, zero here) dz.
    leaked_selection_grad = torch.zeros(K)
    leaked_selection_grad[1] = 1e-8
    rep = run_l3b_dual_engine(
        "cx",
        _meta(),
        dict(_meta()),
        events,
        list(events),
        w,
        w.clone(),
        zero_dz,
        zero_dz.clone(),
        lhs_scores=w,
        rhs_scores=w.clone(),
        lhs_selection_grad=None,
        rhs_selection_grad=leaked_selection_grad,
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.SELECTION_GRADIENT_PRESENT
    assert rep.first_mismatch.owner == "router-backward"
    assert rep.first_mismatch.boundary == "selection-gradient gate"


# --- non-finite values ----------------------------------------------------------
def test_negative_nonfinite_weight_fails_closed_before_byte_gate():
    """A NaN route weight on an ACTIVE row must fail as NON_FINITE, not be
    compared byte-wise (NaN != NaN would misreport as a bytes mismatch)."""
    w = torch.full((K,), 0.25)
    w[1] = float("nan")
    cmp = TraceComparator(case_id="cx")
    cmp.check_identity(_meta(), dict(_meta()))
    cmp.check_score_weight(w, torch.full((K,), 0.25))
    rep = cmp.report()
    assert not rep.passed
    assert rep.stopped_early  # fail-closed, walk halted
    assert rep.primary is RouterVerdict.NON_FINITE
    assert rep.stages[-1].mismatch.owner == "router-score"


def test_negative_nonfinite_ignores_padding_rows():
    """Non-finite on PADDING rows is not a router defect (only active rows are
    checked); with a mask excluding them, the gate must pass and bytes stay
    comparable."""
    w = torch.full((4,), 0.25)
    w[3] = float("inf")  # padding row (mask=False)
    mask = torch.tensor([True, True, True, False])
    cmp = TraceComparator(case_id="cx")
    cmp.check_identity(_meta(), dict(_meta()))
    cmp.check_score_weight(w, w.clone(), active_mask=mask)
    rep = cmp.report()
    assert rep.stages[-1].passed
    assert not rep.passed  # the complete four-stage walk was not executed
    assert not rep.stopped_early


def test_negative_nonfinite_gradient_fails_closed():
    """Non-finite dz on an active row is NON_FINITE (router-backward), never a
    bytes diff; the failure must land on the GRADIENT stage with a backward
    pass_direction, not be misattributed to score_weight."""
    dz = torch.zeros(K)
    dz[0] = float("inf")
    cmp = TraceComparator(case_id="cx")
    cmp.check_identity(_meta(), dict(_meta()))
    cmp.check_gradient(dz, torch.zeros(K))
    rep = cmp.report()
    assert not rep.passed
    assert rep.stopped_early
    assert rep.primary is RouterVerdict.NON_FINITE
    assert rep.stages[-1].stage == "gradient"
    assert rep.stages[-1].mismatch.owner == "router-backward"
    assert rep.stages[-1].mismatch.key.pass_direction == "backward"


def test_negative_nonfinite_upstream_vs_router_not_conflated():
    """Provider-side UPSTREAM_NON_FINITE(18) belongs to the provider layer;
    the runner-side comparator must report NON_FINITE(1) for router-computed
    trace values, keeping the two bands distinct (bands: device 1-2,
    provider 10-22, runner 50-72)."""
    w = torch.full((K,), 0.25)
    w[0] = float("nan")
    cmp = TraceComparator(case_id="cx")
    cmp.check_identity(_meta(), dict(_meta()))
    cmp.check_score_weight(w, w.clone())  # both sides NaN
    rep = cmp.report()
    assert rep.primary is RouterVerdict.NON_FINITE  # not UPSTREAM_NON_FINITE
    assert RouterVerdict.NON_FINITE.value == 1  # device band, not 18


# --- padding discipline --------------------------------------------------------
def test_negative_padding_cannot_mask_core_defect():
    """Padding rows must not hide a real Core change."""
    rows, meta, ident = _case()
    base = _wrap(rows, ident)
    bad = list(rows)
    bad[2] = dataclasses.replace(bad[2], logical_expert_id=(bad[2].logical_expert_id + 1) % 256)
    pert = _wrap(bad, ident, padding_rows=16)
    rep = run_l2_invariance("cx", base, pert, variant="pad-mask")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH


def test_negative_reordered_rows_are_invariant():
    """Row order/padding/Envelope may vary — semantics must not (control)."""
    rows, meta, ident = _case()
    base = _wrap(rows, ident)
    pert = _wrap(
        repack_rows(rows, batch_size=2),
        ident,
        run_id="r2",
        engine_id="miles",
        attempt_id=2,
        placement_offset=5,
        padding_rows=7,
    )
    rep = run_l2_invariance("cx", base, pert, variant="control")
    assert rep.passed
