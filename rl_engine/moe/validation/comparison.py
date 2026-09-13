# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Four-stage ordered comparator for P3 router traces (contract §2.5).

Frozen comparison rules implemented here:

1. Stage order is fixed: **identity -> discrete -> score/weight -> gradient**.
   Any identity / schema / provenance / upstream-verdict failure STOPS all
   later numeric attribution (fail-closed); a numeric mismatch found before an
   identity error is never reported, because the identity error explains it.
2. Comparison modes per field class (§2.5):
   - identity / layer / branch / slot / expert id / Top-K order / tie /
     valid / capacity / canonical key  -> ``exact_discrete``
   - active route weight and declared byte-gate score/gradient -> raw
     ``byte_exact``
   - engine-vs-engine must be byte-exact; oracle-vs-CUDA defaults byte-exact,
     non-bit-defined fields need a frozen ``ulp_bounded(k)`` delta.
3. Strict verdicts are never overridden by tolerance or averaged errors
   (T09 DoD: "strict mismatch 不被 tolerance/平均误差隐藏").
4. Padding rows never enter numeric gates; they are audited only by the
   same-config artifact gate (L1), which lives in the ladder/CLI, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.moe.p3_verdicts import P3Verdict, primary_verdict
from rl_engine.moe.validation.first_mismatch import (
    FirstMismatch,
    MismatchKey,
    TraceEvent,
    first_mismatch,
)

# --- stage identifiers (contract §2.5 order) -------------------------------
STAGE_IDENTITY = "identity"
STAGE_DISCRETE = "discrete"
STAGE_SCORE_WEIGHT = "score_weight"
STAGE_GRADIENT = "gradient"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_IDENTITY, STAGE_DISCRETE, STAGE_SCORE_WEIGHT, STAGE_GRADIENT,
)

# verdicts that stop the walk entirely (infrastructure/identity/schema/upstream
# plus fail-closed device defects: a non-finite P3 value makes later numeric
# attribution meaningless — §6 "非 PASS 的 non-finite fail-closed")
_STOP_VERDICTS: frozenset[P3Verdict] = frozenset({
    P3Verdict.NON_FINITE,
    P3Verdict.IDENTITY_DRIFT,
    P3Verdict.SCHEMA_MISMATCH,
    P3Verdict.STALE_RUN_METADATA,
    P3Verdict.INCOMPLETE_ARTIFACT,
    P3Verdict.CORRUPT_ARTIFACT,
    P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
    P3Verdict.UPSTREAM_VERDICT_MISSING,
    P3Verdict.UPSTREAM_EVIDENCE_MISSING,
    P3Verdict.MISSING_PROVENANCE,
    P3Verdict.MISSING_BOUNDARY_TRACE,
    P3Verdict.MISSING_RANK,
})

# stage -> runner verdict when a discrete mismatch lands there
_DISCRETE_VERDICTS: dict[str, P3Verdict] = {
    "topk_ids": P3Verdict.TOPK_ORDER_MISMATCH,
    "tie_break": P3Verdict.TIE_BREAK_POLICY_MISMATCH,
    "table_slot": P3Verdict.INVALID_DISCRETE_PLAN,
    "logical_expert_id": P3Verdict.INVALID_DISCRETE_PLAN,
    "valid": P3Verdict.INVALID_DISCRETE_PLAN,
}

# stage -> runner verdict when a byte gate fails there
_BYTE_VERDICTS: dict[str, P3Verdict] = {
    "route_weight": P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH,
    "score": P3Verdict.SCORE_BYTES_MISMATCH,
    "gradient": P3Verdict.GRADIENT_BYTES_MISMATCH,
}


@dataclass
class StageResult:
    """Outcome of one comparison stage."""

    stage: str
    passed: bool
    verdict: P3Verdict | None          # None when passed
    mismatch: FirstMismatch | None     # located first divergence, if any
    notes: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full ordered-comparison result for one case."""

    case_id: str
    stages: list[StageResult]
    stopped_early: bool                # identity/schema/upstream gate halted walk

    @property
    def passed(self) -> bool:
        return bool(self.stages) and all(s.passed for s in self.stages)

    @property
    def primary(self) -> P3Verdict | None:
        failures = [s.verdict for s in self.stages if not s.passed and s.verdict]
        return primary_verdict(failures)

    def summary_line(self) -> str:
        flag = "PASS" if self.passed else f"FAIL({self.primary.name if self.primary else '?'})"
        walked = "->".join(
            f"{s.stage}:{'ok' if s.passed else s.verdict.name if s.verdict else 'FAIL'}"
            for s in self.stages
        )
        return f"case={self.case_id} {flag} [{walked}]{' STOP' if self.stopped_early else ''}"


def tensor_byte_exact(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """True iff two tensors have identical dtype/shape/bytes (incl. -0.0 vs 0.0)."""
    if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    return bool(
        torch.equal(lhs.view(torch.uint8).flatten(), rhs.view(torch.uint8).flatten())
    )


def discrete_equal(lhs: Any, rhs: Any) -> bool:
    """``exact_discrete`` for ints/enums/strings/bools and lists thereof."""
    return lhs == rhs


class TraceComparator:
    """Ordered four-stage comparator over two normalized trace streams.

    Usage (mirrors how ``check_p3`` L3a/L3b will drive it)::

        cmp = TraceComparator(case_id="case_x")
        cmp.check_identity(lhs_meta, rhs_meta)      # stops on drift
        cmp.check_discrete(lhs_events, rhs_events)  # ids/order/tie/slots
        cmp.check_score_weight(lhs_w, rhs_w)        # byte gates
        cmp.check_gradient(lhs_dz, rhs_dz)          # byte gates
        report = cmp.report()
    """

    def __init__(self, case_id: str, *, phase: str = "WS1") -> None:
        self._case_id = case_id
        self._phase = phase
        self._stages: list[StageResult] = []
        self._stopped = False

    # -- helpers -----------------------------------------------------------

    def _append(self, result: StageResult) -> None:
        if self._stopped:
            raise RuntimeError("comparator already halted; later stages are unreachable")
        self._stages.append(result)
        if not result.passed and result.verdict in _STOP_VERDICTS:
            self._stopped = True

    def _halted_gate(self, stage: str, verdict: P3Verdict, note: str) -> None:
        self._append(StageResult(stage=stage, passed=False, verdict=verdict,
                                 mismatch=None, notes=[note]))

    # -- stage 1: identity ---------------------------------------------------

    _IDENTITY_FIELDS = (
        "case_id", "checkpoint_id", "weight_id", "absolute_layer",
        "router_mode", "table_fingerprint", "bias_fingerprint",
        "logit_round_point", "tie_break_policy", "capacity_policy",
    )

    def check_identity(self, lhs_meta: dict[str, Any], rhs_meta: dict[str, Any]) -> None:
        """Stage 1: identity/schema/provenance gate. Any drift halts the walk."""
        missing = [k for k in self._IDENTITY_FIELDS if k not in lhs_meta or k not in rhs_meta]
        if missing:
            self._halted_gate(
                STAGE_IDENTITY, P3Verdict.MISSING_PROVENANCE,
                f"identity fields missing on one side: {missing}",
            )
            return
        drifted = [k for k in self._IDENTITY_FIELDS if lhs_meta[k] != rhs_meta[k]]
        if drifted:
            self._halted_gate(
                STAGE_IDENTITY, P3Verdict.IDENTITY_DRIFT,
                f"identity fields differ: { {k: (lhs_meta[k], rhs_meta[k]) for k in drifted} }",
            )
            return
        self._append(StageResult(stage=STAGE_IDENTITY, passed=True, verdict=None, mismatch=None))

    # -- stage 2: discrete ---------------------------------------------------

    def check_discrete(
        self,
        lhs_events: list[TraceEvent],
        rhs_events: list[TraceEvent],
    ) -> None:
        """Stage 2: Top-6 ids/order, tie policy, table slots, valid flags."""
        result = first_mismatch(lhs_events, rhs_events, phase=self._phase,
                                artifact_name="discrete_trace")
        if result.found:
            verdict = _DISCRETE_VERDICTS.get(
                result.detail_kind if hasattr(result, "detail_kind") else "logical_expert_id",
                P3Verdict.INVALID_DISCRETE_PLAN,
            )
            self._append(StageResult(stage=STAGE_DISCRETE, passed=False,
                                     verdict=verdict, mismatch=result))
            return
        self._append(StageResult(stage=STAGE_DISCRETE, passed=True, verdict=None, mismatch=None))

    # -- stage 3: score / weight bytes ---------------------------------------

    # -- non-finite gate (§2.5: forward active values must be finite) --------

    @staticmethod
    def _active_non_finite(t: torch.Tensor, active_mask: torch.Tensor | None,
                           name: str) -> int | None:
        """Index of the first active non-finite element, or None (§2.5).

        ``UPSTREAM_NON_FINITE`` is a *provider* verdict for upstream z /
        dweights; here both sides are P3-computed trace values, so a
        non-finite on either side is P3's own ``NON_FINITE``.
        """
        t = t.detach().to(torch.float32)
        if active_mask is not None:
            t = t[active_mask.to(torch.bool)]
        bad = torch.nonzero(~torch.isfinite(t)).flatten()
        return int(bad[0].item()) if bad.numel() else None

    def _nonfinite_gate(self, lhs: torch.Tensor, rhs: torch.Tensor, *,
                        active_mask: torch.Tensor | None, site: str,
                        owner: str, issue: str) -> bool:
        """Returns True when clean; appends a NON_FINITE stage on failure."""
        for side, tensor in (("lhs", lhs), ("rhs", rhs)):
            idx = self._active_non_finite(tensor, active_mask, site)
            if idx is not None:
                key = MismatchKey(absolute_layer=-1, site=site,
                                  pass_direction="forward", event_index=idx,
                                  global_token_id=-1, rank=-1)
                fm = FirstMismatch(
                    found=True, key=key, owner=owner, issue=issue,
                    boundary="non-finite gate", phase=self._phase,
                    artifact=f"{site}_values",
                    detail=f"{side} {site} has non-finite value at active "
                           f"flat index {idx} (§2.5 fail-closed)",
                )
                self._append(StageResult(stage=STAGE_SCORE_WEIGHT, passed=False,
                                         verdict=P3Verdict.NON_FINITE,
                                         mismatch=fm))
                return False
        return True

    def check_score_weight(
        self,
        lhs_weights: torch.Tensor,
        rhs_weights: torch.Tensor,
        *,
        lhs_scores: torch.Tensor | None = None,
        rhs_scores: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> None:
        """Stage 3: active-row byte gates on route weights (and scores).

        A non-finite active value on either side is ``NON_FINITE`` and
        fails closed *before* any byte comparison (§2.5); the walk stops so
        the defect is never averaged away by later numeric stages.
        """
        if not self._nonfinite_gate(lhs_weights, rhs_weights,
                                    active_mask=active_mask, site="weight",
                                    owner="T02", issue="#41"):
            return
        if lhs_scores is not None and rhs_scores is not None:
            if not self._nonfinite_gate(lhs_scores, rhs_scores,
                                        active_mask=active_mask, site="score",
                                        owner="T02", issue="#41"):
                return
        if active_mask is not None:
            keep = active_mask.to(torch.bool)
            lhs_weights = lhs_weights[keep]
            rhs_weights = rhs_weights[keep]
            if lhs_scores is not None and rhs_scores is not None:
                lhs_scores = lhs_scores[keep]
                rhs_scores = rhs_scores[keep]

        mismatches: list[str] = []
        if not tensor_byte_exact(lhs_weights, rhs_weights):
            mismatches.append("route_weight bytes differ on active rows")

        if lhs_scores is not None and rhs_scores is not None:
            if not tensor_byte_exact(lhs_scores, rhs_scores):
                mismatches.append("score bytes differ on active rows")

        if mismatches:
            key = MismatchKey(absolute_layer=-1, site="weight", pass_direction="forward",
                              event_index=-1, global_token_id=-1, rank=-1)
            fm = FirstMismatch(found=True, key=key, owner="T02", issue="#41",
                               boundary="byte gate", phase=self._phase,
                               artifact="score_weight_bytes", detail="; ".join(mismatches))
            verdict = (_BYTE_VERDICTS["route_weight"]
                       if "route_weight" in mismatches[0] else _BYTE_VERDICTS["score"])
            self._append(StageResult(stage=STAGE_SCORE_WEIGHT, passed=False,
                                     verdict=verdict, mismatch=fm, notes=mismatches))
            return
        self._append(StageResult(stage=STAGE_SCORE_WEIGHT, passed=True, verdict=None,
                                 mismatch=None))

    # -- stage 4: gradient bytes ------------------------------------------------

    def check_gradient(self, lhs_dz: torch.Tensor, rhs_dz: torch.Tensor) -> None:
        """Stage 4: byte gate on active ``ds != 0`` backward outputs (§2.1).

        Non-finite P3-computed gradients fail closed as ``NON_FINITE``
        before byte comparison (§2.5). Upstream z/dweights non-finiteness
        is a provider-side ``UPSTREAM_NON_FINITE`` and never reaches here.
        """
        if not self._nonfinite_gate(lhs_dz, rhs_dz, active_mask=None,
                                    site="bwd", owner="T06", issue="#42/#44"):
            return
        if not tensor_byte_exact(lhs_dz, rhs_dz):
            key = MismatchKey(absolute_layer=-1, site="bwd", pass_direction="backward",
                              event_index=-1, global_token_id=-1, rank=-1)
            fm = FirstMismatch(found=True, key=key, owner="T06", issue="#42/#44",
                               boundary="byte gate", phase=self._phase,
                               artifact="gradient_bytes",
                               detail="dz bytes differ on active rows")
            self._append(StageResult(stage=STAGE_GRADIENT, passed=False,
                                     verdict=_BYTE_VERDICTS["gradient"], mismatch=fm))
            return
        self._append(StageResult(stage=STAGE_GRADIENT, passed=True, verdict=None,
                                 mismatch=None))

    # -- report ---------------------------------------------------------------

    def report(self) -> ComparisonReport:
        return ComparisonReport(
            case_id=self._case_id, stages=list(self._stages), stopped_early=self._stopped,
        )
