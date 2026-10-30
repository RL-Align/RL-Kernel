# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Four-stage ordered comparator for MoE router traces.

Stages run in a fixed order; a failing stage halts the walk and the report
records where it stopped (later stages are reported as not-run).

Frozen comparison rules implemented here:

1. Stage order is fixed: **identity -> discrete -> score/weight -> gradient**.
   Any identity / schema / provenance / upstream-verdict failure STOPS all
   later numeric attribution (fail-closed); a numeric mismatch found before an
   identity error is never reported, because the identity error explains it.
2. Comparison modes per field class:
   - identity / layer / branch / slot / expert id / Top-K order / tie /
     valid / capacity / canonical key  -> ``exact_discrete``
   - active route weight and declared byte-gate score/gradient -> raw
     ``byte_exact``
   - engine-vs-engine must be byte-exact; oracle-vs-CUDA defaults byte-exact,
     non-bit-defined fields need a frozen ``ulp_bounded(k)`` delta.
3. Strict verdicts are never overridden by tolerance or averaged errors
   (a strict mismatch must never be hidden by tolerance or
   averaged-error reporting).
4. Padding rows never enter numeric gates; they are audited only by the
   same-config artifact gate (L1), which lives in the runner/CLI, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.moe.t01_verdicts import RouterVerdict, primary_verdict
from rl_engine.moe.validation.first_mismatch import (
    FirstMismatch,
    MismatchKey,
    TraceEvent,
    first_mismatch,
)

# --- stage identifiers (fixed comparison order) ---------------------------
STAGE_IDENTITY = "identity"
STAGE_DISCRETE = "discrete"
STAGE_SCORE_WEIGHT = "score_weight"
STAGE_GRADIENT = "gradient"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_IDENTITY,
    STAGE_DISCRETE,
    STAGE_SCORE_WEIGHT,
    STAGE_GRADIENT,
)

# verdicts that stop the walk entirely (infrastructure/identity/schema/upstream
# plus fail-closed device defects: a non-finite router value makes later numeric
# attribution meaningless (non-finite values fail closed unless PASS)
_STOP_VERDICTS: frozenset[RouterVerdict] = frozenset(
    {
        RouterVerdict.NON_FINITE,
        RouterVerdict.IDENTITY_DRIFT,
        RouterVerdict.SCHEMA_MISMATCH,
        RouterVerdict.STALE_RUN_METADATA,
        RouterVerdict.INCOMPLETE_ARTIFACT,
        RouterVerdict.CORRUPT_ARTIFACT,
        RouterVerdict.UPSTREAM_CONTRACT_MISMATCH,
        RouterVerdict.UPSTREAM_VERDICT_MISSING,
        RouterVerdict.UPSTREAM_EVIDENCE_MISSING,
        RouterVerdict.MISSING_PROVENANCE,
        RouterVerdict.MISSING_BOUNDARY_TRACE,
        RouterVerdict.MISSING_RANK,
        # a non-auditable substitute path (fast-math, vendor libm/libdevice,
        # fallback kernels) invalidates every later byte comparison: matches are
        # luck, not evidence, so the walk must halt here. The same holds for
        # mutually inconsistent-but-auditable policies (round point / tie break):
        # comparing bytes produced under different policies is meaningless.
        RouterVerdict.SILENT_FALLBACK,
        RouterVerdict.LOGIT_ROUND_POINT_MISMATCH,
        RouterVerdict.TIE_BREAK_POLICY_MISMATCH,
    }
)

# site -> runner verdict when a discrete mismatch lands on that site. The
# verdict stays as specific as the evidence: attribution is never downgraded
# to a generic code while a specific one applies. Band discipline: this
# comparator is runner-level, so only runner-band codes may appear here;
# component attribution is carried by the mismatch's ``owner`` field.
_DISCRETE_VERDICTS: dict[str, RouterVerdict] = {
    "topk": RouterVerdict.TOPK_ORDER_MISMATCH,
    "selection": RouterVerdict.TOPK_ORDER_MISMATCH,
    "hash_lookup": RouterVerdict.INVALID_DISCRETE_PLAN,
    "weight": RouterVerdict.INVALID_DISCRETE_PLAN,
    "handoff": RouterVerdict.INVALID_DISCRETE_PLAN,
}

# stage -> runner verdict when a byte gate fails there
_BYTE_VERDICTS: dict[str, RouterVerdict] = {
    "route_weight": RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH,
    "score": RouterVerdict.SCORE_BYTES_MISMATCH,
    "gradient": RouterVerdict.GRADIENT_BYTES_MISMATCH,
}


@dataclass
class StageResult:
    """Outcome, verdict, and localized diagnostics for one stage."""

    stage: str
    passed: bool
    verdict: RouterVerdict | None  # None when passed
    mismatch: FirstMismatch | None  # located first divergence, if any
    notes: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full ordered-comparison result for one case.

    ``stages`` contains only executed stages. ``stopped_early`` distinguishes
    a fail-closed halt from an ordinary numeric mismatch.
    """

    case_id: str
    stages: list[StageResult]
    stopped_early: bool  # identity/schema/upstream gate halted walk

    @property
    def passed(self) -> bool:
        return [s.stage for s in self.stages] == list(STAGE_ORDER) and all(
            s.passed for s in self.stages
        )

    @property
    def primary(self) -> RouterVerdict | None:
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
    """True iff tensors have identical dtype, shape, and logical bytes."""
    if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    # Recorded tensors may be non-contiguous views.  Comparison is over the
    # canonical logical tensor bytes; layout/stride belongs to provenance.
    return torch.equal(
        lhs.detach().contiguous().view(torch.uint8).flatten(),
        rhs.detach().contiguous().view(torch.uint8).flatten(),
    )


class TraceComparator:
    """Ordered four-stage comparator over two normalized trace streams.

    Usage (mirrors how ``check_router`` L3a/L3b drives it)::

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

    def _halted_gate(self, stage: str, verdict: RouterVerdict, note: str) -> None:
        self._append(
            StageResult(stage=stage, passed=False, verdict=verdict, mismatch=None, notes=[note])
        )

    # -- stage 1: identity ---------------------------------------------------

    _IDENTITY_FIELDS = (
        "case_id",
        "checkpoint_id",
        "weight_id",
        "absolute_layer",
        "router_mode",
        "table_fingerprint",
        "bias_fingerprint",
        "logit_round_point",
        "tie_break_policy",
        "capacity_policy",
    )

    #: The two logit round-point policies are independent groups; anything
    #: else (fast-math, vendor libm/libdevice, fallback kernels) means the
    #: result came from a non-auditable substitute path -> SILENT_FALLBACK.
    _ALLOWED_ROUND_POINTS = frozenset({"fp32_direct", "bf16_round_then_widen"})

    def check_identity(self, lhs_meta: dict[str, Any], rhs_meta: dict[str, Any]) -> None:
        """Stage 1: identity/schema/provenance gate; any drift halts."""
        missing = [k for k in self._IDENTITY_FIELDS if k not in lhs_meta or k not in rhs_meta]
        if missing:
            self._halted_gate(
                STAGE_IDENTITY,
                RouterVerdict.MISSING_PROVENANCE,
                f"identity fields missing on one side: {missing}",
            )
            return
        for side, meta in (("lhs", lhs_meta), ("rhs", rhs_meta)):
            rp = meta["logit_round_point"]
            if rp not in self._ALLOWED_ROUND_POINTS:
                self._halted_gate(
                    STAGE_IDENTITY,
                    RouterVerdict.SILENT_FALLBACK,
                    f"{side} logit_round_point={rp!r} is not an auditable "
                    f"rounding policy (allowed: "
                    f"{sorted(self._ALLOWED_ROUND_POINTS)}); vendor libm or "
                    f"fast-math substitute paths must be flagged, not passed",
                )
                return
        drifted = [k for k in self._IDENTITY_FIELDS if lhs_meta[k] != rhs_meta[k]]
        if drifted:
            # policy fields keep their dedicated codes (14/56) instead of
            # degrading to generic IDENTITY_DRIFT(10):
            # both sides auditable but mutually inconsistent is a policy
            # mismatch, not a generic identity drift
            verdict = RouterVerdict.IDENTITY_DRIFT
            if "logit_round_point" in drifted:
                verdict = RouterVerdict.LOGIT_ROUND_POINT_MISMATCH
            elif "tie_break_policy" in drifted:
                verdict = RouterVerdict.TIE_BREAK_POLICY_MISMATCH
            self._halted_gate(
                STAGE_IDENTITY,
                verdict,
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
        """Stage 2: Top-K ids/order, tie policy, table slots, valid flags."""
        result = first_mismatch(
            lhs_events, rhs_events, phase=self._phase, artifact_name="discrete_trace"
        )
        if result.found:
            verdict = _DISCRETE_VERDICTS.get(
                result.key.site if result.key else "",
                RouterVerdict.INVALID_DISCRETE_PLAN,
            )
            self._append(
                StageResult(stage=STAGE_DISCRETE, passed=False, verdict=verdict, mismatch=result)
            )
            return
        self._append(StageResult(stage=STAGE_DISCRETE, passed=True, verdict=None, mismatch=None))

    # -- stage 3: score / weight bytes ---------------------------------------

    # -- non-finite gate (forward active values must be finite) --------------

    @staticmethod
    def _active_non_finite(
        t: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> int | None:
        """Index of the first active non-finite element, or None.

        ``UPSTREAM_NON_FINITE`` is a *provider* verdict for upstream z /
        dweights; here both sides are router-computed trace values, so a
        non-finite on either side is the router's own ``NON_FINITE``.
        """
        t = t.detach()
        if active_mask is not None:
            t = t[active_mask.to(torch.bool)]
        bad = torch.nonzero(~torch.isfinite(t)).flatten()
        return bad[0].item() if bad.numel() else None

    def _nonfinite_gate(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        active_mask: torch.Tensor | None,
        site: str,
        owner: str,
        stage: str,
        pass_direction: str = "forward",
    ) -> bool:
        """Returns True when clean; appends a NON_FINITE stage on failure."""
        for side, tensor in (("lhs", lhs), ("rhs", rhs)):
            idx = self._active_non_finite(tensor, active_mask)
            if idx is not None:
                key = MismatchKey(
                    absolute_layer=-1,
                    site=site,
                    pass_direction=pass_direction,
                    event_index=idx,
                    global_token_id=-1,
                    rank=-1,
                )
                fm = FirstMismatch(
                    found=True,
                    key=key,
                    owner=owner,
                    boundary="non-finite gate",
                    phase=self._phase,
                    artifact=f"{site}_values",
                    detail=f"{side} {site} has non-finite value at active "
                    f"flat index {idx} (fail-closed)",
                )
                self._append(
                    StageResult(
                        stage=stage, passed=False, verdict=RouterVerdict.NON_FINITE, mismatch=fm
                    )
                )
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
        fails closed *before* any byte comparison; the walk stops so
        the defect is never averaged away by later numeric stages.
        """
        if not self._nonfinite_gate(
            lhs_weights,
            rhs_weights,
            active_mask=active_mask,
            site="weight",
            owner="router-score",
            stage=STAGE_SCORE_WEIGHT,
        ):
            return
        if (lhs_scores is None) != (rhs_scores is None):
            self._halted_gate(
                STAGE_SCORE_WEIGHT,
                RouterVerdict.MISSING_PROVENANCE,
                "score evidence is present on only one side",
            )
            return
        if lhs_scores is not None and rhs_scores is not None:
            if not self._nonfinite_gate(
                lhs_scores,
                rhs_scores,
                active_mask=active_mask,
                site="score",
                owner="router-score",
                stage=STAGE_SCORE_WEIGHT,
            ):
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
            key = MismatchKey(
                absolute_layer=-1,
                site="weight",
                pass_direction="forward",
                event_index=-1,
                global_token_id=-1,
                rank=-1,
            )
            fm = FirstMismatch(
                found=True,
                key=key,
                owner="router-score",
                boundary="byte gate",
                phase=self._phase,
                artifact="score_weight_bytes",
                detail="; ".join(mismatches),
            )
            verdict = (
                _BYTE_VERDICTS["route_weight"]
                if "route_weight" in mismatches[0]
                else _BYTE_VERDICTS["score"]
            )
            self._append(
                StageResult(
                    stage=STAGE_SCORE_WEIGHT,
                    passed=False,
                    verdict=verdict,
                    mismatch=fm,
                    notes=mismatches,
                )
            )
            return
        self._append(
            StageResult(stage=STAGE_SCORE_WEIGHT, passed=True, verdict=None, mismatch=None)
        )

    # -- stage 4: gradient bytes ------------------------------------------------

    def check_gradient(
        self,
        lhs_dz: torch.Tensor,
        rhs_dz: torch.Tensor,
        *,
        lhs_selection_grad: torch.Tensor | None = None,
        rhs_selection_grad: torch.Tensor | None = None,
    ) -> None:
        """Stage 4: byte gate on active ``ds != 0`` backward outputs.

        A non-finite router-computed gradient fails closed as ``NON_FINITE``
        before byte comparison. Upstream z/dweights non-finiteness
        is a provider-side ``UPSTREAM_NON_FINITE`` and never reaches here.

        Selection-gradient leak: the selection/Hash/Top-K/tie paths are
        non-differentiable by contract, so any non-zero gradient recorded on
        the selection path of either engine is ``SELECTION_GRADIENT_PRESENT``
        (61) — it is a defect even when the two engines byte-agree on dz.
        Evidence is explicit (``*_selection_grad``): a bare dz tensor cannot
        attribute where a gradient came from, so byte divergence without
        selection evidence stays ``GRADIENT_BYTES_MISMATCH`` (53).
        """
        if not self._nonfinite_gate(
            lhs_dz,
            rhs_dz,
            active_mask=None,
            site="bwd",
            owner="router-backward",
            stage=STAGE_GRADIENT,
            pass_direction="backward",
        ):
            return
        for side, sg in (("lhs", lhs_selection_grad), ("rhs", rhs_selection_grad)):
            if sg is None:
                continue
            flat = sg.detach().flatten()
            finite = torch.isfinite(flat)
            if not finite.all():
                idx = (~finite).nonzero(as_tuple=False)[0, 0].item()
            elif flat.count_nonzero().item() > 0:
                idx = flat.nonzero(as_tuple=False)[0, 0].item()
            else:
                continue
            key = MismatchKey(
                absolute_layer=-1,
                site="bwd",
                pass_direction="backward",
                event_index=idx,
                global_token_id=-1,
                rank=-1,
            )
            fm = FirstMismatch(
                found=True,
                key=key,
                owner="router-backward",
                boundary="selection-gradient gate",
                phase=self._phase,
                artifact="selection_grad",
                detail=f"{side} selection path produced gradient at "
                f"flat index {idx} (selection/tie must be "
                f"non-differentiable)",
            )
            self._append(
                StageResult(
                    stage=STAGE_GRADIENT,
                    passed=False,
                    verdict=RouterVerdict.SELECTION_GRADIENT_PRESENT,
                    mismatch=fm,
                )
            )
            return
        if not tensor_byte_exact(lhs_dz, rhs_dz):
            key = MismatchKey(
                absolute_layer=-1,
                site="bwd",
                pass_direction="backward",
                event_index=-1,
                global_token_id=-1,
                rank=-1,
            )
            fm = FirstMismatch(
                found=True,
                key=key,
                owner="router-backward",
                boundary="byte gate",
                phase=self._phase,
                artifact="gradient_bytes",
                detail="dz bytes differ on active rows",
            )
            self._append(
                StageResult(
                    stage=STAGE_GRADIENT,
                    passed=False,
                    verdict=_BYTE_VERDICTS["gradient"],
                    mismatch=fm,
                )
            )
            return
        self._append(StageResult(stage=STAGE_GRADIENT, passed=True, verdict=None, mismatch=None))

    # -- report ---------------------------------------------------------------

    def report(self) -> ComparisonReport:
        return ComparisonReport(
            case_id=self._case_id,
            stages=list(self._stages),
            stopped_early=self._stopped,
        )
