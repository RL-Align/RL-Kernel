# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-contract selected-token comparison for cross-configuration cases."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import torch

from rl_engine.contracts.numerical import resolve_logprob_threshold, tolerance_contract_fingerprint
from rl_engine.validation.cross_config.schema import (
    AlignmentResult,
    AlignmentStatus,
    ScoreArtifact,
    ScoreSide,
    SemanticIdentitySpec,
    TokenComparisonArtifact,
)


def semantic_identity_errors(
    rollout: SemanticIdentitySpec,
    training: SemanticIdentitySpec,
) -> tuple[str, ...]:
    """Return every logical identity field that differs between the two sides."""

    return tuple(
        item.name
        for item in fields(SemanticIdentitySpec)
        if getattr(rollout, item.name) != getattr(training, item.name)
    )


def recompute_mismatch_mask(
    rollout_logprobs: torch.Tensor,
    training_logprobs: torch.Tensor,
    active_mask: torch.Tensor,
    fixed_threshold: float,
) -> torch.Tensor:
    """Recompute the sole token mismatch signal from persisted tensors."""

    if rollout_logprobs.shape != training_logprobs.shape:
        raise ValueError("rollout and training logprobs must have identical shapes")
    if active_mask.shape != rollout_logprobs.shape:
        raise ValueError("active_mask shape must match selected logprobs")
    if fixed_threshold < 0.0:
        raise ValueError("fixed_threshold must be non-negative")
    active = active_mask.to(device=rollout_logprobs.device, dtype=torch.bool)
    training = training_logprobs.to(device=rollout_logprobs.device)
    return active & (torch.abs(training - rollout_logprobs) > fixed_threshold)


class FixedThresholdComparator:
    """Compare paired selected logprobs using only the current WS1 contract."""

    def compare(self, rollout: ScoreArtifact, training: ScoreArtifact) -> AlignmentResult:
        contract_fingerprint = tolerance_contract_fingerprint()
        artifact_errors = _artifact_errors(rollout, training)
        if artifact_errors:
            return AlignmentResult(
                case_id=rollout.case_id,
                attempt_id=rollout.attempt_id,
                status=AlignmentStatus.INVALID_ARTIFACT,
                comparable=False,
                passed=False,
                active_token_count=0,
                mismatch_count=0,
                contract_fingerprint=contract_fingerprint,
                artifact_errors=artifact_errors,
            )

        identity_errors = list(semantic_identity_errors(rollout.identity, training.identity))
        identity_errors.extend(_artifact_identity_errors(rollout, training))
        if identity_errors:
            return AlignmentResult(
                case_id=rollout.case_id,
                attempt_id=rollout.attempt_id,
                status=AlignmentStatus.INVALID_IDENTITY,
                comparable=False,
                passed=False,
                active_token_count=0,
                mismatch_count=0,
                contract_fingerprint=contract_fingerprint,
                identity_errors=tuple(dict.fromkeys(identity_errors)),
            )

        threshold, threshold_error = _resolve_fixed_threshold(rollout, training)
        if threshold_error is not None:
            return AlignmentResult(
                case_id=rollout.case_id,
                attempt_id=rollout.attempt_id,
                status=AlignmentStatus.INVALID_ARTIFACT,
                comparable=False,
                passed=False,
                active_token_count=0,
                mismatch_count=0,
                contract_fingerprint=contract_fingerprint,
                artifact_errors=(threshold_error,),
            )
        assert threshold is not None
        fixed_threshold = threshold

        rollout_logprobs = rollout.selected_logprobs.detach().cpu()
        training_logprobs = training.selected_logprobs.detach().cpu()
        active_mask = rollout.active_mask.detach().cpu().to(dtype=torch.bool)
        active_token_count = int(active_mask.sum().item())
        if active_token_count:
            active_rollout = rollout_logprobs[active_mask]
            active_training = training_logprobs[active_mask]
            if not bool(torch.isfinite(active_rollout).all().item()) or not bool(
                torch.isfinite(active_training).all().item()
            ):
                return AlignmentResult(
                    case_id=rollout.case_id,
                    attempt_id=rollout.attempt_id,
                    status=AlignmentStatus.INVALID_ARTIFACT,
                    comparable=False,
                    passed=False,
                    active_token_count=active_token_count,
                    mismatch_count=0,
                    contract_fingerprint=contract_fingerprint,
                    fixed_threshold=fixed_threshold,
                    artifact_errors=("active selected logprobs must be finite",),
                )

        # Inactive positions are outside the numerical contract. Canonicalize
        # them before persistence so an ignored NaN/Inf cannot break strict JSON
        # serialization or make resume artifacts non-reproducible.
        rollout_logprobs = rollout_logprobs.masked_fill(~active_mask, 0.0)
        training_logprobs = training_logprobs.masked_fill(~active_mask, 0.0)
        absolute_diff = torch.abs(training_logprobs - rollout_logprobs)
        mismatch_mask = recompute_mismatch_mask(
            rollout_logprobs,
            training_logprobs,
            active_mask,
            fixed_threshold,
        )
        token_artifact = TokenComparisonArtifact(
            rollout_logprobs=rollout_logprobs,
            training_logprobs=training_logprobs,
            active_mask=active_mask,
            absolute_diff=absolute_diff,
            mismatch_mask=mismatch_mask,
            fixed_threshold=fixed_threshold,
        )
        if active_token_count == 0:
            return AlignmentResult(
                case_id=rollout.case_id,
                attempt_id=rollout.attempt_id,
                status=AlignmentStatus.ZERO_ACTIVE_TOKENS,
                comparable=False,
                passed=False,
                active_token_count=0,
                mismatch_count=0,
                contract_fingerprint=contract_fingerprint,
                fixed_threshold=fixed_threshold,
                token_artifact=token_artifact,
            )

        mismatch_count = int(mismatch_mask.sum().item())
        passed = mismatch_count == 0
        return AlignmentResult(
            case_id=rollout.case_id,
            attempt_id=rollout.attempt_id,
            status=AlignmentStatus.PASS if passed else AlignmentStatus.FAIL,
            comparable=True,
            passed=passed,
            active_token_count=active_token_count,
            mismatch_count=mismatch_count,
            contract_fingerprint=contract_fingerprint,
            fixed_threshold=fixed_threshold,
            diagnostics=_diagnostics(
                rollout_logprobs,
                training_logprobs,
                active_mask,
                absolute_diff,
                mismatch_count,
            ),
            token_artifact=token_artifact,
        )


def compare_score_artifacts(
    rollout: ScoreArtifact,
    training: ScoreArtifact,
) -> AlignmentResult:
    """Convenience wrapper whose API deliberately exposes no threshold override."""

    return FixedThresholdComparator().compare(rollout, training)


def _resolve_fixed_threshold(
    rollout: ScoreArtifact,
    training: ScoreArtifact,
) -> tuple[float | None, str | None]:
    """Resolve one WS1 threshold, rejecting any mixed-dtype ambiguity."""

    try:
        rollout_threshold = resolve_logprob_threshold(rollout.scorer.dtype)
        training_threshold = resolve_logprob_threshold(training.scorer.dtype)
    except ValueError as exc:
        return None, f"fixed WS1 threshold is unavailable: {exc}"
    if rollout_threshold != training_threshold:
        return (
            None,
            "fixed WS1 threshold is ambiguous for scorer dtypes "
            f"rollout={rollout.scorer.dtype!r}, training={training.scorer.dtype!r}",
        )
    return rollout_threshold, None


def _artifact_errors(rollout: ScoreArtifact, training: ScoreArtifact) -> tuple[str, ...]:
    errors: list[str] = []
    if rollout.side is not ScoreSide.ROLLOUT:
        errors.append("first artifact side must be rollout")
    if training.side is not ScoreSide.TRAINING:
        errors.append("second artifact side must be training")
    if rollout.case_id != training.case_id:
        errors.append("case_id")
    if rollout.attempt_id != training.attempt_id:
        errors.append("attempt_id")
    if rollout.selected_logprobs.shape != training.selected_logprobs.shape:
        errors.append("selected_logprobs shape")
    for label, artifact in (("rollout", rollout), ("training", training)):
        expected_dtype = _score_dtype(artifact.scorer.dtype)
        if not artifact.selected_logprobs.is_floating_point():
            errors.append(f"{label}.selected_logprobs must be floating point")
        elif expected_dtype is None:
            errors.append(f"{label}.scorer dtype is unsupported")
        elif artifact.selected_logprobs.dtype != expected_dtype:
            errors.append(
                f"{label}.selected_logprobs dtype does not match scorer dtype "
                f"({artifact.selected_logprobs.dtype} != {expected_dtype})"
            )
    return tuple(errors)


def _score_dtype(value: str) -> torch.dtype | None:
    normalized = str(value).strip().lower().removeprefix("torch.")
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
    }.get(normalized)


def _artifact_identity_errors(
    rollout: ScoreArtifact,
    training: ScoreArtifact,
) -> tuple[str, ...]:
    errors: list[str] = []
    rollout_identity_mask = _identity_mask(rollout.identity)
    training_identity_mask = _identity_mask(training.identity)
    rollout_mask = rollout.active_mask.detach().cpu().to(dtype=torch.bool)
    training_mask = training.active_mask.detach().cpu().to(dtype=torch.bool)
    if rollout_mask.shape != rollout_identity_mask.shape or not torch.equal(
        rollout_mask, rollout_identity_mask
    ):
        errors.append("rollout.active_mask")
    if training_mask.shape != training_identity_mask.shape or not torch.equal(
        training_mask, training_identity_mask
    ):
        errors.append("training.active_mask")
    if rollout_mask.shape != training_mask.shape or not torch.equal(rollout_mask, training_mask):
        errors.append("active_mask")
    return tuple(errors)


def _identity_mask(identity: SemanticIdentitySpec) -> torch.Tensor:
    return torch.tensor(identity.active_mask, dtype=torch.bool)


def _diagnostics(
    rollout_logprobs: torch.Tensor,
    training_logprobs: torch.Tensor,
    active_mask: torch.Tensor,
    absolute_diff: torch.Tensor,
    mismatch_count: int,
) -> dict[str, Any]:
    active_diff = absolute_diff[active_mask].float()
    delta = (training_logprobs[active_mask] - rollout_logprobs[active_mask]).float()
    worst_active_index = int(torch.argmax(active_diff).item())
    active_coordinates = torch.nonzero(active_mask, as_tuple=False)
    worst_coordinate = tuple(int(item) for item in active_coordinates[worst_active_index].tolist())
    approximate_kl = torch.exp(delta.double()) - delta.double() - 1.0
    approximate_kl_mean = _finite_float_or_none(approximate_kl.mean())
    active_count = int(active_diff.numel())
    return {
        "mean_abs_diff": _finite_float_or_none(active_diff.mean()),
        "p95_abs_diff": _finite_float_or_none(torch.quantile(active_diff, 0.95)),
        "p99_abs_diff": _finite_float_or_none(torch.quantile(active_diff, 0.99)),
        "max_abs_diff": _finite_float_or_none(active_diff.max()),
        "mismatch_ratio": mismatch_count / active_count,
        "approximate_kl_mean": approximate_kl_mean,
        "approximate_kl_finite": approximate_kl_mean is not None,
        "worst_token_index": worst_coordinate,
    }


def _finite_float_or_none(value: torch.Tensor) -> float | None:
    result = float(value.item())
    return result if math.isfinite(result) else None


__all__ = [
    "FixedThresholdComparator",
    "compare_score_artifacts",
    "recompute_mismatch_mask",
    "semantic_identity_errors",
]
