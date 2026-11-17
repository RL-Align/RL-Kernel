# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-configuration entry point for the offline MoE logprob diagnostic."""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.validation.cross_config.artifacts import _safe_component
from rl_engine.validation.models.moe_logprob import (
    MoELogprobCase,
    MoELogprobObservation,
    compare_moe_logprobs,
)


def _validated_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized non-empty string")
    return _safe_component(value, label)


def compare_cross_config_moe_logprobs(
    *,
    case_id: str,
    attempt_id: str,
    reference_case: MoELogprobCase,
    reference_values: torch.Tensor,
    candidate_case: MoELogprobCase,
    candidate: MoELogprobObservation,
    expected_mapping_run_id: str,
    expected_model_version: str,
) -> dict[str, Any]:
    """Attach a MoE first-difference report to a cross-config attempt identity."""

    case_id = _validated_identity(case_id, "case_id")
    attempt_id = _validated_identity(attempt_id, "attempt_id")
    report = compare_moe_logprobs(
        reference_case,
        reference_values,
        candidate_case,
        candidate,
        expected_mapping_run_id=expected_mapping_run_id,
        expected_model_version=expected_model_version,
    )
    report["cross_config_identity"] = {"case_id": case_id, "attempt_id": attempt_id}
    return report


__all__ = ["compare_cross_config_moe_logprobs"]
