# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C8 (#274): four-judgment evidence matrix schema.

Cells are ``backend_profile × case_id × op × judgment``. This module builds
and classifies the matrix. GPU execution lives in
``scripts/sweep_ws1_four_judgments.py`` and reuses the C3/C4 CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from rl_engine.kernels.gtest.gradient_adapters import GRADIENT_ADAPTERS, resolve_profile_candidate
from rl_engine.testing.ws1_workload import WS1Manifest, load_manifest

JUDGMENTS = (
    "forward_accuracy",
    "forward_invariance",
    "gradient_accuracy",
    "gradient_invariance",
)
PROFILES = ("cuda_bf16", "triton_cuda_bf16", "ascend_bf16")
TIERS = ("short", "primary")
CELL_STATUSES = (
    "green",
    "red",
    "pending_hopper",
    "N/A",
)

# Required C8 coverage rows (C2 required chain + pack). linear_logp is optional.
C8_REQUIRED_OPS = (
    "embedding",
    "rms_norm",
    "qk_norm",
    "det_gemm",
    "rope",
    "attention",
    "silu",
    "swiglu",
    "lm_head",
    "logp",
    "batch_invariant_logp",
    "pack",
)


@dataclass(frozen=True)
class MatrixCell:
    profile: str
    op_name: str
    judgment: str
    tier: str
    case_id: str | None
    status: str
    detail: str
    candidate: str | None = None
    expected_kernel_config_id: str | None = None
    actual_backend_id: str | None = None
    actual_kernel_config_id: str | None = None
    evidence_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "op_name": self.op_name,
            "judgment": self.judgment,
            "tier": self.tier,
            "case_id": self.case_id,
            "status": self.status,
            "detail": self.detail,
            "candidate": self.candidate,
            "expected_kernel_config_id": self.expected_kernel_config_id,
            "actual_backend_id": self.actual_backend_id,
            "actual_kernel_config_id": self.actual_kernel_config_id,
            "evidence_kind": self.evidence_kind,
        }


@dataclass
class MatrixReport:
    cells: tuple[MatrixCell, ...]
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "counts": dict(self.counts),
        }


def _case_op_name(case: dict[str, Any]) -> str:
    return str(case.get("op_name") or case["operator_spec"])


def _cases_for(manifest: WS1Manifest, *, op_name: str, profile: str) -> dict[str, dict[str, Any]]:
    """Map fixture tier to the complete pinned case for this op/profile."""

    found: dict[str, dict[str, Any]] = {}
    for case in manifest.representative_cases:
        if profile not in case.get("profile_ids", ()):
            continue
        if _case_op_name(case) != op_name:
            continue
        fixture = case["fixture_id"]
        if fixture.startswith("short_"):
            found["short"] = case
        elif fixture.startswith("rep_"):
            found["primary"] = case
    return found


def classify_adapter_cell(
    op_name: str,
    profile: str,
    manifest: WS1Manifest | None = None,
    *,
    allow_sm90: bool = False,
) -> tuple[str, str, str | None]:
    """Return (status, detail, candidate) without running a kernel."""

    adapter = GRADIENT_ADAPTERS[op_name]
    resolved = resolve_profile_candidate(adapter, profile, manifest)
    status = str(resolved["status"])
    candidate = resolved["expected_backend_id"]
    if candidate is not None:
        candidate = str(candidate)
    if adapter.requirement == "layout_supported":
        return (
            "N/A",
            "profile-independent layout helper; C2 declares PyTorch pack and C3/C4 cover it",
            candidate,
        )
    if adapter.requirement == "optional_fused" and status == "optional":
        return "N/A", "optional_fused with no C2 required node", None
    if status == "missing_required":
        return (
            "red",
            "C2 marks this node missing_required; required untested is red, not N/A",
            None,
        )
    if status == "absent_not_required":
        return "red", "not declared supported and differentiable", None
    if candidate == "cuda-sm90" and not allow_sm90:
        return (
            "pending_hopper",
            "declared candidate is cuda-sm90; required Hopper execution remains pending",
            candidate,
        )
    if status == "declared" and candidate:
        return "red", "required cell not yet executed on this host", candidate
    return "red", f"unclassified C2 status {status!r}", candidate


def build_classified_matrix(
    manifest: WS1Manifest | None = None,
    *,
    allow_sm90: bool = False,
    profiles: Sequence[str] = PROFILES,
) -> MatrixReport:
    """Build the C8 grid and classify every cell (no GPU).

    ``profiles`` narrows the grid to the backend profiles a given host can
    actually execute. Each required profile still has to go green somewhere:
    C11 only closes when every profile's own job passes.
    """

    m = manifest if manifest is not None else load_manifest()
    unknown = [p for p in profiles if p not in PROFILES]
    if unknown:
        raise ValueError(f"unknown backend profiles {unknown}")
    cells: list[MatrixCell] = []
    for profile in profiles:
        for op_name in C8_REQUIRED_OPS:
            status, detail, candidate = classify_adapter_cell(
                op_name, profile, m, allow_sm90=allow_sm90
            )
            case_ids = _cases_for(m, op_name=op_name, profile=profile)
            for tier in TIERS:
                case = case_ids.get(tier)
                case_id = None if case is None else str(case["case_id"])
                if status == "N/A":
                    cell_status, cell_detail = status, detail
                elif case_id is None and status != "pending_hopper":
                    cell_status, cell_detail = (
                        "red",
                        "required untested: no C2 case_id for this tier",
                    )
                else:
                    cell_status, cell_detail = status, detail
                    if case is not None and candidate is not None:
                        if str(case.get("expected_backend_id")) != str(candidate):
                            cell_status = "red"
                            cell_detail = (
                                "C2 case candidate does not match this required op/profile; "
                                "cross-path borrowing is forbidden"
                            )
                for judgment in JUDGMENTS:
                    cells.append(
                        MatrixCell(
                            profile=profile,
                            op_name=op_name,
                            judgment=judgment,
                            tier=tier,
                            case_id=case_id,
                            status=cell_status,
                            detail=cell_detail,
                            candidate=candidate,
                            expected_kernel_config_id=(
                                None if case is None else str(case["expected_kernel_config_id"])
                            ),
                            evidence_kind=(
                                "representative_accuracy"
                                if judgment.endswith("accuracy")
                                else "logical_config_invariance"
                            ),
                        )
                    )
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.status] = counts.get(cell.status, 0) + 1
    return MatrixReport(cells=tuple(cells), counts=counts)


def undefined_cells(report: MatrixReport) -> tuple[MatrixCell, ...]:
    return tuple(cell for cell in report.cells if cell.status not in CELL_STATUSES)


def hidden_required_na(report: MatrixReport) -> tuple[MatrixCell, ...]:
    """Required ops must not be N/A without an explicit C2 layout/optional reason."""

    # Reasons written by classify_adapter_cell for legitimate N/A cells.
    allowed_markers = ("layout_supported", "profile-independent", "optional_fused")
    hidden: list[MatrixCell] = []
    for cell in report.cells:
        if cell.op_name == "pack":
            continue
        if cell.status != "N/A":
            continue
        detail = cell.detail or ""
        if any(marker in detail for marker in allowed_markers):
            continue
        hidden.append(cell)
    return tuple(hidden)


__all__ = [
    "C8_REQUIRED_OPS",
    "CELL_STATUSES",
    "JUDGMENTS",
    "MatrixCell",
    "MatrixReport",
    "PROFILES",
    "TIERS",
    "build_classified_matrix",
    "classify_adapter_cell",
    "hidden_required_na",
    "undefined_cells",
]
