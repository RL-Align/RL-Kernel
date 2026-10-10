# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Replay matrix for post-training Attention drift triage.

The matrix deliberately separates a fixed replay baseline, one-at-a-time root-
cause probes, and invariant controls.  It is a reporting/debug contract, not a
second runtime knob catalog and it never implies a Cartesian product.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rl_engine.contracts.diagnostics.attention_axes import (
    ATTENTION_DEBUG_AXES,
    ATTENTION_INVARIANT_CONTROLS,
)
from rl_engine.contracts.operators.attention import AttentionContractError

ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION = "rlkernel.attention.debug_matrix.v1"


@dataclass(frozen=True)
class AttentionDebugMatrixRow:
    """One independent case in the replay matrix."""

    row_id: str
    label: str
    category: str
    probe: str | None = None
    root_cause_axis: str | None = None
    expected: str = "diagnostic"

    def __post_init__(self) -> None:
        if not self.row_id.strip() or not self.label.strip():
            raise ValueError("Attention debug matrix rows need an id and label")
        if self.category not in {
            "baseline",
            "root_cause",
            "comparability_gate",
            "invariant_control",
        }:
            raise ValueError(f"unknown Attention debug matrix category {self.category!r}")
        if self.category == "baseline":
            if self.probe is not None or self.root_cause_axis is not None:
                raise ValueError("baseline rows cannot name a probe or root-cause axis")
            if self.expected != "baseline":
                raise ValueError("baseline rows must use expected='baseline'")
        elif not self.probe or not self.probe.strip():
            raise ValueError("non-baseline rows need a probe")
        if self.category in {"root_cause", "comparability_gate"}:
            if not self.root_cause_axis:
                raise ValueError("root-cause and gate rows need root_cause_axis")
        if self.category == "root_cause":
            if self.expected != "diagnostic":
                raise ValueError("root-cause rows must use expected='diagnostic'")
        if self.category == "comparability_gate" and self.expected != "rejected":
            raise ValueError("comparability gates must use expected='rejected'")
        if self.category == "invariant_control":
            if self.expected != "exact_zero":
                raise ValueError("invariant controls must use expected='exact_zero'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "label": self.label,
            "category": self.category,
            "probe": self.probe,
            "root_cause_axis": self.root_cause_axis,
            "expected": self.expected,
        }


def _build_rows() -> tuple[AttentionDebugMatrixRow, ...]:
    rows: list[AttentionDebugMatrixRow] = [
        AttentionDebugMatrixRow(
            row_id="A0",
            label="Strict replay baseline",
            category="baseline",
            expected="baseline",
        )
    ]
    for index, axis in enumerate(ATTENTION_DEBUG_AXES, start=1):
        is_gate = axis.axis_id == "topology_head_ownership"
        rows.append(
            AttentionDebugMatrixRow(
                row_id=f"A{index}",
                label=axis.label,
                category="comparability_gate" if is_gate else "root_cause",
                probe=axis.representative_subprobe,
                root_cause_axis=axis.axis_id,
                expected="rejected" if is_gate else "diagnostic",
            )
        )
    for index, probe in enumerate(ATTENTION_INVARIANT_CONTROLS):
        rows.append(
            AttentionDebugMatrixRow(
                row_id=f"C{index}",
                label="Invariant control",
                category="invariant_control",
                probe=probe,
                expected="exact_zero",
            )
        )
    return tuple(rows)


ATTENTION_DEBUG_MATRIX = _build_rows()
_ROWS_BY_ID = MappingProxyType({row.row_id: row for row in ATTENTION_DEBUG_MATRIX})


def validate_attention_debug_matrix() -> None:
    """Validate matrix coverage and the no-Cartesian-product invariant."""

    if len(_ROWS_BY_ID) != len(ATTENTION_DEBUG_MATRIX):
        raise RuntimeError("Attention debug matrix row IDs must be unique")
    baselines = [row for row in ATTENTION_DEBUG_MATRIX if row.category == "baseline"]
    if len(baselines) != 1 or baselines[0].row_id != "A0":
        raise RuntimeError("Attention debug matrix must contain exactly one A0 baseline")

    axis_ids = {axis.axis_id for axis in ATTENTION_DEBUG_AXES}
    diagnostic_rows = [
        row
        for row in ATTENTION_DEBUG_MATRIX
        if row.category in {"root_cause", "comparability_gate"}
    ]
    if {row.root_cause_axis for row in diagnostic_rows} != axis_ids:
        raise RuntimeError("every root-cause axis must have one representative row")
    if len(diagnostic_rows) != len(axis_ids):
        raise RuntimeError("root-cause representatives must be one-at-a-time")
    gates = [row for row in diagnostic_rows if row.category == "comparability_gate"]
    if [row.root_cause_axis for row in gates] != ["topology_head_ownership"]:
        raise RuntimeError("only topology/head ownership is a comparability gate")

    controls = [row for row in ATTENTION_DEBUG_MATRIX if row.category == "invariant_control"]
    if {row.probe for row in controls} != set(ATTENTION_INVARIANT_CONTROLS):
        raise RuntimeError("invariant-control coverage is incomplete")
    if any(row.expected != "exact_zero" for row in controls):
        raise RuntimeError("invariant controls must be exact-zero checks")

    axis_by_id = {axis.axis_id: axis for axis in ATTENTION_DEBUG_AXES}
    for row in diagnostic_rows:
        axis_id = row.root_cause_axis
        if axis_id is None:
            raise RuntimeError(f"diagnostic row {row.row_id} must name a root-cause axis")
        axis = axis_by_id[axis_id]
        if row.probe != axis.representative_subprobe:
            raise RuntimeError(
                f"matrix row {row.row_id} is not the representative probe for {axis.axis_id}"
            )


validate_attention_debug_matrix()


def attention_debug_matrix() -> dict[str, Any]:
    """Return the portable matrix manifest used by reports and tooling."""

    return {
        "schema_version": ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION,
        "method": "fixed_replay_one_at_a_time",
        "baseline_row": "A0",
        "cartesian_product": False,
        "replay_identity": (
            "same checkpoint, token IDs, selected-token IDs, masks, positions, "
            "cache metadata, and pre-update model state"
        ),
        "row_baseline": "each diagnostic row is compared with its own phase-local A0 baseline",
        "metrics": [
            "train_rollout_logprob_abs_diff",
            "mismatch_kl",
            "mismatch_k3_kl",
            "out_max_abs",
            "lse_max_abs",
            "dq_max_abs",
            "dk_max_abs",
            "dv_max_abs",
        ],
        "rows": [row.to_dict() for row in ATTENTION_DEBUG_MATRIX],
        "topology_gate": [
            "checkpoint/model/token identity",
            "TP head ownership",
            "CP sequence ownership",
            "actual Split-KV plan",
        ],
    }


def attention_debug_matrix_row(row_id: str) -> AttentionDebugMatrixRow:
    """Look up a stable matrix row by its report ID."""

    if not isinstance(row_id, str) or not row_id.strip():
        raise AttentionContractError("Attention debug matrix row ID must be non-empty")
    try:
        return _ROWS_BY_ID[row_id.strip()]
    except KeyError as exc:
        raise AttentionContractError(f"unknown Attention debug matrix row {row_id!r}") from exc


__all__ = [
    "ATTENTION_DEBUG_MATRIX",
    "ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION",
    "AttentionDebugMatrixRow",
    "attention_debug_matrix",
    "attention_debug_matrix_row",
    "validate_attention_debug_matrix",
]
