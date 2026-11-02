# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from rl_engine.contracts.diagnostics.attention_matrix import (
    ATTENTION_DEBUG_MATRIX,
    ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION,
    attention_debug_matrix,
    attention_debug_matrix_row,
)
from rl_engine.contracts.operators.attention import AttentionContractError


def test_attention_debug_matrix_is_fixed_replay_oat_with_controls():
    manifest = attention_debug_matrix()

    assert manifest["schema_version"] == ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION
    assert manifest["method"] == "fixed_replay_one_at_a_time"
    assert manifest["baseline_row"] == "A0"
    assert manifest["cartesian_product"] is False
    assert [row["id"] for row in manifest["rows"]] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "C0",
        "C1",
        "C2",
    ]
    assert sum(row["category"] == "root_cause" for row in manifest["rows"]) == 6
    assert sum(row["category"] == "comparability_gate" for row in manifest["rows"]) == 1
    assert sum(row["category"] == "invariant_control" for row in manifest["rows"]) == 3


def test_attention_debug_matrix_rows_have_stable_semantics():
    baseline = attention_debug_matrix_row("A0")
    assert baseline.category == "baseline"
    assert baseline.expected == "baseline"

    root = attention_debug_matrix_row("A1")
    assert root.category == "root_cause"
    assert root.probe == "position_ids"
    assert root.root_cause_axis == "position_rope"
    assert root.expected == "diagnostic"

    topology = attention_debug_matrix_row("A4")
    assert topology.category == "comparability_gate"
    assert topology.expected == "rejected"

    control = attention_debug_matrix_row("C0")
    assert control.category == "invariant_control"
    assert control.probe == "tp_partition_control"
    assert control.expected == "exact_zero"

    with pytest.raises(AttentionContractError, match="unknown Attention debug matrix row"):
        attention_debug_matrix_row("A8")


def test_matrix_manifest_matches_taxonomy_representatives():
    from rl_engine.contracts.diagnostics.attention_axes import ATTENTION_DEBUG_AXES

    root_rows = [
        row
        for row in ATTENTION_DEBUG_MATRIX
        if row.category in {"root_cause", "comparability_gate"}
    ]
    assert [row.root_cause_axis for row in root_rows] == [
        axis.axis_id for axis in ATTENTION_DEBUG_AXES
    ]
    assert [row.probe for row in root_rows] == [
        axis.representative_subprobe for axis in ATTENTION_DEBUG_AXES
    ]
