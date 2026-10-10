# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from rl_engine.contracts.diagnostics.attention_axes import (
    ATTENTION_DEBUG_SCHEMA_VERSION,
    attention_debug_probe_metadata,
    attention_debug_taxonomy,
)
from rl_engine.contracts.operators.attention import AttentionContractError


def test_attention_debug_taxonomy_is_compact_complete_and_unambiguous():
    taxonomy = attention_debug_taxonomy()

    assert taxonomy["schema_version"] == ATTENTION_DEBUG_SCHEMA_VERSION
    assert taxonomy["root_cause_axis_count"] == 7
    assert taxonomy["subprobe_count"] == 21
    assert taxonomy["invariant_control_count"] == 3
    assert set(taxonomy["root_cause_axes"]) == {
        "position_rope",
        "qk_preprocessing",
        "mask_sequence_boundary",
        "topology_head_ownership",
        "kv_cache_identity_layout",
        "numerical_policy",
        "distributed_schedule",
    }
    assert "strict_cp_degree_control" not in {
        probe for axis in taxonomy["root_cause_axes"].values() for probe in axis["subprobes"]
    }


def test_attention_debug_probe_metadata_separates_subprobes_and_controls():
    assert attention_debug_probe_metadata("position_ids") == {
        "category": "root_cause_subprobe",
        "root_cause_axis": "position_rope",
        "root_cause_label": "Position / RoPE",
        "representative": True,
    }
    control = attention_debug_probe_metadata("tp_partition_control")
    assert control["category"] == "invariant_control"
    assert control["root_cause_axis"] is None

    with pytest.raises(AttentionContractError, match="unknown Attention debug probe"):
        attention_debug_probe_metadata("strict_cp_degree_control")
