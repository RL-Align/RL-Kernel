# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from rl_engine.contracts.diagnostics.modules import (
    DEBUG_MATRIX_SCHEMA_VERSION,
    MODULE_DEBUG_AXES,
    module_debug_axis,
    module_debug_matrix,
)


def test_module_matrix_is_fixed_replay_and_compact():
    manifest = module_debug_matrix()

    assert manifest["schema_version"] == DEBUG_MATRIX_SCHEMA_VERSION
    assert manifest["method"] == "fixed_replay_one_at_a_time"
    assert manifest["cartesian_product"] is False
    assert manifest["comparison_edges"] == [
        "train_vs_rollout_prefill",
        "rollout_prefill_vs_decode",
    ]
    assert set(manifest["modules"]) == {"attention", "ffn", "logp"}
    assert manifest["modules"]["attention"]["rows"] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
    ]
    assert manifest["modules"]["ffn"]["rows"] == ["F0", "F1", "F2", "F3", "F4"]
    assert manifest["modules"]["logp"]["rows"] == ["L0", "L1", "L2", "L3"]
    assert [row["row"] for row in manifest["modules"]["ffn"]["axes"]] == [
        "F1",
        "F2",
        "F3",
        "F4",
    ]


def test_module_axes_mark_identity_and_topology_as_gates():
    assert module_debug_axis("attention", "topology_head_ownership").kind == "gate"
    assert module_debug_axis("ffn", "weight_shard_ownership").kind == "gate"
    assert module_debug_axis("logp", "vocab_shard_ownership").kind == "gate"
    assert module_debug_axis("logp", "selected_token_identity").kind == "gate"
    assert module_debug_axis("ffn", "gemm_reduction").kind == "diagnostic"

    with pytest.raises(ValueError, match="unknown module debug axis"):
        module_debug_axis("ffn", "not_an_axis")


def test_module_axes_have_unique_ids_and_probes():
    for module, axes in MODULE_DEBUG_AXES.items():
        assert all(axis.module == module for axis in axes)
        assert len({axis.axis_id for axis in axes}) == len(axes)
        assert len({axis.representative_probe for axis in axes}) == len(axes)
