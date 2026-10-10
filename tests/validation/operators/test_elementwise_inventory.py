# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU tests for the WS1 C5 elementwise / RoPE inventory."""

from __future__ import annotations

from rl_engine.config.workload import load_manifest
from rl_engine.validation.operators.elementwise_inventory import (
    inventory_items,
    inventory_names,
    unresolved_needs_fix,
)
from rl_engine.validation.operators.gradient_adapters import get_adapter

_REQUIRED_ITEMS = (
    "rope",
    "silu",
    "swiglu",
    "residual_add",
    "scale",
    "bias",
    "mask_fill",
    "dtype_cast",
)


def test_inventory_covers_c5_required_items():
    assert set(_REQUIRED_ITEMS) <= set(inventory_names())
    assert len(inventory_names()) == len(set(inventory_names()))


def test_every_item_has_a_verdict_or_blocker():
    allowed = {"pass", "blocker", "blocked_hardware", "tracked_red", "absent_not_required"}
    for item in inventory_items():
        assert item.cuda_verdict in allowed, item.name
        assert item.triton_verdict in allowed, item.name
        assert item.entry_point
        assert item.reduction
        assert item.evidence
        if item.cuda_verdict in {"blocker", "blocked_hardware"} or item.triton_verdict in {
            "blocker",
            "blocked_hardware",
        }:
            assert item.blocker, item.name


def test_no_untracked_needs_fix_without_blocker():
    open_items = unresolved_needs_fix()
    assert open_items == ()


def test_on_chain_differentiable_ops_are_c3_c4_enumerable():
    for item in inventory_items():
        if item.name in {"rope", "silu", "swiglu"}:
            adapter = get_adapter(item.name)
            assert adapter.tensors
            assert adapter.requirement == "required"


def test_qk_norm_still_required_on_chain(manifest=None):
    manifest = manifest or load_manifest()
    assert manifest.raw["capabilities"]["qk_norm"]["status"] == "required_on_chain"
    assert manifest.raw["model_identity"]["config_fingerprint"]["attention_bias"] is False
