# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compact root-cause taxonomy for post-training Attention drift reports."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rl_engine.contracts.operators.attention import AttentionContractError

ATTENTION_DEBUG_SCHEMA_VERSION = "rlkernel.attention.debug_taxonomy.v1"


@dataclass(frozen=True)
class AttentionDebugAxisSpec:
    """One first-line post-training Attention drift category."""

    axis_id: str
    label: str
    representative_subprobe: str
    subprobes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("axis_id", "label", "representative_subprobe"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Attention debug {name} must be a non-empty string")
        if not self.subprobes or any(
            not isinstance(probe, str) or not probe.strip() for probe in self.subprobes
        ):
            raise ValueError("Attention debug subprobes must be non-empty strings")
        if len(set(self.subprobes)) != len(self.subprobes):
            raise ValueError(f"Attention debug axis {self.axis_id!r} has duplicate subprobes")
        if self.representative_subprobe not in self.subprobes:
            raise ValueError(
                f"Attention debug representative {self.representative_subprobe!r} "
                f"is not assigned to {self.axis_id!r}"
            )


ATTENTION_DEBUG_AXES = (
    AttentionDebugAxisSpec(
        axis_id="position_rope",
        label="Position / RoPE",
        representative_subprobe="position_ids",
        subprobes=(
            "position_ids",
            "rope_theta",
            "position_offsets",
            "decode_cache_position",
        ),
    ),
    AttentionDebugAxisSpec(
        axis_id="qk_preprocessing",
        label="Q/K preprocessing",
        representative_subprobe="qk_norm_disabled",
        subprobes=(
            "qk_norm_eps",
            "qk_norm_disabled",
            "qk_norm_weight",
            "attention_scale",
            "scale_placement",
        ),
    ),
    AttentionDebugAxisSpec(
        axis_id="mask_sequence_boundary",
        label="Mask / sequence boundary",
        representative_subprobe="causal_mask",
        subprobes=("causal_mask", "key_padding_mask"),
    ),
    AttentionDebugAxisSpec(
        axis_id="topology_head_ownership",
        label="Topology / head ownership",
        representative_subprobe="tp_head_ownership",
        subprobes=("tp_head_ownership",),
    ),
    AttentionDebugAxisSpec(
        axis_id="kv_cache_identity_layout",
        label="KV-cache identity / layout",
        representative_subprobe="kv_page_order",
        subprobes=("kv_page_order", "kv_cache_content"),
    ),
    AttentionDebugAxisSpec(
        axis_id="numerical_policy",
        label="Numerical policy",
        representative_subprobe="accum_dtype",
        subprobes=(
            "accum_dtype",
            "execution_dtype",
            "final_write_dtype",
            "early_downcast",
        ),
    ),
    AttentionDebugAxisSpec(
        axis_id="distributed_schedule",
        label="Distributed schedule",
        representative_subprobe="merge_order",
        subprobes=("nonstrict_cp_degree", "split_kv", "merge_order"),
    ),
)

ATTENTION_INVARIANT_CONTROLS = (
    "tp_partition_control",
    "batch_composition_control",
    "prefill_decode_tail_control",
)

_ATTENTION_DEBUG_AXIS_BY_ID = MappingProxyType(
    {axis.axis_id: axis for axis in ATTENTION_DEBUG_AXES}
)
_ATTENTION_DEBUG_AXIS_BY_PROBE = MappingProxyType(
    {probe: axis for axis in ATTENTION_DEBUG_AXES for probe in axis.subprobes}
)
if len(_ATTENTION_DEBUG_AXIS_BY_ID) != len(ATTENTION_DEBUG_AXES):
    raise RuntimeError("Attention debug axis IDs must be unique")
if len(_ATTENTION_DEBUG_AXIS_BY_PROBE) != sum(len(axis.subprobes) for axis in ATTENTION_DEBUG_AXES):
    raise RuntimeError("Attention debug subprobes must belong to one root-cause axis")
if set(_ATTENTION_DEBUG_AXIS_BY_PROBE) & set(ATTENTION_INVARIANT_CONTROLS):
    raise RuntimeError("Attention debug subprobes cannot also be invariant controls")


def attention_debug_probe_metadata(probe: str) -> dict[str, Any]:
    """Classify one stable debug probe for report aggregation."""

    if not isinstance(probe, str) or not probe.strip():
        raise AttentionContractError("Attention debug probe must be a non-empty string")
    normalized = probe.strip()
    if normalized in ATTENTION_INVARIANT_CONTROLS:
        return {
            "category": "invariant_control",
            "root_cause_axis": None,
            "root_cause_label": None,
            "representative": False,
        }
    axis = _ATTENTION_DEBUG_AXIS_BY_PROBE.get(normalized)
    if axis is None:
        raise AttentionContractError(f"unknown Attention debug probe {probe!r}")
    return {
        "category": "root_cause_subprobe",
        "root_cause_axis": axis.axis_id,
        "root_cause_label": axis.label,
        "representative": normalized == axis.representative_subprobe,
    }


def attention_debug_taxonomy() -> dict[str, Any]:
    """Return the compact JSON schema used by post-training drift reports."""

    return {
        "schema_version": ATTENTION_DEBUG_SCHEMA_VERSION,
        "root_cause_axis_count": len(ATTENTION_DEBUG_AXES),
        "subprobe_count": len(_ATTENTION_DEBUG_AXIS_BY_PROBE),
        "invariant_control_count": len(ATTENTION_INVARIANT_CONTROLS),
        "root_cause_axes": {
            axis.axis_id: {
                "label": axis.label,
                "representative_subprobe": axis.representative_subprobe,
                "subprobes": list(axis.subprobes),
            }
            for axis in ATTENTION_DEBUG_AXES
        },
        "invariant_controls": list(ATTENTION_INVARIANT_CONTROLS),
    }


__all__ = [
    "ATTENTION_DEBUG_AXES",
    "ATTENTION_DEBUG_SCHEMA_VERSION",
    "ATTENTION_INVARIANT_CONTROLS",
    "AttentionDebugAxisSpec",
    "attention_debug_probe_metadata",
    "attention_debug_taxonomy",
]
