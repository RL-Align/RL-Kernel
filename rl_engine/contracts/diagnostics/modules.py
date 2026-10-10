# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compact, module-level mismatch axes for post-training drift triage.

The manifest is deliberately a reporting contract.  It names the first
diagnostic probe for each semantic operator without turning the probes into
runtime configuration or expanding a Cartesian product of settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rl_engine.contracts.diagnostics.attention_axes import ATTENTION_DEBUG_AXES
from rl_engine.contracts.diagnostics.attention_matrix import (
    ATTENTION_DEBUG_MATRIX,
    ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION,
)

DEBUG_MATRIX_SCHEMA_VERSION = "rlkernel.debug_matrix.v1"


@dataclass(frozen=True)
class ModuleDebugAxis:
    """One first-line axis exposed by a semantic operator."""

    module: str
    axis_id: str
    label: str
    representative_probe: str
    kind: str = "diagnostic"

    def __post_init__(self) -> None:
        for field in ("module", "axis_id", "label", "representative_probe"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"module debug {field} must be a non-empty string")
        if self.kind not in {"diagnostic", "gate"}:
            raise ValueError(f"unknown module debug axis kind {self.kind!r}")

    def to_dict(self, *, row_id: str) -> dict[str, str]:
        return {
            "row": row_id,
            "id": self.axis_id,
            "label": self.label,
            "representative_probe": self.representative_probe,
            "kind": self.kind,
        }


def _attention_axes() -> tuple[ModuleDebugAxis, ...]:
    return tuple(
        ModuleDebugAxis(
            module="attention",
            axis_id=axis.axis_id,
            label=axis.label,
            representative_probe=axis.representative_subprobe,
            kind="gate" if axis.axis_id == "topology_head_ownership" else "diagnostic",
        )
        for axis in ATTENTION_DEBUG_AXES
    )


MODULE_DEBUG_AXES: MappingProxyType[str, tuple[ModuleDebugAxis, ...]] = MappingProxyType(
    {
        "attention": _attention_axes(),
        "ffn": (
            ModuleDebugAxis(
                "ffn",
                "weight_shard_ownership",
                "Weight shard / TP ownership",
                "tp_weight_ownership",
                "gate",
            ),
            ModuleDebugAxis(
                "ffn",
                "swiglu_rounding",
                "SwiGLU intermediate rounding",
                "swiglu_one_round",
            ),
            ModuleDebugAxis(
                "ffn",
                "gemm_reduction",
                "GEMM K-reduction / Split-K policy",
                "k_reduction_split_k",
            ),
            ModuleDebugAxis(
                "ffn",
                "token_collective",
                "Token gather / reduce-scatter",
                "sequence_parallel",
            ),
        ),
        "logp": (
            ModuleDebugAxis(
                "logp",
                "vocab_shard_ownership",
                "Vocabulary shard / TP ownership",
                "vocab_shard_bounds",
                "gate",
            ),
            ModuleDebugAxis(
                "logp",
                "selected_token_identity",
                "Selected-token and active-mask identity",
                "selected_token_active_mask",
                "gate",
            ),
            ModuleDebugAxis(
                "logp",
                "vocab_lse_reduction",
                "Vocabulary LSE tile / merge policy",
                "vocab_tile_merge",
            ),
        ),
    }
)


_MODULE_DEBUG_AXIS_BY_ID = MappingProxyType(
    {(axis.module, axis.axis_id): axis for axes in MODULE_DEBUG_AXES.values() for axis in axes}
)


def _module_rows(module: str) -> dict[str, Any]:
    axes = MODULE_DEBUG_AXES[module]
    baseline = {"attention": "A0", "ffn": "F0", "logp": "L0"}[module]
    controls = {
        "attention": ["C0", "C1", "C2"],
        "ffn": ["FC0"],
        "logp": ["LC0"],
    }[module]
    rows = [baseline] + [f"{module[0].upper()}{index}" for index in range(1, len(axes) + 1)]
    return {
        "baseline_row": baseline,
        "rows": rows,
        "invariant_controls": controls,
        "axes": [axis.to_dict(row_id=rows[index]) for index, axis in enumerate(axes, start=1)],
    }


def module_debug_matrix() -> dict[str, Any]:
    """Return the portable matrix manifest shared by all three operators."""

    return {
        "schema_version": DEBUG_MATRIX_SCHEMA_VERSION,
        "method": "fixed_replay_one_at_a_time",
        "cartesian_product": False,
        "comparison_edges": [
            "train_vs_rollout_prefill",
            "rollout_prefill_vs_decode",
        ],
        "replay_identity": (
            "same checkpoint, token IDs, selected-token IDs, masks, positions, "
            "cache metadata, and pre-update model state"
        ),
        "modules": {module: _module_rows(module) for module in MODULE_DEBUG_AXES},
        "attention_compatibility": {
            "schema_version": ATTENTION_DEBUG_MATRIX_SCHEMA_VERSION,
            "rows": [row.to_dict() for row in ATTENTION_DEBUG_MATRIX],
        },
    }


def module_debug_axis(module: str, axis_id: str) -> ModuleDebugAxis:
    """Look up one stable axis by module and identifier."""

    try:
        return _MODULE_DEBUG_AXIS_BY_ID[(module.strip(), axis_id.strip())]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown module debug axis {module!r}/{axis_id!r}") from exc


__all__ = [
    "DEBUG_MATRIX_SCHEMA_VERSION",
    "MODULE_DEBUG_AXES",
    "ModuleDebugAxis",
    "module_debug_axis",
    "module_debug_matrix",
]
