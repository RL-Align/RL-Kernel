# SPDX-License-Identifier: Apache-2.0
"""INK-43: model-specific probes layered on the existing PR230 matrix."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Probe:
    task: str
    row: str
    name: str
    expected: str


PROBES = (
    Probe("INK-44", "A7", "loop_pass_order", "diagnostic"),
    Probe("INK-44", "A4", "loop_pass_count", "rejected"),
    Probe("INK-45", "A2", "loop_message_gate", "diagnostic"),
    Probe("INK-46", "A7", "shared_gradient_order", "diagnostic"),
    Probe("INK-46", "A4", "shared_weight_owner", "rejected"),
    Probe("INK-47", "A5", "logical_visit_cache_identity", "diagnostic"),
    Probe("INK-48", "A6", "attention_residual_normalization", "diagnostic"),
    Probe("INK-48", "A7", "attention_residual_order", "diagnostic"),
    Probe("INK-49", "A3", "local_global_attention_mask", "diagnostic"),
    Probe("INK-49", "A1", "relative_bias_and_log_scaling", "diagnostic"),
    Probe("INK-50", "A3", "short_conv_sequence_boundary", "diagnostic"),
    Probe("INK-51", "A6", "top8_router_weight_policy", "diagnostic"),
    Probe("INK-52", "A6", "low_bit_precision", "diagnostic"),
    Probe("INK-53", "A6", "logit_softcap_padding", "diagnostic"),
    Probe("INK-53", "A4", "tp_ep_cp_ownership", "rejected"),
)


def overlay(matrix: dict) -> dict:
    """Attach subprobes without changing PR230 rows or runtime settings."""
    rows = {row["id"] for row in matrix["rows"]}
    if rows != {f"A{i}" for i in range(8)} | {f"C{i}" for i in range(3)}:
        raise ValueError("expected existing A0-A7/C0-C2 matrix")
    if matrix.get("cartesian_product") is not False:
        raise ValueError("expected one-at-a-time matrix")
    return {
        **matrix,
        "inkling2_subprobes": [vars(probe).copy() for probe in PROBES],
        "inkling2_platforms": ["cuda", "rocm"],
        "inkling2_scope": "diagnostic-only",
    }


def inkling2_debug_matrix() -> dict:
    from rl_engine.contracts.diagnostics.attention_matrix import attention_debug_matrix

    return overlay(attention_debug_matrix())


def compare_boundaries(reference: dict, changed: dict) -> dict:
    """Count exact changed tensor elements, in recorded boundary order."""
    if list(reference) != list(changed) or not reference:
        raise ValueError("incomparable boundary order or coverage")
    counts = {}
    for name, a in reference.items():
        b = changed[name]
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise ValueError(f"incomparable tensor boundary {name}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError(f"nonfinite tensor boundary {name}")
        x = a.detach().contiguous().reshape(-1).view(torch.uint8)
        y = b.detach().contiguous().reshape(-1).view(torch.uint8)
        counts[name] = int((x != y).reshape(-1, a.element_size()).any(-1).sum())
    return {
        "first_drift": next((name for name, n in counts.items() if n), None),
        "mismatches": counts,
        "ws1_certified": False,
    }
