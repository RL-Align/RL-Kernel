# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded MLP-boundary values at the Qwen-Image request layouts."""

import torch

from rl_engine.reference.model_specific.qwenimage.tp_mlp_gemm_gelu import HIDDEN, INTERMEDIATE
from rl_engine.validation.fixtures.qwenimage_attn_out import make_attn_out_workload
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_LAYOUTS as MLP_LAYOUTS
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_WORKLOADS as MLP_WORKLOADS
from rl_engine.validation.fixtures.qwenimage_qkv import (
    canonical_qkv_indices as canonical_mlp_indices,
)
from rl_engine.validation.fixtures.qwenimage_qkv import qkv_workload_rows as mlp_workload_rows


def make_mlp_workload(name, device, seed=386):
    """Reuse logical row identities and call cuts; each stream owns its parameters."""
    fixture = make_attn_out_workload(name, torch.bfloat16, device, seed)
    # Keep weights independent of the reused input fixture's random sequence.
    generator = torch.Generator().manual_seed(seed + 1)

    def rand(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(
            device=device, dtype=torch.bfloat16
        )

    for stream in fixture["streams"].values():
        del stream["weight"], stream["bias"]
        stream["up_weight"] = rand((INTERMEDIATE, HIDDEN), HIDDEN**-0.5)
        stream["up_bias"] = rand((INTERMEDIATE,), 0.1)
        stream["down_weight"] = rand((HIDDEN, INTERMEDIATE), INTERMEDIATE**-0.5)
        stream["down_bias"] = rand((HIDDEN,), 0.1)
    fixture["metadata"]["input_scope"] = "seeded MLP-boundary values at model-derived token layouts"
    for stream in ("image", "text"):
        key = stream + "_attn_out_shape"
        if key in fixture["metadata"]:
            fixture["metadata"][stream + "_mlp_input_shape"] = fixture["metadata"].pop(key)
    fixture["metadata"]["intermediate_width"] = INTERMEDIATE
    fixture["metadata"]["parameter_seed"] = seed + 1
    return fixture


__all__ = [
    "MLP_LAYOUTS",
    "MLP_WORKLOADS",
    "canonical_mlp_indices",
    "make_mlp_workload",
    "mlp_workload_rows",
]
