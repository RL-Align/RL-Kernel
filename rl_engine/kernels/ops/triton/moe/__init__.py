# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MoE kernels.

``fused_mlp`` is the portable production lane: four kernels -- shared
fc1+SwiGLU, shared fc3 (TP invariant), routed fc1+SwiGLU+MX quant, routed fc3.
It uses ``tl.dot`` and fuses the activation into the fc1 epilogue. Written to
be tuned on ROCm, so it is batch-invariant but byte-equal to no other lane.
"""

from .fused_mlp import (
    TP_SEGMENTS,
    fused_shared_fc1_swiglu,
    routed_fc1_swiglu_quant,
    routed_fc3,
    routed_mlp_forward,
    shared_fc3,
)

__all__ = [
    "TP_SEGMENTS",
    "fused_shared_fc1_swiglu",
    "shared_fc3",
    "routed_fc1_swiglu_quant",
    "routed_fc3",
    "routed_mlp_forward",
]
