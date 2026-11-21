# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MoE kernels.

Two lanes with different jobs:

* ``shared_expert`` is the oracle-strict P5-5 reference: one lane per output
  element, serial ascending-k, no ``tl.dot``. Slow by construction, and kept
  because it reproduces ``oracle-fp32-serial-v1`` bit for bit.
* ``fused_mlp`` is the portable production lane, the counterpart of
  ``csrc/cuda/moe/*.cu``: four kernels -- shared fc1+SwiGLU, shared fc3 (TP
  invariant), routed fc1+SwiGLU+MX quant, routed fc3. It uses ``tl.dot`` and
  fuses the activation into the fc1 epilogue. Written to be tuned on ROCm, so
  it is batch-invariant but does not chase byte-equality with the CUDA kernels.
"""

from .fused_mlp import (
    TP_SEGMENTS,
    fused_shared_fc1_swiglu,
    routed_fc1_swiglu_quant,
    routed_fc3,
    routed_mlp_forward,
    shared_fc3,
)
from .mxfp8_act_quant import mxfp8_act_quant_bwd_triton, mxfp8_act_quant_fwd_triton

__all__ = [
    "mxfp8_act_quant_fwd_triton",
    "mxfp8_act_quant_bwd_triton",
    "TP_SEGMENTS",
    "fused_shared_fc1_swiglu",
    "shared_fc3",
    "routed_fc1_swiglu_quant",
    "routed_fc3",
    "routed_mlp_forward",
]
