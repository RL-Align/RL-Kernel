# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch mHC.

- ``dsv4_mhc``: the golden DSv4 mHC (profile ``dsv4-mhc-v1``), forward and dX-only
  backward.
- ``h_aggregate``: the legacy generic ``mhc_pre_h_aggregate`` operator (not used by DSv4;
  design doc section 11), re-exported here for its registry and validation paths.
"""

from .dsv4_mhc import (
    HC,
    HC_EPS,
    NORM_EPS,
    POST_MULT,
    RMS_EPS,
    SINKHORN_ITERS,
    SLICES,
    SPLITS,
    controller,
    fma_f32,
    gemm_partials,
    hc_head,
    hc_head_backward,
    hc_head_from_partials,
    mhc_post,
    mhc_post_backward,
    mhc_pre,
    mhc_pre_backward,
    mhc_pre_from_partials,
    rms_norm,
    rms_norm_backward,
    slice_dot,
    split_sum,
)
from .h_aggregate import (
    MHC_PRE_HC_MULT,
    MHC_PRE_HIDDEN_SIZE,
    MHC_PRE_SUPPORTED_HIDDEN_SIZES,
    NativeMHCPreHAggregateOp,
)

__all__ = [
    "MHC_PRE_HC_MULT",
    "MHC_PRE_HIDDEN_SIZE",
    "MHC_PRE_SUPPORTED_HIDDEN_SIZES",
    "NativeMHCPreHAggregateOp",
    "HC",
    "HC_EPS",
    "NORM_EPS",
    "POST_MULT",
    "RMS_EPS",
    "SINKHORN_ITERS",
    "SLICES",
    "SPLITS",
    "controller",
    "fma_f32",
    "hc_head",
    "hc_head_backward",
    "mhc_post",
    "mhc_post_backward",
    "mhc_pre",
    "mhc_pre_backward",
    "gemm_partials",
    "hc_head_from_partials",
    "mhc_pre_from_partials",
    "rms_norm",
    "rms_norm_backward",
    "split_sum",
    "slice_dot",
]
