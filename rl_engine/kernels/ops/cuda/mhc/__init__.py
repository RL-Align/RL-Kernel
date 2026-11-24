# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 mHC (profile ``dsv4-mhc-v1``)."""

from .dsv4_mhc import (
    PROFILE,
    hc_head,
    hc_head_backward_cuda,
    hc_head_cuda,
    mhc_post,
    mhc_post_backward_cuda,
    mhc_post_cuda,
    mhc_pre,
    mhc_pre_backward_cuda,
    mhc_pre_cuda,
)

__all__ = [
    "PROFILE",
    "hc_head",
    "hc_head_backward_cuda",
    "hc_head_cuda",
    "mhc_post",
    "mhc_post_backward_cuda",
    "mhc_post_cuda",
    "mhc_pre",
    "mhc_pre_backward_cuda",
    "mhc_pre_cuda",
]
