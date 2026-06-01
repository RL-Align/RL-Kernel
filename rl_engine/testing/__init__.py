# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch
from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
)

__all__ = [
    "SyntheticRLKernelBatch",
    "active_token_count",
    "compute_policy_ratio",
    "compute_reference_kl",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
]
