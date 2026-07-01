# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .logprob_parity import compare_selected_logprob_layouts, make_padded_batch_layout
from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
)
from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch

__all__ = [
    "SyntheticRLKernelBatch",
    "active_token_count",
    "compare_selected_logprob_layouts",
    "compute_policy_ratio",
    "compute_reference_kl",
    "make_padded_batch_layout",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
]
