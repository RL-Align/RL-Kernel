# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .forward_invariance import (
    DEFAULT_BATCH_INVARIANT_SWEEP,
    BatchInvariantConfig,
    apply_rope_reference,
    assert_batch_invariant_across_configs,
    build_rope_cache,
    rotate_half,
)
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
    "DEFAULT_BATCH_INVARIANT_SWEEP",
    "BatchInvariantConfig",
    "SyntheticRLKernelBatch",
    "active_token_count",
    "apply_rope_reference",
    "assert_batch_invariant_across_configs",
    "build_rope_cache",
    "compute_policy_ratio",
    "compute_reference_kl",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "rotate_half",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
]
