# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    distributed_active_token_count,
    distributed_masked_mean,
    distributed_masked_sum,
    masked_mean,
    masked_sum,
    owner_ranks_for_token_ids,
    selected_logprobs_distributed_tp_reference,
    selected_logprobs_reference,
    selected_logprobs_tp_reference,
    shard_logits_by_vocab,
    sharded_active_token_count,
    sharded_masked_mean,
    sharded_masked_sum,
    summarize_kernel_drift,
    summarize_tp_logprob_drift,
    vocab_shard_ranges,
)
from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch

__all__ = [
    "SyntheticRLKernelBatch",
    "active_token_count",
    "compute_policy_ratio",
    "compute_reference_kl",
    "distributed_active_token_count",
    "distributed_masked_mean",
    "distributed_masked_sum",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "owner_ranks_for_token_ids",
    "selected_logprobs_distributed_tp_reference",
    "selected_logprobs_reference",
    "selected_logprobs_tp_reference",
    "shard_logits_by_vocab",
    "sharded_active_token_count",
    "sharded_masked_mean",
    "sharded_masked_sum",
    "summarize_kernel_drift",
    "summarize_tp_logprob_drift",
    "vocab_shard_ranges",
]
