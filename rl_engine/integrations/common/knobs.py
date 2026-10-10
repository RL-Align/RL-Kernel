# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 attention knobs for the Qwen3-8B TP=2 CP=2 Megatron + vLLM target.

``V1_KNOBS`` was written against a HuggingFace/FSDP rollout-vs-training pair. Three
of its entries do not survive contact with the frozen Megatron + vLLM target:

* ``training.sharding`` takes ``unsharded``/``fsdp``, neither of which exists in
  Megatron, and is meaningless at DP=1 anyway;
* ``training.attention_backend`` takes HuggingFace names
  (``flash_attention_2``/``sdpa``/``eager``/``model_default``) while Megatron's
  ``AttnBackend`` is ``flash``/``fused``/``unfused``/``local``/``auto``;
* there is no training-side ``tensor_parallel_size`` or ``context_parallel_size``
  at all, so the target configuration cannot even be expressed.

This module is deliberately **additive**: it extends ``V1_KNOBS`` rather than
editing it, and overrides only the normalizer for ``training.attention_backend``.
Deleting the two dead knobs changes ``V1_KNOBS`` itself and would break existing
cross-config tests, so it is left to a follow-up on the framework PR.
"""

from __future__ import annotations

from collections.abc import Mapping

from rl_engine.validation.cross_config.planner import (
    _NORMALIZERS,
    V1_KNOBS,
    Normalizer,
    _normalize_choice,
    _positive_int,
    _strict_bool,
)
from rl_engine.validation.cross_config.schema import IsolationScope, KnobDescriptor

__all__ = [
    "MEGATRON_ATTENTION_BACKENDS",
    "WS2_ATTENTION_KNOBS",
    "WS2_ATTENTION_KNOB_DESCRIPTORS",
    "WS2_ATTENTION_NORMALIZERS",
]


#: ``megatron.core.transformer.enums.AttnBackend``.
MEGATRON_ATTENTION_BACKENDS: tuple[str, ...] = (
    "flash",
    "fused",
    "unfused",
    "local",
    "auto",
)


WS2_ATTENTION_KNOB_DESCRIPTORS: tuple[KnobDescriptor, ...] = (
    # -- training-side parallelism: the target configuration itself ------------
    KnobDescriptor(
        "training.tensor_parallel_size",
        IsolationScope.PROCESS,
        ("training",),
    ),
    KnobDescriptor(
        "training.context_parallel_size",
        IsolationScope.PROCESS,
        ("training",),
    ),
    # -- determinism switches, one per framework ------------------------------
    KnobDescriptor(
        "training.deterministic_mode",
        IsolationScope.PROCESS,
        ("training",),
    ),
    KnobDescriptor(
        "rollout.batch_invariant",
        IsolationScope.PROCESS,
        ("rollout",),
    ),
    # -- reduction knobs: the "turn the noise sources on and off" axis ---------
    # These are what make drift attributable. ``reduction.order=arrival`` in
    # particular is a control group, not a supported production value.
    KnobDescriptor(
        "attention.reduction_acc_dtype",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        allowed_values=("fp32", "bf16"),
    ),
    KnobDescriptor(
        "attention.reduction_order",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        allowed_values=("global_block_index", "arrival"),
    ),
    KnobDescriptor(
        "attention.reduction_downcast_at",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        allowed_values=("final_write", "per_block"),
    ),
    KnobDescriptor(
        "attention.reduction_engine",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        allowed_values=("in_op_reference", "te_oracle"),
    ),
    # -- materialization knobs: differences the experiment measures ------------
    KnobDescriptor(
        "attention.fusion_boundary",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        allowed_values=("unfused_rope_attention", "fused_rope_attention"),
    ),
    KnobDescriptor(
        # Shared logical KV chunk size. Runtime adapters must separately report
        # the actual per-owner boundaries; a configured value is not evidence.
        "attention.split_kv_policy",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
    ),
    KnobDescriptor(
        # vLLM: CacheConfig.block_size -> AttentionContract.kv_cache.page_size
        "rollout.kv_block_size",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout",),
    ),
    # The CP communication group cannot be reconfigured once built; it is bound to
    # the distributed context, not merely to engine construction.
    KnobDescriptor(
        "training.cp_comm_type",
        IsolationScope.DISTRIBUTED_CONTEXT,
        ("training",),
        allowed_values=("p2p", "all_gather", "a2a", "a2a+p2p"),
    ),
)


WS2_ATTENTION_KNOBS: Mapping[str, KnobDescriptor] = {
    **V1_KNOBS,
    **{descriptor.path: descriptor for descriptor in WS2_ATTENTION_KNOB_DESCRIPTORS},
}


WS2_ATTENTION_NORMALIZERS: Mapping[str, Normalizer] = {
    **_NORMALIZERS,
    # Replace, not map: the HuggingFace names have no Megatron counterpart.
    "training.attention_backend": _normalize_choice(*MEGATRON_ATTENTION_BACKENDS),
    "training.tensor_parallel_size": _positive_int,
    "training.context_parallel_size": _positive_int,
    "training.deterministic_mode": _strict_bool,
    "rollout.batch_invariant": _strict_bool,
    # AttentionDType values, not torch dtype names -- these feed ReductionSpec directly.
    "attention.reduction_acc_dtype": _normalize_choice("fp32", "bf16"),
    "attention.reduction_order": _normalize_choice("global_block_index", "arrival"),
    "attention.reduction_downcast_at": _normalize_choice("final_write", "per_block"),
    "attention.reduction_engine": _normalize_choice("in_op_reference", "te_oracle"),
    "attention.fusion_boundary": _normalize_choice(
        "unfused_rope_attention", "fused_rope_attention"
    ),
    "attention.split_kv_policy": _positive_int,
    "rollout.kv_block_size": _positive_int,
    "training.cp_comm_type": _normalize_choice("p2p", "all_gather", "a2a", "a2a+p2p"),
}
