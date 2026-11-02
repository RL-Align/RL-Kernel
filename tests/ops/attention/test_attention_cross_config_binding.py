# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for #235 PR4: rollout/training attention contract binding.

Every test here runs on CPU without Megatron or vLLM installed. That is the point:
the binding rules are contract logic, and contract logic that can only be exercised
on a 2-node x 2-GPU cluster would never be exercised.
"""

from __future__ import annotations

import json
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContractError,
    AttentionMode,
    AttentionRole,
    KVCacheSpec,
    SplitKVExecutionPlan,
    SplitKVRuntimePlanEntry,
    SplitKVRuntimePlanSet,
    SplitKVSpec,
    build_split_kv_runtime_plan_set,
)
from rl_engine.integrations.common.attention_binding import (
    ATTENTION_LSE_DOMAIN,
    AttentionBindingError,
    BindingErrorCode,
    BindingTier,
    bind_attention_contracts,
    bind_attention_runtime_readbacks,
    first_blocking_issue,
    identity_fingerprint,
    summarize_binding,
)
from rl_engine.ops.attention.preprocess import MANDATED_ATTENTION_PREPROCESS_BACKENDS
from rl_engine.ops.attention.projection import (
    O_PROJ_COLLECTIVE_CONTRACT,
    QKV_COLLECTIVE_CONTRACT,
    ProjectionPlan,
)
from rl_engine.validation.cross_config.adapters import (
    QWEN3_8B,
    WS2_ATTENTION_KNOBS,
    AttentionRuntimeReadback,
    MegatronAttentionMaterializer,
    MegatronProvenanceAdapter,
    VllmProvenanceAdapter,
    VllmRolloutMaterializer,
)
from rl_engine.validation.cross_config.determinism import (
    compare_determinism,
    megatron_probe_from_config,
    vllm_probe_from_env,
)
from rl_engine.validation.cross_config.schema import MaterializationStatus

pytestmark = pytest.mark.unit


TRAINING_KNOBS = {
    "batch.size": 2,
    "training.tensor_parallel_size": 2,
    "training.context_parallel_size": 2,
    "training.compute_dtype": "bf16",
    "attention.split_kv_policy": 32,
}

ROLLOUT_KNOBS = {
    "batch.size": 2,
    "rollout.tensor_parallel_size": 2,
    "rollout.context_parallel_size": 2,
    "rollout.dtype": "bf16",
    "attention.split_kv_policy": 32,
}


def _identity(**overrides):
    identity = {
        "checkpoint_id": "qwen3-8b",
        "model_version": "v1",
        "weight_version": 7,
        "tokenizer_fingerprint": "tokenizer-abc",
        "token_ids_fingerprint": "tokens-abc",
        "active_mask_fingerprint": "mask-abc",
        "position_ids_fingerprint": "pos-abc",
        "padding_side": "right",
        "pre_update_state": "pre_update",
        "batch_size": 2,
        "global_token_positions_fingerprint": "gtp-abc",
        "kv_seq_lens_fingerprint": "kvlen-abc",
    }
    identity.update(QWEN3_8B.identity_fields())
    identity.update(overrides)
    return identity


def _contracts():
    training = MegatronAttentionMaterializer().build_contract(TRAINING_KNOBS)
    rollout = VllmRolloutMaterializer().build_contract(ROLLOUT_KNOBS)
    return rollout, training


def _plan_set(contract, *, backend):
    return build_split_kv_runtime_plan_set(
        (contract.sharding.global_sequence_length,) * contract.batch_size,
        tp_world_size=contract.sharding.tp_world_size,
        cp_world_size=contract.sharding.cp_world_size,
        split_kv=contract.split_kv,
        backend=backend,
    )


def _bind(rollout_identity=None, training_identity=None, **kwargs):
    rollout, training = _contracts()
    rollout = kwargs.pop("rollout_contract", rollout)
    training = kwargs.pop("training_contract", training)
    return bind_attention_contracts(
        rollout_contract=rollout,
        training_contract=training,
        rollout_identity=rollout_identity if rollout_identity is not None else _identity(),
        training_identity=training_identity if training_identity is not None else _identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="rlkernel.cp_attention_reference",
        rollout_split_kv_plan_set=kwargs.pop(
            "rollout_split_kv_plan_set", _plan_set(rollout, backend="vllm.readback")
        ),
        training_split_kv_plan_set=kwargs.pop(
            "training_split_kv_plan_set", _plan_set(training, backend="megatron.readback")
        ),
        **kwargs,
    )


# --------------------------------------------------------------------------
# tier 1: identity
# --------------------------------------------------------------------------


def test_matching_identity_and_topology_bind_despite_different_materialization():
    """The core claim of PR4: same identity + same reduction, different runtimes."""

    result = _bind()

    assert result.comparable
    assert result.passed
    assert result.issues == ()
    # Attention mode is a framework materialization difference, while both sides
    # execute the same TP=2/CP=2 local ownership and Split-K schedule.
    assert "mode" in result.recorded_differences
    assert result.recorded_differences["mode"] == {
        "rollout": "chunked_prefill",
        "training": "prefill",
    }


def test_weight_version_mismatch_is_not_comparable():
    result = _bind(rollout_identity=_identity(weight_version=6))

    assert not result.comparable
    assert not result.passed
    codes = {issue.code for issue in result.issues}
    assert BindingErrorCode.IDENTITY_MISMATCH in codes
    blocking = first_blocking_issue(result)
    assert blocking is not None and blocking.tier is BindingTier.IDENTICAL
    assert "NOT COMPARABLE" in summarize_binding(result)


def test_rope_theta_mismatch_is_not_comparable():
    """RoPE math constants are identity, not materialization."""

    result = _bind(training_identity=_identity(rope_theta=10000.0))

    assert not result.comparable
    assert any(issue.field == "rope_theta" for issue in result.issues)


def test_null_rope_scaling_is_a_value_not_an_omission():
    """Qwen3-8B applies no RoPE scaling; ``None`` must not read as undeclared."""

    result = _bind()

    assert not result.issues_by_code(BindingErrorCode.IDENTITY_MISSING)


def test_missing_identity_field_is_reported_per_side():
    identity = _identity()
    del identity["padding_side"]
    result = _bind(rollout_identity=identity, training_identity=identity)

    missing = result.issues_by_code(BindingErrorCode.IDENTITY_MISSING)
    assert {issue.field for issue in missing} == {
        "rollout.padding_side",
        "training.padding_side",
    }
    assert not result.comparable


def test_single_gpu_harness_may_waive_full_identity():
    """#235 PR2 has no KV-cache identity to declare; it opts out explicitly."""

    identity = _identity()
    del identity["global_token_positions_fingerprint"]
    del identity["kv_seq_lens_fingerprint"]

    strict = _bind(rollout_identity=identity, training_identity=identity)
    waived = _bind(
        rollout_identity=identity,
        training_identity=identity,
        require_full_identity=False,
    )

    assert not strict.comparable
    assert waived.comparable and waived.passed


def test_identity_fingerprint_ignores_undeclared_extra_keys():
    base = _identity()
    decorated = dict(base, diagnostic_note="added later")

    assert identity_fingerprint(base) == identity_fingerprint(decorated)


# --------------------------------------------------------------------------
# tier 2: reduction semantics
# --------------------------------------------------------------------------


def test_reduction_semantics_are_bound_and_fingerprinted():
    result = _bind()

    reduction = result.provenance["training"]["contract"]["reduction"]
    assert reduction["merge"] == "online_softmax_lse"
    assert reduction["acc_dtype"] == "fp32"
    assert reduction["order"] == "global_block_index"
    assert reduction["downcast_at"] == "final_write"
    assert result.reduction_fingerprint


def test_reduction_engine_difference_is_recorded_not_rejected():
    """A TE merge oracle on one side must not fail the binding."""

    from rl_engine.integrations.common.attention_binding import (
        RECORDED_FIELDS,
        SEMANTIC_REDUCTION_FIELDS,
    )

    assert "reduction.engine" in RECORDED_FIELDS
    assert "engine" not in SEMANTIC_REDUCTION_FIELDS


def test_lse_domain_is_recorded_as_attention_domain():
    """#235: attention exports attention-domain LSE, not vocab-logprob LSE."""

    result = _bind()

    assert result.provenance["lse_domain"] == ATTENTION_LSE_DOMAIN == "attention"


def test_mixed_dtypes_fail_closed():
    """BF16 rollout against FP16 training produces an unattributable number."""

    rollout = VllmRolloutMaterializer().build_contract(
        {**ROLLOUT_KNOBS, "rollout.dtype": "float16"}
    )
    result = _bind(rollout_contract=rollout)

    assert result.comparable  # identity is fine
    assert not result.passed
    assert any(issue.field == "dtype" for issue in result.issues)


def test_precision_sweep_may_opt_into_mixed_dtypes():
    """#235 PR5 sweeps BF16 against an FP32 reference; it says so explicitly."""

    training = MegatronAttentionMaterializer().build_contract(
        {**TRAINING_KNOBS, "training.compute_dtype": "float32"}
    )
    result = _bind(training_contract=training, allow_dtype_difference=True)

    assert result.passed
    assert result.provenance["dtype"] == "fp32"


def test_batch_size_mismatch_is_not_comparable():
    """Batch invariance is a claim about batch makeup, so it belongs to identity."""

    result = _bind(rollout_identity=_identity(batch_size=4))

    assert not result.comparable
    assert any(issue.field == "batch_size" for issue in result.issues)


def test_missing_split_kv_runtime_evidence_fails_closed():
    result = _bind(rollout_split_kv_plan_set=None)

    assert result.comparable
    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.SPLIT_KV_RUNTIME_MISSING)


def test_split_kv_requested_policy_mismatch_fails_closed():
    rollout, training = _contracts()
    rollout = replace(rollout, split_kv=SplitKVSpec.fixed(16))
    result = _bind(rollout_contract=rollout)

    assert result.comparable
    assert not result.passed
    assert any(issue.field == "split_kv" for issue in result.issues)


def test_split_kv_runtime_boundary_mismatch_fails_closed():
    rollout, training = _contracts()
    mismatched_rollout = build_split_kv_runtime_plan_set(
        (rollout.sharding.global_sequence_length,) * rollout.batch_size,
        tp_world_size=2,
        cp_world_size=2,
        split_kv=SplitKVSpec.fixed(16),
        backend="vllm.readback",
    )
    result = _bind(rollout_split_kv_plan_set=mismatched_rollout)

    assert not result.passed
    issues = result.issues_by_code(BindingErrorCode.SPLIT_KV_MISMATCH)
    assert any(issue.field == "split_kv_runtime_plan_set" for issue in issues)


def test_split_kv_plan_set_must_match_its_own_contract_topology():
    rollout, _ = _contracts()
    wrong_topology = build_split_kv_runtime_plan_set(
        (rollout.sharding.global_sequence_length,) * rollout.batch_size,
        tp_world_size=1,
        cp_world_size=2,
        split_kv=rollout.split_kv,
        backend="vllm.readback",
    )

    result = _bind(rollout_split_kv_plan_set=wrong_topology)

    assert not result.passed
    assert any(
        issue.field == "rollout.split_kv_runtime_plan_set"
        for issue in result.issues_by_code(BindingErrorCode.SPLIT_KV_MISMATCH)
    )


@pytest.mark.parametrize(
    ("field_name", "corrupt_value"),
    [
        ("merge_order", Enum("BadOrder", {"ARRIVAL": "arrival"}).ARRIVAL),
        ("acc_dtype", Enum("BadDType", {"BF16": "bf16"}).BF16),
        ("downcast_at", Enum("BadDowncast", {"PER_BLOCK": "per_block"}).PER_BLOCK),
    ],
)
def test_split_kv_runtime_merge_semantic_corruption_fails_closed(field_name, corrupt_value):
    rollout, _ = _contracts()
    corrupted = _plan_set(rollout, backend="vllm.readback")
    # Runtime reports are deserialized at this boundary. Simulate a corrupted
    # report after construction to prove binding compares the actual fields.
    object.__setattr__(corrupted.entries[0].execution, field_name, corrupt_value)

    result = _bind(rollout_split_kv_plan_set=corrupted)

    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.SPLIT_KV_MISMATCH)


def test_split_kv_runtime_fallback_fails_closed():
    rollout, _ = _contracts()
    plan_set = _plan_set(rollout, backend="vllm.readback")
    fallback_entries = []
    for entry in plan_set.entries:
        execution = entry.execution
        fallback_entries.append(
            SplitKVRuntimePlanEntry(
                coordinate=entry.coordinate,
                expected_kv_range=entry.expected_kv_range,
                execution=SplitKVExecutionPlan(
                    requested_mode=execution.requested_mode,
                    requested_split_size=execution.requested_split_size,
                    actual_mode=execution.actual_mode,
                    actual_split_size=execution.actual_split_size,
                    boundaries=execution.boundaries,
                    merge_order=execution.merge_order,
                    acc_dtype=execution.acc_dtype,
                    downcast_at=execution.downcast_at,
                    backend=execution.backend,
                    source="runtime_fallback",
                    fallback=True,
                    fallback_reason="backend substituted a runtime plan",
                ),
            )
        )
    fallback = SplitKVRuntimePlanSet(
        batch_size=plan_set.batch_size,
        tp_world_size=plan_set.tp_world_size,
        cp_world_size=plan_set.cp_world_size,
        total_kv_tokens=plan_set.total_kv_tokens,
        entries=tuple(fallback_entries),
    )

    result = _bind(rollout_split_kv_plan_set=fallback)

    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.SPLIT_KV_FALLBACK)


@pytest.mark.parametrize(
    "rollout_overrides",
    [
        {"rollout.tensor_parallel_size": 1},
        {"rollout.context_parallel_size": 1},
    ],
)
def test_tp_or_cp_topology_mismatch_is_not_comparable(rollout_overrides):
    rollout = VllmRolloutMaterializer().build_contract({**ROLLOUT_KNOBS, **rollout_overrides})
    result = _bind(rollout_contract=rollout)

    assert not result.comparable
    assert result.issues_by_code(BindingErrorCode.TOPOLOGY_MISMATCH)


# --------------------------------------------------------------------------
# role and input validation
# --------------------------------------------------------------------------


def test_swapped_roles_are_rejected_outright():
    rollout, training = _contracts()

    with pytest.raises(AttentionBindingError):
        bind_attention_contracts(
            rollout_contract=training,
            training_contract=rollout,
            rollout_identity=_identity(),
            training_identity=_identity(),
            rollout_backend_id="a",
            training_backend_id="b",
        )


# --------------------------------------------------------------------------
# determinism cross-check
# --------------------------------------------------------------------------


def _megatron_env():
    return {"NCCL_ALGO": "Tree", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0"}


def _vllm_env(**overrides):
    env = {
        "VLLM_BATCH_INVARIANT": "1",
        "NCCL_ALGO": "allreduce:tree",
        "NCCL_PROTO": "Simple",
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    env.update(overrides)
    return env


def test_nccl_algo_mismatch_blocks_the_binding():
    """Megatron asserts NCCL_ALGO; vLLM hard-sets a different value."""

    training = megatron_probe_from_config(
        SimpleNamespace(deterministic_mode=True), env=_megatron_env()
    )
    rollout = vllm_probe_from_env(_vllm_env())

    report = compare_determinism(rollout=rollout, training=training)

    assert not report.compatible
    fields = {issue.field for issue in report.issues}
    assert "env.NCCL_ALGO" in fields
    assert "env.NCCL_PROTO" in fields


def test_matching_nccl_settings_are_compatible():
    shared = {"NCCL_ALGO": "allreduce:tree", "NCCL_PROTO": "Simple"}
    training = megatron_probe_from_config(
        SimpleNamespace(deterministic_mode=True), env=dict(shared)
    )
    rollout = vllm_probe_from_env({**_vllm_env(**shared), "CUBLAS_WORKSPACE_CONFIG": None})

    report = compare_determinism(rollout=rollout, training=training)

    assert report.compatible, [issue.to_dict() for issue in report.issues]


def test_determinism_switch_off_on_either_side_blocks():
    training = megatron_probe_from_config(
        SimpleNamespace(deterministic_mode=False), env=_megatron_env()
    )
    rollout = vllm_probe_from_env(_vllm_env(VLLM_BATCH_INVARIANT="0"))

    report = compare_determinism(rollout=rollout, training=training)

    fields = {issue.field for issue in report.issues}
    assert "training.deterministic_mode" in fields
    assert "rollout.VLLM_BATCH_INVARIANT" in fields


def test_tf32_asymmetry_is_recorded_not_blocking():
    """Megatron does not manage TF32 at all; vLLM disables it. Record the gap."""

    training = megatron_probe_from_config(
        SimpleNamespace(deterministic_mode=True), env=_megatron_env()
    )
    rollout = vllm_probe_from_env(_vllm_env())

    report = compare_determinism(rollout=rollout, training=training)

    assert training.tf32_disabled is None
    assert rollout.tf32_disabled is True
    assert "tf32_disabled" in report.differences
    assert not any(issue.field == "tf32_disabled" for issue in report.issues)


def test_determinism_issues_flow_into_the_binding():
    training = megatron_probe_from_config(
        SimpleNamespace(deterministic_mode=True), env=_megatron_env()
    )
    rollout = vllm_probe_from_env(_vllm_env())
    report = compare_determinism(rollout=rollout, training=training)

    result = _bind(determinism_issues=report.issues)

    assert result.comparable  # identity is fine
    assert not result.passed  # but the reduction environment is not
    assert result.issues_by_code(BindingErrorCode.DETERMINISM_INCOMPATIBLE)
    assert "FAILED CLOSED" in summarize_binding(result)


# --------------------------------------------------------------------------
# sharding derived from the frozen #239 layout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp_shards_cover_the_global_sequence_without_overlap(cp_rank):
    contract = MegatronAttentionMaterializer(
        cp_rank=cp_rank, global_sequence_length=4096
    ).build_contract(TRAINING_KNOBS)

    sharding = contract.sharding
    assert sharding.local_sequence_length == 2048
    assert sharding.global_block_indices == (cp_rank,)
    assert sharding.global_block_token_starts == (cp_rank * 2048,)
    # The causal offset must be the number of preceding *global* tokens, otherwise
    # rank 1 would mask as if its shard started at position zero.
    assert contract.causal_offsets == (cp_rank * 2048, cp_rank * 2048)


def test_tp_head_shards_split_qwen3_gqa_evenly():
    contract = MegatronAttentionMaterializer(tp_rank=1).build_contract(TRAINING_KNOBS)

    sharding = contract.sharding
    assert (sharding.global_q_heads, sharding.global_kv_heads) == (32, 8)
    assert (sharding.local_q_heads, sharding.local_kv_heads) == (16, 4)
    assert (sharding.local_q_head_start, sharding.local_kv_head_start) == (16, 4)


@pytest.mark.parametrize("tp_world_size", [2, 4, 8])
def test_supported_tp_degrees_shard_qwen3_gqa(tp_world_size):
    """Qwen3-8B has 32 Q heads and 8 KV heads, so TP in {2, 4, 8} all divide."""

    contract = MegatronAttentionMaterializer().build_contract(
        {**TRAINING_KNOBS, "training.tensor_parallel_size": tp_world_size}
    )

    sharding = contract.sharding
    assert sharding.local_q_heads == 32 // tp_world_size
    assert sharding.local_kv_heads == 8 // tp_world_size


@pytest.mark.parametrize(
    ("knob_value", "expected"),
    [("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32"), ("fp16", "fp16")],
)
def test_planner_normalized_dtypes_reach_the_contract(knob_value, expected):
    """The planner emits torch spellings; AttentionDType uses short ones."""

    contract = MegatronAttentionMaterializer().build_contract(
        {**TRAINING_KNOBS, "training.compute_dtype": knob_value}
    )

    assert contract.dtype.value == expected


def test_unknown_dtype_is_rejected_with_the_offending_field():
    with pytest.raises(ValueError, match="training.compute_dtype"):
        MegatronAttentionMaterializer().build_contract(
            {**TRAINING_KNOBS, "training.compute_dtype": "int8"}
        )


def test_indivisible_tp_is_rejected():
    with pytest.raises(ValueError, match="divide evenly"):
        MegatronAttentionMaterializer().build_contract(
            {**TRAINING_KNOBS, "training.tensor_parallel_size": 3}
        )


def test_indivisible_cp_sequence_is_rejected():
    with pytest.raises(ValueError, match="divide evenly"):
        MegatronAttentionMaterializer(global_sequence_length=4097).build_contract(TRAINING_KNOBS)


# --------------------------------------------------------------------------
# materialization: fail closed rather than silently substitute
# --------------------------------------------------------------------------


def _statuses(materialization, path):
    return [app.status for app in materialization.applications if app.path == path]


def _readback(materializer, flat, *, source):
    contract = materializer.build_contract(flat)
    projection_plans = {
        name: ProjectionPlan(
            projection=name,
            backend_id="rlkernel.cuda.det_gemm",
            fallback=True,
            fallback_reason="native projection probe failed",
            probe_id=f"{source}-{name}",
            collective=collective.to_dict(),
        ).to_dict()
        for name, collective in (
            ("qkv", QKV_COLLECTIVE_CONTRACT),
            ("o_proj", O_PROJ_COLLECTIVE_CONTRACT),
        )
    }
    return AttentionRuntimeReadback(
        contract=contract,
        actual_knobs=dict(flat),
        split_kv_plan_set=_plan_set(contract, backend=source),
        source=source,
        frozen_scope_verified=True,
        preprocess_backends=MANDATED_ATTENTION_PREPROCESS_BACKENDS,
        preprocess_fallback=False,
        projection_plans=projection_plans,
        actual_backend=f"reference.{source}",
        communication_backend="none",
        production_ready=False,
    )


def _strict_readback(materializer, flat, *, source):
    readback = _readback(materializer, flat, source=source)
    contract = replace(readback.contract, split_kv=SplitKVSpec.disabled())
    actual_knobs = dict(readback.actual_knobs)
    actual_knobs["attention.split_kv_policy"] = "disabled"
    return replace(
        readback,
        contract=contract,
        actual_knobs=actual_knobs,
        split_kv_plan_set=_plan_set(contract, backend=source),
        strict_mode=True,
        strict_core_id=STRICT_ATTENTION_CORE_ID,
        strict_schedule=STRICT_ATTENTION_SCHEDULE_ID,
        native_attention_arithmetic=False,
        strict_split_kv_policy="disabled",
        actual_backend="rlkernel.cuda.deterministic_attention",
        communication_backend="self_owned_cuda_ag_rs",
        production_ready=True,
    )


def test_configured_contract_without_runtime_readback_is_unobservable():
    normalized = {
        "batch": {"size": 2},
        "training": {
            "tensor_parallel_size": 2,
            "context_parallel_size": 2,
            "compute_dtype": "bf16",
        },
        "attention": {"split_kv_policy": 32},
    }
    materialization = MegatronAttentionMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS)

    assert materialization.applications
    assert {app.status for app in materialization.applications} == {
        MaterializationStatus.UNOBSERVABLE
    }
    assert all(app.actual is None for app in materialization.applications)


@pytest.mark.parametrize(
    ("materializer_type", "flat", "source"),
    [
        (MegatronAttentionMaterializer, TRAINING_KNOBS, "megatron.runtime_readback"),
        (VllmRolloutMaterializer, ROLLOUT_KNOBS, "vllm.runtime_readback"),
    ],
)
def test_runtime_readback_can_verify_materialized_knobs(materializer_type, flat, source):
    configured = materializer_type()
    readback = _readback(configured, flat, source=source)
    materializer = materializer_type(runtime_readback=readback)
    normalized = {}
    for path, value in flat.items():
        section, key = path.split(".", 1)
        normalized.setdefault(section, {})[key] = value

    materialization = materializer.materialize(normalized, WS2_ATTENTION_KNOBS)

    assert materialization.applications
    assert {app.status for app in materialization.applications} == {MaterializationStatus.APPLIED}
    side = "training" if materializer_type is MegatronAttentionMaterializer else "rollout"
    assert materialization.binding.side_configs[side]["runtime_readback"]["source"] == source


def test_runtime_readback_mismatch_is_a_fallback():
    configured = MegatronAttentionMaterializer()
    readback = _readback(configured, TRAINING_KNOBS, source="megatron.runtime_readback")
    actual = dict(readback.actual_knobs)
    actual["training.context_parallel_size"] = 1
    mismatched = AttentionRuntimeReadback(
        contract=readback.contract,
        actual_knobs=actual,
        split_kv_plan_set=readback.split_kv_plan_set,
        source=readback.source,
        frozen_scope_verified=True,
        preprocess_backends=readback.preprocess_backends,
        preprocess_fallback=readback.preprocess_fallback,
    )
    normalized = {
        "batch": {"size": 2},
        "training": {
            "tensor_parallel_size": 2,
            "context_parallel_size": 2,
            "compute_dtype": "bf16",
        },
        "attention": {"split_kv_policy": 32},
    }
    materialization = MegatronAttentionMaterializer(runtime_readback=mismatched).materialize(
        normalized, WS2_ATTENTION_KNOBS
    )

    assert _statuses(materialization, "training.context_parallel_size") == [
        MaterializationStatus.FALLBACK
    ]


def test_decode_split_kv_plan_set_must_match_kv_cache_lengths():
    rollout, _ = _contracts()
    kv_cache = KVCacheSpec(
        cache_positions=(3, 5),
        kv_seq_lens=(4, 6),
        block_table=((0, 1, -1), (2, 3, 4)),
        global_token_positions=tuple(range(4)) + tuple(range(6)),
        page_size=2,
    )
    decode = replace(
        rollout,
        role=AttentionRole.INFER,
        mode=AttentionMode.DECODE,
        query_sequence_length=1,
        causal_offsets=(3, 5),
        kv_cache=kv_cache,
    )
    wrong_lengths = build_split_kv_runtime_plan_set(
        (4, 8),
        tp_world_size=2,
        cp_world_size=2,
        split_kv=decode.split_kv,
        backend="vllm.decode.readback",
    )

    with pytest.raises(ValueError, match="KV-cache lengths"):
        AttentionRuntimeReadback(
            contract=decode,
            actual_knobs={},
            split_kv_plan_set=wrong_lengths,
            source="vllm.decode.readback",
            frozen_scope_verified=True,
        )


def test_strict_runtime_readback_entrypoint_binds_executed_evidence():
    rollout_materializer = VllmRolloutMaterializer()
    training_materializer = MegatronAttentionMaterializer()
    rollout = _readback(rollout_materializer, ROLLOUT_KNOBS, source="vllm.runtime_readback")
    training = _readback(
        training_materializer,
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="rlkernel.cp_attention_reference",
    )

    assert result.passed
    assert result.provenance["split_kv_runtime"]["rollout"]["coverage"] == (
        "complete_batch_tp_cp_owner_cartesian_product"
    )
    assert result.provenance["rollout"]["recorded"]["preprocess.qk_rmsnorm"] == (
        MANDATED_ATTENTION_PREPROCESS_BACKENDS["qk_rmsnorm"]
    )


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        (
            {"strict_core_id": "different.core"},
            BindingErrorCode.ATTENTION_CORE_MISMATCH,
        ),
        (
            {"strict_schedule": "different_schedule"},
            BindingErrorCode.ATTENTION_CORE_SCHEDULE,
        ),
        (
            {"strict_split_kv_policy": "fixed"},
            BindingErrorCode.ATTENTION_CORE_SPLIT_K,
        ),
    ],
)
def test_strict_runtime_readback_rejects_non_shared_attention_arithmetic(changes, expected_code):
    rollout = replace(
        _strict_readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"),
        **changes,
    )
    training = _strict_readback(
        MegatronAttentionMaterializer(),
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="rlkernel.cuda.deterministic_attention",
        training_backend_id="rlkernel.cuda.deterministic_attention",
    )

    assert not result.passed
    assert result.issues_by_code(expected_code)


@pytest.mark.parametrize(
    ("actual_backend", "core_id", "schedule", "communication_backend"),
    [
        (
            "flash_attention_4.cute",
            "rlkernel.attention.flash_attention4.num_splits1.v1",
            "single_batch_flash_attention4_num_splits1",
            "cuda_ag_rs",
        ),
        (
            "aiter.rocm.ck_dense_mha",
            "rlkernel.attention.aiter_ck.num_splits1.v1",
            "single_batch_aiter_ck_num_splits1",
            "rccl_ag_rs",
        ),
    ],
)
def test_strict_runtime_readback_accepts_same_qualified_vendor_core(
    actual_backend, core_id, schedule, communication_backend
):
    rollout = replace(
        _strict_readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"),
        strict_core_id=core_id,
        strict_schedule=schedule,
        native_attention_arithmetic=True,
        actual_backend=actual_backend,
        communication_backend=communication_backend,
    )
    training = replace(
        _strict_readback(
            MegatronAttentionMaterializer(), TRAINING_KNOBS, source="megatron.runtime_readback"
        ),
        strict_core_id=core_id,
        strict_schedule=schedule,
        native_attention_arithmetic=True,
        actual_backend=actual_backend,
        communication_backend=communication_backend,
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id=actual_backend,
        training_backend_id=actual_backend,
    )

    assert result.passed


def test_strict_runtime_readback_rejects_vendor_arithmetic_mismatch():
    rollout = _strict_readback(
        VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"
    )
    training = replace(
        _strict_readback(
            MegatronAttentionMaterializer(), TRAINING_KNOBS, source="megatron.runtime_readback"
        ),
        native_attention_arithmetic=True,
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id=rollout.actual_backend,
        training_backend_id=training.actual_backend,
    )

    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.ATTENTION_NATIVE_ARITHMETIC)


def test_strict_runtime_readback_accepts_shared_no_split_k_core():
    rollout = _strict_readback(
        VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"
    )
    training = _strict_readback(
        MegatronAttentionMaterializer(),
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="rlkernel.cuda.deterministic_attention",
        training_backend_id="rlkernel.cuda.deterministic_attention",
    )

    assert result.passed
    assert result.provenance["rollout"]["recorded"]["strict.core_id"] == (STRICT_ATTENTION_CORE_ID)
    assert result.provenance["rollout"]["recorded"]["runtime.actual_backend"] == (
        "rlkernel.cuda.deterministic_attention"
    )
    assert result.provenance["rollout"]["recorded"]["runtime.communication_backend"] == (
        "self_owned_cuda_ag_rs"
    )
    assert result.provenance["rollout"]["recorded"]["runtime.production_ready"] is True


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        (
            "reference_only",
            True,
            BindingErrorCode.ATTENTION_BACKEND_MISSING,
        ),
        ("attention_fallback", True, BindingErrorCode.ATTENTION_BACKEND_MISSING),
        (
            "communication_backend",
            "p2p_nccl_reference",
            BindingErrorCode.ATTENTION_BACKEND_MISSING,
        ),
        ("production_ready", False, BindingErrorCode.ATTENTION_NOT_PRODUCTION_READY),
    ],
)
def test_strict_runtime_readback_rejects_reference_only_evidence(field, value, error_code):
    rollout = replace(
        _strict_readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"),
        **{field: value},
    )
    training = _strict_readback(
        MegatronAttentionMaterializer(),
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )
    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="rlkernel.cuda.deterministic_attention",
        training_backend_id="rlkernel.cuda.deterministic_attention",
    )
    assert not result.passed
    assert result.issues_by_code(error_code)


def test_strict_runtime_readback_accepts_common_deterministic_preprocess_fallback():
    rollout = replace(
        _readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"),
        preprocess_fallback=True,
        preprocess_fallback_reason="native probe failed",
        preprocess_probe_id="rollout-probe",
    )
    training = replace(
        _readback(
            MegatronAttentionMaterializer(), TRAINING_KNOBS, source="megatron.runtime_readback"
        ),
        preprocess_fallback=True,
        preprocess_fallback_reason="native probe failed",
        preprocess_probe_id="training-probe",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="rlkernel.cp_attention_reference",
    )

    assert result.passed


def test_strict_runtime_readback_rejects_distinct_native_preprocess_backends():
    rollout = replace(
        _readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback"),
        preprocess_backends={
            "qk_rmsnorm": "transformer_engine.rocm.rmsnorm",
            "rope": "rlkernel.rocm.deterministic_rope",
        },
        preprocess_probe_id="rollout-probe",
    )
    training = replace(
        _readback(
            MegatronAttentionMaterializer(), TRAINING_KNOBS, source="megatron.runtime_readback"
        ),
        preprocess_backends={
            "qk_rmsnorm": "transformer_engine.cuda.rmsnorm",
            "rope": "rlkernel.cuda.rope_sm90",
        },
        preprocess_probe_id="training-probe",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="megatron.te",
    )

    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH)


@pytest.mark.parametrize(
    ("backends", "fallback", "expected_code"),
    [
        (
            {"rope": MANDATED_ATTENTION_PREPROCESS_BACKENDS["rope"]},
            False,
            BindingErrorCode.ATTENTION_PREPROCESS_MISSING,
        ),
        (
            {**MANDATED_ATTENTION_PREPROCESS_BACKENDS, "qk_rmsnorm": "unknown.backend"},
            False,
            BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
        ),
        (
            dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS),
            True,
            BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
        ),
    ],
)
def test_strict_runtime_readback_rejects_unverified_preprocess_backend(
    backends, fallback, expected_code
):
    rollout = _readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback")
    rollout = replace(
        rollout,
        preprocess_backends=backends,
        preprocess_fallback=fallback,
        preprocess_fallback_reason=("test fallback" if fallback else None),
    )
    training = _readback(
        MegatronAttentionMaterializer(),
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="rlkernel.cp_attention_reference",
    )

    assert result.comparable
    assert not result.passed
    assert result.issues_by_code(expected_code)


def test_strict_runtime_readback_rejects_projection_split_k_or_missing_plan():
    rollout = _readback(VllmRolloutMaterializer(), ROLLOUT_KNOBS, source="vllm.runtime_readback")
    plans = {name: dict(plan) for name, plan in rollout.projection_plans.items()}
    plans["qkv"]["split_k"] = True
    plans.pop("o_proj")
    rollout = replace(rollout, projection_plans=plans)
    training = _readback(
        MegatronAttentionMaterializer(), TRAINING_KNOBS, source="megatron.runtime_readback"
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="megatron.te",
    )

    assert not result.passed
    assert result.issues_by_code(BindingErrorCode.ATTENTION_PROJECTION_MISMATCH)
    assert result.issues_by_code(BindingErrorCode.ATTENTION_PROJECTION_MISSING)


def test_strict_runtime_readback_entrypoint_rejects_unverified_frozen_scope():
    rollout_materializer = VllmRolloutMaterializer()
    training_materializer = MegatronAttentionMaterializer()
    rollout = _readback(rollout_materializer, ROLLOUT_KNOBS, source="vllm.runtime_readback")
    rollout = replace(rollout, frozen_scope_verified=False)
    training = _readback(
        training_materializer,
        TRAINING_KNOBS,
        source="megatron.runtime_readback",
    )

    result = bind_attention_runtime_readbacks(
        rollout=rollout,
        training=training,
        rollout_identity=_identity(),
        training_identity=_identity(),
        rollout_backend_id="vllm.flash_attn",
        training_backend_id="rlkernel.cp_attention_reference",
    )

    assert result.comparable
    assert not result.passed
    assert any(issue.field == "rollout.frozen_scope_verified" for issue in result.issues)


def test_arrival_merge_order_is_unsupported_not_silently_corrected():
    """The control group must stay distinguishable from the treatment."""

    normalized = {
        "batch": {"size": 2},
        "training": {"tensor_parallel_size": 2, "context_parallel_size": 2},
        "attention": {"reduction_order": "arrival"},
    }
    materialization = MegatronAttentionMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS)

    assert _statuses(materialization, "attention.reduction_order") == [
        MaterializationStatus.UNSUPPORTED
    ]
    assert _statuses(materialization, "training.tensor_parallel_size") == [
        MaterializationStatus.ERROR
    ]
    assert materialization.binding.side_configs["training"]["contract"] is None
    assert "arrival" in materialization.binding.side_configs["training"]["contract_error"]


def test_unsupported_reduction_invalidates_vllm_contract_applications():
    normalized = {
        "batch": {"size": 2},
        "rollout": {"tensor_parallel_size": 2, "context_parallel_size": 2},
        "attention": {"reduction_order": "arrival"},
    }
    materialization = VllmRolloutMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS)

    assert _statuses(materialization, "attention.reduction_order") == [
        MaterializationStatus.UNSUPPORTED
    ]
    assert _statuses(materialization, "rollout.tensor_parallel_size") == [
        MaterializationStatus.ERROR
    ]
    assert materialization.binding.side_configs["rollout"]["contract"] is None


def test_bf16_reduction_accumulation_is_unsupported():
    normalized = {
        "batch": {"size": 2},
        "training": {"tensor_parallel_size": 2, "context_parallel_size": 2},
        "attention": {"reduction_acc_dtype": "bf16"},
    }
    materialization = MegatronAttentionMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS)

    assert _statuses(materialization, "attention.reduction_acc_dtype") == [
        MaterializationStatus.UNSUPPORTED
    ]


def test_te_oracle_engine_is_unsupported_until_pr2_pr3():
    normalized = {
        "batch": {"size": 2},
        "training": {"tensor_parallel_size": 2, "context_parallel_size": 2},
        "attention": {"reduction_engine": "te_oracle"},
    }
    materialization = MegatronAttentionMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS)

    assert _statuses(materialization, "attention.reduction_engine") == [
        MaterializationStatus.UNSUPPORTED
    ]


def test_vllm_cp_falls_back_to_one_in_decode_and_says_why():
    materializer = VllmRolloutMaterializer(mode=AttentionMode.DECODE)
    normalized = {
        "batch": {"size": 2},
        "rollout": {"tensor_parallel_size": 2, "context_parallel_size": 2},
    }

    assert materializer.effective_cp_world_size({"rollout.context_parallel_size": 2}) == 1

    materialization = materializer.materialize(normalized, WS2_ATTENTION_KNOBS)
    contract_error = materialization.binding.side_configs["rollout"]["contract_error"]
    assert "#235 PR6" in contract_error
    assert MaterializationStatus.APPLIED not in {app.status for app in materialization.applications}


def test_decode_contract_is_refused_without_kv_cache_identity():
    with pytest.raises(AttentionContractError, match="PR6"):
        VllmRolloutMaterializer(mode=AttentionMode.DECODE).build_contract(ROLLOUT_KNOBS)


def test_materializers_expose_distinct_implementation_fingerprints():
    megatron = MegatronAttentionMaterializer().implementation_fingerprint
    vllm = VllmRolloutMaterializer().implementation_fingerprint

    assert megatron and vllm and megatron != vllm


def test_runtime_binding_reports_the_frozen_topology():
    normalized = {
        "batch": {"size": 2},
        "training": {"tensor_parallel_size": 2, "context_parallel_size": 2},
    }
    binding = MegatronAttentionMaterializer().materialize(normalized, WS2_ATTENTION_KNOBS).binding

    topology = binding.topology["training"]
    assert topology["tensor_parallel_size"] == 2
    assert topology["context_parallel_size"] == 2
    assert topology["world_size"] == 4
    assert topology["pipeline_parallel_size"] == 1
    assert topology["data_parallel_size"] == 1


# --------------------------------------------------------------------------
# provenance adapters
# --------------------------------------------------------------------------


def test_megatron_provenance_flags_undeclared_frozen_scope_fields():
    adapter = MegatronProvenanceAdapter(SimpleNamespace(deterministic_mode=True))

    violations = adapter.frozen_scope_violations()

    # Nothing is declared, so every assertion reads as unknown rather than as met.
    assert any("expert_model_parallel_size" in text for text in violations)
    assert any("fp8" in text for text in violations)


def test_megatron_provenance_accepts_a_conforming_dense_config():
    adapter = MegatronProvenanceAdapter(
        SimpleNamespace(
            deterministic_mode=True,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            fp8=None,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
    )

    assert adapter.frozen_scope_violations() == ("fp8 is not declared (expected None)",)


def test_megatron_construction_fingerprint_tracks_fusion_changes():
    base = SimpleNamespace(deterministic_mode=True, apply_rope_fusion=False)
    fused = SimpleNamespace(deterministic_mode=True, apply_rope_fusion=True)

    assert (
        MegatronProvenanceAdapter(base).construction_fingerprint
        != MegatronProvenanceAdapter(fused).construction_fingerprint
    )


def test_vllm_provenance_reads_page_size_and_split_kv_policy():
    adapter = VllmProvenanceAdapter(
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=32),
    )

    assert adapter.kv_page_size == 16
    assert adapter.split_kv_policy == 32
    assert adapter.to_dict()["flash_attn_max_num_splits_for_cuda_graph"] == 32


def test_vllm_provenance_flags_fp8_kv_cache_and_cascade_attention():
    adapter = VllmProvenanceAdapter(
        model_config=SimpleNamespace(quantization=None, disable_cascade_attn=False),
        cache_config=SimpleNamespace(
            cache_dtype="fp8", calculate_kv_scales=False, sliding_window=None
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
    )

    violations = adapter.frozen_scope_violations()

    assert any("cache_dtype" in text for text in violations)
    assert any("disable_cascade_attn" in text for text in violations)


# --------------------------------------------------------------------------
# scenario config
# --------------------------------------------------------------------------


SCENARIO = (
    Path(__file__).resolve().parents[3]
    / "configs/experiments/cross_config/cross_config_qwen3_8b_megatron_tp2_cp2_vllm.json"
)


def test_scenario_uses_megatron_vocabulary_only():
    config = json.loads(SCENARIO.read_text(encoding="utf-8"))
    training = config["baseline"]["training"]

    assert training["attention_backend"] in {"flash", "fused", "unfused", "local", "auto"}
    assert training["tensor_parallel_size"] == 2
    assert training["context_parallel_size"] == 2
    assert config["baseline"]["rollout"]["batch_invariant"] is True
    assert config["scenario"]["debug_matrix"]["modules"] == ["attention", "ffn", "logp"]
    assert config["scenario"]["debug_matrix"]["cartesian_product"] is False


def test_scenario_knob_paths_all_exist():
    config = json.loads(SCENARIO.read_text(encoding="utf-8"))

    def paths(mapping, prefix=""):
        for key, value in mapping.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                yield from paths(value, f"{path}.")
            else:
                yield path

    declared = set(paths(config["baseline"]))
    unknown = declared - set(WS2_ATTENTION_KNOBS)
    assert not unknown, f"scenario declares unknown knobs: {sorted(unknown)}"

    for intervention in config["interventions"]:
        assert intervention["path"] in WS2_ATTENTION_KNOBS
