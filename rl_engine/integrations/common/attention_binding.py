# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Three-tier binding between rollout-side and training-side attention contracts.

Issue #235 PR4 requires that "rollout and training descriptors bind to the same
semantic attention contract". Under the frozen Megatron + vLLM deployment the two
sides can never produce *identical* :class:`AttentionContract` instances: training
runs full-sequence prefill over a CP-sharded sequence, while rollout runs vLLM
paged-KV chunked prefill and decode. Taking "same contract" literally would make
the target configuration permanently unbindable.

This module therefore splits binding into three tiers:

``IDENTICAL``
    Logical identity. Both sides must agree bit for bit, otherwise the pair is not
    comparable at all and no drift number from it means anything.

``SEMANTIC``
    The WS2 numerical claim: merge semantics, accumulation dtype, reduction order
    and downcast point are decided by the contract, not by the implementation.
    Both sides must carry the same values *and* those values must match the WS2
    mandate, otherwise the comparison fails closed.

``RECORDED``
    Materialization facts that the two sides are expected to differ on -- attention
    mode, RoPE fusion boundary, KV-cache paging, backend id, reduction engine. These
    differences are exactly what the experiment measures, so they are recorded into
    provenance rather than rejected.

Deliberately *not* in ``SEMANTIC``: ``engine``. Training may run the in-op
deterministic reference while rollout runs a Transformer Engine merge oracle; forcing
those equal would defeat the purpose of the oracle comparison in #235 PR2/3/5/6.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

from rl_engine.contracts.operators.attention import (
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMerge,
    AttentionRole,
    DowncastPoint,
    ReductionOrder,
    SplitKVRuntimePlanSet,
    validate_split_kv_plan_set_alignment,
)
from rl_engine.ops.attention.preprocess import (
    ALLOWED_ATTENTION_PREPROCESS_BACKENDS,
    MANDATED_ATTENTION_PREPROCESS_BACKENDS,
    PREPROCESS_POLICY_ID,
)
from rl_engine.ops.attention.projection import (
    CUDA_DETERMINISTIC_PROJECTION_BACKEND_ID,
    O_PROJ_COLLECTIVE_CONTRACT,
    PROJECTION_POLICY_ID,
    QKV_COLLECTIVE_CONTRACT,
    ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID,
)

__all__ = [
    "ATTENTION_LSE_DOMAIN",
    "AttentionBindingError",
    "AttentionBindingResult",
    "AttentionRuntimeReadback",
    "BindingErrorCode",
    "BindingIssue",
    "BindingTier",
    "IDENTITY_FIELDS",
    "NULLABLE_IDENTITY_FIELDS",
    "RECORDED_FIELDS",
    "SEMANTIC_CONTRACT_FIELDS",
    "SEMANTIC_REDUCTION_FIELDS",
    "TOPOLOGY_FIELDS",
    "WS2_ATTENTION_REDUCTION_MANDATE",
    "bind_attention_contracts",
    "bind_attention_runtime_readbacks",
    "first_blocking_issue",
    "identity_fingerprint",
    "summarize_binding",
]


class AttentionBindingError(ValueError):
    """Raised when a caller supplies structurally unusable binding inputs."""


class BindingTier(str, Enum):
    """Which rule a field is governed by."""

    IDENTICAL = "identical"
    SEMANTIC = "semantic"
    RECORDED = "recorded"


class BindingErrorCode(str, Enum):
    """Stable, machine-readable reasons a binding is rejected.

    Callers branch on these; they are part of the artifact schema and must not be
    renamed without a schema version bump.
    """

    IDENTITY_MISSING = "IDENTITY_MISSING"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    REDUCTION_SEMANTIC_MISMATCH = "REDUCTION_SEMANTIC_MISMATCH"
    REDUCTION_MANDATE_VIOLATION = "REDUCTION_MANDATE_VIOLATION"
    LSE_NOT_EXPORTED = "LSE_NOT_EXPORTED"
    ROLE_COLLISION = "ROLE_COLLISION"
    DETERMINISM_INCOMPATIBLE = "DETERMINISM_INCOMPATIBLE"
    TOPOLOGY_MISMATCH = "TOPOLOGY_MISMATCH"
    SPLIT_KV_RUNTIME_MISSING = "SPLIT_KV_RUNTIME_MISSING"
    SPLIT_KV_MISMATCH = "SPLIT_KV_MISMATCH"
    SPLIT_KV_FALLBACK = "SPLIT_KV_FALLBACK"
    ATTENTION_PREPROCESS_MISSING = "ATTENTION_PREPROCESS_MISSING"
    ATTENTION_PREPROCESS_MISMATCH = "ATTENTION_PREPROCESS_MISMATCH"
    ATTENTION_PREPROCESS_FALLBACK = "ATTENTION_PREPROCESS_FALLBACK"
    ATTENTION_PROJECTION_MISSING = "ATTENTION_PROJECTION_MISSING"
    ATTENTION_PROJECTION_MISMATCH = "ATTENTION_PROJECTION_MISMATCH"
    ATTENTION_CORE_MISSING = "ATTENTION_CORE_MISSING"
    ATTENTION_CORE_MISMATCH = "ATTENTION_CORE_MISMATCH"
    ATTENTION_CORE_SCHEDULE = "ATTENTION_CORE_SCHEDULE"
    ATTENTION_NATIVE_ARITHMETIC = "ATTENTION_NATIVE_ARITHMETIC"
    ATTENTION_CORE_SPLIT_K = "ATTENTION_CORE_SPLIT_K"
    ATTENTION_BACKEND_MISSING = "ATTENTION_BACKEND_MISSING"
    ATTENTION_NOT_PRODUCTION_READY = "ATTENTION_NOT_PRODUCTION_READY"


@dataclass(frozen=True)
class AttentionRuntimeReadback:
    """Actual attention contract and all-rank Split-KV evidence from one engine."""

    contract: AttentionContract
    actual_knobs: Mapping[str, Any]
    split_kv_plan_set: SplitKVRuntimePlanSet
    source: str
    frozen_scope_verified: bool
    preprocess_backends: Mapping[str, str] = field(default_factory=dict)
    preprocess_fallback: bool = False
    preprocess_fallback_reason: str | None = None
    preprocess_probe_id: str = ""
    preprocess_policy_id: str = PREPROCESS_POLICY_ID
    projection_plans: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    strict_mode: bool = False
    strict_core_id: str | None = None
    strict_schedule: str | None = None
    native_attention_arithmetic: bool = True
    strict_split_kv_policy: str | None = None
    actual_backend: str | None = None
    communication_backend: str | None = None
    production_ready: bool = False
    attention_fallback: bool = False
    reference_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.contract, AttentionContract):
            raise TypeError("runtime readback contract must be an AttentionContract")
        if not isinstance(self.actual_knobs, Mapping):
            raise TypeError("runtime readback actual_knobs must be a mapping")
        if not isinstance(self.split_kv_plan_set, SplitKVRuntimePlanSet):
            raise TypeError("runtime readback requires a complete SplitKVRuntimePlanSet")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("runtime readback source must be a non-empty string")
        if not isinstance(self.frozen_scope_verified, bool):
            raise TypeError("frozen_scope_verified must be a bool")
        if not isinstance(self.preprocess_backends, Mapping):
            raise TypeError("runtime readback preprocess_backends must be a mapping")
        for name, backend in self.preprocess_backends.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("preprocess backend names must be non-empty strings")
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError("preprocess backend IDs must be non-empty strings")
        if not isinstance(self.preprocess_fallback, bool):
            raise TypeError("preprocess_fallback must be a bool")
        if self.preprocess_fallback and not self.preprocess_fallback_reason:
            raise ValueError(
                "preprocess_fallback_reason is required when preprocess_fallback is true"
            )
        if not isinstance(self.preprocess_probe_id, str):
            raise TypeError("preprocess_probe_id must be a string")
        if not isinstance(self.preprocess_policy_id, str) or not self.preprocess_policy_id.strip():
            raise ValueError("preprocess_policy_id must be a non-empty string")
        if not isinstance(self.projection_plans, Mapping):
            raise TypeError("projection_plans must be a mapping")
        if not isinstance(self.strict_mode, bool):
            raise TypeError("strict_mode must be a bool")
        if self.strict_core_id is not None and (
            not isinstance(self.strict_core_id, str) or not self.strict_core_id.strip()
        ):
            raise ValueError("strict_core_id must be a non-empty string when provided")
        if self.strict_schedule is not None and (
            not isinstance(self.strict_schedule, str) or not self.strict_schedule.strip()
        ):
            raise ValueError("strict_schedule must be a non-empty string when provided")
        if not isinstance(self.native_attention_arithmetic, bool):
            raise TypeError("native_attention_arithmetic must be a bool")
        if self.strict_split_kv_policy is not None and self.strict_split_kv_policy not in {
            "disabled",
            "fixed",
            "auto",
        }:
            raise ValueError("strict_split_kv_policy must be disabled, fixed, or auto")
        for name, value in (
            ("actual_backend", self.actual_backend),
            ("communication_backend", self.communication_backend),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if not isinstance(self.production_ready, bool):
            raise TypeError("production_ready must be a bool")
        if not isinstance(self.attention_fallback, bool):
            raise TypeError("attention_fallback must be a bool")
        if not isinstance(self.reference_only, bool):
            raise TypeError("reference_only must be a bool")
        normalized_projection_plans: dict[str, Mapping[str, Any]] = {}
        for name, plan in self.projection_plans.items():
            if not isinstance(name, str) or not isinstance(plan, Mapping):
                raise TypeError("projection_plans must map projection names to mappings")
            normalized_projection_plans[name] = MappingProxyType(dict(plan))

        plan_error = _split_kv_plan_contract_error(self.contract, self.split_kv_plan_set)
        if plan_error is not None:
            raise ValueError(plan_error)
        object.__setattr__(self, "actual_knobs", MappingProxyType(dict(self.actual_knobs)))
        object.__setattr__(
            self,
            "preprocess_backends",
            MappingProxyType(dict(self.preprocess_backends)),
        )
        object.__setattr__(
            self,
            "projection_plans",
            MappingProxyType(normalized_projection_plans),
        )

    @property
    def split_kv_fallback(self) -> bool:
        return bool(_split_kv_fallbacks(self.split_kv_plan_set))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frozen_scope_verified": self.frozen_scope_verified,
            "contract": self.contract.to_dict(),
            "actual_knobs": dict(self.actual_knobs),
            "attention_preprocess": {
                "backends": dict(self.preprocess_backends),
                "fallback": self.preprocess_fallback,
                "fallback_reason": self.preprocess_fallback_reason,
                "probe_id": self.preprocess_probe_id,
                "policy_id": self.preprocess_policy_id,
            },
            "attention_projections": {
                name: dict(plan) for name, plan in self.projection_plans.items()
            },
            "strict_attention": {
                "enabled": self.strict_mode,
                "core_id": self.strict_core_id,
                "schedule": self.strict_schedule,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "split_kv_policy": self.strict_split_kv_policy,
            },
            "runtime_backend": {
                "actual_backend": self.actual_backend,
                "communication_backend": self.communication_backend,
                "production_ready": self.production_ready,
                "fallback": self.attention_fallback,
                "reference_only": self.reference_only,
            },
            "split_kv_runtime_plan_set": self.split_kv_plan_set.to_dict(),
        }


#: Attention exports attention-domain LSE, never vocab-logprob LSE (#235).
#: Recorded explicitly so a future ``LogprobContract`` binding cannot be confused
#: with this one purely because both set ``export_lse=True``.
ATTENTION_LSE_DOMAIN = "attention"


#: Fields both sides must agree on bit for bit before any comparison is meaningful.
#: Sourced from #235 "Numerical Contract" preconditions plus the vime-owned rollout
#: provenance (weight version, sampling, padding) that the issue assumes but does
#: not enumerate.
IDENTITY_FIELDS: tuple[str, ...] = (
    "checkpoint_id",
    "model_version",
    "weight_version",
    "tokenizer_fingerprint",
    "token_ids_fingerprint",
    "active_mask_fingerprint",
    "position_ids_fingerprint",
    "padding_side",
    "pre_update_state",
    # model semantics that decide what attention *means*
    "q_heads",
    "kv_heads",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "rotary_dim",
    "qk_layernorm",
    # batch composition: batch-invariance is a claim about results not changing with
    # batch makeup, so two sides scoring different batches are not comparable at all
    "batch_size",
    # decode replay identity (#235 PR6)
    "global_token_positions_fingerprint",
    "kv_seq_lens_fingerprint",
)


#: Reduction fields that decide the numerical result. Both sides must carry the
#: same value, and that value must satisfy :data:`WS2_ATTENTION_REDUCTION_MANDATE`.
SEMANTIC_REDUCTION_FIELDS: tuple[str, ...] = (
    "merge",
    "acc_dtype",
    "order",
    "downcast_at",
)


#: Contract fields outside ``ReductionSpec`` that still decide the numerical result.
#: ``dtype`` is here rather than in :data:`RECORDED_FIELDS` because comparing a BF16
#: rollout against an FP16 training pass produces a real drift number attributable to
#: nothing. #235 PR5 does sweep BF16 against an FP32 reference; that sweep opts in via
#: ``allow_dtype_difference`` instead of loosening the default.
SEMANTIC_CONTRACT_FIELDS: tuple[str, ...] = ("dtype",)


#: Sharding fields that determine local GQA head and sequence ownership. These are
#: comparison preconditions, not harmless backend provenance: a TP/CP mismatch
#: means the two ranks did not evaluate the same local attention problem.
TOPOLOGY_FIELDS: tuple[str, ...] = (
    "tp_rank",
    "tp_world_size",
    "cp_rank",
    "cp_world_size",
    "global_q_heads",
    "global_kv_heads",
    "local_q_head_start",
    "local_q_heads",
    "local_kv_head_start",
    "local_kv_heads",
    "global_sequence_length",
    "local_sequence_length",
    "global_block_indices",
    "global_block_token_starts",
    "local_block_offsets",
    "packed_sequence_offsets",
)


#: The WS2 mandate itself. ``#236`` currently declares single-member enums for
#: ``merge`` / ``order`` / ``downcast_at``, so those checks are tautological today;
#: they are written out anyway so that widening any of those enums later fails here
#: instead of silently admitting a non-conforming backend.
WS2_ATTENTION_REDUCTION_MANDATE: Mapping[str, str] = {
    "merge": AttentionMerge.ONLINE_SOFTMAX_LSE.value,
    "acc_dtype": AttentionDType.FP32.value,
    "order": ReductionOrder.GLOBAL_BLOCK_INDEX.value,
    "downcast_at": DowncastPoint.FINAL_WRITE.value,
}


#: Materialization facts the two sides are expected to differ on. Recorded into
#: provenance; never a rejection reason.
RECORDED_FIELDS: tuple[str, ...] = (
    "mode",
    "backend_id",
    "reduction.engine",
    "rope.fusion_boundary",
    "rope.q_state",
    "rope.k_state",
    "rope.k_cache_state",
    "rope.cast_at",
    "rope.output_dtype",
    "preprocess.qk_rmsnorm",
    "preprocess.rope",
    "preprocess.fallback",
    "kv_cache.page_size",
    "kv_cache.prefix_cache_enabled",
    "kv_cache.block_table_shape",
)


@dataclass(frozen=True)
class BindingIssue:
    """One reason a binding is not comparable or not admissible."""

    code: BindingErrorCode
    tier: BindingTier
    field: str
    rollout: Any = None
    training: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "tier": self.tier.value,
            "field": self.field,
            "rollout": self.rollout,
            "training": self.training,
            "message": self.message,
        }


@dataclass(frozen=True)
class AttentionBindingResult:
    """Outcome of binding one rollout contract to one training contract.

    ``comparable`` and ``passed`` are deliberately separate. A pair whose identity
    does not match is *not comparable* -- reporting a drift number for it would be
    meaningless. A pair that is comparable but violates the reduction mandate *is*
    comparable yet must still fail closed, because the whole WS2 claim is that
    reduction order and accumulation precision come from the contract.
    """

    comparable: bool
    passed: bool
    issues: tuple[BindingIssue, ...] = ()
    identity_fingerprint: str = ""
    reduction_fingerprint: str = ""
    binding_fingerprint: str = ""
    recorded_differences: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.attention_binding.v3"

    def issues_by_code(self, code: BindingErrorCode) -> tuple[BindingIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code is code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparable": self.comparable,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "identity_fingerprint": self.identity_fingerprint,
            "reduction_fingerprint": self.reduction_fingerprint,
            "binding_fingerprint": self.binding_fingerprint,
            "recorded_differences": {
                key: dict(value) for key, value in self.recorded_differences.items()
            },
            "provenance": dict(self.provenance),
        }


def _canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def identity_fingerprint(identity: Mapping[str, Any]) -> str:
    """Fingerprint only the declared :data:`IDENTITY_FIELDS`, in a fixed order.

    Extra keys in ``identity`` are ignored on purpose: callers pass whole
    provenance bundles, and the fingerprint must not drift when an unrelated
    diagnostic field is added.
    """

    return _canonical_fingerprint({name: identity.get(name) for name in IDENTITY_FIELDS})


def _reduction_view(contract: AttentionContract) -> dict[str, Any]:
    reduction = contract.reduction
    return {
        "merge": reduction.merge.value,
        "acc_dtype": reduction.acc_dtype.value,
        "order": reduction.order.value,
        "downcast_at": reduction.downcast_at.value,
        "engine": reduction.engine.value,
    }


def _recorded_view(
    contract: AttentionContract,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    rope = contract.rope
    kv_cache = contract.kv_cache
    view: dict[str, Any] = {
        "mode": contract.mode.value,
        "backend_id": None,
        "reduction.engine": contract.reduction.engine.value,
    }
    if rope is not None:
        view.update(
            {
                "rope.fusion_boundary": rope.fusion_boundary.value,
                "rope.q_state": rope.q_state.value,
                "rope.k_state": rope.k_state.value,
                "rope.k_cache_state": rope.k_cache_state.value,
                "rope.cast_at": rope.cast_at.value,
                "rope.output_dtype": rope.output_dtype.value,
            }
        )
    if kv_cache is not None:
        view.update(
            {
                "kv_cache.page_size": kv_cache.page_size,
                "kv_cache.prefix_cache_enabled": kv_cache.prefix_cache_enabled,
                "kv_cache.block_table_shape": [
                    len(kv_cache.block_table),
                    max((len(row) for row in kv_cache.block_table), default=0),
                ],
            }
        )
    if extra:
        view.update(extra)
    return view


def _topology_view(contract: AttentionContract) -> dict[str, Any]:
    sharding = contract.sharding
    return {name: getattr(sharding, name) for name in TOPOLOGY_FIELDS}


def _split_kv_fallbacks(plan_set: SplitKVRuntimePlanSet) -> list[dict[str, Any]]:
    return [
        entry.to_dict()
        for entry in plan_set.entries
        if entry.execution.fallback
        or entry.execution.actual_mode is None
        or entry.execution.actual_mode is not entry.execution.requested_mode
        or entry.execution.actual_split_size != entry.execution.requested_split_size
    ]


def _split_kv_plan_contract_error(
    contract: AttentionContract,
    plan_set: SplitKVRuntimePlanSet,
) -> str | None:
    sharding = contract.sharding
    expected_topology = (
        contract.batch_size,
        sharding.tp_world_size,
        sharding.cp_world_size,
    )
    actual_topology = (
        plan_set.batch_size,
        plan_set.tp_world_size,
        plan_set.cp_world_size,
    )
    if actual_topology != expected_topology:
        return (
            "Split-KV plan-set batch/TP/CP topology does not match the attention "
            f"contract: actual={actual_topology}, expected={expected_topology}"
        )
    if contract.mode.value in {"prefill", "chunked_prefill"}:
        expected_totals = (sharding.global_sequence_length,) * contract.batch_size
        if plan_set.total_kv_tokens != expected_totals:
            return (
                "Split-KV plan-set KV lengths do not match the prefill attention "
                f"contract: actual={plan_set.total_kv_tokens}, expected={expected_totals}"
            )
    elif contract.kv_cache is not None:
        expected_totals = contract.kv_cache.kv_seq_lens
        if plan_set.total_kv_tokens != expected_totals:
            return (
                "Split-KV plan-set KV lengths do not match decode KV-cache lengths: "
                f"actual={plan_set.total_kv_tokens}, expected={expected_totals}"
            )
    for entry in plan_set.entries:
        execution = entry.execution
        if (
            execution.requested_mode is not contract.split_kv.mode
            or execution.requested_split_size != contract.split_kv.fixed_split_size
        ):
            return (
                "Split-KV runtime request does not match the first-class attention "
                f"contract at {entry.coordinate}"
            )
    return None


#: Identity fields where ``None`` is a real value rather than an omission. Qwen3-8B
#: applies no RoPE scaling, so ``rope_scaling=None`` must not read as "undeclared" --
#: both sides still have to agree on it, which the equality pass below handles.
NULLABLE_IDENTITY_FIELDS: frozenset[str] = frozenset({"rope_scaling"})


def _missing_identity_fields(identity: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name in IDENTITY_FIELDS
        if name not in NULLABLE_IDENTITY_FIELDS and identity.get(name) is None
    )


def bind_attention_contracts(
    *,
    rollout_contract: AttentionContract,
    training_contract: AttentionContract,
    rollout_identity: Mapping[str, Any],
    training_identity: Mapping[str, Any],
    rollout_backend_id: str,
    training_backend_id: str,
    determinism_issues: Sequence[BindingIssue] = (),
    require_full_identity: bool = True,
    allow_dtype_difference: bool = False,
    rollout_split_kv_plan_set: Optional[SplitKVRuntimePlanSet] = None,
    training_split_kv_plan_set: Optional[SplitKVRuntimePlanSet] = None,
    rollout_recorded_extra: Optional[Mapping[str, Any]] = None,
    training_recorded_extra: Optional[Mapping[str, Any]] = None,
) -> AttentionBindingResult:
    """Bind a rollout attention contract to a training attention contract.

    ``determinism_issues`` is threaded in from
    :mod:`rl_engine.validation.cross_config.determinism` rather than computed here,
    so that this module stays free of framework probing and remains testable
    without Megatron or vLLM present.

    ``require_full_identity`` exists for the single-GPU harness in #235 PR2, which
    legitimately has no KV-cache or decode identity to declare. Distributed callers
    must leave it at ``True``.

    ``allow_dtype_difference`` exists for the #235 PR5 sweep that deliberately scores
    a BF16 path against an FP32 reference. It must stay ``False`` everywhere else.

    Strict binding requires complete actual Split-KV plan sets from both runtimes.
    A configured policy is insufficient because auto-selection, graph capture, and
    backend fallbacks can change the executed boundaries. The plan sets cover the
    complete batch x TP x CP x KV-owner Cartesian product.

    ``rollout_recorded_extra`` / ``training_recorded_extra`` are diagnostic-only
    backend facts. They can never make a semantic mismatch admissible.
    """

    if rollout_contract.role is not AttentionRole.INFER:
        raise AttentionBindingError(
            f"rollout_contract.role must be {AttentionRole.INFER.value!r}, "
            f"got {rollout_contract.role.value!r}"
        )
    if training_contract.role is not AttentionRole.TRAIN:
        raise AttentionBindingError(
            f"training_contract.role must be {AttentionRole.TRAIN.value!r}, "
            f"got {training_contract.role.value!r}"
        )

    issues: list[BindingIssue] = []

    # ---- tier 1: identity, bit for bit -------------------------------------
    if require_full_identity:
        for side, identity in (("rollout", rollout_identity), ("training", training_identity)):
            for name in _missing_identity_fields(identity):
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.IDENTITY_MISSING,
                        tier=BindingTier.IDENTICAL,
                        field=f"{side}.{name}",
                        message=f"{side} identity does not declare {name!r}",
                    )
                )

    for name in IDENTITY_FIELDS:
        rollout_value = rollout_identity.get(name)
        training_value = training_identity.get(name)
        if rollout_value != training_value:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.IDENTITY_MISMATCH,
                    tier=BindingTier.IDENTICAL,
                    field=name,
                    rollout=rollout_value,
                    training=training_value,
                    message=(
                        f"{name!r} differs between sides; the pair is not comparable "
                        "and any drift computed from it is meaningless"
                    ),
                )
            )

    comparable = not any(issue.tier is BindingTier.IDENTICAL for issue in issues)

    rollout_topology = _topology_view(rollout_contract)
    training_topology = _topology_view(training_contract)
    for name in TOPOLOGY_FIELDS:
        if rollout_topology[name] != training_topology[name]:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.TOPOLOGY_MISMATCH,
                    tier=BindingTier.IDENTICAL,
                    field=f"sharding.{name}",
                    rollout=rollout_topology[name],
                    training=training_topology[name],
                    message=(
                        f"sharding.{name} changes TP/CP ownership; the pair is not "
                        "the same local attention problem"
                    ),
                )
            )

    comparable = not any(issue.tier is BindingTier.IDENTICAL for issue in issues)

    # ---- tier 2: reduction semantics, and the WS2 mandate -------------------
    rollout_reduction = _reduction_view(rollout_contract)
    training_reduction = _reduction_view(training_contract)

    for name in SEMANTIC_REDUCTION_FIELDS:
        if rollout_reduction[name] != training_reduction[name]:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.REDUCTION_SEMANTIC_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"reduction.{name}",
                    rollout=rollout_reduction[name],
                    training=training_reduction[name],
                    message=(
                        f"reduction.{name!r} must be decided by the contract, not by the "
                        "backend; the two sides disagree"
                    ),
                )
            )
        mandated = WS2_ATTENTION_REDUCTION_MANDATE[name]
        for side, view in (("rollout", rollout_reduction), ("training", training_reduction)):
            if view[name] != mandated:
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.REDUCTION_MANDATE_VIOLATION,
                        tier=BindingTier.SEMANTIC,
                        field=f"{side}.reduction.{name}",
                        rollout=rollout_reduction[name],
                        training=training_reduction[name],
                        message=(
                            f"WS2 requires reduction.{name} == {mandated!r}; "
                            f"{side} declares {view[name]!r}"
                        ),
                    )
                )

    if not allow_dtype_difference and rollout_contract.dtype is not training_contract.dtype:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.REDUCTION_SEMANTIC_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="dtype",
                rollout=rollout_contract.dtype.value,
                training=training_contract.dtype.value,
                message=(
                    "the two sides compute in different dtypes; the resulting drift is "
                    "not attributable. Pass allow_dtype_difference=True only for a "
                    "deliberate precision sweep"
                ),
            )
        )

    if rollout_contract.split_kv != training_contract.split_kv:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.SPLIT_KV_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="split_kv",
                rollout=rollout_contract.split_kv.to_dict(),
                training=training_contract.split_kv.to_dict(),
                message="training and rollout must request the same first-class Split-KV policy",
            )
        )

    for side, plan_set in (
        ("rollout", rollout_split_kv_plan_set),
        ("training", training_split_kv_plan_set),
    ):
        if plan_set is None:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.SPLIT_KV_RUNTIME_MISSING,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.split_kv_runtime_plan_set",
                    message=(
                        f"{side} did not report a complete actual Split-KV plan set; "
                        "configured policy alone is not runtime evidence"
                    ),
                )
            )
            continue
        contract = rollout_contract if side == "rollout" else training_contract
        contract_error = _split_kv_plan_contract_error(contract, plan_set)
        if contract_error is not None:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.SPLIT_KV_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.split_kv_runtime_plan_set",
                    rollout=plan_set.to_dict() if side == "rollout" else None,
                    training=plan_set.to_dict() if side == "training" else None,
                    message=contract_error,
                )
            )
        fallbacks = _split_kv_fallbacks(plan_set)
        if fallbacks:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.SPLIT_KV_FALLBACK,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.split_kv_runtime_plan_set",
                    rollout=fallbacks if side == "rollout" else None,
                    training=fallbacks if side == "training" else None,
                    message=f"{side} Split-KV runtime used an unknown or fallback plan",
                )
            )

    if rollout_split_kv_plan_set is not None and training_split_kv_plan_set is not None:
        try:
            validate_split_kv_plan_set_alignment(
                training_split_kv_plan_set,
                rollout_split_kv_plan_set,
            )
        except AttentionContractError as exc:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.SPLIT_KV_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field="split_kv_runtime_plan_set",
                    rollout=rollout_split_kv_plan_set.to_dict(),
                    training=training_split_kv_plan_set.to_dict(),
                    message=str(exc),
                )
            )

    for side, contract in (("rollout", rollout_contract), ("training", training_contract)):
        if not contract.export_lse:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.LSE_NOT_EXPORTED,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.export_lse",
                    message=(
                        "attention-domain LSE must be exported; without it the deterministic "
                        "CP merge cannot be validated"
                    ),
                )
            )

    if rollout_backend_id == training_backend_id and rollout_backend_id:
        # Not an error, but worth surfacing: an identical backend on both sides means
        # the experiment is not actually measuring a cross-implementation difference.
        pass

    issues.extend(determinism_issues)

    # ---- tier 3: recorded differences --------------------------------------
    rollout_recorded = _recorded_view(rollout_contract, rollout_recorded_extra)
    rollout_recorded["backend_id"] = rollout_backend_id
    training_recorded = _recorded_view(training_contract, training_recorded_extra)
    training_recorded["backend_id"] = training_backend_id

    recorded_differences: dict[str, dict[str, Any]] = {}
    for name in RECORDED_FIELDS:
        rollout_value = rollout_recorded.get(name)
        training_value = training_recorded.get(name)
        if rollout_value != training_value:
            recorded_differences[name] = {
                "rollout": rollout_value,
                "training": training_value,
            }

    identity_fp = identity_fingerprint(training_identity if comparable else rollout_identity)
    reduction_fp = _canonical_fingerprint(
        {name: training_reduction[name] for name in SEMANTIC_REDUCTION_FIELDS}
    )
    passed = comparable and not any(issue.tier is BindingTier.SEMANTIC for issue in issues)

    provenance = {
        "lse_domain": ATTENTION_LSE_DOMAIN,
        "dtype": training_contract.dtype.value,
        "split_kv_runtime": {
            "rollout": (
                None if rollout_split_kv_plan_set is None else rollout_split_kv_plan_set.to_dict()
            ),
            "training": (
                None if training_split_kv_plan_set is None else training_split_kv_plan_set.to_dict()
            ),
        },
        "rollout": {
            "contract": rollout_contract.to_dict(),
            "backend_id": rollout_backend_id,
            "recorded": rollout_recorded,
        },
        "training": {
            "contract": training_contract.to_dict(),
            "backend_id": training_backend_id,
            "recorded": training_recorded,
        },
    }

    return AttentionBindingResult(
        comparable=comparable,
        passed=passed,
        issues=tuple(issues),
        identity_fingerprint=identity_fp,
        reduction_fingerprint=reduction_fp,
        binding_fingerprint=_canonical_fingerprint(
            {
                "identity": identity_fp,
                "reduction": reduction_fp,
                "topology": training_topology,
                "split_kv": provenance["split_kv_runtime"],
                "lse_domain": ATTENTION_LSE_DOMAIN,
                "rollout_backend": rollout_backend_id,
                "training_backend": training_backend_id,
                "attention_preprocess": {
                    "rollout": {
                        name: rollout_recorded.get(f"preprocess.{name}")
                        for name in (
                            *MANDATED_ATTENTION_PREPROCESS_BACKENDS,
                            "fallback",
                        )
                    },
                    "training": {
                        name: training_recorded.get(f"preprocess.{name}")
                        for name in (
                            *MANDATED_ATTENTION_PREPROCESS_BACKENDS,
                            "fallback",
                        )
                    },
                },
            }
        ),
        recorded_differences=recorded_differences,
        provenance=provenance,
    )


def bind_attention_runtime_readbacks(
    *,
    rollout: AttentionRuntimeReadback,
    training: AttentionRuntimeReadback,
    rollout_identity: Mapping[str, Any],
    training_identity: Mapping[str, Any],
    rollout_backend_id: str,
    training_backend_id: str,
    determinism_issues: Sequence[BindingIssue] = (),
) -> AttentionBindingResult:
    """Strict public handoff from executed framework runtimes to PR4 binding.

    The Megatron/vLLM launchers remain environment-owned. Once both launchers have
    reconstructed their actual contracts and all-rank Split-KV reports, this entry
    point performs the complete comparison without accepting configured-only data.
    """

    missing_scope_evidence = []
    for side, readback in (("rollout", rollout), ("training", training)):
        if not readback.frozen_scope_verified:
            missing_scope_evidence.append(
                BindingIssue(
                    code=BindingErrorCode.DETERMINISM_INCOMPATIBLE,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.frozen_scope_verified",
                    message=f"{side} runtime did not verify the frozen attention scope",
                )
            )
        missing_scope_evidence.extend(_attention_preprocess_issues(side, readback))
        missing_scope_evidence.extend(_attention_projection_issues(side, readback))
        missing_scope_evidence.extend(_strict_attention_core_issues(side, readback))
    missing_scope_evidence.extend(_strict_attention_core_pair_issues(rollout, training))
    if rollout.preprocess_fallback != training.preprocess_fallback:
        missing_scope_evidence.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="preprocess.fallback",
                rollout=rollout.preprocess_fallback,
                training=training.preprocess_fallback,
                message=(
                    "both runtimes must either pass the platform vendor bitwise probe or "
                    "use the same deterministic preprocess fallback"
                ),
            )
        )
    if rollout.preprocess_policy_id != training.preprocess_policy_id:
        missing_scope_evidence.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="preprocess.policy_id",
                rollout=rollout.preprocess_policy_id,
                training=training.preprocess_policy_id,
                message="QK-Norm/RoPE policy IDs differ between runtimes",
            )
        )
    for name in MANDATED_ATTENTION_PREPROCESS_BACKENDS:
        rollout_backend = rollout.preprocess_backends.get(name)
        training_backend = training.preprocess_backends.get(name)
        if rollout_backend != training_backend:
            missing_scope_evidence.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"preprocess.{name}",
                    rollout=rollout_backend,
                    training=training_backend,
                    message="training and rollout must execute the same preprocess backend",
                )
            )
    for projection in ("qkv", "o_proj"):
        rollout_plan = rollout.projection_plans.get(projection, {})
        training_plan = training.projection_plans.get(projection, {})
        rollout_fallback = bool(rollout_plan.get("fallback", False))
        training_fallback = bool(training_plan.get("fallback", False))
        if rollout_fallback != training_fallback:
            missing_scope_evidence.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PROJECTION_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"projection.{projection}.fallback",
                    rollout=rollout_fallback,
                    training=training_fallback,
                    message=(
                        "QKV/o_proj must use the same native-or-deterministic " "path on both sides"
                    ),
                )
            )
        for field_name in ("backend_id", "policy_id", "split_k", "reduction_order"):
            if rollout_plan.get(field_name) != training_plan.get(field_name):
                missing_scope_evidence.append(
                    BindingIssue(
                        code=BindingErrorCode.ATTENTION_PROJECTION_MISMATCH,
                        tier=BindingTier.SEMANTIC,
                        field=f"projection.{projection}.{field_name}",
                        rollout=rollout_plan.get(field_name),
                        training=training_plan.get(field_name),
                        message="projection execution evidence differs between sides",
                    )
                )
    return bind_attention_contracts(
        rollout_contract=rollout.contract,
        training_contract=training.contract,
        rollout_identity=rollout_identity,
        training_identity=training_identity,
        rollout_backend_id=rollout_backend_id,
        training_backend_id=training_backend_id,
        determinism_issues=tuple(determinism_issues) + tuple(missing_scope_evidence),
        rollout_split_kv_plan_set=rollout.split_kv_plan_set,
        training_split_kv_plan_set=training.split_kv_plan_set,
        rollout_recorded_extra={
            **{
                f"preprocess.{name}": backend
                for name, backend in rollout.preprocess_backends.items()
            },
            "preprocess.fallback": rollout.preprocess_fallback,
            "preprocess.fallback_reason": rollout.preprocess_fallback_reason,
            "preprocess.probe_id": rollout.preprocess_probe_id,
            "preprocess.policy_id": rollout.preprocess_policy_id,
            "strict.enabled": rollout.strict_mode,
            "strict.core_id": rollout.strict_core_id,
            "strict.schedule": rollout.strict_schedule,
            "strict.native_attention_arithmetic": rollout.native_attention_arithmetic,
            "strict.split_kv_policy": rollout.strict_split_kv_policy,
            "runtime.actual_backend": rollout.actual_backend,
            "runtime.communication_backend": rollout.communication_backend,
            "runtime.production_ready": rollout.production_ready,
            "runtime.fallback": rollout.attention_fallback,
            "runtime.reference_only": rollout.reference_only,
            **{
                f"projection.{projection}": dict(plan)
                for projection, plan in rollout.projection_plans.items()
            },
        },
        training_recorded_extra={
            **{
                f"preprocess.{name}": backend
                for name, backend in training.preprocess_backends.items()
            },
            "preprocess.fallback": training.preprocess_fallback,
            "preprocess.fallback_reason": training.preprocess_fallback_reason,
            "preprocess.probe_id": training.preprocess_probe_id,
            "preprocess.policy_id": training.preprocess_policy_id,
            "strict.enabled": training.strict_mode,
            "strict.core_id": training.strict_core_id,
            "strict.schedule": training.strict_schedule,
            "strict.native_attention_arithmetic": training.native_attention_arithmetic,
            "strict.split_kv_policy": training.strict_split_kv_policy,
            "runtime.actual_backend": training.actual_backend,
            "runtime.communication_backend": training.communication_backend,
            "runtime.production_ready": training.production_ready,
            "runtime.fallback": training.attention_fallback,
            "runtime.reference_only": training.reference_only,
            **{
                f"projection.{projection}": dict(plan)
                for projection, plan in training.projection_plans.items()
            },
        },
    )


def _strict_attention_core_issues(
    side: str,
    readback: AttentionRuntimeReadback,
) -> list[BindingIssue]:
    if not readback.strict_mode:
        return []
    issues = []
    if not readback.actual_backend:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_BACKEND_MISSING,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.runtime.actual_backend",
                rollout=readback.actual_backend if side == "rollout" else None,
                training=readback.actual_backend if side == "training" else None,
                message=(f"{side} strict Attention did not report its executed production core"),
            )
        )
    supported_communication = {"self_owned_cuda_ag_rs", "cuda_ag_rs", "rccl_ag_rs"}
    if (
        readback.contract.sharding.cp_world_size > 1
        and readback.communication_backend not in supported_communication
    ):
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_BACKEND_MISSING,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.runtime.communication_backend",
                rollout=readback.communication_backend if side == "rollout" else None,
                training=readback.communication_backend if side == "training" else None,
                message=(
                    f"{side} strict CP Attention did not execute a supported self-owned AG/RS path"
                ),
            )
        )
    if not readback.production_ready:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_NOT_PRODUCTION_READY,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.runtime.production_ready",
                rollout=False if side == "rollout" else None,
                training=False if side == "training" else None,
                message=(f"{side} evidence is reference-only and cannot close the production gate"),
            )
        )
    if not readback.strict_core_id:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_MISSING,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.strict.core_id",
                message=(f"{side} strict Attention did not report its exact shared core identity"),
            )
        )
    if not readback.strict_schedule:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_SCHEDULE,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.strict.schedule",
                message=(f"{side} strict Attention did not report its exact reduction schedule"),
            )
        )
    if readback.attention_fallback or readback.reference_only:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_BACKEND_MISSING,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.runtime.production_path",
                message=f"{side} strict Attention executed a fallback or reference-only path",
            )
        )
    if (
        readback.strict_split_kv_policy != "disabled"
        or readback.contract.split_kv.mode.value != "disabled"
    ):
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_SPLIT_K,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.strict.split_kv_policy",
                message=(
                    f"{side} strict Attention did not prove Split-KV disabled "
                    "in both runtime evidence and AttentionContract"
                ),
            )
        )
    return issues


def _strict_attention_core_pair_issues(
    rollout: AttentionRuntimeReadback,
    training: AttentionRuntimeReadback,
) -> list[BindingIssue]:
    if rollout.strict_mode != training.strict_mode:
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="strict.enabled",
                rollout=rollout.strict_mode,
                training=training.strict_mode,
                message="training and rollout must use the same strict Attention mode",
            )
        ]
    if rollout.strict_mode and rollout.strict_core_id != training.strict_core_id:
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="strict.core_id",
                rollout=rollout.strict_core_id,
                training=training.strict_core_id,
                message="training and rollout executed different Attention cores",
            )
        ]
    if rollout.strict_mode and rollout.strict_schedule != training.strict_schedule:
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_SCHEDULE,
                tier=BindingTier.SEMANTIC,
                field="strict.schedule",
                rollout=rollout.strict_schedule,
                training=training.strict_schedule,
                message="training and rollout executed different strict Attention schedules",
            )
        ]
    if (
        rollout.strict_mode
        and rollout.native_attention_arithmetic != training.native_attention_arithmetic
    ):
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_NATIVE_ARITHMETIC,
                tier=BindingTier.SEMANTIC,
                field="strict.native_attention_arithmetic",
                rollout=rollout.native_attention_arithmetic,
                training=training.native_attention_arithmetic,
                message="training and rollout disagree on vendor Attention arithmetic",
            )
        ]
    if rollout.strict_mode and rollout.actual_backend != training.actual_backend:
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="runtime.actual_backend",
                rollout=rollout.actual_backend,
                training=training.actual_backend,
                message="training and rollout executed different Attention backends",
            )
        ]
    if rollout.strict_mode and rollout.communication_backend != training.communication_backend:
        return [
            BindingIssue(
                code=BindingErrorCode.ATTENTION_CORE_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="runtime.communication_backend",
                rollout=rollout.communication_backend,
                training=training.communication_backend,
                message="training and rollout executed different Attention communication backends",
            )
        ]
    return []


def _attention_preprocess_issues(
    side: str,
    readback: AttentionRuntimeReadback,
) -> list[BindingIssue]:
    issues: list[BindingIssue] = []
    for name, mandated in MANDATED_ATTENTION_PREPROCESS_BACKENDS.items():
        actual = readback.preprocess_backends.get(name)
        if actual is None:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PREPROCESS_MISSING,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.preprocess.{name}",
                    message=(
                        f"{side} did not report the executed {name} backend; "
                        "runtime-native execution cannot validate the Attention input boundary"
                    ),
                )
            )
        elif actual not in ALLOWED_ATTENTION_PREPROCESS_BACKENDS[name]:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PREPROCESS_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.preprocess.{name}",
                    rollout=actual if side == "rollout" else None,
                    training=actual if side == "training" else None,
                    message=(
                        f"{side} executed {actual!r}; "
                        f"the experiment requires {mandated!r} or a verified platform backend"
                    ),
                )
            )
    if readback.preprocess_fallback and any(
        str(readback.preprocess_backends.get(name, "")).startswith("transformer_engine.")
        for name in MANDATED_ATTENTION_PREPROCESS_BACKENDS
    ):
        issues.append(
            BindingIssue(
                code=BindingErrorCode.ATTENTION_PREPROCESS_FALLBACK,
                tier=BindingTier.SEMANTIC,
                field=f"{side}.preprocess.fallback",
                message=(
                    f"{side} reported fallback while still claiming a vendor preprocess backend"
                ),
            )
        )
    return issues


def _attention_projection_issues(
    side: str,
    readback: AttentionRuntimeReadback,
) -> list[BindingIssue]:
    issues: list[BindingIssue] = []
    expected_collectives = {
        "qkv": QKV_COLLECTIVE_CONTRACT.to_dict(),
        "o_proj": O_PROJ_COLLECTIVE_CONTRACT.to_dict(),
    }
    fixed_fields = {
        "input_dtype": "torch.bfloat16",
        "weight_dtype": "torch.bfloat16",
        "output_dtype": "torch.bfloat16",
        "accumulation_dtype": "torch.float32",
        "reduction_order": "k_ascending",
        "split_k": False,
        "policy_id": PROJECTION_POLICY_ID,
    }
    for projection, expected_collective in expected_collectives.items():
        plan = readback.projection_plans.get(projection)
        if plan is None:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PROJECTION_MISSING,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.projection.{projection}",
                    message=f"{side} did not report the executed {projection} projection plan",
                )
            )
            continue
        for field_name, expected in fixed_fields.items():
            actual = plan.get(field_name)
            if actual != expected:
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.ATTENTION_PROJECTION_MISMATCH,
                        tier=BindingTier.SEMANTIC,
                        field=f"{side}.projection.{projection}.{field_name}",
                        rollout=actual if side == "rollout" else None,
                        training=actual if side == "training" else None,
                        message=f"{side} {projection} {field_name} must be {expected!r}",
                    )
                )
        collective = plan.get("collective")
        if not isinstance(collective, Mapping) or dict(collective) != expected_collective:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PROJECTION_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.projection.{projection}.collective",
                    message=f"{side} {projection} TP/SP collective directions are invalid",
                )
            )
        backend_id = plan.get("backend_id")
        if not isinstance(backend_id, str) or not backend_id.strip():
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PROJECTION_MISSING,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.projection.{projection}.backend_id",
                    message=f"{side} {projection} backend identity is missing",
                )
            )
        if not isinstance(plan.get("probe_id"), str) or not plan.get("probe_id"):
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.ATTENTION_PROJECTION_MISSING,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.projection.{projection}.probe_id",
                    message=f"{side} {projection} bitwise probe identity is missing",
                )
            )
        if plan.get("fallback"):
            deterministic_backends = {
                CUDA_DETERMINISTIC_PROJECTION_BACKEND_ID,
                ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID,
            }
            if backend_id not in deterministic_backends or not plan.get("fallback_reason"):
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.ATTENTION_PROJECTION_MISMATCH,
                        tier=BindingTier.SEMANTIC,
                        field=f"{side}.projection.{projection}.fallback",
                        message=(
                            f"{side} {projection} fallback must execute DetGemmOp and record why"
                        ),
                    )
                )
    return issues


def summarize_binding(result: AttentionBindingResult) -> str:
    """One-line human summary for CLI output and failure messages."""

    if result.passed:
        return (
            f"attention binding OK "
            f"(identity={result.identity_fingerprint[:12]}, "
            f"{len(result.recorded_differences)} recorded difference(s))"
        )
    if not result.comparable:
        fields = ", ".join(
            sorted({issue.field for issue in result.issues if issue.tier is BindingTier.IDENTICAL})
        )
        return f"attention binding NOT COMPARABLE; identity problems: {fields}"
    fields = ", ".join(
        sorted({issue.field for issue in result.issues if issue.tier is BindingTier.SEMANTIC})
    )
    return f"attention binding FAILED CLOSED; semantic problems: {fields}"


def first_blocking_issue(
    result: AttentionBindingResult,
) -> Optional[BindingIssue]:
    """Return the issue a caller should report, preferring identity over semantics."""

    for tier in (BindingTier.IDENTICAL, BindingTier.SEMANTIC):
        for issue in result.issues:
            if issue.tier is tier:
                return issue
    return None
