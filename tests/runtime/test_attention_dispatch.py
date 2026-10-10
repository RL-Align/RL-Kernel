# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contract-aware attention dispatch: registration, policy, and fail-closed.

These cases use a fresh ``KernelRegistry`` so they never mutate the process
singleton, and they assert the property that motivates a separate dispatch
entry point: a WS2 attention caller must never be served by a backend that
declares different reduction, Split-KV, or LSE-export semantics.
"""

from __future__ import annotations

import pytest

from rl_engine.contracts.operators.attention import (
    AttentionBackendCapability,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
    validate_cross_config_alignment,
)
from rl_engine.runtime.registry import KernelRegistry, OpBackend, _rocm_strict_attention_available


def _contract(*, cp_world_size: int = 1, seq_len: int = 128) -> AttentionContract:
    sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=1,
        cp_rank=0,
        cp_world_size=cp_world_size,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=0,
        local_q_heads=32,
        local_kv_head_start=0,
        local_kv_heads=8,
        global_sequence_length=seq_len,
        local_sequence_length=seq_len,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, seq_len),
    )
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=seq_len,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )


def _capability(**overrides) -> AttentionBackendCapability:
    fields = {
        "backend_id": "test.strict.core",
        "roles": frozenset({AttentionRole.TRAIN, AttentionRole.INFER}),
        "modes": frozenset({AttentionMode.PREFILL}),
        "dtypes": frozenset({AttentionDType.BF16}),
        "cp_world_sizes": (1,),
        "exports_attention_lse": True,
        "reports_actual_split_kv_plan": True,
        "implementation_kind": "production",
    }
    fields.update(overrides)
    return AttentionBackendCapability(**fields)


class _FakeCore:
    pass


@pytest.fixture()
def registry(monkeypatch):
    fresh = KernelRegistry()
    # Serve a stand-in instance so dispatch never imports a vendor stack.
    monkeypatch.setattr(fresh, "_get_or_create_backend", lambda backend: _FakeCore())
    return fresh


def _platform(registry) -> str:
    return registry._platform()


def test_strict_rocm_core_is_registered_only_when_the_vendor_stack_loads():
    """The strict core is conditional; every other candidate is static.

    ``ws2_attention`` is a static priority list, so a backend that exists only
    on some machines cannot be declared there. It registers itself at runtime,
    and only when ``aiter.ops.mha`` really loaded - otherwise dispatch would
    offer a backend that fails at materialization.
    """

    fresh = KernelRegistry()
    expected = _rocm_strict_attention_available()

    rocm = fresh._priority_map["rocm"].get("ws2_attention", [])
    assert (OpBackend.ROCM_STRICT_ATTENTION in rocm) is expected
    if expected:
        # It must lead: a strict caller should not land on a reference first.
        assert rocm[0] is OpBackend.ROCM_STRICT_ATTENTION
        assert OpBackend.ROCM_STRICT_ATTENTION in fresh._attention_capabilities

    # It is a ROCm backend and must never appear on another platform.
    for platform in ("cuda", "cpu"):
        assert OpBackend.ROCM_STRICT_ATTENTION not in fresh._priority_map[platform].get(
            "ws2_attention", []
        )


def test_rocm_strict_cp_capability_matches_the_transport():
    """The declared CP degrees must be the ones the RCCL transport accepts.

    CP is supplied by StrictRocmAttentionRuntime wrapping this core in the RCCL
    AG/RS transport. Declaring a degree the transport rejects would make
    dispatch hand back a backend that fails at materialization; declaring fewer
    would hide working CP behind an "unsupported" rejection.
    """

    if not _rocm_strict_attention_available():
        pytest.skip("strict ROCm attention requires a ROCm device with aiter.ops.mha")

    capability = KernelRegistry()._attention_capabilities[OpBackend.ROCM_STRICT_ATTENTION]

    assert capability.cp_world_sizes == (1, 2, 4, 8)
    # The merge order is the transport's fixed balanced rank tree, so the CP
    # combine is deterministic even though the core is single-rank arithmetic.
    assert capability.deterministic_cp_merge is True
    assert capability.exports_attention_lse is True


def test_registered_backend_resolves_with_provenance(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    result = registry.get_attention_op(_contract(), requested_backend="test.strict.core")

    assert result.capability.backend_id == "test.strict.core"
    assert result.provenance["actual_backend"] == "test.strict.core"
    assert result.provenance["fallback"] is False
    assert result.provenance["requested_backend"] == "test.strict.core"
    assert result.provenance["contract"]["lse_domain"] == "attention"


def test_unregistered_contract_fails_loudly_instead_of_falling_back(registry):
    # With no attention candidate at all, dispatch must raise rather than reach
    # into the legacy priority lists for something with other semantics.
    for ops in registry._priority_map.values():
        ops["ws2_attention"] = []

    with pytest.raises(RuntimeError, match="No attention backend supports"):
        registry.get_attention_op(_contract(), requested_backend="auto")


def test_explicit_backend_id_never_resolves_to_a_different_backend(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    with pytest.raises(RuntimeError, match="does not match requested_backend"):
        registry.get_attention_op(_contract(), requested_backend="some.other.backend")


def test_capability_mismatch_is_rejected_rather_than_approximated(registry):
    # A backend that cannot export attention-domain LSE must not serve a
    # contract that requires it.
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(exports_attention_lse=False),
        platform=_platform(registry),
    )

    with pytest.raises(RuntimeError, match="LSE export is unsupported"):
        registry.get_attention_op(_contract(), requested_backend="test.strict.core")


def test_cp_contract_requires_deterministic_merge_support(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(cp_world_sizes=(1, 2)),
        platform=_platform(registry),
    )

    with pytest.raises(RuntimeError, match="deterministic CP"):
        registry.get_attention_op(_contract(cp_world_size=2), requested_backend="test.strict.core")


def test_auto_is_rejected_under_context_parallelism(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    with pytest.raises(AttentionContractError, match="Unsafe dispatch"):
        registry.get_attention_op(_contract(cp_world_size=2), requested_backend="auto")


def test_deterministic_is_a_valid_attention_policy(registry):
    """``deterministic`` is a real ``implementation_kind`` for attention.

    It is not a policy for logprob dispatch, but ``AttentionBackendCapability``
    admits it, and it is the default here, so requesting it must select a
    deterministic backend rather than being rejected.
    """

    platform = _platform(registry)
    registry._priority_map[platform]["ws2_attention"] = [OpBackend.ROCM_STRICT_ATTENTION]
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(implementation_kind="deterministic"),
        platform=platform,
    )

    result = registry.get_attention_op(_contract(), requested_backend="deterministic")
    assert result.capability.implementation_kind == "deterministic"

    with pytest.raises(RuntimeError, match="does not satisfy requested_backend=production"):
        registry.get_attention_op(_contract(), requested_backend="production")


def test_implementation_kind_policy_filters_without_marking_fallback(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    result = registry.get_attention_op(_contract(), requested_backend="production")
    assert result.provenance["fallback"] is False

    with pytest.raises(RuntimeError, match="does not satisfy requested_backend=reference"):
        registry.get_attention_op(_contract(), requested_backend="reference")


def test_reregistration_replaces_capability_without_duplicating(registry):
    platform = _platform(registry)
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=platform
    )
    first = list(registry._priority_map[platform]["ws2_attention"])
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(implementation_kind="reference"),
        platform=platform,
    )
    second = registry._priority_map[platform]["ws2_attention"]

    assert first == second
    assert second.count(OpBackend.ROCM_STRICT_ATTENTION) == 1
    assert (
        registry._attention_capabilities[OpBackend.ROCM_STRICT_ATTENTION].implementation_kind
        == "reference"
    )


def test_register_rejects_wrong_types_and_unknown_platforms(registry):
    with pytest.raises(AttentionContractError, match="must be an OpBackend"):
        registry.register_attention_backend("not-a-backend", _capability())
    with pytest.raises(AttentionContractError, match="must be an AttentionBackendCapability"):
        registry.register_attention_backend(OpBackend.ROCM_STRICT_ATTENTION, object())
    with pytest.raises(AttentionContractError, match="unsupported platform"):
        registry.register_attention_backend(
            OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform="quantum"
        )


def test_registration_touches_only_the_ws2_attention_list(registry):
    """Registering must not perturb any legacy dispatch key.

    ``ws2_attention`` lives inside the priority map, so registration does write
    there - but the SDPA-shaped ``attn`` / ``attention`` keys that legacy
    ``get_op`` callers resolve through must be left exactly as they were.
    """

    platform = _platform(registry)
    before = {op: list(v) for op, v in registry._priority_map[platform].items()}

    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=platform
    )

    after = {op: list(v) for op, v in registry._priority_map[platform].items()}
    changed = {op for op in after if before.get(op) != after[op]}
    assert changed <= {"ws2_attention"}
    for legacy in ("attn", "attention", "cp_attention", "kv_cache_attention"):
        if legacy in before:
            assert before[legacy] == after[legacy]
            assert OpBackend.ROCM_STRICT_ATTENTION not in after[legacy]


def test_contract_fingerprint_is_rank_independent():
    left = _contract()
    right_sharding = ShardingSpec(
        tp_rank=1,
        tp_world_size=2,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=16,
        local_q_heads=16,
        local_kv_head_start=4,
        local_kv_heads=4,
        global_sequence_length=128,
        local_sequence_length=128,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, 128),
    )
    left_tp2 = AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=128,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=2,
            cp_rank=0,
            cp_world_size=1,
            global_q_heads=32,
            global_kv_heads=8,
            local_q_head_start=0,
            local_q_heads=16,
            local_kv_head_start=0,
            local_kv_heads=4,
            global_sequence_length=128,
            local_sequence_length=128,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, 128),
        ),
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )
    right_tp2 = AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=128,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=right_sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )

    # Both TP ranks of one logical invocation agree; a different TP degree does not.
    assert left_tp2.cross_rank_fingerprint() == right_tp2.cross_rank_fingerprint()
    assert left.cross_rank_fingerprint() != left_tp2.cross_rank_fingerprint()


# ---------------------------------------------------------------------------
# Cross-config binding: train and rollout must use the same parallel degrees
# ---------------------------------------------------------------------------


def _tp_contract(tp_world_size: int, tp_rank: int = 0, seq_len: int = 128) -> AttentionContract:
    """A contract for one TP rank of a 32Q/8KV layout."""

    local_q = 32 // tp_world_size
    local_kv = 8 // tp_world_size
    sharding = ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=tp_rank * local_q,
        local_q_heads=local_q,
        local_kv_head_start=tp_rank * local_kv,
        local_kv_heads=local_kv,
        global_sequence_length=seq_len,
        local_sequence_length=seq_len,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, seq_len),
    )
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=seq_len,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )


def test_matching_contracts_pass_cross_config_alignment():
    validate_cross_config_alignment(_tp_contract(4, tp_rank=0), _tp_contract(4, tp_rank=3))


def test_differing_tp_degrees_are_comparable():
    """A TP-degree difference must NOT be rejected.

    The provider pins every launch to one batch row and one KV group, which
    makes a head shard's result independent of the TP degree that produced it
    (verified bitwise at TP=1/2/4/8 on MI300X). Rejecting the comparison would
    refuse results that are in fact identical.
    """

    validate_cross_config_alignment(_tp_contract(4), _tp_contract(8))
    validate_cross_config_alignment(_tp_contract(1), _tp_contract(8))


def test_head_layout_mismatch_fails_closed():
    train = _tp_contract(2)
    rollout_sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=2,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=16,
        global_kv_heads=8,
        local_q_head_start=0,
        local_q_heads=8,
        local_kv_head_start=0,
        local_kv_heads=4,
        global_sequence_length=128,
        local_sequence_length=128,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, 128),
    )
    rollout = AttentionContract(
        role=AttentionRole.INFER,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=128,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=rollout_sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )
    with pytest.raises(AttentionContractError, match="global head layouts"):
        validate_cross_config_alignment(train, rollout)


def test_contract_fingerprint_is_a_per_invocation_rank_preflight():
    """The fingerprint agrees across ranks of one invocation, and only there.

    It still separates TP degrees, which is correct for its purpose: every rank
    of a single logical invocation must agree on one topology. It is not a
    train-vs-rollout equality token -- those may legitimately run different TP
    degrees and still be bitwise equal.
    """

    assert (
        _tp_contract(4, tp_rank=0).cross_rank_fingerprint()
        == _tp_contract(4, tp_rank=3).cross_rank_fingerprint()
    )
    assert _tp_contract(4).cross_rank_fingerprint() != _tp_contract(8).cross_rank_fingerprint()


def test_dtype_and_split_kv_mismatches_are_explained():
    train = _tp_contract(2)
    rollout = AttentionContract(
        role=train.role,
        mode=train.mode,
        dtype=AttentionDType.FP16,
        batch_size=train.batch_size,
        query_sequence_length=train.query_sequence_length,
        head_dim=train.head_dim,
        causal=train.causal,
        causal_offsets=train.causal_offsets,
        sharding=train.sharding,
        reduction=train.reduction,
        split_kv=train.split_kv,
        export_lse=True,
    )
    with pytest.raises(AttentionContractError, match="dtype"):
        validate_cross_config_alignment(train, rollout)
