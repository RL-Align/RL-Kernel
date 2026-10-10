# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 TP-aware logprob contract and contract-aware dispatch tests (issue #241)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_engine.contracts.operators.logprob import (
    DeterminismScope,
    LogprobBackendCapability,
    LogprobContract,
    LogprobContractError,
    LogprobDType,
    LogprobOutputSpec,
    LogprobRole,
    MaskMode,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
    TPPlacement,
)
from rl_engine.runtime.registry import KernelRegistry, OpBackend

QWEN3_REAL_VOCAB = 151936
QWEN3_PADDED_VOCAB = 152064


def _even_bounds(padded_vocab: int, tp_world_size: int) -> tuple[tuple[int, int], ...]:
    shard = padded_vocab // tp_world_size
    return tuple(
        (rank * shard, padded_vocab if rank == tp_world_size - 1 else (rank + 1) * shard)
        for rank in range(tp_world_size)
    )


def _sharding(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 2,
    cp_rank: int = 0,
    cp_world_size: int = 2,
    real_vocab_size: int = QWEN3_REAL_VOCAB,
    padded_vocab_size: int = QWEN3_PADDED_VOCAB,
    vocab_shard_bounds: tuple[tuple[int, int], ...] | None = None,
) -> ShardingSpec:
    return ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        vocab_shard_bounds=(
            vocab_shard_bounds
            if vocab_shard_bounds is not None
            else _even_bounds(padded_vocab_size, tp_world_size)
        ),
        real_vocab_size=real_vocab_size,
        padded_vocab_size=padded_vocab_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )


def _mask(
    *,
    num_tokens: int = 8,
    active_mask: tuple[bool, ...] | None = None,
    ignore_index: int = -100,
) -> MaskSpec:
    return MaskSpec(
        num_tokens=num_tokens,
        active_mask=(
            active_mask
            if active_mask is not None
            else (False, False, True, True, True, True, True, False)
        ),
        ignore_index=ignore_index,
    )


def _contract(
    *,
    role: str = "train",
    dtype: str = "bf16",
    mask: MaskSpec | None = None,
    sharding: ShardingSpec | None = None,
    reduction: ReductionSpec | None = None,
) -> LogprobContract:
    return LogprobContract(
        role=role,
        dtype=dtype,
        mask=mask if mask is not None else _mask(),
        sharding=sharding if sharding is not None else _sharding(),
        reduction=reduction if reduction is not None else ReductionSpec(),
    )


def _declared_tp_backend() -> LogprobBackendCapability:
    return LogprobBackendCapability(
        backend_id="test-deterministic-tp-logprob",
        roles=frozenset({LogprobRole.TRAIN, LogprobRole.INFER}),
        dtypes=frozenset({LogprobDType.BF16}),
        tp_world_sizes=(1, 2, 4),
        cp_world_sizes=None,
        supports_vocab_padding=True,
        mask_modes=frozenset({MaskMode.EXPLICIT_ACTIVE_MASK, MaskMode.IGNORE_INDEX}),
        exports_vocab_lse=True,
        determinism_scopes=frozenset(
            {DeterminismScope.CROSS_TP_BITWISE, DeterminismScope.FIXED_TOPOLOGY}
        ),
        implementation_kind="reference",
    )


def test_qwen3_tp2_bf16_contract_is_representable_and_serializable():
    contract = _contract()

    assert contract.sharding.tp_world_size == 2
    assert contract.sharding.cp_world_size == 2
    assert contract.sharding.local_vocab_start == 0
    assert contract.sharding.local_vocab_end == QWEN3_PADDED_VOCAB // 2
    assert contract.sharding.local_vocab_size == QWEN3_PADDED_VOCAB // 2
    assert contract.mask.active_token_count == 5
    assert contract.reduction.acc_dtype is LogprobDType.FP32
    assert contract.to_dict()["reduction"] == {
        "merge": "max_sumexp",
        "merge_axis": "tp_vocab",
        "acc_dtype": "fp32",
        "order": "global_vocab_shard_index",
        "transport": "all_gather",
        "downcast_at": "final_write",
        "engine": "in_op_reference",
        "determinism_scope": "cross_tp_bitwise",
        "cp_is_merge_axis": False,
    }
    json.dumps(contract.to_dict())


@pytest.mark.parametrize("tp_world_size", [1, 2, 4])
def test_pr4_sweep_tp_degrees_are_representable(tp_world_size):
    sharding = _sharding(tp_world_size=tp_world_size, cp_world_size=1)

    assert len(sharding.vocab_shard_bounds) == tp_world_size
    assert sharding.vocab_shard_bounds[-1][1] == QWEN3_PADDED_VOCAB
    assert sharding.owner_rank(QWEN3_REAL_VOCAB - 1) == tp_world_size - 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tp_rank", 2, "tp_rank=2"),
        ("cp_rank", 2, "cp_rank=2"),
        ("real_vocab_size", 0, "positive integer"),
        ("padded_vocab_size", QWEN3_REAL_VOCAB - 1, "must not be smaller"),
    ],
)
def test_invalid_rank_and_vocab_metadata_fail_loudly(field, value, message):
    values = {
        "tp_rank": 0,
        "tp_world_size": 2,
        "cp_rank": 0,
        "cp_world_size": 2,
        "real_vocab_size": QWEN3_REAL_VOCAB,
        "padded_vocab_size": QWEN3_PADDED_VOCAB,
        "vocab_shard_bounds": _even_bounds(QWEN3_PADDED_VOCAB, 2),
    }
    values[field] = value

    with pytest.raises(LogprobContractError, match=message):
        ShardingSpec(**values)


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((), "one \\(start, end\\) pair per TP rank"),
        (((0, 76032),), "one \\(start, end\\) pair per TP rank"),
        (((0, 76032), (76032, 76032)), "end > start"),
        (((0, 76000), (76032, 152064)), "contiguous"),
        (((0, 76064), (76032, 152064)), "contiguous"),
        (((0, 76032), (76032, 152000)), "cover padded_vocab_size exactly"),
    ],
)
def test_incomplete_or_overlapping_vocab_shard_bounds_fail_loudly(bounds, message):
    with pytest.raises(LogprobContractError, match=message):
        _sharding(vocab_shard_bounds=bounds)


def test_owner_rank_is_unique_and_rejects_out_of_real_vocab_targets():
    sharding = _sharding()

    assert sharding.owner_rank(0) == 0
    assert sharding.owner_rank(QWEN3_PADDED_VOCAB // 2 - 1) == 0
    assert sharding.owner_rank(QWEN3_PADDED_VOCAB // 2) == 1
    assert sharding.owner_rank(QWEN3_REAL_VOCAB - 1) == 1

    with pytest.raises(LogprobContractError, match="outside the real vocabulary"):
        sharding.owner_rank(-1)
    with pytest.raises(LogprobContractError, match="outside the real vocabulary"):
        sharding.owner_rank(QWEN3_REAL_VOCAB)


def test_active_token_mask_metadata_is_validated():
    with pytest.raises(LogprobContractError, match="one entry per token"):
        _mask(num_tokens=4)

    with pytest.raises(LogprobContractError, match="must be a bool"):
        MaskSpec(num_tokens=2, active_mask=(True, 1))

    all_inactive = _mask(num_tokens=3, active_mask=(False, False, False))
    assert all_inactive.active_token_count == 0


def test_reduction_requires_fp32_accumulation_and_known_semantics():
    with pytest.raises(LogprobContractError, match="must be fp32"):
        ReductionSpec(acc_dtype="bf16")

    with pytest.raises(LogprobContractError, match="merge must be one of"):
        ReductionSpec(merge="lse_average")

    with pytest.raises(LogprobContractError, match="transport must be one of"):
        ReductionSpec(transport="all_reduce")


def test_contract_component_types_and_lse_export_are_enforced():
    with pytest.raises(LogprobContractError, match="mask must be a MaskSpec"):
        LogprobContract(
            role="train",
            dtype="bf16",
            mask=None,
            sharding=_sharding(),
            reduction=ReductionSpec(),
        )

    with pytest.raises(LogprobContractError, match="export_lse must be True"):
        replace(_contract(), export_lse=False)


def test_ignore_index_must_not_collide_with_the_real_vocabulary():
    with pytest.raises(LogprobContractError, match="must not collide"):
        _contract(mask=_mask(ignore_index=5))

    padding_column = QWEN3_REAL_VOCAB + 1
    contract = _contract(mask=_mask(ignore_index=padding_column))
    assert contract.mask.ignore_index == padding_column


def _restrict_to_ws1_candidates(registry: KernelRegistry) -> None:
    """Drop all WS2 vocab-parallel backends so only WS1 backends remain."""

    platform = registry._platform()
    ws2_backends = {
        OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP,
        OpBackend.ROCM_VOCAB_PARALLEL_LOGP,
        OpBackend.TRITON_VOCAB_PARALLEL_LOGP,
    }
    registry._logprob_candidates[platform] = [
        backend for backend in registry._logprob_candidates[platform] if backend not in ws2_backends
    ]


def test_current_ws1_backend_rejects_strict_tp_contract_without_fallback():
    registry = KernelRegistry()
    _restrict_to_ws1_candidates(registry)

    with pytest.raises(RuntimeError) as exc_info:
        registry.get_logprob_op(_contract())

    message = str(exc_info.value)
    assert "TP=2 is unsupported" in message
    assert "vocab-domain LSE export is unsupported" in message
    assert "determinism_scope=cross_tp_bitwise is unsupported" in message
    assert "padded-vs-real vocab masking is unsupported" in message


def test_current_ws1_backend_rejects_padded_vocab_even_at_tp1():
    registry = KernelRegistry()
    _restrict_to_ws1_candidates(registry)
    contract = _contract(sharding=_sharding(tp_world_size=1, cp_world_size=1))

    with pytest.raises(RuntimeError) as exc_info:
        registry.get_logprob_op(contract)

    message = str(exc_info.value)
    assert "TP=1 is unsupported" not in message
    assert "padded-vs-real vocab masking is unsupported" in message


def test_ws1_rejections_recorded_when_vocab_parallel_reference_resolves():
    """The WS1 backends still reject strict contracts; they are skipped with
    recorded reasons while dispatch resolves the #241 PR3 reference."""

    registry = KernelRegistry()
    platform = registry._platform()
    # Order the WS1 backends ahead of the reference so their rejections are
    # exercised on the way to a successful resolution.
    candidates = registry._logprob_candidates[platform]
    candidates.remove(OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP)
    candidates.append(OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP)

    result = registry.get_logprob_op(_contract(), requested_backend="reference")
    assert result.capability.backend_id == "pytorch-vocab-parallel-logp-ws2"
    assert result.provenance["fallback"] is True
    rejections = " | ".join(result.provenance["prior_rejections"])
    assert "vocab-domain LSE export is unsupported" in rejections


def test_undeclared_backend_capability_is_never_selected():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = [OpBackend.PYTORCH_NATIVE]

    with pytest.raises(RuntimeError, match="no LogprobBackendCapability declared"):
        registry.get_logprob_op(_contract())


def test_declared_compatible_backend_resolves_and_records_provenance():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    result = registry.get_logprob_op(_contract(), requested_backend="reference")

    assert result.op is not None
    assert result.capability.backend_id == "test-deterministic-tp-logprob"
    assert result.provenance["requested_backend"] == "reference"
    assert result.provenance["actual_backend"] == "test-deterministic-tp-logprob"
    assert result.provenance["fallback"] is False
    assert result.provenance["contract"]["sharding"]["tp_world_size"] == 2
    assert result.provenance["contract"]["sharding"]["real_vocab_size"] == QWEN3_REAL_VOCAB
    assert result.provenance["contract"]["reduction"]["cp_is_merge_axis"] is False
    json.dumps(result.provenance)


def test_requested_stable_backend_id_is_enforced():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    with pytest.raises(RuntimeError, match="does not match requested_backend=another-backend"):
        registry.get_logprob_op(_contract(), requested_backend="another-backend")

    result = registry.get_logprob_op(_contract(), requested_backend="test-deterministic-tp-logprob")
    assert result.provenance["actual_backend"] == "test-deterministic-tp-logprob"


def test_cp_is_a_non_merge_axis_and_cp_agnostic_backends_accept_any_cp_degree():
    capability = _declared_tp_backend()
    cp2_contract = _contract(sharding=_sharding(cp_world_size=2, cp_rank=1))

    assert capability.incompatibilities(cp2_contract) == ()

    cp_restricted = replace(capability, cp_world_sizes=(1,))
    assert cp_restricted.incompatibilities(cp2_contract) == ("CP=2 is unsupported",)


def test_inactive_tokens_require_explicit_active_mask_support():
    capability = replace(_declared_tp_backend(), mask_modes=frozenset({MaskMode.IGNORE_INDEX}))
    contract = _contract()

    assert "explicit active-token masking is unsupported" in (
        capability.incompatibilities(contract)
    )

    fully_active = _contract(mask=_mask(num_tokens=3, active_mask=(True, True, True)))
    assert capability.incompatibilities(fully_active) == ()


def test_backend_id_must_not_shadow_a_reserved_policy_keyword():
    with pytest.raises(LogprobContractError, match="reserved dispatch policy keyword"):
        replace(_declared_tp_backend(), backend_id="Deterministic")


def test_default_auto_policy_resolves_any_compatible_implementation_kind():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
        replace(_declared_tp_backend(), implementation_kind="reference"),
        platform=platform,
    )

    result = registry.get_logprob_op(_contract())

    assert result.provenance["requested_backend"] == "auto"
    assert result.capability.implementation_kind == "reference"


def test_policy_keywords_are_case_insensitive_but_backend_ids_are_exact():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    result = registry.get_logprob_op(_contract(), requested_backend="REFERENCE")
    assert result.capability.backend_id == "test-deterministic-tp-logprob"

    with pytest.raises(RuntimeError, match="does not match requested_backend"):
        registry.get_logprob_op(_contract(), requested_backend="Test-Deterministic-TP-Logprob")


def test_policy_only_skips_are_not_reported_as_fallback():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.TRITON_BATCH_INVARIANT_LOGP,
        replace(_declared_tp_backend(), backend_id="other-compatible-backend"),
        platform=platform,
    )
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    result = registry.get_logprob_op(_contract(), requested_backend="test-deterministic-tp-logprob")

    assert result.provenance["fallback"] is False
    assert len(result.provenance["prior_rejections"]) == 1


def test_capability_rejections_are_reported_as_fallback():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.TRITON_BATCH_INVARIANT_LOGP,
        replace(_declared_tp_backend(), backend_id="tp1-only-backend", tp_world_sizes=(1,)),
        platform=platform,
    )
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    result = registry.get_logprob_op(_contract())

    assert result.provenance["fallback"] is True
    assert "TP=2 is unsupported" in result.provenance["prior_rejections"][0]


def test_ws2_candidate_list_is_decoupled_from_the_legacy_priority_map():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform].insert(0, OpBackend.PYTORCH_NATIVE)

    legacy = registry._priority_map[platform]["batch_invariant_logp"]
    assert OpBackend.PYTORCH_NATIVE not in legacy

    legacy.insert(0, OpBackend.PYTORCH_GEMM)
    assert OpBackend.PYTORCH_GEMM not in registry._logprob_candidates[platform]


def test_register_logprob_backend_is_the_public_registration_seam():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    capability = _declared_tp_backend()

    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, capability, platform=platform
    )
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
        replace(capability, backend_id="replacement-backend"),
        platform=platform,
    )

    assert registry._logprob_candidates[platform] == [OpBackend.PYTORCH_BATCH_INVARIANT_LOGP]
    result = registry.get_logprob_op(_contract())
    assert result.capability.backend_id == "replacement-backend"

    with pytest.raises(LogprobContractError, match="capability must be"):
        registry.register_logprob_backend(OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, None)


def test_backend_id_whitespace_is_normalized_for_dispatch():
    capability = replace(_declared_tp_backend(), backend_id="  padded-id  ")
    assert capability.backend_id == "padded-id"


def test_capabilities_are_scoped_per_platform():
    registry = KernelRegistry()
    platform = registry._platform()
    other = "rocm" if platform != "rocm" else "cpu"
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
        replace(_declared_tp_backend(), backend_id="other-platform-backend"),
        platform=other,
    )

    result = registry.get_logprob_op(_contract())

    assert result.capability.backend_id == "test-deterministic-tp-logprob"
    assert (
        registry._logprob_capabilities[other][OpBackend.PYTORCH_BATCH_INVARIANT_LOGP].backend_id
        == "other-platform-backend"
    )


def test_register_logprob_backend_rejects_unknown_platform():
    registry = KernelRegistry()

    with pytest.raises(LogprobContractError, match="unsupported platform"):
        registry.register_logprob_backend(
            OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
            _declared_tp_backend(),
            platform="cuda-typo",
        )


def test_non_iterable_roles_and_dtypes_raise_contract_errors():
    with pytest.raises(LogprobContractError, match="roles and dtypes must be iterables"):
        replace(_declared_tp_backend(), roles=None)

    with pytest.raises(LogprobContractError, match="roles and dtypes must be iterables"):
        replace(_declared_tp_backend(), dtypes=42)


def test_requested_deterministic_policy_is_a_loud_error():
    registry = KernelRegistry()

    with pytest.raises(LogprobContractError, match="determinism_scope"):
        registry.get_logprob_op(_contract(), requested_backend="deterministic")


def test_determinism_scope_is_part_of_the_typed_contract():
    fixed_only = replace(
        _declared_tp_backend(),
        determinism_scopes=frozenset({DeterminismScope.FIXED_TOPOLOGY}),
    )

    assert "determinism_scope=cross_tp_bitwise is unsupported" in (
        fixed_only.incompatibilities(_contract())
    )

    relaxed = _contract(reduction=ReductionSpec(determinism_scope="fixed_topology"))
    assert fixed_only.incompatibilities(relaxed) == ()


def test_policy_filtered_candidates_never_count_toward_fallback():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._logprob_candidates[platform] = []
    registry.register_logprob_backend(
        OpBackend.TRITON_BATCH_INVARIANT_LOGP,
        replace(_declared_tp_backend(), backend_id="tp1-only-backend", tp_world_sizes=(1,)),
        platform=platform,
    )
    registry.register_logprob_backend(
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP, _declared_tp_backend(), platform=platform
    )

    result = registry.get_logprob_op(_contract(), requested_backend="test-deterministic-tp-logprob")

    assert result.provenance["fallback"] is False
    assert len(result.provenance["prior_rejections"]) == 1


def test_output_spec_is_pinned_to_fp32_replicated():
    with pytest.raises(LogprobContractError, match="must be fp32"):
        LogprobOutputSpec(selected_logp_dtype="bf16")
    with pytest.raises(LogprobContractError, match="must be fp32"):
        LogprobOutputSpec(lse_dtype="bf16")

    assert LogprobOutputSpec().tp_placement is TPPlacement.REPLICATED
    assert _contract().to_dict()["output"] == {
        "selected_logp_dtype": "fp32",
        "lse_dtype": "fp32",
        "tp_placement": "replicated",
    }


def test_cross_rank_fingerprint_is_rank_independent_and_content_sensitive():
    rank0 = _contract(sharding=_sharding(tp_rank=0))
    rank1 = _contract(sharding=_sharding(tp_rank=1, cp_rank=1))

    assert rank0.cross_rank_fingerprint() == rank1.cross_rank_fingerprint()

    different_mask = _contract(
        mask=_mask(active_mask=(True, True, True, True, True, True, True, False))
    )
    assert rank0.cross_rank_fingerprint() != different_mask.cross_rank_fingerprint()


def test_provenance_records_the_active_mask_digest():
    provenance_mask = _contract().to_dict()["mask"]

    assert provenance_mask["active_mask_sha256"] == _mask().active_mask_sha256
    assert len(provenance_mask["active_mask_sha256"]) == 64

    same_count_different_mask = _mask(
        active_mask=(True, True, True, True, True, False, False, False)
    )
    assert same_count_different_mask.active_token_count == _mask().active_token_count
    assert same_count_different_mask.active_mask_sha256 != _mask().active_mask_sha256


def test_padding_only_shard_is_constructible_for_the_identity_partial():
    sharding = _sharding(
        vocab_shard_bounds=((0, QWEN3_REAL_VOCAB), (QWEN3_REAL_VOCAB, QWEN3_PADDED_VOCAB)),
    )

    assert sharding.local_vocab_start == 0
    assert sharding.vocab_shard_bounds[1] == (QWEN3_REAL_VOCAB, QWEN3_PADDED_VOCAB)
    assert sharding.owner_rank(QWEN3_REAL_VOCAB - 1) == 0
