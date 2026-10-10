# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 Attention CP contract and contract-aware dispatch tests (issue #235)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionBackendCapability,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionProjectionSpec,
    AttentionRole,
    KVCacheSpec,
    ProjectionCollective,
    ReductionSpec,
    RoPEFusionBoundary,
    RoPESpec,
    ShardingSpec,
    SplitKVExecutionPlan,
    SplitKVRuntimePlanSet,
    SplitKVSpec,
    build_split_kv_runtime_plan_set,
    validate_split_kv_alignment,
    validate_split_kv_plan_set_alignment,
)
from rl_engine.runtime.registry import KernelRegistry, OpBackend


def test_strict_attention_identity_pins_core_and_arithmetic_schedule():
    assert STRICT_ATTENTION_CORE_ID == "rlkernel.attention.deterministic_core.v1"
    assert STRICT_ATTENTION_SCHEDULE_ID == "single_batch_single_query_global_kv_blocks"


def _sharding(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 2,
    cp_rank: int = 0,
    cp_world_size: int = 2,
    global_sequence_length: int = 4096,
    local_sequence_length: int = 2048,
    global_block_indices: tuple[int, ...] = (0,),
    global_block_token_starts: tuple[int, ...] = (0,),
    local_block_offsets: tuple[int, ...] = (0, 2048),
    packed_sequence_offsets: tuple[int, ...] | None = None,
    sp_rank: int = 0,
    sp_world_size: int = 1,
) -> ShardingSpec:
    local_q_heads = 32 // tp_world_size
    local_kv_heads = 8 // tp_world_size
    return ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=tp_rank * local_q_heads,
        local_q_heads=local_q_heads,
        local_kv_head_start=tp_rank * local_kv_heads,
        local_kv_heads=local_kv_heads,
        global_sequence_length=global_sequence_length,
        local_sequence_length=local_sequence_length,
        global_block_indices=global_block_indices,
        global_block_token_starts=global_block_token_starts,
        local_block_offsets=local_block_offsets,
        packed_sequence_offsets=packed_sequence_offsets,
        sp_rank=sp_rank,
        sp_world_size=sp_world_size,
    )


def _contract(
    *,
    role: str = "infer",
    mode: str = "prefill",
    sharding: ShardingSpec | None = None,
    kv_cache: KVCacheSpec | None = None,
    causal_offsets: tuple[int, ...] = (0,),
    batch_size: int = 1,
    query_sequence_length: int | None = None,
    rope: RoPESpec | None = None,
) -> AttentionContract:
    resolved_sharding = sharding or _sharding()
    return AttentionContract(
        role=role,
        mode=mode,
        dtype="bf16",
        batch_size=batch_size,
        query_sequence_length=(
            query_sequence_length
            if query_sequence_length is not None
            else (1 if mode == "decode" else resolved_sharding.local_sequence_length)
        ),
        head_dim=128,
        causal=True,
        causal_offsets=causal_offsets,
        sharding=resolved_sharding,
        reduction=ReductionSpec(),
        kv_cache=kv_cache,
        rope=rope,
    )


def _declared_cp_backend() -> AttentionBackendCapability:
    return AttentionBackendCapability(
        backend_id="test-deterministic-cp-attention",
        roles=frozenset({AttentionRole.TRAIN, AttentionRole.INFER}),
        modes=frozenset(
            {AttentionMode.PREFILL, AttentionMode.CHUNKED_PREFILL, AttentionMode.DECODE}
        ),
        dtypes=frozenset({AttentionDType.BF16}),
        tp_world_sizes=(2,),
        cp_world_sizes=(1, 2),
        exports_attention_lse=True,
        deterministic_cp_merge=True,
        supports_packed_varlen=True,
        supports_kv_cache=True,
        supports_split_kv_fixed=True,
        reports_actual_split_kv_plan=True,
        implementation_kind="deterministic",
    )


def test_qwen3_tp2_cp2_contract_is_representable_and_serializable():
    contract = _contract()

    assert contract.sharding.local_q_heads == 16
    assert contract.sharding.local_kv_heads == 4
    assert contract.sharding.tp_world_size == 2
    assert contract.sharding.cp_world_size == 2
    assert contract.reduction.acc_dtype is AttentionDType.FP32
    assert contract.to_dict()["reduction"] == {
        "merge": "online_softmax_lse",
        "acc_dtype": "fp32",
        "order": "global_block_index",
        "downcast_at": "final_write",
        "engine": "in_op_reference",
    }
    json.dumps(contract.to_dict())


def test_rope_metadata_is_part_of_attention_contract_provenance():
    rope = RoPESpec(
        q_state="post_rope",
        k_state="post_rope",
        k_cache_state="post_rope",
        theta=1.0e6,
        rotary_dim=128,
        query_position_offsets=(0,),
        key_position_offsets=(0,),
        cast_at="after_rope",
        output_dtype="bf16",
        fusion_boundary="unfused_rope_attention",
    )

    contract = _contract(rope=rope)
    payload = contract.to_dict()

    assert payload["rope"] == {
        "q_state": "post_rope",
        "k_state": "post_rope",
        "k_cache_state": "post_rope",
        "theta": 1.0e6,
        "rotary_dim": 128,
        "rope_scaling": None,
        "position_ids": None,
        "query_position_offsets": [0],
        "key_position_offsets": [0],
        "cast_at": "after_rope",
        "output_dtype": "bf16",
        "fusion_boundary": "unfused_rope_attention",
    }
    json.dumps(payload)


def test_rope_position_metadata_is_validated_against_contract_shape():
    with pytest.raises(AttentionContractError, match="rotary_dim=256"):
        _contract(rope=RoPESpec(rotary_dim=256))

    with pytest.raises(AttentionContractError, match="query_position_offsets"):
        _contract(batch_size=2, causal_offsets=(0, 0), rope=RoPESpec(query_position_offsets=(0,)))

    with pytest.raises(AttentionContractError, match="position_ids"):
        _contract(rope=RoPESpec(position_ids=(0, 1, 2)))

    valid = _contract(rope=RoPESpec(position_ids=tuple(range(2048))))
    assert valid.rope is not None
    assert valid.rope.position_ids == tuple(range(2048))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tp_rank", 2, "tp_rank=2"),
        ("cp_rank", 2, "cp_rank=2"),
        ("global_block_indices", (), "must not be empty"),
        ("global_block_indices", (1, 0), "strictly increasing"),
    ],
)
def test_invalid_rank_and_cp_order_metadata_fail_loudly(field, value, message):
    values = {
        "tp_rank": 0,
        "tp_world_size": 2,
        "cp_rank": 0,
        "cp_world_size": 2,
        "global_q_heads": 32,
        "global_kv_heads": 8,
        "local_q_head_start": 0,
        "local_q_heads": 16,
        "local_kv_head_start": 0,
        "local_kv_heads": 4,
        "global_sequence_length": 4096,
        "local_sequence_length": 2048,
        "global_block_indices": (0,),
        "global_block_token_starts": (0,),
        "local_block_offsets": (0, 2048),
    }
    values[field] = value

    with pytest.raises(AttentionContractError, match=message):
        ShardingSpec(**values)


def test_tp_local_heads_must_preserve_global_gqa_mapping():
    with pytest.raises(AttentionContractError, match="local TP head counts"):
        replace(_sharding(), local_q_heads=7)

    with pytest.raises(AttentionContractError, match="head starts"):
        replace(_sharding(tp_rank=1), local_q_head_start=0)


def test_sequence_range_and_packed_offsets_are_validated():
    with pytest.raises(AttentionContractError, match="exceeds global_sequence_length"):
        _sharding(global_block_token_starts=(4000,))

    with pytest.raises(AttentionContractError, match="final packed_sequence_offsets"):
        _sharding(packed_sequence_offsets=(0, 512))

    sharding = _sharding(packed_sequence_offsets=(0, 512, 2048))
    assert sharding.packed_sequence_offsets == (0, 512, 2048)


def test_non_contiguous_cp_blocks_have_explicit_global_and_local_offsets():
    sharding = _sharding(
        global_block_indices=(0, 3),
        global_block_token_starts=(0, 3072),
        local_block_offsets=(0, 1024, 2048),
    )

    assert sharding.global_block_indices == (0, 3)
    assert sharding.global_block_token_starts == (0, 3072)
    assert sharding.local_block_offsets == (0, 1024, 2048)

    with pytest.raises(AttentionContractError, match="non-overlapping and ordered"):
        _sharding(
            global_block_indices=(0, 1),
            global_block_token_starts=(0, 512),
            local_block_offsets=(0, 1024, 2048),
        )


def test_reduction_requires_fp32_accumulation():
    with pytest.raises(AttentionContractError, match="must be fp32"):
        ReductionSpec(acc_dtype="bf16")


def test_split_kv_policy_is_a_first_class_strict_contract():
    contract = _contract()
    assert contract.to_dict()["split_kv"] == {
        "mode": "disabled",
        "fixed_split_size": None,
        "strict_consistency": True,
    }

    fixed = replace(contract, split_kv=SplitKVSpec.fixed(128))
    assert fixed.to_dict()["split_kv"]["mode"] == "fixed"
    assert fixed.to_dict()["split_kv"]["fixed_split_size"] == 128

    with pytest.raises(AttentionContractError, match="auto Split-KV"):
        SplitKVSpec.auto(strict_consistency=True)


def test_split_kv_execution_plan_records_actual_logical_schedule():
    plan = SplitKVSpec.fixed(4).resolve(10, backend="training-reference")

    assert plan.actual_split_count == 3
    assert plan.to_dict()["actual_split_boundaries"] == [[0, 4], [4, 8], [8, 10]]
    assert plan.to_dict()["split_kv_merge_order"] == "global_block_index"
    assert plan.to_dict()["split_kv_accum_dtype"] == "fp32"
    assert plan.to_dict()["split_kv_downcast_at"] == "final_write"


def test_strict_split_kv_alignment_rejects_unknown_or_mismatched_actual_plan():
    training = SplitKVSpec.fixed(4).resolve(10, backend="training")
    rollout = SplitKVSpec.fixed(4).resolve(10, backend="rollout")
    validate_split_kv_alignment(training, rollout)

    unknown = SplitKVSpec.auto().resolve(10, backend="rollout")
    with pytest.raises(AttentionContractError, match="actual runtime plans"):
        validate_split_kv_alignment(training, unknown)

    mismatched = SplitKVSpec.fixed(5).resolve(10, backend="rollout")
    with pytest.raises(AttentionContractError, match="differ"):
        validate_split_kv_alignment(training, mismatched)

    with pytest.raises(AttentionContractError, match="contiguous"):
        SplitKVExecutionPlan(
            requested_mode="fixed",
            requested_split_size=4,
            actual_mode="fixed",
            actual_split_size=4,
            boundaries=((0, 4), (5, 10)),
        )


def test_complete_split_kv_plan_set_covers_batch_tp_cp_and_owner_coordinates():
    plan_set = build_split_kv_runtime_plan_set(
        (8, 10),
        tp_world_size=2,
        cp_world_size=2,
        split_kv=SplitKVSpec.fixed(2),
        backend="training-reference",
    )

    assert len(plan_set.entries) == 16
    assert plan_set.to_dict()["coverage"] == ("complete_batch_tp_cp_owner_cartesian_product")
    assert {
        tuple(entry["expected_kv_range"])
        for entry in plan_set.to_dict()["entries"]
        if entry["batch_index"] == 0
    } == {(0, 4), (4, 8)}


def test_split_kv_plan_set_alignment_rejects_missing_and_mismatched_rank_plans():
    training = build_split_kv_runtime_plan_set(
        (8,),
        tp_world_size=2,
        cp_world_size=2,
        split_kv=SplitKVSpec.fixed(2),
        backend="training",
    )
    rollout = build_split_kv_runtime_plan_set(
        (8,),
        tp_world_size=2,
        cp_world_size=2,
        split_kv=SplitKVSpec.fixed(2),
        backend="rollout",
    )
    validate_split_kv_plan_set_alignment(training, rollout)

    with pytest.raises(AttentionContractError, match="coordinate coverage is incomplete"):
        SplitKVRuntimePlanSet(
            batch_size=training.batch_size,
            tp_world_size=training.tp_world_size,
            cp_world_size=training.cp_world_size,
            total_kv_tokens=training.total_kv_tokens,
            entries=training.entries[:-1],
        )

    mismatched = build_split_kv_runtime_plan_set(
        (8,),
        tp_world_size=2,
        cp_world_size=2,
        split_kv=SplitKVSpec.fixed(1),
        backend="rollout",
    )
    with pytest.raises(AttentionContractError, match="plan differs"):
        validate_split_kv_plan_set_alignment(training, mismatched)


def test_backend_must_support_policy_and_actual_plan_provenance():
    fixed = replace(_contract(), split_kv=SplitKVSpec.fixed(128))
    capability = replace(
        _declared_cp_backend(),
        supports_split_kv_fixed=False,
        reports_actual_split_kv_plan=False,
    )

    assert capability.incompatibilities(fixed)[-2:] == (
        "Split-KV policy=fixed is unsupported",
        "actual Split-KV execution-plan provenance is unsupported",
    )


def test_causal_attention_requires_explicit_offset():
    contract = _contract()
    with pytest.raises(AttentionContractError, match="causal_offsets are required"):
        replace(contract, causal_offsets=None)


def test_full_prefill_query_length_must_match_local_sequence_length():
    with pytest.raises(AttentionContractError, match="prefill query_sequence_length must equal"):
        _contract(mode="prefill", query_sequence_length=1024)

    chunked = _contract(mode="chunked_prefill", query_sequence_length=512)
    decode = _contract(
        mode="decode",
        query_sequence_length=1,
        kv_cache=KVCacheSpec(
            cache_positions=(16,),
            kv_seq_lens=(17,),
            block_table=((0, 1),),
            global_token_positions=tuple(range(17)),
            page_size=16,
        ),
    )
    assert chunked.query_sequence_length == 512
    assert decode.query_sequence_length == 1


def test_decode_requires_complete_kv_cache_identity():
    with pytest.raises(AttentionContractError, match="kv_cache metadata is required"):
        _contract(mode="decode")

    cache = KVCacheSpec(
        cache_positions=(16,),
        kv_seq_lens=(17,),
        block_table=((0, 1, -1),),
        global_token_positions=tuple(range(17)),
        page_size=16,
        prefix_cache_enabled=True,
        prefix_cache_key="prefix:sample-0",
    )
    contract = _contract(mode="decode", kv_cache=cache)
    assert contract.to_dict()["kv_cache"]["block_table"] == [[0, 1, -1]]


def test_prefix_cache_key_is_required_only_when_prefix_cache_is_enabled():
    with pytest.raises(AttentionContractError, match="prefix_cache_key is required"):
        KVCacheSpec(
            cache_positions=(16,),
            kv_seq_lens=(17,),
            block_table=((0, 1),),
            global_token_positions=tuple(range(17)),
            page_size=16,
            prefix_cache_enabled=True,
        )


def test_cache_positions_must_match_kv_sequence_count():
    with pytest.raises(AttentionContractError, match="one entry per kv_seq_lens"):
        KVCacheSpec(
            cache_positions=(1,),
            kv_seq_lens=(2, 2),
            block_table=((0,), (1,)),
            global_token_positions=(0, 1, 0, 1),
            page_size=2,
        )


def test_cache_position_must_match_terminal_global_token_position():
    with pytest.raises(AttentionContractError, match="terminal global token position"):
        KVCacheSpec(
            cache_positions=(999,),
            kv_seq_lens=(17,),
            block_table=((0, 1),),
            global_token_positions=tuple(range(17)),
            page_size=16,
        )


@pytest.mark.parametrize("positions", [(7, 6), (7, 7)])
def test_kv_cache_positions_must_be_strictly_increasing_per_sequence(positions):
    with pytest.raises(AttentionContractError, match="strictly increasing"):
        KVCacheSpec(
            cache_positions=(7,),
            kv_seq_lens=(2,),
            block_table=((0,),),
            global_token_positions=positions,
            page_size=2,
        )


@pytest.mark.parametrize(
    ("block_table", "message"),
    [
        ((0, -1, 1), "padding must be trailing"),
        ((0, 0, -1), "duplicate active page ids"),
        ((0, -1, -1), "active page count"),
    ],
)
def test_kv_cache_block_table_page_mapping_is_validated(block_table, message):
    with pytest.raises(AttentionContractError, match=message):
        KVCacheSpec(
            cache_positions=(16,),
            kv_seq_lens=(17,),
            block_table=(block_table,),
            global_token_positions=tuple(range(17)),
            page_size=16,
        )


def test_prefix_pages_may_be_shared_across_sequences():
    cache = KVCacheSpec(
        cache_positions=(1, 1),
        kv_seq_lens=(2, 2),
        block_table=((3,), (3,)),
        global_token_positions=(0, 1, 0, 1),
        page_size=2,
        prefix_cache_enabled=True,
        prefix_cache_key="shared-prefix",
        shared_prefix_page_count=1,
    )

    assert cache.block_table == ((3,), (3,))
    assert cache.shared_prefix_page_count == 1


def test_non_prefix_cache_rejects_cross_sequence_page_sharing():
    with pytest.raises(AttentionContractError, match="only when declared"):
        KVCacheSpec(
            cache_positions=(1, 1),
            kv_seq_lens=(2, 2),
            block_table=((3,), (3,)),
            global_token_positions=(0, 1, 0, 1),
            page_size=2,
            prefix_cache_enabled=False,
        )


def test_prefix_cache_requires_explicit_shared_page_count():
    with pytest.raises(AttentionContractError, match="only when declared"):
        KVCacheSpec(
            cache_positions=(1, 1),
            kv_seq_lens=(2, 2),
            block_table=((3,), (3,)),
            global_token_positions=(0, 1, 0, 1),
            page_size=2,
            prefix_cache_enabled=True,
            prefix_cache_key="shared-prefix",
            shared_prefix_page_count=0,
        )


def test_prefix_cache_rejects_shared_writable_suffix_pages():
    with pytest.raises(AttentionContractError, match="only when declared"):
        KVCacheSpec(
            cache_positions=(3, 3),
            kv_seq_lens=(4, 4),
            block_table=((3, 4), (3, 4)),
            global_token_positions=(0, 1, 2, 3, 0, 1, 2, 3),
            page_size=2,
            prefix_cache_enabled=True,
            prefix_cache_key="shared-prefix",
            shared_prefix_page_count=1,
        )


def test_shared_prefix_identity_must_match_pages_and_positions():
    with pytest.raises(AttentionContractError, match="page ids must match"):
        KVCacheSpec(
            cache_positions=(3, 3),
            kv_seq_lens=(4, 4),
            block_table=((3, 4), (5, 6)),
            global_token_positions=(0, 1, 2, 3, 0, 1, 2, 3),
            page_size=2,
            prefix_cache_enabled=True,
            prefix_cache_key="shared-prefix",
            shared_prefix_page_count=1,
        )

    with pytest.raises(AttentionContractError, match="token positions must match"):
        KVCacheSpec(
            cache_positions=(3, 13),
            kv_seq_lens=(4, 4),
            block_table=((3, 4), (3, 5)),
            global_token_positions=(0, 1, 2, 3, 10, 11, 12, 13),
            page_size=2,
            prefix_cache_enabled=True,
            prefix_cache_key="shared-prefix",
            shared_prefix_page_count=1,
        )


def test_shared_prefix_pages_must_be_fully_populated():
    with pytest.raises(AttentionContractError, match="fully populated and read-only"):
        KVCacheSpec(
            cache_positions=(0, 0),
            kv_seq_lens=(1, 1),
            block_table=((3,), (3,)),
            global_token_positions=(0, 0),
            page_size=2,
            prefix_cache_enabled=True,
            prefix_cache_key="partial-prefix-page",
            shared_prefix_page_count=1,
        )


def test_current_ws1_backend_rejects_strict_cp_contract_without_fallback():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._priority_map[platform]["ws2_attention"] = [OpBackend.PYTORCH_NATIVE_ATTENTION]

    with pytest.raises(RuntimeError) as exc_info:
        registry.get_attention_op(_contract())

    message = str(exc_info.value)
    assert "CP=2 is unsupported" in message
    assert "attention-domain LSE export is unsupported" in message
    assert "deterministic CP (out, lse) merge is unsupported" in message


def test_undeclared_backend_capability_is_never_selected():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._priority_map[platform]["ws2_attention"] = [OpBackend.PYTORCH_ATTN]

    with pytest.raises(RuntimeError, match="no AttentionBackendCapability declared"):
        registry.get_attention_op(_contract())


def test_declared_compatible_backend_resolves_and_records_provenance():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._priority_map[platform]["ws2_attention"] = [OpBackend.PYTORCH_NATIVE_ATTENTION]
    registry._attention_capabilities[OpBackend.PYTORCH_NATIVE_ATTENTION] = _declared_cp_backend()

    result = registry.get_attention_op(_contract(), requested_backend="deterministic")

    assert result.op is not None
    assert result.capability.backend_id == "test-deterministic-cp-attention"
    assert result.provenance["requested_backend"] == "deterministic"
    assert result.provenance["actual_backend"] == "test-deterministic-cp-attention"
    assert result.provenance["fallback"] is False
    assert result.provenance["contract"]["sharding"]["tp_world_size"] == 2
    assert result.provenance["contract"]["sharding"]["cp_world_size"] == 2
    json.dumps(result.provenance)


def test_requested_stable_backend_id_is_enforced():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._priority_map[platform]["ws2_attention"] = [OpBackend.PYTORCH_NATIVE_ATTENTION]
    registry._attention_capabilities[OpBackend.PYTORCH_NATIVE_ATTENTION] = _declared_cp_backend()

    with pytest.raises(RuntimeError, match="does not match requested_backend=another-backend"):
        registry.get_attention_op(_contract(), requested_backend="another-backend")

    result = registry.get_attention_op(
        _contract(), requested_backend="test-deterministic-cp-attention"
    )
    assert result.provenance["actual_backend"] == "test-deterministic-cp-attention"


def test_packed_layout_requires_declared_backend_support():
    capability = replace(_declared_cp_backend(), supports_packed_varlen=False)
    contract = _contract(
        sharding=_sharding(packed_sequence_offsets=(0, 512, 2048)),
        causal_offsets=(0, 0),
        batch_size=2,
    )

    assert capability.incompatibilities(contract) == ("packed varlen layout is unsupported",)


def test_rope_contract_requires_declared_backend_support():
    contract = _contract(rope=RoPESpec())
    capability = _declared_cp_backend()

    assert capability.incompatibilities(contract) == ("RoPE/position metadata is unsupported",)

    supported = replace(capability, supports_rope_metadata=True)
    assert supported.incompatibilities(contract) == ()


def test_fused_rope_attention_boundary_requires_declared_backend_support():
    contract = _contract(rope=RoPESpec(fusion_boundary=RoPEFusionBoundary.FUSED_ROPE_ATTENTION))
    capability = replace(_declared_cp_backend(), supports_rope_metadata=True)

    assert capability.incompatibilities(contract) == (
        "fused RoPE+Attention boundary is unsupported",
    )

    supported = replace(capability, supports_fused_rope_attention=True)
    assert supported.incompatibilities(contract) == ()


def test_packed_sequence_count_must_match_logical_batch_size():
    sharding = _sharding(packed_sequence_offsets=(0, 512, 2048))

    with pytest.raises(AttentionContractError, match="must equal logical batch_size"):
        _contract(sharding=sharding, causal_offsets=(0, 0), batch_size=1)

    contract = _contract(sharding=sharding, causal_offsets=(0, 0), batch_size=2)
    assert contract.batch_size == 2


def test_projection_contract_pins_gemm_and_tp_sp_collectives():
    contract = _contract(sharding=_sharding(sp_rank=1, sp_world_size=2))
    provenance = contract.to_dict()

    assert provenance["sharding"]["sp_rank"] == 1
    assert provenance["sharding"]["sp_world_size"] == 2
    assert provenance["projections"]["qkv"] == {
        "input_dtype": "bf16",
        "output_dtype": "bf16",
        "acc_dtype": "fp32",
        "split_kv": "disabled",
        "k_order": "ascending",
        "backend_policy": "native_verified_then_common_deterministic",
        "deterministic_backend": "rlkernel.cuda.det_gemm",
        "tp_forward_collective": "none",
        "tp_backward_dgrad_collective": "all_reduce",
        "sp_forward_collective": "all_gather",
        "sp_backward_collective": "reduce_scatter",
        "qkv_split_order": ["q", "k", "v"],
        "require_runtime_readback": True,
    }
    assert provenance["projections"]["o_proj"]["tp_forward_collective"] == "all_reduce"
    assert provenance["projections"]["o_proj"]["sp_forward_collective"] == "reduce_scatter"
    assert provenance["projections"]["o_proj"]["sp_backward_collective"] == "all_gather"


def test_projection_contract_rejects_split_k_and_wrong_collective_identity():
    with pytest.raises(AttentionContractError, match="disable Split-K"):
        replace(AttentionProjectionSpec.qkv(), split_kv="fixed")

    with pytest.raises(AttentionContractError, match="qkv_projection"):
        replace(
            _contract(),
            qkv_projection=replace(
                AttentionProjectionSpec.output(),
                tp_forward_collective=ProjectionCollective.NONE,
            ),
        )


def test_sharding_rejects_out_of_range_sp_rank():
    with pytest.raises(AttentionContractError, match="sp_rank=2"):
        _sharding(sp_rank=2, sp_world_size=2)
