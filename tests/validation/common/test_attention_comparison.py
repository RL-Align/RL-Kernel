# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from dataclasses import replace
from typing import Literal

import pytest
import torch

from rl_engine.contracts.operators.attention import SplitKVSpec
from rl_engine.reference.rope.rope import NativeRoPEOp
from rl_engine.validation.common.attention_comparison import (
    AttentionComparisonInputs,
    DecodeAttentionInputs,
    DecodeKVCacheMetadata,
    compare_decode_kv_replay,
    compare_single_gpu_attention,
    compare_single_gpu_rope_attention,
    decode_kv_cache_fingerprint,
    decode_prefix_cache_fingerprint,
    decode_rope_identity_fingerprint,
    run_decode_kv_replay,
    run_paged_kv_attention,
)
from rl_engine.validation.operators import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

_TE_CONTEXT_PARALLEL_MODULE = (
    "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
)


def _qkv(*, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 4, 6, 8, generator=gen)
    k = torch.randn(2, 2, 6, 8, generator=gen)
    v = torch.randn(2, 2, 6, 8, generator=gen)
    return q, k, v


def _comparison_inputs() -> AttentionComparisonInputs:
    q, k, v = _qkv()
    gen = torch.Generator().manual_seed(2)
    lm_head_weight = torch.randn(13, q.size(1) * q.size(3), generator=gen)
    target_ids = torch.randint(0, 13, (q.size(0), q.size(2)), generator=gen)
    active_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [False, True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    return AttentionComparisonInputs(
        q=q,
        k=k,
        v=v,
        causal=True,
        lm_head_weight=lm_head_weight,
        target_ids=target_ids,
        active_token_mask=active_mask,
    )


def _decode_inputs(
    *,
    page_order: tuple[int, ...] = (0, 1, 2),
    prefix_cache_enabled: bool = False,
    q_rope_state: Literal["pre_rope", "post_rope"] = "post_rope",
    k_cache_rope_state: Literal["pre_rope", "post_rope"] = "post_rope",
    position_offset: int = 0,
) -> DecodeAttentionInputs:
    q, logical_k, logical_v = _qkv(seed=17)
    q = q[:, :, 4:6, :]
    page_size = 2
    physical_k = torch.empty_like(logical_k)
    physical_v = torch.empty_like(logical_v)
    positions = torch.full((2, 6), -1, dtype=torch.long)
    for logical_page, physical_page in enumerate(page_order):
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(physical_page * page_size, (physical_page + 1) * page_size)
        physical_k[:, :, physical_slice, :] = logical_k[:, :, logical_slice, :]
        physical_v[:, :, physical_slice, :] = logical_v[:, :, logical_slice, :]
        positions[:, physical_slice] = torch.arange(
            position_offset + logical_page * page_size,
            position_offset + (logical_page + 1) * page_size,
        )
    inputs = DecodeAttentionInputs(
        q=q,
        k_cache=physical_k,
        v_cache=physical_v,
        metadata=DecodeKVCacheMetadata(
            cache_position=torch.tensor(
                [[position_offset + 4, position_offset + 5]] * 2,
                dtype=torch.long,
            ),
            kv_seq_lens=torch.tensor([6, 6], dtype=torch.long),
            block_table=torch.tensor([page_order, page_order], dtype=torch.long),
            global_token_positions=positions,
            query_position_ids=torch.tensor(
                [[position_offset + 4, position_offset + 5]] * 2,
                dtype=torch.long,
            ),
            key_position_ids=positions.clone(),
            page_size=page_size,
            q_rope_state=q_rope_state,
            k_cache_rope_state=k_cache_rope_state,
            cp_block_owners=torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long),
            cp_world_size=2,
        ),
        lm_head_weight=torch.randn(
            11, q.size(1) * q.size(3), generator=torch.Generator().manual_seed(18)
        ),
        target_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        active_token_mask=torch.tensor([[True, True], [False, True]], dtype=torch.bool),
    )
    if not prefix_cache_enabled:
        return inputs
    prefix_length = 4
    with_prefix_identity = replace(
        inputs,
        metadata=replace(
            inputs.metadata,
            prefix_cache_enabled=True,
            prefix_cache_key="shared-prefix",
            prefix_length=prefix_length,
        ),
    )
    fingerprint = decode_prefix_cache_fingerprint(
        with_prefix_identity,
        prefix_length=prefix_length,
    )
    return replace(
        with_prefix_identity,
        metadata=replace(
            with_prefix_identity.metadata,
            prefix_cache_fingerprint=fingerprint,
        ),
    )


def test_single_gpu_attention_harness_reports_out_lse_and_dlogp_drift():
    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=2,
        kv_page_size=3,
    )

    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert set(by_name) == {"chunked_prefill", "rl_kernel_paged_kv"}
    for drift in by_name.values():
        assert drift.out.max_abs <= 1.0e-6
        assert drift.lse.max_abs <= 1.0e-6
        assert drift.dlogp is not None
        assert drift.dlogp.active_count == 7
        assert drift.dlogp.p95_abs <= 1.0e-6
        assert drift.provenance["execution_scope"] == "single_device_correctness_reference"
        assert drift.provenance["native_backend_executed"] is False
        assert drift.provenance["preprocess_fallback"] is True
        assert drift.provenance["qkv_projection_executed"] is False

    payload = report.to_dict()
    assert payload["reference_name"] == "full_prefill"
    assert payload["drifts"][0]["out"]["p99_abs"] >= 0.0
    json.dumps(payload)


def test_strict_shared_core_is_bitwise_across_full_chunked_and_paged_layouts():
    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=2,
        kv_page_size=3,
        strict_bitwise=True,
    )

    assert report.reference_name == "strict_shared_core_full_prefill"
    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert set(by_name) == {
        "strict_shared_core_chunked_prefill",
        "strict_shared_core_paged_kv",
    }
    for drift in by_name.values():
        assert drift.out.max_abs == 0.0
        assert drift.lse.max_abs == 0.0
        assert drift.provenance["strict_core_id"] == ("rlkernel.attention.deterministic_core.v1")
        assert drift.provenance["strict_schedule"] == ("single_batch_single_query_global_kv_blocks")
        assert drift.provenance["split_kv_policy"] == "disabled"


def test_single_gpu_attention_harness_preserves_key_padding_mask():
    q, k, v = _qkv(seed=3)
    key_padding_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, False, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    report = compare_single_gpu_attention(
        AttentionComparisonInputs(
            q=q,
            k=k,
            v=v,
            causal=True,
            key_padding_mask=key_padding_mask,
        ),
        query_chunk_size=4,
        kv_page_size=2,
    )

    assert report.unavailable == ()
    for drift in report.drifts:
        assert drift.out.max_abs <= 1.0e-6
        assert drift.lse.max_abs <= 1.0e-6


def test_single_gpu_rope_attention_harness_reports_rope_and_attention_drift():
    base = _comparison_inputs()
    report = compare_single_gpu_rope_attention(
        AttentionComparisonInputs(
            q=base.q,
            k=base.k,
            v=base.v,
            causal=True,
            lm_head_weight=base.lm_head_weight,
            target_ids=base.target_ids,
            active_token_mask=base.active_token_mask,
            rope_positions=torch.arange(base.q.size(2), dtype=torch.long),
            rope_theta=1_000_000.0,
            rope_rotary_dim=base.q.size(-1),
            rope_output_dtype=torch.float32,
        )
    )

    assert report.reference_name == "unfused_rope_attention"
    assert len(report.drifts) == 1
    drift = report.drifts[0]
    assert drift.candidate_name == "fused_like_rope_attention"
    assert drift.post_rope_q is not None
    assert drift.post_rope_k is not None
    assert drift.post_rope_q.max_abs <= 1.0e-6
    assert drift.post_rope_k.max_abs <= 1.0e-6
    assert drift.out.max_abs <= 1.0e-6
    assert drift.lse.max_abs <= 1.0e-6
    assert drift.dlogp is not None
    assert drift.dlogp.max_abs <= 1.0e-6
    assert drift.provenance["position_kind"] == "position_ids"
    assert drift.provenance["position_ids_shape"] == [base.q.size(2)]
    assert drift.provenance["rotary_dim"] == base.q.size(-1)
    assert drift.provenance["rope_cast_at"] == "after_rope"
    assert drift.provenance["fusion_boundary"] == "fused_rope_attention"
    assert drift.provenance["preprocess_backends"] == {
        "rope": "rlkernel.pytorch.rope_reference",
        "qk_rmsnorm": "not_executed_projected_qk_input",
    }
    assert drift.provenance["preprocess_policy"] == "reference_only_not_production"

    payload = report.to_dict()
    assert payload["drifts"][0]["post_rope_q"]["active_count"] == base.q.numel()
    json.dumps(payload)


def test_single_gpu_rope_attention_requires_position_metadata():
    base = _comparison_inputs()

    with pytest.raises(ValueError, match="rope_positions are required"):
        compare_single_gpu_rope_attention(AttentionComparisonInputs(q=base.q, k=base.k, v=base.v))


def test_decode_replay_matches_full_prefill_for_single_and_few_query():
    inputs = _decode_inputs()
    report = compare_decode_kv_replay(inputs)

    assert report.reference_name == "full_prefill_decode_reference"
    drift = report.drifts[0]
    assert drift.candidate_name == "rl_kernel_decode_kv_replay"
    assert drift.out.max_abs <= 1.0e-6
    assert drift.lse.max_abs <= 1.0e-6
    assert drift.dlogp is not None
    assert drift.dlogp.max_abs <= 3.0e-6
    assert drift.dlogp.active_count == 3
    assert drift.provenance["attention_mode"] == "decode"
    assert drift.provenance["cache_position"] == [[4, 5], [4, 5]]
    assert drift.provenance["cp_block_owners"] == [[0, 1, 0], [0, 1, 0]]
    assert drift.provenance["merge_order"] == "global_block_index"
    assert drift.provenance["accum_dtype"] == "fp32"
    assert drift.provenance["lse_dtype"] == "fp32"
    assert drift.provenance["downcast_at"] == "final_write"
    assert drift.provenance["output_dtype"] == "float32"
    assert drift.provenance["scale"] == pytest.approx(1.0 / (8.0**0.5))
    assert drift.provenance["q_dtype"] == "float32"
    assert drift.provenance["k_cache_dtype"] == "float32"
    assert drift.provenance["v_cache_dtype"] == "float32"
    assert drift.provenance["execution_scope"] == "single_device_logical_reference"
    assert drift.provenance["runtime_verified"] is False
    assert drift.provenance["supports_backward"] is False
    assert drift.provenance["communication"] == "none"
    assert drift.provenance["cache_execution_fingerprint"] == decode_kv_cache_fingerprint(inputs)
    assert drift.provenance["kv_cache_identity_fingerprint"] == decode_kv_cache_fingerprint(inputs)
    assert drift.provenance["rope_identity_fingerprint"] == decode_rope_identity_fingerprint(inputs)
    assert drift.provenance["preprocess_policy"] == "reference_only_not_production"
    assert drift.provenance["preprocess_fallback"] is True
    assert drift.provenance["logical_merge_orders"] == [
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    json.dumps(report.to_dict())

    single_query = DecodeAttentionInputs(
        q=inputs.q[:, :, -1:, :],
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        metadata=DecodeKVCacheMetadata(
            cache_position=inputs.metadata.cache_position[:, -1:],
            kv_seq_lens=inputs.metadata.kv_seq_lens,
            block_table=inputs.metadata.block_table,
            global_token_positions=inputs.metadata.global_token_positions,
            query_position_ids=inputs.metadata.query_position_ids[:, -1:],
            key_position_ids=inputs.metadata.key_position_ids,
            page_size=inputs.metadata.page_size,
            cp_block_owners=inputs.metadata.cp_block_owners,
            cp_world_size=inputs.metadata.cp_world_size,
        ),
    )
    single_report = compare_decode_kv_replay(single_query)
    assert single_report.drifts[0].out.max_abs <= 1.0e-6
    assert single_report.drifts[0].lse.max_abs <= 1.0e-6


def test_strict_decode_replay_is_bitwise_across_logical_and_paged_materialization():
    report = compare_decode_kv_replay(_decode_inputs(page_order=(2, 0, 1)), strict_bitwise=True)
    assert report.reference_name == "strict_shared_core_logical_prefill"
    assert len(report.drifts) == 1
    drift = report.drifts[0]
    assert drift.candidate_name == "strict_shared_core_paged_kv_layout"
    assert drift.out.max_abs == 0.0
    assert drift.lse.max_abs == 0.0
    assert drift.provenance["strict_core_id"] == ("rlkernel.attention.deterministic_core.v1")
    assert drift.provenance["strict_schedule"] == ("single_batch_single_query_global_kv_blocks")
    assert drift.provenance["split_kv_policy"] == "disabled"


def test_decode_replay_is_invariant_to_physical_page_and_prefix_layout():
    contiguous = run_decode_kv_replay(_decode_inputs())
    permuted_inputs = _decode_inputs(page_order=(2, 0, 1), prefix_cache_enabled=True)
    permuted = run_decode_kv_replay(permuted_inputs)

    torch.testing.assert_close(permuted.out, contiguous.out, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(permuted.lse, contiguous.lse, atol=1.0e-6, rtol=0.0)
    assert permuted.provenance["prefix_cache_enabled"] is True
    assert permuted.provenance["prefix_cache_key"] == "shared-prefix"
    assert permuted.provenance["prefix_length"] == 4
    assert permuted.provenance["prefix_cache_fingerprint"] == decode_prefix_cache_fingerprint(
        permuted_inputs,
        prefix_length=permuted_inputs.metadata.prefix_length,
    )


def test_decode_replay_supports_nonzero_global_position_origin():
    base = compare_decode_kv_replay(_decode_inputs())
    offset = compare_decode_kv_replay(_decode_inputs(position_offset=128))

    assert offset.drifts[0].out.max_abs <= 1.0e-6
    assert offset.drifts[0].lse.max_abs <= 1.0e-6
    assert base.drifts[0].out.max_abs <= 1.0e-6


def test_decode_cache_execution_fingerprint_is_layout_invariant_and_content_bound():
    contiguous = _decode_inputs()
    permuted = _decode_inputs(page_order=(2, 0, 1))

    assert decode_kv_cache_fingerprint(contiguous) == decode_kv_cache_fingerprint(permuted)
    assert run_decode_kv_replay(contiguous).provenance[
        "cache_execution_fingerprint"
    ] == decode_kv_cache_fingerprint(contiguous)

    changed_v = contiguous.v_cache.clone()
    first_slot = int(contiguous.metadata.block_table[0, 0].item()) * contiguous.metadata.page_size
    changed_v[0, 0, first_slot, 0] += 1.0
    changed = replace(contiguous, v_cache=changed_v)
    assert decode_kv_cache_fingerprint(changed) != decode_kv_cache_fingerprint(contiguous)

    contiguous_prefix = _decode_inputs(prefix_cache_enabled=True)
    permuted_prefix = _decode_inputs(page_order=(2, 0, 1), prefix_cache_enabled=True)
    assert contiguous_prefix.metadata.prefix_cache_fingerprint == (
        permuted_prefix.metadata.prefix_cache_fingerprint
    )


def test_decode_cache_fingerprint_ignores_inactive_physical_page_content():
    inputs = _decode_inputs()
    generator = torch.Generator().manual_seed(91)
    extra_k = torch.randn(
        inputs.k_cache.size(0),
        inputs.k_cache.size(1),
        inputs.metadata.page_size,
        inputs.k_cache.size(3),
        generator=generator,
    )
    extra_v = torch.randn(extra_k.shape, generator=generator)
    extended = replace(
        inputs,
        k_cache=torch.cat((inputs.k_cache, extra_k), dim=2),
        v_cache=torch.cat((inputs.v_cache, extra_v), dim=2),
        metadata=replace(
            inputs.metadata,
            block_table=torch.cat(
                (inputs.metadata.block_table, torch.full((2, 1), -1, dtype=torch.long)),
                dim=1,
            ),
            global_token_positions=torch.cat(
                (
                    inputs.metadata.global_token_positions,
                    torch.full((2, inputs.metadata.page_size), -1, dtype=torch.long),
                ),
                dim=1,
            ),
            key_position_ids=torch.cat(
                (
                    inputs.metadata.key_position_ids,
                    torch.full((2, inputs.metadata.page_size), -1, dtype=torch.long),
                ),
                dim=1,
            ),
            cp_block_owners=torch.cat(
                (
                    inputs.metadata.cp_block_owners,
                    torch.full((2, 1), -1, dtype=torch.long),
                ),
                dim=1,
            ),
        ),
    )
    changed_inactive = replace(
        extended,
        k_cache=extended.k_cache.clone(),
        v_cache=extended.v_cache.clone(),
    )
    changed_inactive.k_cache[:, :, -2:, :].fill_(123.0)
    changed_inactive.v_cache[:, :, -2:, :].fill_(-456.0)

    assert decode_kv_cache_fingerprint(extended) == decode_kv_cache_fingerprint(changed_inactive)
    extended_result = run_decode_kv_replay(extended)
    changed_result = run_decode_kv_replay(changed_inactive)
    torch.testing.assert_close(extended_result.out, changed_result.out)
    torch.testing.assert_close(extended_result.lse, changed_result.lse)


def test_decode_replay_supports_per_batch_lengths_and_partial_final_page():
    inputs = _decode_inputs()
    metadata = replace(
        inputs.metadata,
        cache_position=torch.tensor([[3, 4], [4, 5]], dtype=torch.long),
        query_position_ids=torch.tensor([[3, 4], [4, 5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([5, 6], dtype=torch.long),
        global_token_positions=inputs.metadata.global_token_positions.clone(),
        key_position_ids=inputs.metadata.key_position_ids.clone(),
    )
    first_unused_slot = int(metadata.block_table[0, 2].item()) * metadata.page_size + 1
    metadata.global_token_positions[0, first_unused_slot] = -1
    metadata.key_position_ids[0, first_unused_slot] = -1
    shortened = replace(inputs, metadata=metadata)

    report = compare_decode_kv_replay(shortened)

    assert report.drifts[0].out.max_abs <= 1.0e-6
    assert report.drifts[0].lse.max_abs <= 1.0e-6
    assert report.drifts[0].provenance["kv_seq_lens"] == [5, 6]


@pytest.mark.parametrize("field_name", ["q", "k_cache", "v_cache"])
def test_decode_replay_rejects_non_finite_active_values(field_name):
    inputs = _decode_inputs()
    tensor = getattr(inputs, field_name).clone()
    tensor[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        run_decode_kv_replay(replace(inputs, **{field_name: tensor}))


def test_decode_replay_rejects_finite_inputs_that_overflow_attention():
    inputs = _decode_inputs()
    huge = torch.full_like(inputs.q, torch.finfo(inputs.q.dtype).max)

    with pytest.raises(ValueError, match="produced non-finite"):
        compare_decode_kv_replay(replace(inputs, q=huge))


def test_decode_replay_rejects_prefix_cache_key_drift():
    inputs = _decode_inputs(prefix_cache_enabled=True)
    changed_key = replace(
        inputs,
        metadata=replace(inputs.metadata, prefix_cache_key="different-prefix"),
    )

    with pytest.raises(ValueError, match="prefix_cache_fingerprint"):
        run_decode_kv_replay(changed_key)


def test_decode_replay_is_invariant_to_equivalent_cp_block_ownership():
    cp2_inputs = _decode_inputs()
    cp1_inputs = replace(
        cp2_inputs,
        metadata=replace(
            cp2_inputs.metadata,
            cp_block_owners=torch.zeros_like(cp2_inputs.metadata.cp_block_owners),
        ),
    )

    cp1 = run_decode_kv_replay(cp1_inputs)
    cp2 = run_decode_kv_replay(cp2_inputs)
    torch.testing.assert_close(cp2.out, cp1.out, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(cp2.lse, cp1.lse, atol=1.0e-6, rtol=0.0)
    assert cp1.provenance["cp_block_owners"] == [[0, 0, 0], [0, 0, 0]]
    assert cp2.provenance["cp_block_owners"] == [[0, 1, 0], [0, 1, 0]]


def test_decode_replay_pre_rope_cache_matches_equivalent_post_rope_cache():
    pre_rope = _decode_inputs(q_rope_state="pre_rope", k_cache_rope_state="pre_rope")
    rope = NativeRoPEOp()
    post_q = rope.forward_fp32(
        pre_rope.q,
        pre_rope.metadata.query_position_ids,
        theta=pre_rope.rope_theta,
    )
    post_k = rope.forward_fp32(
        pre_rope.k_cache,
        pre_rope.metadata.key_position_ids,
        theta=pre_rope.rope_theta,
    )
    post_rope = replace(
        pre_rope,
        q=post_q,
        k_cache=post_k,
        metadata=replace(
            pre_rope.metadata,
            q_rope_state="post_rope",
            k_cache_rope_state="post_rope",
        ),
    )

    pre_result = run_decode_kv_replay(pre_rope)
    post_result = run_decode_kv_replay(post_rope)
    torch.testing.assert_close(post_result.out, pre_result.out, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(post_result.lse, pre_result.lse, atol=1.0e-6, rtol=0.0)


def test_decode_replay_preserves_separate_q_and_k_rope_output_dtypes():
    base = _decode_inputs(q_rope_state="pre_rope", k_cache_rope_state="pre_rope")
    mixed = replace(
        base,
        k_cache=base.k_cache.to(torch.bfloat16),
        v_cache=base.v_cache.to(torch.bfloat16),
    )
    rope = NativeRoPEOp()
    post = replace(
        mixed,
        q=rope.forward_fp32(
            mixed.q,
            mixed.metadata.query_position_ids,
            theta=mixed.rope_theta,
        ).to(torch.float32),
        k_cache=rope.forward_fp32(
            mixed.k_cache,
            mixed.metadata.key_position_ids,
            theta=mixed.rope_theta,
        ).to(torch.bfloat16),
        metadata=replace(
            mixed.metadata,
            q_rope_state="post_rope",
            k_cache_rope_state="post_rope",
        ),
    )

    pre_result = run_decode_kv_replay(mixed)
    post_result = run_decode_kv_replay(post)
    torch.testing.assert_close(post_result.out, pre_result.out, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(post_result.lse, pre_result.lse, atol=1.0e-6, rtol=0.0)
    assert pre_result.provenance["q_rope_output_dtype"] == "float32"
    assert pre_result.provenance["k_cache_rope_output_dtype"] == "bfloat16"


def test_decode_replay_rejects_stale_prefix_cache_content():
    inputs = _decode_inputs(prefix_cache_enabled=True)
    stale_k = inputs.k_cache.clone()
    first_prefix_slot = int(inputs.metadata.block_table[0, 0].item()) * inputs.metadata.page_size
    stale_k[0, 0, first_prefix_slot, 0] += 1.0

    with pytest.raises(ValueError, match="prefix_cache_fingerprint"):
        run_decode_kv_replay(replace(inputs, k_cache=stale_k))


def test_decode_replay_rejects_prefix_cache_theta_drift():
    inputs = _decode_inputs(prefix_cache_enabled=True)

    with pytest.raises(ValueError, match="prefix_cache_fingerprint"):
        run_decode_kv_replay(replace(inputs, rope_theta=10_000.0))


def test_decode_replay_rejects_prefix_cache_storage_dtype_drift():
    inputs = _decode_inputs(prefix_cache_enabled=True)

    with pytest.raises(ValueError, match="prefix_cache_fingerprint"):
        run_decode_kv_replay(replace(inputs, k_cache=inputs.k_cache.to(torch.bfloat16)))


def test_decode_replay_rejects_unsupported_rope_cast_boundary():
    inputs = _decode_inputs(prefix_cache_enabled=True)

    with pytest.raises(ValueError, match="rope_cast_at"):
        run_decode_kv_replay(replace(inputs, rope_cast_at="before_rope"))


@pytest.mark.parametrize("theta", [float("nan"), float("inf"), float("-inf")])
def test_decode_replay_rejects_non_finite_rope_theta(theta):
    with pytest.raises(ValueError, match="rope_theta"):
        run_decode_kv_replay(replace(_decode_inputs(), rope_theta=theta))


def test_decode_replay_rejects_noncanonical_inactive_page_metadata():
    inputs = _decode_inputs()
    bad_metadata = replace(
        inputs.metadata,
        block_table=torch.cat(
            [inputs.metadata.block_table, torch.zeros((2, 1), dtype=torch.long)], dim=1
        ),
        cp_block_owners=torch.cat(
            [inputs.metadata.cp_block_owners, torch.full((2, 1), -1, dtype=torch.long)], dim=1
        ),
    )

    with pytest.raises(ValueError, match="unused block_table"):
        run_decode_kv_replay(replace(inputs, metadata=bad_metadata))


def test_decode_replay_rejects_negative_active_cp_owner():
    inputs = _decode_inputs()
    bad_owners = inputs.metadata.cp_block_owners.clone()
    bad_owners[0, 1] = -1

    with pytest.raises(ValueError, match="active cp_block_owners"):
        run_decode_kv_replay(
            replace(inputs, metadata=replace(inputs.metadata, cp_block_owners=bad_owners))
        )


def test_decode_replay_rejects_noncanonical_inactive_cp_owner():
    inputs = _decode_inputs()
    bad_metadata = replace(
        inputs.metadata,
        block_table=torch.cat(
            [inputs.metadata.block_table, torch.full((2, 1), -1, dtype=torch.long)], dim=1
        ),
        cp_block_owners=torch.cat(
            [inputs.metadata.cp_block_owners, torch.zeros((2, 1), dtype=torch.long)], dim=1
        ),
    )

    with pytest.raises(ValueError, match="unused cp_block_owners"):
        run_decode_kv_replay(replace(inputs, metadata=bad_metadata))


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -1.0])
def test_decode_replay_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        run_decode_kv_replay(replace(_decode_inputs(), scale=scale))


def test_decode_replay_rejects_non_floating_cache_dtypes():
    inputs = _decode_inputs()
    with pytest.raises(ValueError, match="floating-point dtypes"):
        run_decode_kv_replay(replace(inputs, q=inputs.q.to(torch.int32)))


def test_decode_replay_fails_loudly_on_position_identity_mismatch():
    inputs = _decode_inputs()
    bad_query_positions = inputs.metadata.query_position_ids.clone()
    bad_query_positions[0, -1] = 4
    bad_metadata = DecodeKVCacheMetadata(
        cache_position=inputs.metadata.cache_position,
        kv_seq_lens=inputs.metadata.kv_seq_lens,
        block_table=inputs.metadata.block_table,
        global_token_positions=inputs.metadata.global_token_positions,
        query_position_ids=bad_query_positions,
        key_position_ids=inputs.metadata.key_position_ids,
        page_size=inputs.metadata.page_size,
        cp_block_owners=inputs.metadata.cp_block_owners,
        cp_world_size=inputs.metadata.cp_world_size,
    )

    with pytest.raises(ValueError, match="cache_position and query_position_ids"):
        run_decode_kv_replay(
            DecodeAttentionInputs(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                metadata=bad_metadata,
            )
        )


def test_decode_replay_fails_loudly_on_invalid_page_identity():
    inputs = _decode_inputs()
    bad_positions = inputs.metadata.global_token_positions.clone()
    bad_positions[:, 0] = 1
    bad_metadata = DecodeKVCacheMetadata(
        cache_position=inputs.metadata.cache_position,
        kv_seq_lens=inputs.metadata.kv_seq_lens,
        block_table=inputs.metadata.block_table,
        global_token_positions=bad_positions,
        query_position_ids=inputs.metadata.query_position_ids,
        key_position_ids=bad_positions.clone(),
        page_size=inputs.metadata.page_size,
        cp_block_owners=inputs.metadata.cp_block_owners,
        cp_world_size=inputs.metadata.cp_world_size,
    )

    with pytest.raises(ValueError, match="reconstruct logical positions"):
        compare_decode_kv_replay(
            DecodeAttentionInputs(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                metadata=bad_metadata,
            )
        )


def test_decode_replay_covers_qwen3_gqa_head_layout():
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(1, 32, 1, 128, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4, 128, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 4, 128, generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
    inputs = DecodeAttentionInputs(
        q=q,
        k_cache=k,
        v_cache=v,
        metadata=DecodeKVCacheMetadata(
            cache_position=torch.tensor([[3]], dtype=torch.long),
            kv_seq_lens=torch.tensor([4], dtype=torch.long),
            block_table=torch.tensor([[0, 1]], dtype=torch.long),
            global_token_positions=positions,
            query_position_ids=torch.tensor([[3]], dtype=torch.long),
            key_position_ids=positions.clone(),
            page_size=2,
            cp_block_owners=torch.tensor([[0, 1]], dtype=torch.long),
            cp_world_size=2,
        ),
        output_dtype=torch.bfloat16,
    )

    report = compare_decode_kv_replay(inputs)
    assert report.drifts[0].out.max_abs <= 2 * torch.finfo(torch.bfloat16).eps
    assert report.drifts[0].lse.max_abs <= 1.0e-6


def test_decode_append_matches_full_prefill_suffix():
    generator = torch.Generator().manual_seed(71)
    q = torch.randn(1, 4, 2, 8, generator=generator)
    k_past = torch.randn(1, 2, 4, 8, generator=generator)
    v_past = torch.randn(1, 2, 4, 8, generator=generator)
    k_new = torch.randn(1, 2, 2, 8, generator=generator)
    v_new = torch.randn(1, 2, 2, 8, generator=generator)
    inputs = DecodeAttentionInputs(
        q=q,
        k_cache=k_past,
        v_cache=v_past,
        k_new=k_new,
        v_new=v_new,
        metadata=DecodeKVCacheMetadata(
            cache_position=torch.tensor([[104, 105]], dtype=torch.long),
            kv_seq_lens=torch.tensor([4], dtype=torch.long),
            block_table=torch.tensor([[0, 1]], dtype=torch.long),
            global_token_positions=torch.tensor([[100, 101, 102, 103]], dtype=torch.long),
            query_position_ids=torch.tensor([[104, 105]], dtype=torch.long),
            key_position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long),
            page_size=2,
            cp_block_owners=torch.tensor([[0, 1]], dtype=torch.long),
            cp_world_size=2,
        ),
        split_kv=SplitKVSpec.fixed(2),
    )

    report = compare_decode_kv_replay(inputs)
    drift = report.drifts[0]
    assert drift.out.max_abs <= 1.0e-6
    assert drift.lse.max_abs <= 1.0e-6
    assert drift.provenance["decode_semantics"] == "past_kv_plus_new_kv_append"
    assert drift.provenance["past_kv_lengths"] == [4]
    assert drift.provenance["new_kv_length"] == 2
    assert drift.provenance["actual_split_kv_plans"][0][1]["actual_split_boundaries"] == [
        [0, 2],
        [2, 4],
        [4, 6],
    ]


def test_decode_replay_supports_nonzero_global_position_offset():
    base = _decode_inputs()
    offset = 4096
    active = base.metadata.global_token_positions >= 0
    positions = torch.where(
        active,
        base.metadata.global_token_positions + offset,
        base.metadata.global_token_positions,
    )
    inputs = replace(
        base,
        metadata=replace(
            base.metadata,
            cache_position=base.metadata.cache_position + offset,
            query_position_ids=base.metadata.query_position_ids + offset,
            global_token_positions=positions,
            key_position_ids=positions.clone(),
        ),
    )

    report = compare_decode_kv_replay(inputs)
    assert report.drifts[0].out.max_abs <= 1.0e-6
    assert report.drifts[0].provenance["global_token_positions"][0][0] >= offset


def test_decode_split_k_disabled_and_fixed_share_cache_layout():
    base = _decode_inputs()
    disabled = run_decode_kv_replay(replace(base, split_kv=SplitKVSpec.disabled()))
    fixed = run_decode_kv_replay(replace(base, split_kv=SplitKVSpec.fixed(2)))

    torch.testing.assert_close(fixed.out, disabled.out, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(fixed.lse, disabled.lse, atol=1.0e-6, rtol=0.0)
    assert disabled.provenance["requested_split_kv_policy"] == "disabled"
    assert fixed.provenance["requested_split_kv_policy"] == "fixed"
    assert disabled.provenance["block_table"] == fixed.provenance["block_table"]


def test_decode_transformer_engine_oracle_reuses_sorted_partial_states(monkeypatch):
    calls = {"lse": 0, "out": 0}

    def lse_correction(softmax_lse, softmax_lse_per_step):
        calls["lse"] += 1
        softmax_lse.copy_(torch.logaddexp(softmax_lse, softmax_lse_per_step))

    def out_correction_init(out_init_step, softmax_lse, softmax_lse_init_step, seq_dim):
        scale = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
        return out_init_step * scale.unsqueeze(-1)

    def out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step, seq_dim):
        calls["out"] += 1
        scale = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
        out.add_(out_per_step * scale.unsqueeze(-1))

    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lse_correction,
            flash_attn_fwd_out_correction_init=out_correction_init,
            flash_attn_fwd_out_correction=out_correction,
        ),
    )

    report = compare_decode_kv_replay(
        _decode_inputs(page_order=(2, 0, 1)),
        include_transformer_engine=True,
    )

    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert set(by_name) == {
        "rl_kernel_decode_kv_replay",
        "transformer_engine_decode_kv_replay",
    }
    assert by_name["transformer_engine_decode_kv_replay"].out.max_abs <= 1.0e-6
    assert by_name["transformer_engine_decode_kv_replay"].lse.max_abs <= 1.0e-6
    assert by_name["transformer_engine_decode_kv_replay"].provenance["logical_merge_orders"] == [
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    assert calls["lse"] > 0
    assert calls["out"] > 0
    assert report.unavailable == ()


def test_decode_transformer_engine_unavailable_is_reported(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == _TE_CONTEXT_PARALLEL_MODULE:
            raise ImportError("decode TE unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    report = compare_decode_kv_replay(
        _decode_inputs(),
        include_transformer_engine=True,
    )

    assert {drift.candidate_name for drift in report.drifts} == {"rl_kernel_decode_kv_replay"}
    assert report.unavailable == ("transformer_engine_decode_kv_replay: decode TE unavailable",)


def test_transformer_engine_merge_oracle_can_be_reused_when_available(monkeypatch):
    calls = {"lse": 0, "out": 0}

    def lse_correction(softmax_lse, softmax_lse_per_step):
        calls["lse"] += 1
        softmax_lse.copy_(torch.logaddexp(softmax_lse, softmax_lse_per_step))

    def out_correction_init(out_init_step, softmax_lse, softmax_lse_init_step, seq_dim):
        scale = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
        return out_init_step * scale.unsqueeze(-1)

    def out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step, seq_dim):
        calls["out"] += 1
        scale = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
        out.add_(out_per_step * scale.unsqueeze(-1))

    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lse_correction,
            flash_attn_fwd_out_correction_init=out_correction_init,
            flash_attn_fwd_out_correction=out_correction,
        ),
    )

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert "transformer_engine_paged_kv" in by_name
    assert by_name["transformer_engine_paged_kv"].out.max_abs <= 1.0e-6
    assert by_name["transformer_engine_paged_kv"].lse.max_abs <= 1.0e-6
    provenance = by_name["transformer_engine_paged_kv"].provenance
    assert provenance["te_available"] is True
    assert provenance["te_module"] == _TE_CONTEXT_PARALLEL_MODULE
    assert provenance["te_capability_probe"] == "passed"
    assert provenance["te_signature_checked"] is True
    assert provenance["te_numeric_selftest"] == "passed"
    assert provenance["actual_backend"] == "te_context_parallel_merge_helpers"
    assert provenance["actual_backend_source"] == "rl_kernel_te_context_parallel_adapter"
    assert provenance["accum_dtype"] == "fp32"
    assert provenance["downcast_at"] == "final_write"
    assert calls["lse"] > 0
    assert calls["out"] > 0
    assert report.unavailable == ()


def test_transformer_engine_merge_oracle_keeps_all_masked_rows_stable(monkeypatch):
    def lse_correction(softmax_lse, softmax_lse_per_step):
        max_scale = torch.max(softmax_lse, softmax_lse_per_step)
        min_scale = torch.min(softmax_lse, softmax_lse_per_step)
        softmax_lse.copy_(max_scale + torch.log1p(torch.exp(min_scale - max_scale)))

    def out_correction_init(out_init_step, softmax_lse, softmax_lse_init_step, seq_dim):
        scale = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
        return out_init_step * scale.unsqueeze(-1)

    def out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step, seq_dim):
        scale = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
        out.add_(out_per_step * scale.unsqueeze(-1))

    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lse_correction,
            flash_attn_fwd_out_correction_init=out_correction_init,
            flash_attn_fwd_out_correction=out_correction,
        ),
    )

    q, k, v = _qkv(seed=11)
    inputs = AttentionComparisonInputs(
        q=q,
        k=k,
        v=v,
        causal=False,
        key_padding_mask=torch.zeros(q.size(0), k.size(2), dtype=torch.bool),
    )

    te_result = run_paged_kv_attention(
        inputs,
        kv_page_size=2,
        merge_backend="transformer_engine",
    )
    assert torch.equal(te_result.out, torch.zeros_like(te_result.out))
    assert torch.isneginf(te_result.lse).all()

    report = compare_single_gpu_attention(
        inputs,
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )
    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert by_name["transformer_engine_paged_kv"].out.max_abs == 0.0
    assert by_name["transformer_engine_paged_kv"].lse.max_abs == 0.0
    assert report.unavailable == ()


def test_transformer_engine_path_reports_unavailable_without_failing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == _TE_CONTEXT_PARALLEL_MODULE:
            raise ImportError("test TE unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    assert {drift.candidate_name for drift in report.drifts} == {
        "chunked_prefill",
        "rl_kernel_paged_kv",
    }
    assert report.unavailable == ("transformer_engine_paged_kv: test TE unavailable",)


def test_transformer_engine_path_reports_missing_helpers(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lambda softmax_lse, per_step: None,
        ),
    )

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    assert len(report.unavailable) == 1
    assert "missing required helpers" in report.unavailable[0]


def test_transformer_engine_path_reports_incompatible_helper_signature(monkeypatch):
    def lse_correction(wrong_name, softmax_lse_per_step):
        wrong_name.copy_(torch.logaddexp(wrong_name, softmax_lse_per_step))

    def out_correction_init(out_init_step, softmax_lse, softmax_lse_init_step, seq_dim):
        scale = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
        return out_init_step * scale.unsqueeze(-1)

    def out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step, seq_dim):
        scale = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
        out.add_(out_per_step * scale.unsqueeze(-1))

    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lse_correction,
            flash_attn_fwd_out_correction_init=out_correction_init,
            flash_attn_fwd_out_correction=out_correction,
        ),
    )

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    assert len(report.unavailable) == 1
    assert "incompatible signature" in report.unavailable[0]


def test_operator_comparison_specs_register_attention():
    args = argparse.Namespace(
        op="attention",
        candidate="pytorch",
        arch_key=None,
        batch=1,
        seq=3,
        vocab=17,
        seed=7,
        input_mode="random",
        constant_value=0.5,
        token_value=3,
        normalized_dim=128,
        k_dim=16,
        n_dim=32,
        theta=1.0e6,
        eps=1.0e-6,
    )

    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    candidate = make_candidate(args)
    report = run_operator_suite("attention", candidates=[candidate], cases=[case])

    assert report.passed
    assert report.candidates[0].cases[0].op_class == "attention"
