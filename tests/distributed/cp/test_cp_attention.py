# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the WS2 deterministic CP attention reference.

The implementation is a correctness-first prefill/chunked-prefill reference:
local KV blocks produce ``(out, lse)`` partial states and CP merges those states
with fp32 online-softmax arithmetic in logical global-block order.
"""

import contextlib
import json
import math

import pytest
import torch

from rl_engine.backends.cuda.attention import deterministic_attn as deterministic_attn_module
from rl_engine.backends.cuda.attention.deterministic_attn import RLKernelDeterministicAttentionCore
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    SplitKVSpec,
)
from rl_engine.reference.attention.cp_attention import (
    AttentionPartialState,
    AttentionRingSchedule,
    AttentionSavedForwardState,
    DeterministicAttentionCore,
    DeterministicCPAttentionReferenceOp,
    compare_cp_attention_backward,
    merge_attention_partial_states,
    split_kv_execution_plan_provenance,
)
from rl_engine.reference.attention.standard_attn import NativeAttentionOp
from rl_engine.reference.rope.rope import NativeRoPEOp
from rl_engine.runtime.registry import kernel_registry

_N_HEADS = 32
_N_KV = 8
_HEAD_DIM = 128
_ATOL = 3.0e-6
_GRAD_ATOL = 1.0e-5


def test_ring_schedule_separates_compute_and_merge_order():
    schedule = AttentionRingSchedule.build(12, cp_world_size=2, kv_chunk_size=2)

    assert schedule.schedule_id == "rlkernel.attention.strict_ring_state.v1"
    assert schedule.compute_communication == "decoupled"
    assert schedule.overlap == "disabled"
    assert schedule.merge_order == tuple(range(6))
    assert schedule.compute_order == (0, 5, 1, 4, 2, 3)
    assert [block.owner_cp_rank for block in schedule.blocks] == [0, 0, 0, 1, 1, 1]


@contextlib.contextmanager
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _qkv(
    batch,
    sq,
    skv,
    *,
    seed,
    dtype=torch.float32,
    heads=_N_HEADS,
    kv_heads=_N_KV,
    dim=_HEAD_DIM,
):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, sq, dim, generator=gen, dtype=dtype)
    k = torch.randn(batch, kv_heads, skv, dim, generator=gen, dtype=dtype)
    v = torch.randn(batch, kv_heads, skv, dim, generator=gen, dtype=dtype)
    return q, k, v


def _full_lse(q, k, *, causal, scale=None, key_padding_mask=None):
    qf, kf = q.float(), k.float()
    hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
    hkv, skv = kf.shape[1], kf.shape[2]
    if hq % hkv != 0:
        raise ValueError("invalid GQA shape")
    if hq != hkv:
        kf = kf.repeat_interleave(hq // hkv, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (
        scale if scale is not None else 1.0 / math.sqrt(dim)
    )
    if causal:
        query_pos = torch.arange(skv - sq, skv)
        key_pos = torch.arange(skv)
        scores = scores.masked_fill(
            (key_pos[None, :] > query_pos[:, None])[None, None, :, :],
            float("-inf"),
        )
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    return torch.logsumexp(scores, dim=-1)


def test_strict_attention_core_freezes_plan_and_final_write():
    q, k, v = _qkv(1, 3, 4, seed=17, dtype=torch.bfloat16)
    core = DeterministicAttentionCore()
    result = core.forward_with_lse(q, k, v, output_dtype=torch.bfloat16)

    assert result.out.dtype is torch.bfloat16
    assert result.lse.dtype is torch.float32
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_SCHEDULE_ID
    assert result.provenance["merge_order"] == "global_block_index"
    assert result.provenance["accum_dtype"] == "fp32"
    assert result.provenance["downcast_at"] == "final_write"
    assert result.provenance["fallback"] is False
    assert result.provenance["native_attention_arithmetic"] is False


@pytest.mark.parametrize("split_kv", [SplitKVSpec.fixed(2), SplitKVSpec.auto()])
def test_strict_attention_core_rejects_split_kv(split_kv):
    with pytest.raises(ValueError, match="requires Split-KV to be disabled"):
        DeterministicAttentionCore(split_kv=split_kv)


def test_cp1_matches_native_attention_and_exports_lse():
    op = DeterministicCPAttentionReferenceOp()
    native = NativeAttentionOp()
    q, k, v = _qkv(2, 8, 8, seed=1)

    with _single_thread():
        out, lse = op.forward_fp32_with_lse(q, k, v, causal=True, cp_world_size=1)
        want = native.forward_fp32(q, k, v, causal=True)
        want_lse = _full_lse(q, k, causal=True)

    torch.testing.assert_close(out, want, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(lse, want_lse, atol=_ATOL, rtol=0.0)
    assert lse.dtype == torch.float32
    assert lse.shape == q.shape[:3]


def test_cp2_prefill_matches_cp1_reference():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(2, 9, 9, seed=2)

    with _single_thread():
        out1, lse1 = op.forward_fp32_with_lse(q, k, v, causal=True, cp_world_size=1)
        out2, lse2 = op.forward_fp32_with_lse(q, k, v, causal=True, cp_world_size=2)

    torch.testing.assert_close(out2, out1, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(lse2, lse1, atol=_ATOL, rtol=0.0)


def test_cp2_consumes_post_rope_qk_with_shared_global_position_metadata():
    op = DeterministicCPAttentionReferenceOp()
    rope = NativeRoPEOp()
    pre_rope_q, pre_rope_k, v = _qkv(2, 7, 7, seed=14, heads=4, kv_heads=2, dim=8)
    position_offsets = torch.tensor([17, 103], dtype=torch.long)
    positions = position_offsets[:, None] + torch.arange(pre_rope_q.size(2), dtype=torch.long)
    q = rope.forward_fp32(pre_rope_q, positions, theta=1_000_000.0)
    k = rope.forward_fp32(pre_rope_k, positions, theta=1_000_000.0)

    assert not torch.equal(q, pre_rope_q.float())
    assert not torch.equal(k, pre_rope_k.float())

    with _single_thread():
        out1, lse1 = op.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=True,
            query_position_offsets=position_offsets,
            key_position_offsets=position_offsets,
            cp_world_size=1,
        )
        out2, lse2 = op.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=True,
            query_position_offsets=position_offsets,
            key_position_offsets=position_offsets,
            cp_world_size=2,
            kv_chunk_size=2,
        )

    torch.testing.assert_close(out2, out1, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(lse2, lse1, atol=_ATOL, rtol=0.0)


def test_chunked_prefill_replay_matches_unchunked_cp2():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(2, 10, 10, seed=3)

    with _single_thread():
        unchunked_out, unchunked_lse = op.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=True,
            cp_world_size=2,
        )
        chunked_out, chunked_lse = op.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=True,
            cp_world_size=2,
            kv_chunk_size=3,
        )

    torch.testing.assert_close(chunked_out, unchunked_out, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(chunked_lse, unchunked_lse, atol=_ATOL, rtol=0.0)


def test_causal_mask_uses_global_positions_across_cp_boundary():
    op = DeterministicCPAttentionReferenceOp()
    batch, heads, kv_heads, seq, dim = 1, 2, 1, 5, 3
    q = torch.zeros(batch, heads, seq, dim)
    k = torch.zeros(batch, kv_heads, seq, dim)
    v = torch.arange(seq * dim, dtype=torch.float32).reshape(1, 1, seq, dim)
    out = op.forward_fp32(q, k, v, causal=True, cp_world_size=2)

    expected = torch.stack([v[0, 0, : index + 1].mean(dim=0) for index in range(seq)])
    expected = expected.reshape(1, 1, seq, dim).repeat(1, heads, 1, 1)
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=0.0)


def test_position_offsets_apply_varlen_causal_metadata_per_batch_row():
    op = DeterministicCPAttentionReferenceOp()
    q = torch.zeros(2, 2, 2, 1)
    k = torch.zeros(2, 1, 4, 1)
    v = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4, 1)

    out, lse = op.forward_fp32_with_lse(
        q,
        k,
        v,
        causal=True,
        query_position_offsets=torch.tensor([0, 11]),
        key_position_offsets=torch.tensor([0, 10]),
        cp_world_size=2,
        kv_chunk_size=1,
    )

    expected = torch.tensor([0.0, 0.5, 4.5, 5.0]).reshape(2, 1, 2, 1).repeat(1, 2, 1, 1)
    expected_lse = torch.log(torch.tensor([1.0, 2.0, 2.0, 3.0])).reshape(2, 1, 2)
    expected_lse = expected_lse.repeat(1, 2, 1)
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(lse, expected_lse, atol=1.0e-6, rtol=0.0)


def test_merge_order_uses_global_block_index_not_arrival_order():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 6, 6, seed=4)
    first = op.local_partial_state(
        q,
        k[:, :, :3],
        v[:, :, :3],
        q_start=0,
        k_start=0,
        total_kv_len=6,
        causal=True,
    )
    second = op.local_partial_state(
        q,
        k[:, :, 3:],
        v[:, :, 3:],
        q_start=0,
        k_start=3,
        total_kv_len=6,
        causal=True,
    )

    forward = merge_attention_partial_states([first, second])
    reversed_arrival = merge_attention_partial_states([second, first])
    assert torch.equal(forward.out, reversed_arrival.out)
    assert torch.equal(forward.lse, reversed_arrival.lse)


def test_key_padding_mask_and_all_masked_rows_are_stable():
    op = DeterministicCPAttentionReferenceOp()
    native = NativeAttentionOp()
    q, k, v = _qkv(2, 6, 6, seed=5)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    with _single_thread():
        out, lse = op.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=False,
            key_padding_mask=mask,
            cp_world_size=2,
            kv_chunk_size=2,
        )
        want = native.forward_fp32(q, k, v, causal=False, key_padding_mask=mask)

    torch.testing.assert_close(out[:1], want[:1], atol=_ATOL, rtol=0.0)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isneginf(lse[1]).all()
    assert torch.isfinite(out).all()


def test_empty_query_and_empty_kv_edges_are_stable():
    op = DeterministicCPAttentionReferenceOp()
    q_empty = torch.randn(1, 2, 0, 4, requires_grad=True)
    k_empty = torch.randn(1, 1, 0, 4, requires_grad=True)
    v_empty = torch.randn(1, 1, 0, 4, requires_grad=True)
    out, lse = op.forward_fp32_with_lse(q_empty, k_empty, v_empty, cp_world_size=2)
    assert out.shape == (1, 2, 0, 4)
    assert lse.shape == (1, 2, 0)
    assert out.requires_grad
    out.sum().backward()
    assert torch.equal(q_empty.grad, torch.zeros_like(q_empty))
    assert torch.equal(k_empty.grad, torch.zeros_like(k_empty))
    assert torch.equal(v_empty.grad, torch.zeros_like(v_empty))

    q = torch.randn(1, 2, 3, 4)
    out, lse = op.forward_fp32_with_lse(q, k_empty, v_empty, causal=False, cp_world_size=4)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()


def test_empty_kv_backward_returns_zero_grads():
    op = DeterministicCPAttentionReferenceOp()
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k = torch.randn(1, 1, 0, 4, requires_grad=True)
    v = torch.randn(1, 1, 0, 4, requires_grad=True)

    out = op.forward_fp32(q, k, v, causal=False, cp_world_size=4)
    assert out.requires_grad
    out.sum().backward()

    assert torch.equal(q.grad, torch.zeros_like(q))
    assert torch.equal(k.grad, torch.zeros_like(k))
    assert torch.equal(v.grad, torch.zeros_like(v))


def test_bf16_forward_uses_fp32_merge_then_final_write():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(2, 8, 8, seed=6, dtype=torch.bfloat16)

    out, lse = op.forward_with_lse(q, k, v, causal=True, cp_world_size=2, kv_chunk_size=2)
    fp32_out, fp32_lse = op.forward_fp32_with_lse(
        q,
        k,
        v,
        causal=True,
        cp_world_size=2,
        kv_chunk_size=2,
    )

    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert torch.equal(out, fp32_out.to(torch.bfloat16))
    assert torch.equal(lse, fp32_lse)


def test_cp2_chunked_gradients_match_cp1_reference():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 5, 5, seed=12, heads=4, kv_heads=2, dim=8)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_cp = q.detach().clone().requires_grad_(True)
    k_cp = k.detach().clone().requires_grad_(True)
    v_cp = v.detach().clone().requires_grad_(True)
    gen = torch.Generator().manual_seed(13)
    dy = torch.randn(1, 4, 5, 8, generator=gen)

    with _single_thread():
        out_ref = op.forward_fp32(q_ref, k_ref, v_ref, causal=True, cp_world_size=1)
        out_cp = op.forward_fp32(
            q_cp,
            k_cp,
            v_cp,
            causal=True,
            cp_world_size=2,
            kv_chunk_size=2,
        )
        out_ref.backward(dy)
        out_cp.backward(dy)

    torch.testing.assert_close(out_cp, out_ref, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(q_cp.grad, q_ref.grad, atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(k_cp.grad, k_ref.grad, atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(v_cp.grad, v_ref.grad, atol=1.0e-5, rtol=0.0)


def test_backward_report_cp2_prefill_matches_cp1_reference():
    q, k, v = _qkv(1, 5, 5, seed=15, heads=4, kv_heads=2, dim=8)
    dout = torch.randn(1, 4, 5, 8, generator=torch.Generator().manual_seed(16))

    with _single_thread():
        report = compare_cp_attention_backward(
            q,
            k,
            v,
            dout,
            causal=True,
            candidate_cp_world_size=2,
            output_dtype=torch.float32,
        )

    assert report.reference_name == "cp1_backward_reference"
    drift = report.drifts[0]
    assert drift.candidate_name == "cp2_backward"
    assert drift.dq.max_abs <= _GRAD_ATOL
    assert drift.dk.max_abs <= _GRAD_ATOL
    assert drift.dv.max_abs <= _GRAD_ATOL
    assert drift.out.max_abs <= _ATOL
    assert drift.lse.max_abs <= _ATOL
    assert len(drift.per_rank) == 2
    assert drift.per_rank[0].dq.active_count > 0
    assert drift.per_rank[1].dk.active_count > 0
    assert drift.provenance["saved_forward_state"][0] == "out"
    assert drift.provenance["merge_order"] == "global_block_index"
    assert drift.provenance["te_backward_oracle"] == "not_used"
    assert drift.provenance["decode_backward"] == "not_supported"
    assert drift.provenance["projection_scope"] == "attention_core_only"
    assert drift.provenance["qkv_projection_backward_dgrad_collective"] == "all_reduce"
    assert drift.provenance["qkv_projection_sp_backward_collective"] == "reduce_scatter"
    assert drift.provenance["o_proj_backward_dgrad_collective"] == "none"
    assert drift.provenance["o_proj_sp_backward_collective"] == "all_gather"
    assert drift.provenance["projection_collectives_executed"] is False
    json.dumps(report.to_dict())


def test_backward_report_cp2_chunked_prefill_matches_cp1_reference():
    q, k, v = _qkv(1, 6, 6, seed=17, heads=4, kv_heads=2, dim=8)
    dout = torch.randn(1, 4, 6, 8, generator=torch.Generator().manual_seed(18))

    with _single_thread():
        report = compare_cp_attention_backward(
            q,
            k,
            v,
            dout,
            causal=True,
            candidate_cp_world_size=2,
            candidate_kv_chunk_size=2,
            output_dtype=torch.float32,
        )

    drift = report.drifts[0]
    assert drift.candidate_name == "cp2_chunked_backward"
    assert drift.dq.max_abs <= _GRAD_ATOL
    assert drift.dk.max_abs <= _GRAD_ATOL
    assert drift.dv.max_abs <= _GRAD_ATOL
    assert drift.provenance["attention_mode"] == "chunked_prefill"
    assert drift.provenance["kv_chunk_size"] == 2
    assert drift.provenance["requested_split_kv_policy"] == "fixed"
    assert drift.provenance["actual_split_kv_plans"] == [
        {
            "owner_cp_rank": 0,
            "requested_split_kv_policy": "fixed",
            "requested_split_kv_size": 2,
            "actual_split_kv_policy": "fixed",
            "actual_split_kv_size": 2,
            "actual_split_kv_count": 2,
            "actual_split_boundaries": [[0, 2], [2, 3]],
            "split_kv_merge_order": "global_block_index",
            "split_kv_accum_dtype": "fp32",
            "split_kv_downcast_at": "final_write",
            "split_kv_backend": "deterministic_cp_backward_reference",
            "split_kv_plan_source": "reference_execution",
            "split_kv_fallback": False,
            "split_kv_fallback_reason": None,
        },
        {
            "owner_cp_rank": 1,
            "requested_split_kv_policy": "fixed",
            "requested_split_kv_size": 2,
            "actual_split_kv_policy": "fixed",
            "actual_split_kv_size": 2,
            "actual_split_kv_count": 2,
            "actual_split_boundaries": [[3, 5], [5, 6]],
            "split_kv_merge_order": "global_block_index",
            "split_kv_accum_dtype": "fp32",
            "split_kv_downcast_at": "final_write",
            "split_kv_backend": "deterministic_cp_backward_reference",
            "split_kv_plan_source": "reference_execution",
            "split_kv_fallback": False,
            "split_kv_fallback_reason": None,
        },
    ]


def test_saved_forward_backward_matches_independent_dense_autograd():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(2, 5, 7, seed=31, heads=4, kv_heads=2, dim=8)
    mask = torch.tensor(
        [[True, True, True, True, True, True, False], [True] * 7],
        dtype=torch.bool,
    )
    query_offsets = torch.tensor([11, 23], dtype=torch.long)
    key_offsets = torch.tensor([9, 21], dtype=torch.long)
    dout = torch.randn(q.shape, generator=torch.Generator().manual_seed(32))

    state = op.save_forward_state(
        q,
        k,
        v,
        causal=True,
        scale=0.37,
        key_padding_mask=mask,
        query_position_offsets=query_offsets,
        key_position_offsets=key_offsets,
        cp_world_size=2,
        kv_chunk_size=2,
    )
    result = op.backward_reference(
        q,
        k,
        v,
        dout,
        causal=True,
        scale=0.37,
        key_padding_mask=mask,
        query_position_offsets=query_offsets,
        key_position_offsets=key_offsets,
        cp_world_size=2,
        kv_chunk_size=2,
        saved_forward_state=state,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    k_expanded = k_ref.repeat_interleave(2, dim=1)
    v_expanded = v_ref.repeat_interleave(2, dim=1)
    scores = torch.matmul(q_ref, k_expanded.transpose(-1, -2)) * 0.37
    q_pos = query_offsets[:, None] + torch.arange(q.size(2))
    k_pos = key_offsets[:, None] + torch.arange(k.size(2))
    scores = scores.masked_fill(
        (k_pos[:, None, :] > q_pos[:, :, None])[:, None, :, :],
        float("-inf"),
    )
    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    out_ref = torch.matmul(torch.softmax(scores, dim=-1), v_expanded)
    out_ref.backward(dout)

    assert isinstance(result.saved_forward_state, AttentionSavedForwardState)
    assert result.saved_forward_state is state
    assert result.provenance["saved_forward_state_source"] == "caller"
    torch.testing.assert_close(result.out, out_ref, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(result.gradients.dq, q_ref.grad, atol=_GRAD_ATOL, rtol=0.0)
    torch.testing.assert_close(result.gradients.dk, k_ref.grad, atol=_GRAD_ATOL, rtol=0.0)
    torch.testing.assert_close(result.gradients.dv, v_ref.grad, atol=_GRAD_ATOL, rtol=0.0)


@pytest.mark.parametrize(
    ("tensor_name", "message"),
    [("q", "q_fingerprint"), ("k", "k_fingerprint"), ("v", "v_fingerprint")],
)
def test_saved_forward_state_rejects_stale_qkv(tensor_name, message):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=33, heads=4, kv_heads=2, dim=8)
    state = op.save_forward_state(q, k, v, cp_world_size=2, kv_chunk_size=2)
    inputs = {"q": q.clone(), "k": k.clone(), "v": v.clone()}
    inputs[tensor_name].flatten()[0] += 1.0

    with pytest.raises(ValueError, match=message):
        op.backward_reference(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            torch.ones_like(q),
            cp_world_size=2,
            kv_chunk_size=2,
            saved_forward_state=state,
        )


@pytest.mark.parametrize(
    ("tensor_name", "message"),
    [
        ("out", "out_fingerprint"),
        ("lse", "lse_fingerprint"),
        ("key_padding_mask", "key_padding_mask_fingerprint"),
        ("query_position_offsets", "query_position_offsets_fingerprint"),
        ("key_position_offsets", "key_position_offsets_fingerprint"),
    ],
)
def test_saved_forward_state_rejects_mutated_saved_tensors(tensor_name, message):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=34, heads=4, kv_heads=2, dim=8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    offsets = torch.tensor([7], dtype=torch.long)
    state = op.save_forward_state(
        q,
        k,
        v,
        key_padding_mask=mask,
        query_position_offsets=offsets,
        key_position_offsets=offsets,
        cp_world_size=2,
        kv_chunk_size=2,
    )
    tensor = getattr(state, tensor_name)
    if tensor.dtype == torch.bool:
        tensor.flatten()[0].logical_not_()
    else:
        tensor.flatten()[0].add_(1)

    with pytest.raises(ValueError, match=message):
        op.backward_reference(
            q,
            k,
            v,
            torch.ones_like(q),
            key_padding_mask=mask,
            query_position_offsets=offsets,
            key_position_offsets=offsets,
            cp_world_size=2,
            kv_chunk_size=2,
            saved_forward_state=state,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"causal": False}, "causal"),
        ({"scale": 0.5}, "scale"),
        ({"cp_world_size": 1}, "cp_world_size"),
        ({"kv_chunk_size": None}, "kv_chunk_size"),
        ({"query_position_offsets": torch.tensor([8])}, "query_position_offsets"),
        ({"key_position_offsets": torch.tensor([8])}, "key_position_offsets"),
    ],
)
def test_saved_forward_state_rejects_execution_metadata_mismatch(override, message):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=35, heads=4, kv_heads=2, dim=8)
    offsets = torch.tensor([7], dtype=torch.long)
    state = op.save_forward_state(
        q,
        k,
        v,
        query_position_offsets=offsets,
        key_position_offsets=offsets,
        cp_world_size=2,
        kv_chunk_size=2,
    )
    kwargs = {
        "causal": True,
        "scale": None,
        "query_position_offsets": offsets,
        "key_position_offsets": offsets,
        "cp_world_size": 2,
        "kv_chunk_size": 2,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        op.backward_reference(
            q,
            k,
            v,
            torch.ones_like(q),
            saved_forward_state=state,
            **kwargs,
        )


def test_split_kv_plan_never_crosses_cp_owner_boundaries():
    plans = split_kv_execution_plan_provenance(
        10,
        cp_world_size=3,
        kv_chunk_size=3,
        backend="test-reference",
    )

    assert [plan["actual_split_boundaries"] for plan in plans] == [
        [[0, 3], [3, 4]],
        [[4, 7]],
        [[7, 10]],
    ]
    assert [plan["owner_cp_rank"] for plan in plans] == [0, 1, 2]


def test_backward_report_preserves_post_rope_position_metadata():
    rope = NativeRoPEOp()
    pre_rope_q, pre_rope_k, v = _qkv(2, 5, 5, seed=19, heads=4, kv_heads=2, dim=8)
    position_offsets = torch.tensor([23, 101], dtype=torch.long)
    positions = position_offsets[:, None] + torch.arange(pre_rope_q.size(2), dtype=torch.long)
    q = rope.forward_fp32(pre_rope_q, positions, theta=1_000_000.0)
    k = rope.forward_fp32(pre_rope_k, positions, theta=1_000_000.0)
    dout = torch.randn(2, 4, 5, 8, generator=torch.Generator().manual_seed(20))

    with _single_thread():
        report = compare_cp_attention_backward(
            q,
            k,
            v,
            dout,
            causal=True,
            query_position_offsets=position_offsets,
            key_position_offsets=position_offsets,
            candidate_cp_world_size=2,
            candidate_kv_chunk_size=2,
            output_dtype=torch.float32,
        )

    drift = report.drifts[0]
    assert drift.dq.max_abs <= _GRAD_ATOL
    assert drift.dk.max_abs <= _GRAD_ATOL
    assert drift.dv.max_abs <= _GRAD_ATOL


def test_qwen3_8b_local_tp2_cp2_bf16_backward_report_smoke():
    # Qwen3-8B global Hq/Hkv is 32/8. A TP=2 local shard owns 16/4 heads.
    q, k, v = _qkv(
        1,
        4,
        4,
        seed=21,
        dtype=torch.bfloat16,
        heads=16,
        kv_heads=4,
        dim=_HEAD_DIM,
    )
    dout = torch.randn(
        1,
        16,
        4,
        _HEAD_DIM,
        generator=torch.Generator().manual_seed(22),
        dtype=torch.bfloat16,
    )

    with _single_thread():
        report = compare_cp_attention_backward(
            q,
            k,
            v,
            dout,
            causal=True,
            candidate_cp_world_size=2,
            candidate_kv_chunk_size=2,
            output_dtype=torch.bfloat16,
        )

    drift = report.drifts[0]
    assert drift.provenance["q_dtype"] == "bfloat16"
    assert drift.provenance["output_dtype"] == "bfloat16"
    assert drift.provenance["downcast_at"] == "final_write"
    assert drift.dq.max_abs <= 5.0e-2
    assert drift.dk.max_abs <= 5.0e-2
    assert drift.dv.max_abs <= 5.0e-2


def test_backward_report_validates_dout_shape_and_dtype():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=23, heads=4, kv_heads=2, dim=8)

    with pytest.raises(ValueError, match="dout must have shape"):
        op.backward_reference(q, k, v, torch.randn(1, 4, 3, 8), cp_world_size=2)

    with pytest.raises(ValueError, match="dout must be a real floating-point tensor"):
        op.backward_reference(
            q,
            k,
            v,
            torch.ones(1, 4, 4, 8, dtype=torch.long),
            cp_world_size=2,
        )

    with pytest.raises(ValueError, match="dout must have the same dtype"):
        op.backward_reference(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            torch.ones_like(q),
            cp_world_size=2,
        )


def test_inputs_are_not_mutated():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(2, 6, 6, seed=7)
    mask = torch.ones(2, 6, dtype=torch.bool)
    qc, kc, vc, mc = q.clone(), k.clone(), v.clone(), mask.clone()

    op.forward_fp32_with_lse(q, k, v, causal=True, key_padding_mask=mask, cp_world_size=2)

    assert torch.equal(q, qc)
    assert torch.equal(k, kc)
    assert torch.equal(v, vc)
    assert torch.equal(mask, mc)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cp_world_size": 0}, "cp_world_size"),
        ({"cp_world_size": 2, "kv_chunk_size": 0}, "kv_chunk_size"),
    ],
)
def test_invalid_parallelism_arguments_raise(kwargs, message):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=8)
    with pytest.raises(ValueError, match=message):
        op.forward_fp32_with_lse(q, k, v, causal=True, **kwargs)


def test_invalid_gqa_and_mask_shapes_raise():
    op = DeterministicCPAttentionReferenceOp()
    q = torch.randn(1, 6, 4, _HEAD_DIM)
    k = torch.randn(1, 4, 4, _HEAD_DIM)
    v = torch.randn(1, 4, 4, _HEAD_DIM)
    with pytest.raises(ValueError, match="not divisible"):
        op.forward_fp32_with_lse(q, k, v, causal=True)

    q, k, v = _qkv(1, 4, 4, seed=9)
    with pytest.raises(ValueError, match="key_padding_mask"):
        op.forward_fp32_with_lse(q, k, v, key_padding_mask=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="key_padding_mask"):
        op.forward_fp32_with_lse(q, k, v, key_padding_mask=torch.ones(1, 4))
    with pytest.raises(ValueError, match="query_position_offsets"):
        op.forward_fp32_with_lse(
            q,
            k,
            v,
            query_position_offsets=torch.ones(2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="key_position_offsets"):
        op.forward_fp32_with_lse(
            q,
            k,
            v,
            key_position_offsets=torch.ones(1, dtype=torch.float32),
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True, "bad"])
def test_invalid_scale_fails_before_attention_math(scale):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=24, heads=4, kv_heads=2, dim=8)

    with pytest.raises(ValueError, match="scale must be a positive finite number"):
        op.forward_fp32_with_lse(q, k, v, scale=scale)


def test_forward_reference_provenance_does_not_claim_production_communication():
    provenance = DeterministicCPAttentionReferenceOp.execution_provenance(
        8,
        cp_world_size=2,
        kv_chunk_size=2,
    )

    assert provenance["execution_scope"] == "logical_single_process_cp_reference"
    assert provenance["query_scope"] == "logical_global_query_reference"
    assert provenance["kv_scope"] == "logical_owner_local_cp_shards"
    assert provenance["production_cp_protocol"] == "ag_query_local_kv_rs_out_lse"
    assert provenance["communication_executed"] == "none"
    assert provenance["merge_order"] == "global_block_index"
    plans = provenance["actual_split_kv_plans"]
    assert [plan["owner_cp_rank"] for plan in plans] == [0, 1]
    assert [plan["actual_split_boundaries"] for plan in plans] == [
        [[0, 2], [2, 4]],
        [[4, 6], [6, 8]],
    ]


def test_qkv_dtype_and_floating_contract_fails_closed():
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=25, heads=4, kv_heads=2, dim=8)

    with pytest.raises(ValueError, match="same dtype"):
        op.forward_fp32_with_lse(q, k.to(torch.bfloat16), v)
    with pytest.raises(ValueError, match="real floating-point"):
        op.forward_fp32_with_lse(q.to(torch.long), k.to(torch.long), v.to(torch.long))


@pytest.mark.parametrize("kwargs", [{"cp_world_size": True}, {"kv_chunk_size": True}])
def test_boolean_parallelism_arguments_fail_closed(kwargs):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=26, heads=4, kv_heads=2, dim=8)

    with pytest.raises(ValueError):
        op.forward_fp32_with_lse(q, k, v, **kwargs)


@pytest.mark.parametrize("output_dtype", [torch.long, torch.complex64, "fp32"])
def test_nonfloating_output_dtype_fails_closed(output_dtype):
    op = DeterministicCPAttentionReferenceOp()
    q, k, v = _qkv(1, 4, 4, seed=27, heads=4, kv_heads=2, dim=8)

    with pytest.raises(ValueError, match="output_dtype must be a real floating-point"):
        op.forward_with_lse(q, k, v, output_dtype=output_dtype)


def test_partial_states_must_remain_fp32_and_colocated():
    out = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="must remain FP32"):
        AttentionPartialState(out=out, lse=lse, block_start=0, block_end=1)


def test_overlapping_partial_ranges_raise():
    out = torch.zeros(1, 1, 1, 1)
    lse = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="overlap"):
        merge_attention_partial_states(
            [
                AttentionPartialState(out=out, lse=lse, block_start=0, block_end=3),
                AttentionPartialState(out=out, lse=lse, block_start=2, block_end=4),
            ]
        )


def test_gapped_partial_ranges_raise():
    out = torch.zeros(1, 1, 1, 1)
    lse = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="gap-free"):
        merge_attention_partial_states(
            [
                AttentionPartialState(out=out, lse=lse, block_start=0, block_end=2),
                AttentionPartialState(out=out, lse=lse, block_start=3, block_end=4),
            ]
        )


def test_registry_dispatches_cp_attention_reference():
    assert isinstance(kernel_registry.get_op("cp_attention"), DeterministicCPAttentionReferenceOp)


def test_shared_strict_core_reports_canonical_cuda_schedule(monkeypatch):
    class FakeCUDAAttentionOp:
        @staticmethod
        def forward_with_lse(q, k, v, **_kwargs):
            del k, v
            return (
                torch.zeros_like(q),
                torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device),
            )

    monkeypatch.setattr(
        deterministic_attn_module,
        "DeterministicAttentionOp",
        FakeCUDAAttentionOp,
    )
    q, k, v = _qkv(1, 3, 4, seed=41, heads=4, kv_heads=2, dim=8)
    q_positions = torch.arange(1, 4).view(1, -1)
    k_positions = torch.arange(4).view(1, -1)
    result = RLKernelDeterministicAttentionCore().forward_with_lse(
        q,
        k,
        v,
        query_position_ids=q_positions,
        key_position_ids=k_positions,
    )
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_SCHEDULE_ID


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_strict_forward_is_bitwise_invariant_to_batch_cp_and_chunk(dtype):
    q, k, v = _qkv(2, 5, 9, seed=41, dtype=dtype, heads=4, kv_heads=2, dim=8)
    op = DeterministicCPAttentionReferenceOp(strict_bitwise=True)

    full_out, full_lse = op.forward_with_lse(q, k, v, cp_world_size=1)
    chunked_out, chunked_lse = op.forward_with_lse(
        q,
        k,
        v,
        cp_world_size=2,
        kv_chunk_size=3,
    )
    single_out, single_lse = op.forward_with_lse(
        q[:1],
        k[:1],
        v[:1],
        cp_world_size=4,
        kv_chunk_size=1,
    )
    assert torch.equal(full_out, chunked_out)
    assert torch.equal(full_lse, chunked_lse)
    assert torch.equal(full_out[:1], single_out)
    assert torch.equal(full_lse[:1], single_lse)


def test_strict_backward_is_bitwise_invariant_to_batch_cp_and_chunk():
    q, k, v = _qkv(2, 5, 9, seed=42, heads=4, kv_heads=2, dim=8)
    dout = torch.randn(q.shape, generator=torch.Generator().manual_seed(43))
    op = DeterministicCPAttentionReferenceOp(strict_bitwise=True)

    full = op.backward_reference(q, k, v, dout, cp_world_size=1)
    chunked = op.backward_reference(
        q,
        k,
        v,
        dout,
        cp_world_size=2,
        kv_chunk_size=3,
    )
    single = op.backward_reference(
        q[:1],
        k[:1],
        v[:1],
        dout[:1],
        cp_world_size=4,
        kv_chunk_size=1,
    )

    for full_tensor, chunked_tensor, single_tensor in (
        (full.out, chunked.out, single.out),
        (full.lse, chunked.lse, single.lse),
        (full.gradients.dq, chunked.gradients.dq, single.gradients.dq),
        (full.gradients.dk, chunked.gradients.dk, single.gradients.dk),
        (full.gradients.dv, chunked.gradients.dv, single.gradients.dv),
    ):
        assert torch.equal(full_tensor, chunked_tensor)
        assert torch.equal(full_tensor[:1], single_tensor)

    assert chunked.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert chunked.provenance["strict_schedule"] == STRICT_ATTENTION_SCHEDULE_ID
    assert chunked.provenance["actual_split_kv_policy"] == "disabled"
    assert chunked.provenance["backward_algorithm"] == ("saved_out_lse_canonical_row_reference")
    assert all(
        plan["actual_split_kv_policy"] == "disabled"
        and plan["actual_split_boundaries"] == [[0, k.size(2)]]
        for plan in chunked.provenance["actual_split_kv_plans"]
    )


def test_strict_backward_rejects_non_strict_saved_forward_state():
    q, k, v = _qkv(1, 4, 6, seed=44, heads=4, kv_heads=2, dim=8)
    state = DeterministicCPAttentionReferenceOp().save_forward_state(q, k, v)

    with pytest.raises(ValueError, match="strict_bitwise"):
        DeterministicCPAttentionReferenceOp(strict_bitwise=True).backward_reference(
            q,
            k,
            v,
            torch.ones_like(q),
            saved_forward_state=state,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_strict_core_is_bitwise_invariant_to_batch_and_cp_schedule(dtype):
    """The strict candidate must not change arithmetic with batch/CP shape."""

    q, k, v = _qkv(2, 5, 9, seed=41, dtype=dtype, heads=4, kv_heads=2, dim=8)
    op = DeterministicCPAttentionReferenceOp(strict_bitwise=True)

    cp1_out, cp1_lse = op.forward_with_lse(q, k, v, cp_world_size=1, kv_chunk_size=3)
    cp2_out, cp2_lse = op.forward_with_lse(q, k, v, cp_world_size=2, kv_chunk_size=3)
    assert torch.equal(cp1_out, cp2_out)
    assert torch.equal(cp1_lse, cp2_lse)

    single_out, single_lse = op.forward_with_lse(
        q[:1],
        k[:1],
        v[:1],
        cp_world_size=1,
        kv_chunk_size=1,
    )
    assert torch.equal(single_out, cp1_out[:1])
    assert torch.equal(single_lse, cp1_lse[:1])


def test_strict_core_backward_is_bitwise_invariant_to_cp_schedule():
    q, k, v = _qkv(1, 4, 8, seed=42, dtype=torch.float32, heads=4, kv_heads=2, dim=8)
    dout = torch.randn_like(q)
    op = DeterministicCPAttentionReferenceOp(strict_bitwise=True)

    cp1 = op.backward_reference(q, k, v, dout, cp_world_size=1, kv_chunk_size=None)
    cp2 = op.backward_reference(q, k, v, dout, cp_world_size=2, kv_chunk_size=3)
    assert torch.equal(cp1.out, cp2.out)
    assert torch.equal(cp1.lse, cp2.lse)
    assert torch.equal(cp1.gradients.dq, cp2.gradients.dq)
    assert torch.equal(cp1.gradients.dk, cp2.gradients.dk)
    assert torch.equal(cp1.gradients.dv, cp2.gradients.dv)
