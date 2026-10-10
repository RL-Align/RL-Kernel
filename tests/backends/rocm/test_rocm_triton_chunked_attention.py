# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bitwise contract tests for the ROCm Triton chunked flash attention."""

from __future__ import annotations

import math

import pytest
import torch

IS_ROCM = getattr(torch.version, "hip", None) is not None
IS_GFX942 = (
    IS_ROCM
    and torch.cuda.is_available()
    and str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx942")
)

pytestmark = pytest.mark.skipif(not IS_GFX942, reason="chunked flash attention targets ROCm gfx942")

if IS_GFX942:
    from rl_engine.backends.rocm.attention import chunked_flash_attn as A

DEV = "cuda"
HQ, HKV, D = 8, 2, 128  # Qwen3-8B TP4 local heads
PAGE = 16
SCALE = 1.0 / math.sqrt(D)


def _paged(seqs, *, extra_pages=7):
    total_pages = sum((L + PAGE - 1) // PAGE for L in seqs) + extra_pages
    perm = torch.randperm(total_pages, device=DEV)
    k_cache = (torch.randn(total_pages, PAGE, HKV, D, device=DEV) * 0.5).bfloat16()
    v_cache = (torch.randn(total_pages, PAGE, HKV, D, device=DEV) * 0.5).bfloat16()
    max_pages = max((L + PAGE - 1) // PAGE for L in seqs)
    table = torch.zeros(len(seqs), max_pages, dtype=torch.int32, device=DEV)
    cursor = 0
    dense = []
    for row, length in enumerate(seqs):
        count = (length + PAGE - 1) // PAGE
        pages = perm[cursor : cursor + count]
        cursor += count
        table[row, :count] = pages.to(torch.int32)
        table[row, count:] = pages[0]
        dense.append(
            (
                k_cache[pages].reshape(-1, HKV, D)[:length],
                v_cache[pages].reshape(-1, HKV, D)[:length],
            )
        )
    return k_cache, v_cache, table, dense


def _reference(q, k, v, q_pos):
    group = HQ // HKV
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * SCALE
    key_pos = torch.arange(k.size(0), device=DEV)
    scores = scores.masked_fill((key_pos[None, :] > q_pos[:, None])[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vf)


def _run(q, k_cache, v_cache, table, cu, seqlen_k, max_q, *, schedule="auto"):
    return A.paged_attention_forward(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu,
        block_table=table,
        seqlen_k=seqlen_k,
        max_seqlen_q=max_q,
        scale=SCALE,
        schedule=schedule,
    )


@pytest.mark.parametrize("length", [1, 17, 64, 65, 500, 1024, 1500, 4096])
def test_prefill_decode_and_extend_rows_are_bitwise_identical(length):
    torch.manual_seed(length)
    k_cache, v_cache, table, dense = _paged([length])
    q = (torch.randn(length, HQ, D, device=DEV) * 0.5).bfloat16()
    cu = torch.tensor([0, length], dtype=torch.int32, device=DEV)
    seqlen_k = torch.tensor([length], dtype=torch.int32, device=DEV)
    out_prefill, lse_prefill = _run(
        q, k_cache, v_cache, table, cu, seqlen_k, length, schedule="monolithic"
    )
    reference = _reference(q, dense[0][0], dense[0][1], torch.arange(length, device=DEV))
    assert (out_prefill.float() - reference).abs().max().item() <= 1e-2

    positions = sorted(
        {
            p
            for p in (
                0,
                1,
                15,
                16,
                63,
                64,
                65,
                127,
                128,
                511,
                512,
                513,
                1023,
                1024,
                length - 2,
                length - 1,
            )
            if 0 <= p < length
        }
    )
    q_decode = q[positions]
    cu_decode = torch.arange(len(positions) + 1, dtype=torch.int32, device=DEV)
    seqlen_decode = torch.tensor([p + 1 for p in positions], dtype=torch.int32, device=DEV)
    table_decode = table.expand(len(positions), -1).contiguous()
    for schedule in ("split", "monolithic"):
        out_decode, lse_decode = _run(
            q_decode, k_cache, v_cache, table_decode, cu_decode, seqlen_decode, 1, schedule=schedule
        )
        assert torch.equal(out_decode, out_prefill[positions]), schedule
        assert torch.equal(lse_decode, lse_prefill[:, positions]), schedule

    if length > 40:
        start = length - 37
        cu_extend = torch.tensor([0, length - start], dtype=torch.int32, device=DEV)
        for schedule in ("monolithic", "split"):
            if schedule == "split" and (length - start) * (HQ // HKV) > A.BLOCK_M:
                continue
            out_extend, _ = _run(
                q[start:],
                k_cache,
                v_cache,
                table,
                cu_extend,
                seqlen_k,
                length - start,
                schedule=schedule,
            )
            assert torch.equal(out_extend, out_prefill[start:]), schedule


def test_batch_composition_is_invariant():
    torch.manual_seed(7)
    lengths = [700, 1, 4096, 33]
    k_cache, v_cache, table, dense = _paged(lengths)
    queries = [(torch.randn(L, HQ, D, device=DEV) * 0.5).bfloat16() for L in lengths]
    q_all = torch.cat(queries)
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], dtype=torch.int32, device=DEV
    )
    seqlen_k = torch.tensor(lengths, dtype=torch.int32, device=DEV)
    out_all, _ = _run(
        q_all, k_cache, v_cache, table, cu, seqlen_k, max(lengths), schedule="monolithic"
    )
    offset = 0
    for index, length in enumerate(lengths):
        out_one, _ = _run(
            queries[index],
            k_cache,
            v_cache,
            table[index : index + 1],
            torch.tensor([0, length], dtype=torch.int32, device=DEV),
            seqlen_k[index : index + 1],
            length,
            schedule="monolithic",
        )
        assert torch.equal(out_one, out_all[offset : offset + length])
        reference = _reference(
            queries[index], dense[index][0], dense[index][1], torch.arange(length, device=DEV)
        )
        assert (out_one.float() - reference).abs().max().item() <= 1e-2
        offset += length
    q_last = torch.stack([queries[index][-1] for index in range(4)])
    out_decode, _ = _run(
        q_last,
        k_cache,
        v_cache,
        table,
        torch.arange(5, dtype=torch.int32, device=DEV),
        seqlen_k,
        1,
        schedule="split",
    )
    for index in range(4):
        assert torch.equal(out_decode[index], out_all[cu[index + 1] - 1])


def test_mixed_decode_extend_prefill_batch_matches_isolated_rows():
    torch.manual_seed(11)
    lengths = [2048, 300, 900]
    query_lengths = [1, 17, 900]  # decode, extend, fresh prefill
    k_cache, v_cache, table, _dense = _paged(lengths)
    full = [(torch.randn(L, HQ, D, device=DEV) * 0.5).bfloat16() for L in lengths]
    isolated = []
    for index, length in enumerate(lengths):
        out, _ = _run(
            full[index],
            k_cache,
            v_cache,
            table[index : index + 1],
            torch.tensor([0, length], dtype=torch.int32, device=DEV),
            torch.tensor([length], dtype=torch.int32, device=DEV),
            length,
            schedule="monolithic",
        )
        isolated.append(out[length - query_lengths[index] :])
    q_mixed = torch.cat([full[i][lengths[i] - query_lengths[i] :] for i in range(3)])
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(query_lengths), 0).tolist()], dtype=torch.int32, device=DEV
    )
    seqlen_k = torch.tensor(lengths, dtype=torch.int32, device=DEV)
    out_mixed, _ = _run(q_mixed, k_cache, v_cache, table, cu, seqlen_k, max(query_lengths))
    assert torch.equal(out_mixed, torch.cat(isolated))


@pytest.mark.parametrize("length", [1, 64, 65, 700, 1500, 4096])
def test_unmasked_fast_path_matches_masked_path_bitwise(length):
    torch.manual_seed(length + 100)
    k_cache, v_cache, table, _dense = _paged([length, 33])
    q = (torch.randn(length + 33, HQ, D, device=DEV) * 0.5).bfloat16()
    cu = torch.tensor([0, length, length + 33], dtype=torch.int32, device=DEV)
    seqlen_k = torch.tensor([length, 33], dtype=torch.int32, device=DEV)
    max_q = max(length, 33)
    for schedule in ("monolithic", "split"):
        if schedule == "split" and max_q * (HQ // HKV) > A.BLOCK_M:
            continue
        slow, slow_lse = A.paged_attention_forward(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu,
            block_table=table,
            seqlen_k=seqlen_k,
            max_seqlen_q=max_q,
            scale=SCALE,
            schedule=schedule,
            fast_path=False,
        )
        fast, fast_lse = A.paged_attention_forward(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu,
            block_table=table,
            seqlen_k=seqlen_k,
            max_seqlen_q=max_q,
            scale=SCALE,
            schedule=schedule,
            fast_path=True,
        )
        assert torch.equal(slow, fast), schedule
        assert torch.equal(slow_lse, fast_lse), schedule
    # decode rows through the split schedule against the masked prefill rows
    positions = [p for p in (0, 63, 64, 511, 512, 1000, length - 1) if 0 <= p < length]
    q_decode = q[positions]
    cu_decode = torch.arange(len(positions) + 1, dtype=torch.int32, device=DEV)
    seqlen_decode = torch.tensor([p + 1 for p in positions], dtype=torch.int32, device=DEV)
    table_decode = table[:1].expand(len(positions), -1).contiguous()
    prefill, _ = A.paged_attention_forward(
        q[:length],
        k_cache,
        v_cache,
        cu_seqlens_q=cu[:2],
        block_table=table[:1],
        seqlen_k=seqlen_k[:1],
        max_seqlen_q=length,
        scale=SCALE,
        schedule="monolithic",
        fast_path=False,
    )
    decode, _ = A.paged_attention_forward(
        q_decode,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_decode,
        block_table=table_decode,
        seqlen_k=seqlen_decode,
        max_seqlen_q=1,
        scale=SCALE,
        schedule="split",
        fast_path=True,
    )
    assert torch.equal(decode, prefill[positions])


def test_decode_sequence_metadata_graph_replay():
    torch.manual_seed(19)
    k_cache, v_cache, table, _dense = _paged([1500, 700, 33])
    q = torch.randn(3, HQ, D, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(4, device=DEV, dtype=torch.int32)
    lengths = torch.tensor([1500, 700, 33], device=DEV, dtype=torch.int32)
    mapping = torch.arange(3, device=DEV, dtype=torch.int32)

    def run(seq_of_token):
        return A.paged_attention_forward(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu,
            block_table=table,
            seqlen_k=lengths,
            max_seqlen_q=1,
            scale=SCALE,
            seq_of_token=seq_of_token,
        )

    run(None)
    run(mapping)
    torch.cuda.synchronize()
    implicit_graph, explicit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(implicit_graph):
        implicit = run(None)
    with torch.cuda.graph(explicit_graph):
        explicit = run(mapping)
    for live_lengths in ([1500, 700, 33], [1024, 513, 17], [1025, 64, 1]):
        q.normal_()
        lengths.copy_(torch.tensor(live_lengths, device=DEV, dtype=torch.int32))
        implicit_graph.replay()
        explicit_graph.replay()
        reference = _run(q, k_cache, v_cache, table, cu, lengths, 1, schedule="monolithic")
        for a, b, expected in zip(implicit, explicit, reference, strict=True):
            bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            assert torch.equal(a.view(bits), expected.view(bits))
            assert torch.equal(b.view(bits), expected.view(bits))


def test_output_buffer_and_padded_rows_are_left_alone():
    torch.manual_seed(3)
    k_cache, v_cache, table, _dense = _paged([100, 50])
    q = (torch.randn(2, HQ, D, device=DEV) * 0.5).bfloat16()
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEV)
    seqlen_k = torch.tensor([100, 50], dtype=torch.int32, device=DEV)
    expected, _ = _run(q, k_cache, v_cache, table, cu, seqlen_k, 1)
    buffer = torch.zeros((4, HQ, D), dtype=torch.bfloat16, device=DEV)
    out, _ = A.paged_attention_forward(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu,
        block_table=table,
        seqlen_k=seqlen_k,
        max_seqlen_q=1,
        scale=SCALE,
        out=buffer.narrow(0, 0, 2),
    )
    assert out.data_ptr() == buffer.data_ptr()
    assert torch.equal(buffer[:2], expected)
    assert torch.equal(buffer[2:], torch.zeros_like(buffer[2:]))


def test_aiter_shaped_entry_point_and_warmup():
    torch.manual_seed(5)
    k_cache, v_cache, table, _dense = _paged([64, 64])
    q = (torch.randn(128, HQ, D, device=DEV) * 0.5).bfloat16()
    cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=DEV)
    seqlen_k = torch.tensor([64, 64], dtype=torch.int32, device=DEV)
    indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEV)
    out, lse, mask, rng = A.triton_paged_prefill(
        q,
        k_cache,
        v_cache,
        cu,
        indptr,
        table.reshape(-1),
        64,
        64,
        0.0,
        SCALE,
        0.0,
        False,
        True,
        -1,
        -1,
        0,
        True,
        False,
        block_table=table,
        seqlen_k=seqlen_k,
    )
    expected, expected_lse = _run(q, k_cache, v_cache, table, cu, seqlen_k, 64)
    assert torch.equal(out, expected)
    assert torch.equal(lse, expected_lse)
    assert mask.numel() == 0 and rng.shape == (2,)
    with pytest.raises(ValueError):
        A.triton_paged_prefill(
            q,
            k_cache,
            v_cache,
            cu,
            indptr,
            table.reshape(-1),
            64,
            64,
            0.1,
            SCALE,
            0.0,
            False,
            True,
            -1,
            -1,
            0,
            True,
            False,
            block_table=table,
            seqlen_k=seqlen_k,
        )
    A.warmup(torch.device(DEV), num_q_heads=HQ, num_kv_heads=HKV, dtype=torch.bfloat16)


def test_strict_core_binds_the_triton_contract(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTENTION_BACKEND", "triton")
    monkeypatch.setenv("RL_KERNEL_ROCM_FIXED_PAGED_TILE", "128")
    from rl_engine.backends.rocm.attention import flash_attn as F

    core = F.StrictRocmAiterCKAttentionCore()
    assert core.attention_backend == "triton"
    assert core.paged_kernel_id == A.CHUNKED_FLASH_ATTENTION_CONTRACT_ID
    assert core.paged_entrypoint_id == A.CHUNKED_FLASH_ATTENTION_CONTRACT_ID
    assert core.supports_paged_schedule and core.supports_mixed_paged_batches
    torch.manual_seed(9)
    k_cache, v_cache, table, _dense = _paged([48])
    q = (torch.randn(48, HQ, D, device=DEV) * 0.5).bfloat16()
    cu = torch.tensor([0, 48], dtype=torch.int32, device=DEV)
    seqlen_k = torch.tensor([48], dtype=torch.int32, device=DEV)
    indptr = torch.tensor([0, table.size(1)], dtype=torch.int32, device=DEV)
    with torch.inference_mode():
        result = core.forward_paged_varlen_with_lse(
            q,
            k_cache,
            v_cache,
            page_table=table,
            seqused_k=seqlen_k,
            cu_seqlens_q=cu,
            kv_indptr=indptr,
            max_seqlen_q=48,
            max_seqlen_k=48,
            causal=True,
            scale=SCALE,
        )
    expected, expected_lse = _run(q, k_cache, v_cache, table, cu, seqlen_k, 48)
    assert torch.equal(result.out, expected)
    assert torch.equal(result.lse, expected_lse)
    assert result.provenance["forward_entrypoint"] == A.CHUNKED_FLASH_ATTENTION_CONTRACT_ID
    assert result.provenance["paged_kernel"] == A.CHUNKED_FLASH_ATTENTION_CONTRACT_ID
    # dense training-style forward routes through the same paged contract
    dense_q = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    dense_k = k_cache[table[0, :3]].reshape(-1, HKV, D).permute(1, 0, 2).unsqueeze(0).contiguous()
    dense_v = v_cache[table[0, :3]].reshape(-1, HKV, D).permute(1, 0, 2).unsqueeze(0).contiguous()
    positions = torch.arange(48, device=DEV).unsqueeze(0)
    dense = core.forward_with_lse(
        dense_q,
        dense_k,
        dense_v,
        causal=True,
        scale=SCALE,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    assert torch.equal(dense.out.squeeze(0).permute(1, 0, 2), expected)
    assert dense.provenance["forward_entrypoint"] == A.CHUNKED_FLASH_ATTENTION_CONTRACT_ID
