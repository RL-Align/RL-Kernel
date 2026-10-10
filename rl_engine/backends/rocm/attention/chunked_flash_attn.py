# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Chunked-KV flash attention for ROCm with one arithmetic contract for every role.

Contract ``rlkernel.rocm.triton_chunked_flash_attention.v1``:

* BF16/FP16 Q, K, V with ``head_dim=128`` read from vLLM-style token-major
  pages ``[pages, 16, kv_heads, head_dim]`` through a 2-D page table.
* Scores ``Q.K^T`` on ``v_mfma_f32_16x16x16`` (``matrix_instr_nonkdim=16``,
  ``kpack=2``), scaled once in FP32 by ``scale * log2(e)``.
* The key axis is consumed in ascending ``BLOCK_N=64`` blocks inside fixed
  ``CHUNK_KV=512``-token chunks.  Every chunk runs the online softmax from an
  empty state (``m=-inf, l=0, acc=0``); chunk states are merged in ascending
  order with the exact FA2 rescale.  ``P`` is rounded to the input dtype before
  ``P.V``; ``exp2`` is the hardware instruction; FP contraction is disabled.
* Blocks and chunks that are entirely masked for a row leave that row's state
  bit-for-bit untouched, so a row's result never depends on which other rows
  share its tile.

Every query row's output is therefore a pure function of its own Q, the
visible K/V prefix and the contract.  The monolithic schedule (one program per
query tile, chunks evaluated sequentially) and the split schedule (one program
per ``(sequence, kv head, chunk)`` plus an ascending merge) produce identical
bits, which is what lets a Megatron full-sequence forward, a vLLM prefill, a
prefix-cached extend and a single-token paged decode agree exactly.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _TRITON_AVAILABLE = False

CHUNKED_FLASH_ATTENTION_CONTRACT_ID = "rlkernel.rocm.triton_chunked_flash_attention.v1"
BLOCK_M = 64
BLOCK_N = 64
CHUNK_KV = 512
HEAD_DIM = 128
PAGE_SIZE = 16
NUM_WARPS = 4
WAVES_PER_EU = 2
MATRIX_INSTR_NONKDIM = 16
KPACK = 2
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_COMPILE_OPTIONS: dict[str, int | str] = dict(
    num_warps=NUM_WARPS,
    waves_per_eu=WAVES_PER_EU,
    matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM,
    kpack=KPACK,
    enable_fp_fusion=False,
)


if _TRITON_AVAILABLE:

    @triton.jit
    def _kv_tile_ptrs(
        base_ptr,
        block_table_ptr,
        bt_stride,
        seq,
        start_n,
        stride_page,
        stride_tok,
        stride_head,
        kv_head,
        BLOCK_N: tl.constexpr,
        PAGE: tl.constexpr,
        D: tl.constexpr,
    ):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        page = tl.load(block_table_ptr + seq * bt_stride + offs_n // PAGE)
        rows = (
            page.to(tl.int64) * stride_page + (offs_n % PAGE) * stride_tok + kv_head * stride_head
        )
        offs_d = tl.arange(0, D)
        return base_ptr + rows[:, None] + offs_d[None, :]

    @triton.jit
    def _full_block_update(
        q,
        k,
        v,
        m_i,
        l_i,
        acc,
        scale_log2,
    ):
        """One key block that every row sees completely (no masking needed).

        Identical operation sequence to the masked update with an all-true
        mask, so the two paths are bit-for-bit interchangeable.
        """

        x = tl.dot(q, tl.trans(k)) * scale_log2
        row_max = tl.max(x, 1)
        m_new = tl.maximum(m_i, row_max)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(x - m_safe[:, None])
        l_ij = tl.sum(p, 1)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv
        l_i = l_i * alpha + l_ij
        return m_new, l_i, acc

    @triton.jit
    def _chunk_state(
        q,
        k_ptr,
        v_ptr,
        block_table_ptr,
        bt_stride,
        seq,
        kv_head,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_vp,
        stride_vt,
        stride_vh,
        chunk_start,
        chunk_end,
        kv_len,
        q_pos,
        full_limit,
        scale_log2,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PAGE: tl.constexpr,
        D: tl.constexpr,
        FAST_PATH: tl.constexpr,
    ):
        """Online softmax over keys ``[chunk_start, chunk_end)`` from an empty state.

        ``full_limit`` is a BLOCK_N multiple below which every row of the tile
        sees every key; blocks under it skip the masking work when
        ``FAST_PATH`` is set.  Block boundaries are the same in both paths.
        """

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        masked_start = chunk_start
        if FAST_PATH:
            full_end = tl.minimum(chunk_end, full_limit)
            for start_n in range(chunk_start, full_end, BLOCK_N):
                k_ptrs = _kv_tile_ptrs(
                    k_ptr,
                    block_table_ptr,
                    bt_stride,
                    seq,
                    start_n,
                    stride_kp,
                    stride_kt,
                    stride_kh,
                    kv_head,
                    BLOCK_N,
                    PAGE,
                    D,
                )
                v_ptrs = _kv_tile_ptrs(
                    v_ptr,
                    block_table_ptr,
                    bt_stride,
                    seq,
                    start_n,
                    stride_vp,
                    stride_vt,
                    stride_vh,
                    kv_head,
                    BLOCK_N,
                    PAGE,
                    D,
                )
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
                m_i, l_i, acc = _full_block_update(q, k, v, m_i, l_i, acc, scale_log2)
            masked_start = tl.maximum(chunk_start, full_end)
        for start_n in range(masked_start, chunk_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = _kv_tile_ptrs(
                k_ptr,
                block_table_ptr,
                bt_stride,
                seq,
                start_n,
                stride_kp,
                stride_kt,
                stride_kh,
                kv_head,
                BLOCK_N,
                PAGE,
                D,
            )
            v_ptrs = _kv_tile_ptrs(
                v_ptr,
                block_table_ptr,
                bt_stride,
                seq,
                start_n,
                stride_vp,
                stride_vt,
                stride_vh,
                kv_head,
                BLOCK_N,
                PAGE,
                D,
            )
            kv_valid = offs_n < kv_len
            k = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)
            x = tl.dot(q, tl.trans(k)) * scale_log2
            visible = kv_valid[None, :] & (offs_n[None, :] <= q_pos[:, None])
            x = tl.where(visible, x, float("-inf"))
            row_max = tl.max(x, 1)
            block_empty = row_max == float("-inf")
            m_new = tl.maximum(m_i, row_max)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(x - m_safe[:, None])
            l_ij = tl.sum(p, 1)
            pv = tl.dot(p.to(v.dtype), v)
            acc_new = acc * alpha[:, None] + pv
            l_new = l_i * alpha + l_ij
            acc = tl.where(block_empty[:, None], acc, acc_new)
            l_i = tl.where(block_empty, l_i, l_new)
            m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _merge_state(m, row_sum, acc, mc, lc, accc):
        """Merge chunk state ``(mc, lc, accc)`` into ``(m, l, acc)``; empty chunks are no-ops."""

        chunk_empty = mc == float("-inf")
        prev_empty = m == float("-inf")
        m_new = tl.maximum(m, mc)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp2(m - m_safe)
        b = tl.exp2(mc - m_safe)
        acc_merged = acc * a[:, None] + accc * b[:, None]
        l_merged = row_sum * a + lc * b
        acc_out = tl.where(
            chunk_empty[:, None], acc, tl.where(prev_empty[:, None], accc, acc_merged)
        )
        l_out = tl.where(chunk_empty, row_sum, tl.where(prev_empty, lc, l_merged))
        return m_new, l_out, acc_out

    @triton.jit
    def _attn_fwd_monolithic_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        cu_seqlens_q_ptr,
        seqlen_k_ptr,
        block_table_ptr,
        bt_stride,
        stride_qt,
        stride_qh,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_vp,
        stride_vt,
        stride_vh,
        stride_ot,
        stride_oh,
        stride_lh,
        stride_lt,
        scale_log2,
        group_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CHUNK: tl.constexpr,
        PAGE: tl.constexpr,
        D: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        FAST_PATH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        head = tl.program_id(1)
        seq = tl.program_id(2)
        q_start = tl.load(cu_seqlens_q_ptr + seq)
        q_len = tl.load(cu_seqlens_q_ptr + seq + 1) - q_start
        start_m = tile * BLOCK_M
        if start_m >= q_len:
            return
        kv_len = tl.load(seqlen_k_ptr + seq)
        kv_head = head // group_size
        offs_m = start_m + tl.arange(0, BLOCK_M)
        row_valid = offs_m < q_len
        offs_d = tl.arange(0, D)
        q_rows = (q_start + offs_m).to(tl.int64)
        q = tl.load(
            q_ptr + q_rows[:, None] * stride_qt + head * stride_qh + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        q_pos = kv_len - q_len + offs_m
        tile_kv_limit = tl.minimum(kv_len, kv_len - q_len + start_m + BLOCK_M)
        # Every row of this tile sees all keys below the first row's causal
        # bound; round down to a block boundary so both paths share blocks.
        first_row_visible = kv_len - q_len + start_m + 1
        full_limit = tl.maximum((tl.minimum(first_row_visible, kv_len) // BLOCK_N) * BLOCK_N, 0)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        num_chunks = tl.cdiv(tile_kv_limit, CHUNK)
        for chunk in range(0, num_chunks):
            chunk_start = chunk * CHUNK
            chunk_end = tl.minimum(chunk_start + CHUNK, tile_kv_limit)
            mc, lc, accc = _chunk_state(
                q,
                k_ptr,
                v_ptr,
                block_table_ptr,
                bt_stride,
                seq,
                kv_head,
                stride_kp,
                stride_kt,
                stride_kh,
                stride_vp,
                stride_vt,
                stride_vh,
                chunk_start,
                chunk_end,
                kv_len,
                q_pos,
                full_limit,
                scale_log2,
                BLOCK_M,
                BLOCK_N,
                PAGE,
                D,
                FAST_PATH,
            )
            m_i, l_i, acc = _merge_state(m_i, l_i, acc, mc, lc, accc)
        out = acc / l_i[:, None]
        tl.store(
            o_ptr + q_rows[:, None] * stride_ot + head * stride_oh + offs_d[None, :],
            out.to(o_ptr.dtype.element_ty),
            mask=row_valid[:, None],
        )
        if WRITE_LSE:
            lse = m_i * 0.6931471805599453 + tl.log(l_i)
            tl.store(lse_ptr + head * stride_lh + q_rows * stride_lt, lse, mask=row_valid)

    @triton.jit
    def _attn_fwd_split_partial_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pm_ptr,
        pl_ptr,
        pacc_ptr,
        cu_seqlens_q_ptr,
        seqlen_k_ptr,
        block_table_ptr,
        bt_stride,
        stride_qt,
        stride_qh,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_vp,
        stride_vt,
        stride_vh,
        stride_pc,
        stride_pt,
        stride_ph,
        stride_pacc_c,
        stride_pacc_t,
        stride_pacc_h,
        scale_log2,
        group_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CHUNK: tl.constexpr,
        PAGE: tl.constexpr,
        D: tl.constexpr,
        FAST_PATH: tl.constexpr,
    ):
        """One program per ``(sequence, kv head, chunk)``; rows pack the GQA group x queries."""

        chunk = tl.program_id(0)
        kv_head = tl.program_id(1)
        seq = tl.program_id(2)
        q_start = tl.load(cu_seqlens_q_ptr + seq)
        q_len = tl.load(cu_seqlens_q_ptr + seq + 1) - q_start
        kv_len = tl.load(seqlen_k_ptr + seq)
        chunk_start = chunk * CHUNK
        if chunk_start >= kv_len:
            return
        chunk_end = tl.minimum(chunk_start + CHUNK, kv_len)
        rows = tl.arange(0, BLOCK_M)
        row_q = rows // group_size
        row_h = rows % group_size
        row_valid = row_q < q_len
        head = kv_head * group_size + row_h
        offs_d = tl.arange(0, D)
        tok = (q_start + row_q).to(tl.int64)
        q = tl.load(
            q_ptr + tok[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        q_pos = tl.where(row_valid, kv_len - q_len + row_q, -1)
        first_row_visible = kv_len - q_len + 1
        full_limit = tl.maximum((tl.minimum(first_row_visible, kv_len) // BLOCK_N) * BLOCK_N, 0)
        mc, lc, accc = _chunk_state(
            q,
            k_ptr,
            v_ptr,
            block_table_ptr,
            bt_stride,
            seq,
            kv_head,
            stride_kp,
            stride_kt,
            stride_kh,
            stride_vp,
            stride_vt,
            stride_vh,
            chunk_start,
            chunk_end,
            kv_len,
            q_pos,
            full_limit,
            scale_log2,
            BLOCK_M,
            BLOCK_N,
            PAGE,
            D,
            FAST_PATH,
        )
        state = chunk * stride_pc + tok * stride_pt + head * stride_ph
        tl.store(pm_ptr + state, mc, mask=row_valid)
        tl.store(pl_ptr + state, lc, mask=row_valid)
        tl.store(
            pacc_ptr
            + chunk * stride_pacc_c
            + tok[:, None] * stride_pacc_t
            + head[:, None] * stride_pacc_h
            + offs_d[None, :],
            accc,
            mask=row_valid[:, None],
        )

    @triton.jit
    def _attn_fwd_split_merge_kernel(
        pm_ptr,
        pl_ptr,
        pacc_ptr,
        o_ptr,
        lse_ptr,
        seq_of_token_ptr,
        seqlen_k_ptr,
        stride_pc,
        stride_pt,
        stride_ph,
        stride_pacc_c,
        stride_pacc_t,
        stride_pacc_h,
        stride_ot,
        stride_oh,
        stride_lh,
        stride_lt,
        CHUNK: tl.constexpr,
        D: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        IDENTITY_SEQ: tl.constexpr = False,
    ):
        tok = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        seq = tok if IDENTITY_SEQ else tl.load(seq_of_token_ptr + tok)
        kv_len = tl.load(seqlen_k_ptr + seq)
        num_chunks = tl.cdiv(kv_len, CHUNK)
        offs_d = tl.arange(0, D)
        state = tok * stride_pt + head * stride_ph
        acc_base = pacc_ptr + tok * stride_pacc_t + head * stride_pacc_h + offs_d
        m = tl.load(pm_ptr + state)
        row_sum = tl.load(pl_ptr + state)
        acc = tl.load(acc_base)
        for chunk in range(1, num_chunks):
            mc = tl.load(pm_ptr + chunk * stride_pc + state)
            lc = tl.load(pl_ptr + chunk * stride_pc + state)
            accc = tl.load(acc_base + chunk * stride_pacc_c)
            chunk_empty = mc == float("-inf")
            prev_empty = m == float("-inf")
            m_new = tl.maximum(m, mc)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            a = tl.exp2(m - m_safe)
            b = tl.exp2(mc - m_safe)
            acc_merged = acc * a + accc * b
            l_merged = row_sum * a + lc * b
            acc = tl.where(chunk_empty, acc, tl.where(prev_empty, accc, acc_merged))
            row_sum = tl.where(chunk_empty, row_sum, tl.where(prev_empty, lc, l_merged))
            m = m_new
        out = acc / row_sum
        tl.store(
            o_ptr + tok * stride_ot + head * stride_oh + offs_d, out.to(o_ptr.dtype.element_ty)
        )
        if WRITE_LSE:
            lse = m * 0.6931471805599453 + tl.log(row_sum)
            tl.store(lse_ptr + head * stride_lh + tok * stride_lt, lse)


def _validate(q, k_cache, v_cache, cu_seqlens_q, block_table, seqlen_k, max_seqlen_q):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    if q.ndim != 3 or q.size(-1) != HEAD_DIM:
        raise ValueError(f"packed Q must be [tokens, heads, {HEAD_DIM}]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape or k_cache.size(1) != PAGE_SIZE:
        raise ValueError(f"paged K/V must be [pages, {PAGE_SIZE}, kv_heads, {HEAD_DIM}]")
    if k_cache.size(-1) != HEAD_DIM or q.size(1) % k_cache.size(2):
        raise ValueError("paged Q/K head counts or dimensions are incompatible")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
        raise ValueError("chunked flash attention supports one BF16/FP16 dtype for Q/K/V")
    if v_cache.dtype != q.dtype:
        raise ValueError("chunked flash attention supports one BF16/FP16 dtype for Q/K/V")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("Q/K/V head dimension must be contiguous")
    batch = block_table.size(0)
    if block_table.ndim != 2 or block_table.stride(-1) != 1:
        raise ValueError("block_table must be a 2-D row-major page table")
    if tuple(cu_seqlens_q.shape) != (batch + 1,) or tuple(seqlen_k.shape) != (batch,):
        raise ValueError("cu_seqlens_q and seqlen_k must carry batch + 1 and batch entries")
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqlen_k", seqlen_k),
        ("block_table", block_table),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the Q device")
    if not cu_seqlens_q.is_contiguous() or not seqlen_k.is_contiguous():
        raise ValueError("cu_seqlens_q and seqlen_k must be contiguous")
    if max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive")


def paged_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    out: torch.Tensor | None = None,
    return_lse: bool = True,
    schedule: str = "auto",
    seq_of_token: torch.Tensor | None = None,
    fast_path: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal paged attention; returns ``(out [tokens, heads, D], lse [heads, tokens])``.

    Query ``i`` of a sequence sits at key position ``seqlen_k - q_len + i``,
    which covers full prefill, prefix-cached extend and single-token decode.
    """

    _validate(q, k_cache, v_cache, cu_seqlens_q, block_table, seqlen_k, max_seqlen_q)
    total_q, num_q_heads, _ = q.shape
    num_kv_heads = k_cache.size(2)
    group = num_q_heads // num_kv_heads
    batch = block_table.size(0)
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("out must match Q shape and dtype with a contiguous head dimension")
    lse = torch.empty(
        (num_q_heads, total_q) if return_lse else (0,), dtype=torch.float32, device=q.device
    )
    if total_q == 0:
        return out, lse
    scale_log2 = float(scale) * _LOG2E
    if schedule == "auto":
        schedule = "split" if max_seqlen_q * group <= BLOCK_M else "monolithic"
    if schedule not in ("split", "monolithic"):
        raise ValueError("schedule must be 'auto', 'split' or 'monolithic'")
    common = dict(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CHUNK=CHUNK_KV,
        PAGE=PAGE_SIZE,
        D=HEAD_DIM,
        FAST_PATH=bool(fast_path),
    )
    if schedule == "monolithic":
        grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_q_heads, batch)
        _attn_fwd_monolithic_kernel[grid](
            q,
            k_cache,
            v_cache,
            out,
            lse,
            cu_seqlens_q,
            seqlen_k,
            block_table,
            block_table.stride(0),
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            out.stride(0),
            out.stride(1),
            lse.stride(0) if return_lse else 0,
            1,
            scale_log2,
            group,
            WRITE_LSE=return_lse,
            **common,
            **_COMPILE_OPTIONS,
        )
        return out, lse
    if max_seqlen_q * group > BLOCK_M:
        raise ValueError("split schedule requires max_seqlen_q * gqa_group <= BLOCK_M")
    # Decode packs only a few query/head rows. A smaller tile avoids masked
    # MFMA work without changing the per-row key traversal or reduction tree.
    split_options = _COMPILE_OPTIONS
    if max_seqlen_q * group <= 16:
        common["BLOCK_M"] = 16
        split_options = {
            **_COMPILE_OPTIONS,
            "num_warps": 2,
            "schedule_hint": "attention,memory-bound-attention",
        }
    num_chunks = triton.cdiv(int(block_table.size(1)) * PAGE_SIZE, CHUNK_KV)
    pm = torch.empty((num_chunks, total_q, num_q_heads), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty(
        (num_chunks, total_q, num_q_heads, HEAD_DIM), dtype=torch.float32, device=q.device
    )
    grid = (num_chunks, num_kv_heads, batch)
    _attn_fwd_split_partial_kernel[grid](
        q,
        k_cache,
        v_cache,
        pm,
        pl,
        pacc,
        cu_seqlens_q,
        seqlen_k,
        block_table,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        pm.stride(0),
        pm.stride(1),
        pm.stride(2),
        pacc.stride(0),
        pacc.stride(1),
        pacc.stride(2),
        scale_log2,
        group,
        **common,
        **split_options,
    )
    identity_seq = seq_of_token is None and total_q == batch and max_seqlen_q == 1
    if seq_of_token is None and not identity_seq:
        if total_q == batch:
            seq_of_token = torch.arange(total_q, dtype=torch.int32, device=q.device)
        else:
            tokens = torch.arange(total_q, dtype=torch.int32, device=q.device)
            seq_of_token = torch.searchsorted(cu_seqlens_q[1:], tokens, right=True).to(torch.int32)
    _attn_fwd_split_merge_kernel[(total_q, num_q_heads)](
        pm,
        pl,
        pacc,
        out,
        lse,
        seq_of_token,
        seqlen_k,
        pm.stride(0),
        pm.stride(1),
        pm.stride(2),
        pacc.stride(0),
        pacc.stride(1),
        pacc.stride(2),
        out.stride(0),
        out.stride(1),
        lse.stride(0) if return_lse else 0,
        1,
        CHUNK=CHUNK_KV,
        D=HEAD_DIM,
        WRITE_LSE=return_lse,
        IDENTITY_SEQ=identity_seq,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out, lse


def triton_paged_prefill(
    q,
    k,
    v,
    cuq,
    indptr,
    flat_pages,
    maxq,
    maxk,
    dropout,
    scale,
    softcap,
    zero_tensors,
    causal,
    window_left,
    window_right,
    sink,
    return_lse,
    return_dropout,
    *,
    block_table,
    seqlen_k,
    out=None,
):
    """AITER ``mha_batch_prefill``-shaped entry point over the chunked contract.

    Mirrors the fixed CK entry point so the strict core can bind either one.
    Single-token rows see their whole prefix under causal masking, so the
    ``causal=False`` decode request is served by the same causal kernel.
    """

    del indptr, flat_pages, maxk
    if dropout or softcap or sink or return_dropout or zero_tensors:
        raise ValueError(
            "chunked flash attention requires no dropout, softcap, sink or dropout mask"
        )
    if window_left != -1 or window_right != -1:
        raise ValueError("chunked flash attention does not support sliding windows")
    if not causal and int(maxq) != 1:
        raise ValueError("non-causal attention is only served for single-token decode rows")
    resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
    out, lse = paged_attention_forward(
        q,
        k,
        v,
        cu_seqlens_q=cuq,
        block_table=block_table,
        seqlen_k=seqlen_k,
        max_seqlen_q=int(maxq),
        scale=resolved_scale,
        out=out,
        return_lse=bool(return_lse),
    )
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, lse, torch.empty((0,), dtype=q.dtype, device=q.device), rng_state


def warmup(
    device: torch.device, *, num_q_heads: int, num_kv_heads: int, dtype: torch.dtype
) -> None:
    """Compile both schedules so no Triton JIT happens inside HIP Graph capture."""

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm chunked flash attention before HIP Graph capture")
    group = num_q_heads // num_kv_heads
    pages = 3
    k_cache = torch.zeros((pages, PAGE_SIZE, num_kv_heads, HEAD_DIM), device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(pages, device=device, dtype=torch.int32).reshape(1, pages)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    with torch.inference_mode():
        for q_len in (1, BLOCK_M // group, BLOCK_M + 1):
            q = torch.zeros((q_len, num_q_heads, HEAD_DIM), device=device, dtype=dtype)
            cu = torch.tensor((0, q_len), device=device, dtype=torch.int32)
            seqlen = torch.tensor((pages * PAGE_SIZE,), device=device, dtype=torch.int32)
            for schedule in ("split", "monolithic"):
                if schedule == "split" and q_len * group > BLOCK_M:
                    continue
                for return_lse in (False, True):
                    paged_attention_forward(
                        q,
                        k_cache,
                        v_cache,
                        cu_seqlens_q=cu,
                        block_table=block_table,
                        seqlen_k=seqlen,
                        max_seqlen_q=q_len,
                        scale=scale,
                        return_lse=return_lse,
                        schedule=schedule,
                    )
    torch.cuda.synchronize(device)


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "CHUNKED_FLASH_ATTENTION_CONTRACT_ID",
    "CHUNK_KV",
    "HEAD_DIM",
    "PAGE_SIZE",
    "paged_attention_forward",
    "triton_paged_prefill",
    "warmup",
]
