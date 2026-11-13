# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modifications copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant DeepSeek-V4 mHC pre on ROCm (Triton).

Copied from aiter 326cffa: the kernels from
aiter/ops/triton/_triton_kernels/fusions/mhc.py (``_mhc_fused_split_kernel``,
``_mhc_reduce_apply_kernel``, ``_mhc_asymmetric_sinkhorn_kernel``) and the
launchers from aiter/ops/triton/fusions/mhc.py (``mhc``, ``mhc_pre_dsv4``). aiter's
arithmetic is kept except for three changes, so the results differ from aiter's
only by the order of the sums and by the fp32 projection
(docs/operators/dsv4-mhc.md, section "Accuracy"):

1. Batch invariance by construction. Every M runs the same two kernels: a split
   kernel over NUM_KSPLIT = 4 segments of K, then an apply kernel that adds the
   segments in segment order (aiter switches between a fused kernel and the split
   pair, and picks the split count by M). Every sum has its order written in the
   source instead of leaving it to ``tl.sum``, whose tree follows the tile layout:
   the sum of squares is accumulated element-wise with ``tl.fma`` over the K tiles
   of a segment, then the BLOCK_K = 128 columns are added pairwise; the weighted
   sum of the n streams is one ``tl.fma`` per stream in stream order; the Sinkhorn
   row and column sums are pairwise. BLOCK_K and NUM_KSPLIT are module constants;
   the other launch parameters only regroup tokens and hidden positions.
2. Dynamic shape. M is a runtime argument of every kernel and is not specialized
   (``do_not_specialize``), and the partial results are stored [M, split, column],
   so no stride depends on M. A new token count does not compile anything.
3. Projection precision of the DeepSeek-V4 model: ``fn`` stays fp32 and the
   projection is an fp32 ``tl.dot`` with ``input_precision="ieee"`` (aiter rounds
   ``fn`` to bf16). Its MFMA shape is fixed (``matrix_instr_nonkdim=16``,
   ``kpack=1``).

Backward is RL-Kernel's own (aiter recomputes the formula in PyTorch): Triton
kernels that compute the exact derivative of the forward above, row by row.

- Saved, not recomputed: with grad enabled the forward kernels also store, per
  token, the sigmoids of the pre and post logits, the raw
  comb logits, the projection ``acc`` and the RMS factor ``r`` (49 floats for
  mHC pre). The values are those the forward used; without grad
  nothing is stored (``SAVE`` is a constexpr, the stores are the only difference).
- Sinkhorn backward replays the rounds from the saved logits with the forward's
  own helpers, once per step it walks back through (2 * rounds steps, so about
  (2 * rounds)^2 / 2 replayed steps per token, all in registers).
- Every reduction has its order written in the source, as in the forward, and
  none depends on M or on a launch parameter: sums over C (one per stream) are
  accumulated element-wise with ``tl.fma`` over C
  tiles of DOT_BLOCK = 128, then the 128 columns are added pairwise; sums over the
  rows of one group of logits are pairwise; the transposed projection
  ``fn^T @ d acc`` is one ``tl.fma`` per ``fn`` row in row order; sums over the n
  streams are one ``tl.fma`` per stream in stream order. The backward kernels run
  with ``enable_fp_fusion=False``, so only the written ``tl.fma`` fuse.
- Parameter gradients (``fn``, ``scale``, ``base``) are computed only when asked
  for, and only through the canonical backward session
  (rl_engine/ops/autograd/canonical_backward.py): the forward registers the
  caller's logical keys, the backward submits per-row pieces, and the session
  reduces them in logical-key order (``fn``: the fixed-order fp32 det_gemm;
  ``scale``, ``base``: a serial fold). Without a session they raise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from torch.autograd.function import once_differentiable

from rl_engine.contracts.operators.mhc import PROFILE
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.utils.logger import logger

# Part of the arithmetic: changing them changes the bits. These values, and the
# launch configs below, are validated on gfx942 only; the op constructors raise on
# any other architecture (_require_rocm).
VALIDATED_ARCH = "gfx942"
NUM_KSPLIT = 4
BLOCK_K = 128
DOT_OPTIONS = {"matrix_instr_nonkdim": 16, "kpack": 1}
# Backward: width of the element-wise accumulators of every sum over C.
DOT_BLOCK = 128

# Launch configs. They do not change the bits (only how tokens and hidden
# positions are grouped into workgroups); one config for every M, so each kernel
# compiles once per process.
PRE_CONFIG = {
    "BLOCK_M": 32,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "APPLY_BLOCK_M": 32,
    "APPLY_BLOCK_C": 128,
    "apply_num_warps": 4,
    "SINKHORN_BLOCK_M": 16,
}

# Backward launch configs. MIX_* is the kernel that
# reduces over C per token (one workgroup per MIX_BLOCK_M tokens, the whole of C);
# DX_* the element-wise input-gradient kernels. They do not change the bits either.
BACKWARD_CONFIG = {
    "MIX_BLOCK_M": 8,
    "mix_num_warps": 4,
    "DX_BLOCK_M": 32,
    "DX_BLOCK_C": 128,
    "dx_num_warps": 4,
}
# Every backward kernel rounds each operation on its own except the written tl.fma.
_BACKWARD_LAUNCH = {"enable_fp_fusion": False}

# DeepSeek-V4's hc_mult; the ops are validated with it only.
SUPPORTED_N = 4


@triton.jit
def _pairwise_sum_last(v, ROWS: tl.constexpr, WIDTH: tl.constexpr, LOG2_WIDTH: tl.constexpr):
    """[ROWS, WIDTH] -> [ROWS]: add adjacent pairs, LOG2_WIDTH levels.

    Each ``tl.sum`` adds exactly two numbers, which rounds the same way whatever
    lanes hold them, so the order is fixed by WIDTH alone.
    """
    for i in tl.static_range(LOG2_WIDTH):
        v = tl.sum(tl.reshape(v, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
    return tl.reshape(v, (ROWS,))


@triton.jit
def _sum_splits(ptr, offsets, mask, stride_split, NUM_SPLITS: tl.constexpr):
    """Partial results of the K segments added in segment order."""
    total = tl.load(ptr + offsets, mask=mask, other=0.0)
    for s in tl.static_range(1, NUM_SPLITS):
        total += tl.load(ptr + s * stride_split + offsets, mask=mask, other=0.0)
    return total


@triton.jit(do_not_specialize=["M"])
def _mhc_fused_split_kernel(
    x_ptr,
    phi_ptr,  # fn in checkpoint layout (N, K), fp32
    acc_ptr,  # (M, NUM_KSPLIT, N) fp32
    acc_sq_ptr,  # (M, NUM_KSPLIT) fp32
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_phi_k,
    stride_phi_n,
    stride_acc_m,
    stride_acc_k,
    stride_acc_n,
    stride_acc_sq_m,
    stride_acc_sq_k,
    BLOCK_M: tl.constexpr,
    N_TOTAL_POW2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2_BLOCK_K: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
):
    """Partial projection x @ fn^T and partial sum of x^2 of one K segment.

    Grid: (cdiv(M, BLOCK_M), NUM_KSPLIT). SPLITK_BLOCK_SIZE is a multiple of
    BLOCK_K; K positions past the end of K are loaded as zeros.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = tl.arange(0, N_TOTAL_POW2)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)

    acc = tl.zeros([BLOCK_M, N_TOTAL_POW2], dtype=tl.float32)
    # One accumulator per element of the tile: no sum inside the K loop.
    acc_sq = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for k_idx in range(SPLITK_BLOCK_SIZE // BLOCK_K):
        rk = split_k_start + k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < split_k_end),
            other=0.0,
        ).to(tl.float32)
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn[None, :] * stride_phi_n,
            mask=(rk[:, None] < split_k_end) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(x_tile, phi_tile, acc=acc, input_precision="ieee")
        acc_sq = tl.fma(x_tile, x_tile, acc_sq)

    tl.store(
        acc_ptr + rm[:, None] * stride_acc_m + pid_k * stride_acc_k + rn[None, :] * stride_acc_n,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
    tl.store(
        acc_sq_ptr + rm * stride_acc_sq_m + pid_k * stride_acc_sq_k,
        _pairwise_sum_last(acc_sq, BLOCK_M, BLOCK_K, LOG2_BLOCK_K),
        mask=rm < M,
    )


@triton.jit(do_not_specialize=["M"])
def _mhc_reduce_apply_kernel(
    acc_ptr,  # (M, NUM_KSPLIT, n + n + n_squared), layout [pre | post | res]
    acc_sq_ptr,  # (M, NUM_KSPLIT)
    alpha_ptr,  # (3,) fp32: alpha_pre, alpha_post, alpha_res
    bias_ptr,  # (n + n + n_squared,) fp32
    x_ptr,  # (M, n*C)
    out_ptr,  # (M, n + n_squared), layout [post | res logits]
    layer_input_ptr,  # (M, C) in x.dtype
    act_ptr,  # SAVE: (M, 2n + n_squared) fp32 [sigmoid(pre) | sigmoid(post) | res logits]
    proj_ptr,  # SAVE: (M, 2n + n_squared) fp32, the projection acc
    rsigma_ptr,  # SAVE: (M,) fp32
    M,
    K: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    stride_acc_m,
    stride_acc_k,
    stride_acc_n,
    stride_acc_sq_m,
    stride_acc_sq_k,
    stride_xm,
    stride_xk,
    stride_out_m,
    stride_out_n,
    stride_li_m,
    stride_li_c,
    stride_save_m,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    N_POW2_RES: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RES_PID_C: tl.constexpr,
    SAVE: tl.constexpr,
):
    """Add the K segments, RMS factor, the three streams and the apply step.

    Grid: ``(cdiv(M, BLOCK_M), cdiv(C, BLOCK_C))``. Every program computes the pre
    weights and its BLOCK_C slice of ``layer_input``; ``pid_c == 0`` also writes
    the post weights, ``pid_c == RES_PID_C`` the raw comb logits (Sinkhorn runs
    in ``_mhc_sinkhorn_kernel``). With SAVE, the same programs also store the
    values the backward needs (``act``, ``proj``, ``rsigma``); nothing else changes.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    alpha_pre_val = tl.load(alpha_ptr)
    alpha_post_val = tl.load(alpha_ptr + 1)
    alpha_res_val = tl.load(alpha_ptr + 2)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M

    # --- 1) RMS factor ---
    acc_sq = _sum_splits(acc_sq_ptr, rm * stride_acc_sq_m, row_ok, stride_acc_sq_k, NUM_SPLITS)
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms
    if SAVE:
        if pid_c == 0:
            tl.store(rsigma_ptr + rm, rsigma, mask=row_ok)

    # --- 2) Pre weights and the apply step, one stream at a time ---
    #   layer_input = pre_0 * x_0, then fma(pre_i, x_i, layer_input) for i = 1 .. n-1
    tile_ok = row_ok[:, None] & (rc[None, :] < C)
    layer_input = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(n):
        acc_pre = _sum_splits(
            acc_ptr, rm * stride_acc_m + i * stride_acc_n, row_ok, stride_acc_k, NUM_SPLITS
        )
        h_pre = rsigma * alpha_pre_val * acc_pre + tl.load(bias_ptr + i)
        sig_pre = tl.sigmoid(h_pre)
        pre_mix = sig_pre + hc_pre_eps
        if SAVE:
            if pid_c == 0:
                tl.store(act_ptr + rm * stride_save_m + i, sig_pre, mask=row_ok)
                tl.store(proj_ptr + rm * stride_save_m + i, acc_pre, mask=row_ok)
        x_i = tl.load(
            x_ptr + rm[:, None] * stride_xm + (i * C + rc[None, :]) * stride_xk,
            mask=tile_ok,
            other=0.0,
        ).to(tl.float32)
        if i == 0:
            layer_input = pre_mix[:, None] * x_i
        else:
            layer_input = tl.fma(pre_mix[:, None], x_i, layer_input)
    tl.store(
        layer_input_ptr + rm[:, None] * stride_li_m + rc[None, :] * stride_li_c,
        layer_input.to(layer_input_ptr.dtype.element_ty),
        mask=tile_ok,
    )

    # --- 3) Post weights on pid_c == 0 ---
    if pid_c == 0:
        rn_post_local = tl.arange(0, N_POW2)
        rn_post_global = n + rn_post_local
        post_mask = row_ok[:, None] & (rn_post_local[None, :] < n)
        acc_post = _sum_splits(
            acc_ptr,
            rm[:, None] * stride_acc_m + rn_post_global[None, :] * stride_acc_n,
            post_mask,
            stride_acc_k,
            NUM_SPLITS,
        )
        bias_post = tl.load(bias_ptr + rn_post_global, mask=rn_post_local < n, other=0.0)
        h_post = rsigma[:, None] * alpha_post_val * acc_post + bias_post[None, :]
        sig_post = tl.sigmoid(h_post)
        out_post = sig_post * hc_post_mult_value
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + rn_post_local[None, :] * stride_out_n,
            out_post,
            mask=post_mask,
        )
        if SAVE:
            save_post = rm[:, None] * stride_save_m + rn_post_global[None, :]
            tl.store(act_ptr + save_post, sig_post, mask=post_mask)
            tl.store(proj_ptr + save_post, acc_post, mask=post_mask)

    # --- 4) Comb logits on pid_c == RES_PID_C (Sinkhorn runs in its own kernel) ---
    if pid_c == RES_PID_C:
        rn_res_local = tl.arange(0, N_POW2_RES)
        rn_res_global = 2 * n + rn_res_local
        res_mask = row_ok[:, None] & (rn_res_local[None, :] < n_squared)
        acc_res = _sum_splits(
            acc_ptr,
            rm[:, None] * stride_acc_m + rn_res_global[None, :] * stride_acc_n,
            res_mask,
            stride_acc_k,
            NUM_SPLITS,
        )
        bias_res = tl.load(bias_ptr + rn_res_global, mask=rn_res_local < n_squared, other=0.0)
        h_res = rsigma[:, None] * alpha_res_val * acc_res + bias_res[None, :]
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + (n + rn_res_local[None, :]) * stride_out_n,
            h_res,
            mask=res_mask,
        )
        if SAVE:
            save_res = rm[:, None] * stride_save_m + rn_res_global[None, :]
            tl.store(act_ptr + save_res, h_res, mask=res_mask)
            tl.store(proj_ptr + save_res, acc_res, mask=res_mask)


@triton.jit
def _row_sums(p, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    """[BLOCK_M, n, n] -> [BLOCK_M, n]: sum over the last axis, pairwise."""
    flat = tl.reshape(p, (BLOCK_M * n, n))
    return tl.reshape(_pairwise_sum_last(flat, BLOCK_M * n, n, LOG2_N), (BLOCK_M, n))


@triton.jit
def _col_sums(p, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    """[BLOCK_M, n, n] -> [BLOCK_M, n]: sum over the middle axis, pairwise."""
    return _row_sums(tl.permute(p, (0, 2, 1)), BLOCK_M, n, LOG2_N)


# The Sinkhorn steps, shared by the forward and the backward replay, so both run
# the same arithmetic. None has a multiply followed by an add, so fp fusion
# cannot change them either.
@triton.jit
def _sinkhorn_softmax(values, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    """Row softmax of the logits, [BLOCK_M, n, n]."""
    row_max = tl.max(values, axis=2)
    P = tl.exp(values - row_max[:, :, None])
    return P / _row_sums(P, BLOCK_M, n, LOG2_N)[:, :, None]


@triton.jit
def _sinkhorn_divide_rows(P, eps, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    return P / (_row_sums(P, BLOCK_M, n, LOG2_N)[:, :, None] + eps)


@triton.jit
def _sinkhorn_divide_cols(P, eps, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    return P / (_col_sums(P, BLOCK_M, n, LOG2_N)[:, None, :] + eps)


@triton.jit(do_not_specialize=["M"])
def _mhc_sinkhorn_kernel(
    logits_ptr,
    out_ptr,
    M,
    stride_logits_m,
    stride_logits_n,
    stride_out_m,
    stride_out_n,
    n: tl.constexpr,
    LOG2_N: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """DeepSeek-V4's asymmetric Sinkhorn on the comb logits, BLOCK_M tokens per program.

    aiter's ``_mhc_asymmetric_sinkhorn_kernel`` with the row and column sums
    written as pairwise additions; the maxima need no fixed order.
    """
    pid = tl.program_id(0)
    rm = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    ri = tl.arange(0, n)
    entry = ri[None, :, None] * n + ri[None, None, :]
    mask = (rm[:, None, None] < M) & (entry >= 0)
    values = tl.load(
        logits_ptr + rm[:, None, None] * stride_logits_m + entry * stride_logits_n,
        mask=mask,
        other=0.0,
    )
    if NUM_SINKHORN_ITERS > 0:
        P = _sinkhorn_softmax(values, BLOCK_M, n, LOG2_N) + eps
        P = _sinkhorn_divide_cols(P, eps, BLOCK_M, n, LOG2_N)
        for _ in range(NUM_SINKHORN_ITERS - 1):
            P = _sinkhorn_divide_rows(P, eps, BLOCK_M, n, LOG2_N)
            P = _sinkhorn_divide_cols(P, eps, BLOCK_M, n, LOG2_N)
        values = P
    tl.store(out_ptr + rm[:, None, None] * stride_out_m + entry * stride_out_n, values, mask=mask)


# --- backward kernels ------------------------------------------------------------------------
# Row-major buffers allocated by the launchers (and the contiguous gradients) are
# passed with their row stride only; their column stride is 1.


@triton.jit
def _stream_dots(
    g_ptr,
    x_ptr,
    rm,
    row_ok,
    stride_gm,
    stride_xm,
    C: tl.constexpr,
    n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DOT_BLOCK: tl.constexpr,
    LOG2_DOT_BLOCK: tl.constexpr,
):
    """[BLOCK_M, n]: sum over c of g[m, c] * x[m, i * C + c] for every stream i.

    One fp32 accumulator per (token, stream, column of the tile), updated with
    ``tl.fma`` over the C tiles of DOT_BLOCK in order, then the DOT_BLOCK columns
    added pairwise. Positions past C are loaded as zeros.
    """
    rn = tl.arange(0, n)
    acc = tl.zeros([BLOCK_M, n, DOT_BLOCK], dtype=tl.float32)
    for c_start in range(0, C, DOT_BLOCK):
        rc = c_start + tl.arange(0, DOT_BLOCK)
        c_ok = row_ok[:, None] & (rc[None, :] < C)
        g = tl.load(g_ptr + rm[:, None] * stride_gm + rc[None, :], mask=c_ok, other=0.0)
        x = tl.load(
            x_ptr + rm[:, None, None] * stride_xm + rn[None, :, None] * C + rc[None, None, :],
            mask=c_ok[:, None, :],
            other=0.0,
        )
        acc = tl.fma(g.to(tl.float32)[:, None, :], x.to(tl.float32), acc)
    flat = tl.reshape(acc, (BLOCK_M * n, DOT_BLOCK))
    return tl.reshape(
        _pairwise_sum_last(flat, BLOCK_M * n, DOT_BLOCK, LOG2_DOT_BLOCK), (BLOCK_M, n)
    )


@triton.jit
def _sinkhorn_backward(
    values,
    grad,
    eps,
    NUM_SINKHORN_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    n: tl.constexpr,
    LOG2_N: tl.constexpr,
):
    """Gradient of the comb logits ``values`` [BLOCK_M, n, n] from the gradient of comb.

    The 2 * NUM_SINKHORN_ITERS steps are: step 0 the row softmax plus eps, then
    columns (odd steps) and rows (even steps) divided by (sum + eps). The input
    of each step is replayed from the logits with the forward's helpers, and the
    step is walked back with its exact derivative: for Y = P / d, d = sum(P) + eps
    along the step's axis, dP = (dY - sum(dY * Y)) / d; for Y = S + eps with S the
    softmax, d logits = S * (dY - row sum(dY * S)). Every sum is pairwise.
    """
    dP = grad
    if NUM_SINKHORN_ITERS > 0:
        for k in range(0, 2 * NUM_SINKHORN_ITERS - 1):
            step = 2 * NUM_SINKHORN_ITERS - 1 - k
            # Input of this step: steps 0 .. step - 1 replayed.
            P = _sinkhorn_softmax(values, BLOCK_M, n, LOG2_N) + eps
            for t in range(1, step):
                if t % 2 == 1:
                    P = _sinkhorn_divide_cols(P, eps, BLOCK_M, n, LOG2_N)
                else:
                    P = _sinkhorn_divide_rows(P, eps, BLOCK_M, n, LOG2_N)
            if step % 2 == 1:
                d_col = _col_sums(P, BLOCK_M, n, LOG2_N)[:, None, :] + eps
                y_col = P / d_col
                dP = (dP - _col_sums(dP * y_col, BLOCK_M, n, LOG2_N)[:, None, :]) / d_col
            else:
                d_row = _row_sums(P, BLOCK_M, n, LOG2_N)[:, :, None] + eps
                y_row = P / d_row
                dP = (dP - _row_sums(dP * y_row, BLOCK_M, n, LOG2_N)[:, :, None]) / d_row
        S = _sinkhorn_softmax(values, BLOCK_M, n, LOG2_N)
        dP = S * (dP - _row_sums(dP * S, BLOCK_M, n, LOG2_N)[:, :, None])
    return dP


@triton.jit(do_not_specialize=["M"])
def _mhc_pre_backward_mix_kernel(
    x_ptr,  # (M, n*C) residual
    grad_li_ptr,  # (M, C) gradient of layer_input
    grad_post_ptr,  # (M, n) fp32
    grad_comb_ptr,  # (M, n*n) fp32
    act_ptr,  # (M, 2n + n*n) saved [sigmoid(pre) | sigmoid(post) | res logits]
    proj_ptr,  # (M, 2n + n*n) saved projection acc
    rsigma_ptr,  # (M,) saved
    alpha_ptr,  # (3,) scale
    dlogit_ptr,  # out (M, 2n + n*n): gradient of the logits (per-row base gradient)
    dproj_ptr,  # out (M, 2n + n*n): gradient of acc
    dscale_ptr,  # out (M, 3): per-row scale gradient
    rms_coef_ptr,  # out (M,): coefficient of x in dResidual from the RMS factor
    M,
    K: tl.constexpr,
    n: tl.constexpr,
    C: tl.constexpr,
    LOG2_N: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_eps: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    stride_xm,
    stride_gm,
    stride_gpm,
    stride_gcm,
    stride_save_m,
    BLOCK_M: tl.constexpr,
    DOT_BLOCK: tl.constexpr,
    LOG2_DOT_BLOCK: tl.constexpr,
):
    """mHC pre backward up to d acc, BLOCK_M tokens per program over the whole of C.

    Per token, with s the saved sigmoids, a = scale and r the RMS factor:
      d pre_i  = sum_c g[c] * x_i[c]                          (_stream_dots)
      d h_pre  = d pre * (s_pre * (1 - s_pre))
      d h_post = (g_post * hc_post_mult_value) * (s_post * (1 - s_post))
      d h_res  = Sinkhorn backward of g_comb                  (_sinkhorn_backward)
      d acc    = d h * (r * a_group);  t_group = pairwise sum of d h * acc
      d scale  = r * t_group;  d r = fma(a_res, t_res, fma(a_post, t_post, a_pre * t_pre))
      rms_coef = -(d r * ((r * r) * r)) / K, since d r / d x = -r^3 x / K
    """
    pid = tl.program_id(0)
    rm = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = tl.arange(0, n)
    row_ok = rm < M
    vec_ok = row_ok[:, None] & (rn[None, :] < n)
    entry = rn[:, None] * n + rn[None, :]
    mat_ok = row_ok[:, None, None] & (entry[None, :, :] < n * n)
    alpha_pre = tl.load(alpha_ptr)
    alpha_post = tl.load(alpha_ptr + 1)
    alpha_res = tl.load(alpha_ptr + 2)
    rsigma = tl.load(rsigma_ptr + rm, mask=row_ok, other=0.0)

    save_pre = rm[:, None] * stride_save_m + rn[None, :]
    save_post = save_pre + n
    save_res = rm[:, None, None] * stride_save_m + 2 * n + entry[None, :, :]

    # --- 1) Pre rows ---
    dpre = _stream_dots(
        grad_li_ptr, x_ptr, rm, row_ok, stride_gm, stride_xm, C, n, BLOCK_M, DOT_BLOCK,
        LOG2_DOT_BLOCK,
    )
    s_pre = tl.load(act_ptr + save_pre, mask=vec_ok, other=0.0)
    dh_pre = dpre * (s_pre * (1.0 - s_pre))
    acc_pre = tl.load(proj_ptr + save_pre, mask=vec_ok, other=0.0)
    t_pre = _pairwise_sum_last(dh_pre * acc_pre, BLOCK_M, n, LOG2_N)
    tl.store(dlogit_ptr + save_pre, dh_pre, mask=vec_ok)
    tl.store(dproj_ptr + save_pre, dh_pre * (rsigma * alpha_pre)[:, None], mask=vec_ok)

    # --- 2) Post rows ---
    g_post = tl.load(grad_post_ptr + rm[:, None] * stride_gpm + rn[None, :], mask=vec_ok, other=0.0)
    s_post = tl.load(act_ptr + save_post, mask=vec_ok, other=0.0)
    dh_post = (g_post * hc_post_mult_value) * (s_post * (1.0 - s_post))
    acc_post = tl.load(proj_ptr + save_post, mask=vec_ok, other=0.0)
    t_post = _pairwise_sum_last(dh_post * acc_post, BLOCK_M, n, LOG2_N)
    tl.store(dlogit_ptr + save_post, dh_post, mask=vec_ok)
    tl.store(dproj_ptr + save_post, dh_post * (rsigma * alpha_post)[:, None], mask=vec_ok)

    # --- 3) Comb rows ---
    logits = tl.load(act_ptr + save_res, mask=mat_ok, other=0.0)
    g_comb = tl.load(
        grad_comb_ptr + rm[:, None, None] * stride_gcm + entry[None, :, :], mask=mat_ok, other=0.0
    )
    dh_res = _sinkhorn_backward(
        logits, g_comb, sinkhorn_eps, NUM_SINKHORN_ITERS, BLOCK_M, n, LOG2_N
    )
    acc_res = tl.load(proj_ptr + save_res, mask=mat_ok, other=0.0)
    t_res = _pairwise_sum_last(
        tl.reshape(dh_res * acc_res, (BLOCK_M, n * n)), BLOCK_M, n * n, 2 * LOG2_N
    )
    tl.store(dlogit_ptr + save_res, dh_res, mask=mat_ok)
    tl.store(dproj_ptr + save_res, dh_res * (rsigma * alpha_res)[:, None, None], mask=mat_ok)

    # --- 4) Scale and RMS factor ---
    tl.store(dscale_ptr + rm * 3, rsigma * t_pre, mask=row_ok)
    tl.store(dscale_ptr + rm * 3 + 1, rsigma * t_post, mask=row_ok)
    tl.store(dscale_ptr + rm * 3 + 2, rsigma * t_res, mask=row_ok)
    d_rsigma = tl.fma(alpha_res, t_res, tl.fma(alpha_post, t_post, alpha_pre * t_pre))
    tl.store(rms_coef_ptr + rm, -(d_rsigma * ((rsigma * rsigma) * rsigma)) / K, mask=row_ok)


@triton.jit(do_not_specialize=["M"])
def _mhc_backward_dx_kernel(
    x_ptr,  # (M, n*C) residual
    grad_ptr,  # (M, C) gradient of layer_input
    act_ptr,  # (M, >= n) saved; columns 0 .. n-1 are the sigmoids of the stream weights
    dproj_ptr,  # (M, N_ROWS) gradient of acc
    rms_coef_ptr,  # (M,)
    fn_ptr,  # (N_ROWS, n*C) fp32
    dx_ptr,  # out (M, n*C) in x.dtype
    M,
    n: tl.constexpr,
    C: tl.constexpr,
    N_ROWS: tl.constexpr,
    pre_eps: tl.constexpr,
    stride_xm,
    stride_gm,
    stride_save_m,
    stride_dproj_m,
    stride_fn_n,
    stride_dxm,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """dResidual of mHC pre, element-wise; one (M tile, C tile) per program.

      dx_i[c] = w_i * g[c]                                    w_i = s_i + pre_eps
      then fma(fn[j, i*C + c], d acc_j, dx_i[c]) for j = 0 .. N_ROWS - 1 in order
      then fma(rms_coef, x_i[c], dx_i[c]), rounded to x.dtype once.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M
    c_ok = rc < C
    tile_ok = row_ok[:, None] & c_ok[None, :]
    g = tl.load(grad_ptr + rm[:, None] * stride_gm + rc[None, :], mask=tile_ok, other=0.0)
    g = g.to(tl.float32)
    rms_coef = tl.load(rms_coef_ptr + rm, mask=row_ok, other=0.0)
    for i in tl.static_range(n):
        w = tl.load(act_ptr + rm * stride_save_m + i, mask=row_ok, other=0.0) + pre_eps
        dx = w[:, None] * g
        for j in tl.static_range(N_ROWS):
            dacc = tl.load(dproj_ptr + rm * stride_dproj_m + j, mask=row_ok, other=0.0)
            fn_j = tl.load(fn_ptr + j * stride_fn_n + i * C + rc, mask=c_ok, other=0.0)
            dx = tl.fma(dacc[:, None], fn_j[None, :], dx)
        x_i = tl.load(x_ptr + rm[:, None] * stride_xm + i * C + rc[None, :], mask=tile_ok, other=0.0)
        dx = tl.fma(rms_coef[:, None], x_i.to(tl.float32), dx)
        tl.store(
            dx_ptr + rm[:, None] * stride_dxm + i * C + rc[None, :],
            dx.to(dx_ptr.dtype.element_ty),
            mask=tile_ok,
        )


@triton.jit(do_not_specialize=["T"])
def _mhc_fold_rows_kernel(rows_ptr, out_ptr, T, W: tl.constexpr, stride_t, BLOCK_W: tl.constexpr):
    """out[w] = sum over t of rows[t, w], one row at a time in ascending t (canonical order)."""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = cols < W
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    ptrs = rows_ptr + cols
    for _ in tl.range(0, T, loop_unroll_factor=1):
        acc = acc + tl.load(ptrs, mask=mask, other=0.0)
        ptrs += stride_t
    tl.store(out_ptr + cols, acc, mask=mask)


# --- launchers -------------------------------------------------------------------------------


def _log2(v: int) -> int:
    if v <= 0 or v & (v - 1):
        raise ValueError(f"{v} is not a power of two")
    return v.bit_length() - 1


def _split(x: torch.Tensor, fn: torch.Tensor, config: dict):
    """Partial projections (M, NUM_KSPLIT, N) and sums of squares (M, NUM_KSPLIT)."""
    M, K = x.shape
    N = fn.shape[0]
    # A multiple of BLOCK_K, from K only.
    splitk_block_size = triton.cdiv(triton.cdiv(K, NUM_KSPLIT), BLOCK_K) * BLOCK_K
    acc = torch.empty((M, NUM_KSPLIT, N), dtype=torch.float32, device=x.device)
    acc_sq = torch.empty((M, NUM_KSPLIT), dtype=torch.float32, device=x.device)
    BLOCK_M = config["BLOCK_M"]
    _mhc_fused_split_kernel[(triton.cdiv(M, BLOCK_M), NUM_KSPLIT)](
        x,
        fn,
        acc,
        acc_sq,
        M,
        K=K,
        N=N,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_phi_k=fn.stride(1),
        stride_phi_n=fn.stride(0),
        stride_acc_m=acc.stride(0),
        stride_acc_k=acc.stride(1),
        stride_acc_n=acc.stride(2),
        stride_acc_sq_m=acc_sq.stride(0),
        stride_acc_sq_k=acc_sq.stride(1),
        BLOCK_M=BLOCK_M,
        N_TOTAL_POW2=max(16, triton.next_power_of_2(N)),
        BLOCK_K=BLOCK_K,
        LOG2_BLOCK_K=_log2(BLOCK_K),
        SPLITK_BLOCK_SIZE=splitk_block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        **DOT_OPTIONS,
    )
    return acc, acc_sq


def _saved_buffers(M, rows, device, save):
    """(act, proj, rsigma) the forward stores for the backward, or None without SAVE."""
    if not save:
        return None
    return (
        torch.empty((M, rows), dtype=torch.float32, device=device),
        torch.empty((M, rows), dtype=torch.float32, device=device),
        torch.empty((M,), dtype=torch.float32, device=device),
    )


def _mhc_pre_dsv4_forward(
    residual,
    fn,
    scale,
    base,
    eps,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_iters,
    config,
    save=False,
):
    """(post_mix, comb_mix, layer_input, saved); ``saved`` is (act, proj, rsigma) with
    ``save``, else None."""
    M, n, C = residual.shape
    n_squared = n * n
    saved = _saved_buffers(M, 2 * n + n_squared, residual.device, save)
    if M == 0:
        return (
            torch.empty((0, n, 1), dtype=torch.float32, device=residual.device),
            torch.empty((0, n, n), dtype=torch.float32, device=residual.device),
            torch.empty((0, C), dtype=residual.dtype, device=residual.device),
            saved,
        )
    cfg = dict(PRE_CONFIG, **(config or {}))
    x = residual.view(M, n * C)
    acc, acc_sq = _split(x, fn, cfg)

    out = torch.empty((M, n + n_squared), dtype=torch.float32, device=x.device)
    layer_input = torch.empty((M, C), dtype=x.dtype, device=x.device)
    # Without SAVE the three pointers are never written; any fp32 buffer will do.
    act, proj, rsigma = saved if save else (out, out, out)
    BLOCK_M, BLOCK_C = cfg["APPLY_BLOCK_M"], cfg["APPLY_BLOCK_C"]
    # Post weights on pid_c == 0, comb logits on pid_c == 1 when there are two C tiles.
    RES_PID_C = 0 if triton.cdiv(C, BLOCK_C) == 1 else 1
    _mhc_reduce_apply_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_C))](
        acc,
        acc_sq,
        scale,
        base,
        x,
        out,
        layer_input,
        act,
        proj,
        rsigma,
        M,
        K=n * C,
        n=n,
        n_squared=n_squared,
        C=C,
        eps=eps,
        hc_pre_eps=pre_eps,
        hc_post_mult_value=post_multiplier,
        stride_acc_m=acc.stride(0),
        stride_acc_k=acc.stride(1),
        stride_acc_n=acc.stride(2),
        stride_acc_sq_m=acc_sq.stride(0),
        stride_acc_sq_k=acc_sq.stride(1),
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        stride_li_m=layer_input.stride(0),
        stride_li_c=layer_input.stride(1),
        stride_save_m=act.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        N_POW2=triton.next_power_of_2(n),
        N_POW2_RES=triton.next_power_of_2(n_squared),
        NUM_SPLITS=NUM_KSPLIT,
        RES_PID_C=RES_PID_C,
        SAVE=save,
        num_warps=cfg["apply_num_warps"],
    )

    raw_comb = out[:, n:]
    comb = torch.empty((M, n, n), dtype=torch.float32, device=x.device)
    comb_flat = comb.view(M, n_squared)
    SINKHORN_BLOCK_M = cfg["SINKHORN_BLOCK_M"]
    _mhc_sinkhorn_kernel[(triton.cdiv(M, SINKHORN_BLOCK_M),)](
        raw_comb,
        comb_flat,
        M,
        stride_logits_m=raw_comb.stride(0),
        stride_logits_n=raw_comb.stride(1),
        stride_out_m=comb_flat.stride(0),
        stride_out_n=comb_flat.stride(1),
        n=n,
        LOG2_N=_log2(n),
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        eps=sinkhorn_eps,
        BLOCK_M=SINKHORN_BLOCK_M,
        num_warps=1,
    )
    return out[:, :n].unsqueeze(-1), comb, layer_input, saved


# --- backward launchers ----------------------------------------------------------------------


def _backward_config(backward_config):
    return dict(BACKWARD_CONFIG, **(backward_config or {}))


def _mhc_backward_dx(x, grad, act, dproj, rms_coef, fn, n, C, pre_eps, cfg):
    """dResidual [M, n*C] of mHC pre from the per-token results of a mix kernel."""
    M = x.shape[0]
    dx = torch.empty_like(x)
    BLOCK_M, BLOCK_C = cfg["DX_BLOCK_M"], cfg["DX_BLOCK_C"]
    _mhc_backward_dx_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_C))](
        x,
        grad,
        act,
        dproj,
        rms_coef,
        fn,
        dx,
        M,
        n=n,
        C=C,
        N_ROWS=fn.shape[0],
        pre_eps=pre_eps,
        stride_xm=x.stride(0),
        stride_gm=grad.stride(0),
        stride_save_m=act.stride(0),
        stride_dproj_m=dproj.stride(0),
        stride_fn_n=fn.stride(0),
        stride_dxm=dx.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=cfg["dx_num_warps"],
        **_BACKWARD_LAUNCH,
    )
    return dx


def _mhc_pre_dsv4_backward(
    residual,
    fn,
    scale,
    saved,
    grad_post,
    grad_comb,
    grad_layer,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_iters,
    cfg,
    need_dx,
):
    """(dResidual or None, d logits [M, 2n+n^2], d acc [M, 2n+n^2], d scale per row [M, 3]).

    Contiguous inputs; grad_post [M, n], grad_comb [M, n*n], grad_layer [M, C]."""
    M, n, C = residual.shape
    act, proj, rsigma = saved
    x = residual.view(M, n * C)
    dlogit = torch.empty_like(act)
    dproj = torch.empty_like(act)
    dscale = torch.empty((M, 3), dtype=torch.float32, device=x.device)
    rms_coef = torch.empty((M,), dtype=torch.float32, device=x.device)
    if M == 0:
        return (torch.zeros_like(residual) if need_dx else None), dlogit, dproj, dscale
    BLOCK_M = cfg["MIX_BLOCK_M"]
    _mhc_pre_backward_mix_kernel[(triton.cdiv(M, BLOCK_M),)](
        x,
        grad_layer,
        grad_post,
        grad_comb,
        act,
        proj,
        rsigma,
        scale,
        dlogit,
        dproj,
        dscale,
        rms_coef,
        M,
        K=n * C,
        n=n,
        C=C,
        LOG2_N=_log2(n),
        hc_post_mult_value=post_multiplier,
        sinkhorn_eps=sinkhorn_eps,
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        stride_xm=x.stride(0),
        stride_gm=grad_layer.stride(0),
        stride_gpm=grad_post.stride(0),
        stride_gcm=grad_comb.stride(0),
        stride_save_m=act.stride(0),
        BLOCK_M=BLOCK_M,
        DOT_BLOCK=DOT_BLOCK,
        LOG2_DOT_BLOCK=_log2(DOT_BLOCK),
        num_warps=cfg["mix_num_warps"],
        **_BACKWARD_LAUNCH,
    )
    dx = None
    if need_dx:
        dx = _mhc_backward_dx(x, grad_layer, act, dproj, rms_coef, fn, n, C, pre_eps, cfg)
        dx = dx.view(M, n, C)
    return dx, dlogit, dproj, dscale


# --- parameter gradients through the canonical backward session ---------------------------------

_PARAMETERS = ("fn", "scale", "base")
_FOLD_BLOCK_W = 32


def _fold_rows(rows):
    """Sum of rows [T, W] one row at a time in their (canonical) order, fp32: scale and base."""
    rows = rows.float().contiguous()
    T, W = rows.shape
    out = torch.empty((W,), dtype=torch.float32, device=rows.device)
    _mhc_fold_rows_kernel[(triton.cdiv(W, _FOLD_BLOCK_W),)](
        rows,
        out,
        T,
        W=W,
        stride_t=rows.stride(0),
        BLOCK_W=_FOLD_BLOCK_W,
        num_warps=1,
        **_BACKWARD_LAUNCH,
    )
    return out


def _fn_gradient(rows, grads):
    """d fn = grads^T @ rows over the tokens in canonical order, with RL-Kernel's fixed-order
    fp32 det_gemm (as the other ROCm VJPs); rows are the bf16 residual rows, grads d acc."""
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

    return _triton_gemm_fp32(grads.t().contiguous(), rows)


def _bind_rows(needs_param_grad, logical_keys, parameter_id, residual):
    """(keys, canonical) for one forward call.

    ``keys`` is a copy of the caller's logical keys (rows with key[:, 0] < 0 are padding:
    their upstream gradients are zeroed in backward), or None. ``canonical`` is
    (session, parameter_id, {name: slot}) for the parameters that need a gradient,
    registered in the active canonical backward session, or None.
    """
    M = residual.shape[0]
    keys = None
    if logical_keys is not None:
        if (
            logical_keys.dtype != torch.int64
            or logical_keys.dim() != 2
            or logical_keys.shape[1] not in (2, 3)
            or logical_keys.shape[0] != M
        ):
            raise ValueError(
                f"logical_keys must be int64 [{M}, 2 or 3], one row per token, got "
                f"{logical_keys.dtype} {tuple(logical_keys.shape)}"
            )
        if logical_keys.device != residual.device:
            raise ValueError("logical_keys must be on the device of residual")
        keys = logical_keys.detach().clone()
    names = [name for name, need in zip(_PARAMETERS, needs_param_grad) if need]
    if not names:
        return keys, None
    session = active_session()
    if session is None:
        raise RuntimeError(
            f"trainable mHC {', '.join(names)} require canonical_backward_session: their "
            "gradient is a sum over tokens, reduced in logical-key order"
        )
    if keys is None or not isinstance(parameter_id, str) or not parameter_id:
        raise ValueError("trainable mHC parameters require logical_keys and a parameter_id")
    slots = {name: session.register(f"{parameter_id}.{name}", keys) for name in names}
    return keys, (session, parameter_id, slots)


def _mask_padding(keys, grads):
    """Upstream gradients with the rows of padding tokens (key[:, 0] < 0) zeroed."""
    if keys is None:
        return grads
    valid = keys[:, 0] >= 0
    return tuple(
        torch.where(valid.view(-1, *([1] * (g.dim() - 1))), g, torch.zeros_like(g)) for g in grads
    )


def _parameter_gradients(canonical, x, dproj, dscale, dlogit):
    """(d fn, d scale, d base) from the session; None for a frozen parameter and until the
    last use of a parameter has been submitted."""
    if canonical is None:
        return None, None, None
    session, parameter_id, slots = canonical
    grads = {name: None for name in _PARAMETERS}
    if "fn" in slots:
        grads["fn"] = session.submit_linear(
            f"{parameter_id}.fn", slots["fn"], x, dproj, _fn_gradient
        )
    for name, rows in (("scale", dscale), ("base", dlogit)):
        if name in slots:
            grads[name] = session.submit_rows(f"{parameter_id}.{name}", slots[name], rows, _fold_rows)
    return grads["fn"], grads["scale"], grads["base"]


def _kernel_ids(*kernels):
    return "+".join(f"{__name__}.{k.fn.__name__}" for k in kernels)


class _MHCPreDSV4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residual,
        fn,
        scale,
        base,
        eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        sinkhorn_iters,
        config,
        backward_config,
        save,
        logical_keys,
        parameter_id,
    ):
        post, comb, layer_input, saved = _mhc_pre_dsv4_forward(
            residual,
            fn,
            scale,
            base,
            eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_iters,
            config,
            save,
        )
        if save:
            ctx.save_for_backward(residual, fn, scale, *saved)
            ctx.params = (pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters)
            ctx.backward_config = backward_config
            ctx.keys, ctx.canonical = _bind_rows(
                ctx.needs_input_grad[1:4], logical_keys, parameter_id, residual
            )
        return post, comb, layer_input

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_post, grad_comb, grad_layer):
        residual, fn, scale, *saved = ctx.saved_tensors
        M, n, C = residual.shape
        grads = (grad_post.reshape(M, n), grad_comb.reshape(M, n * n), grad_layer)
        grad_post, grad_comb, grad_layer = (g.contiguous() for g in _mask_padding(ctx.keys, grads))
        dx, dlogit, dproj, dscale = _mhc_pre_dsv4_backward(
            residual,
            fn.contiguous(),
            scale,
            saved,
            grad_post.float(),
            grad_comb.float(),
            grad_layer,
            *ctx.params,
            _backward_config(ctx.backward_config),
            ctx.needs_input_grad[0],
        )
        record_backward(
            "mhc_pre",
            kernel_id=_kernel_ids(_mhc_pre_backward_mix_kernel, _mhc_backward_dx_kernel),
            impl=f"{PROFILE}:rocm-triton:saved",
            family="rocm-triton",
        )
        dfn, dscale_total, dbase = _parameter_gradients(
            ctx.canonical, residual.view(M, n * C), dproj, dscale, dlogit
        )
        return (dx, dfn, dscale_total, dbase) + (None,) * 10


def _needs_backward(*tensors):
    """True when autograd will record this call, so the forward must store what backward needs."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def mhc_pre_dsv4(
    residual,
    fn,
    scale,
    base,
    *,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    config=None,
    backward_config=None,
    logical_keys=None,
    parameter_id=None,
):
    """residual [M, n, C] bf16 contiguous, fn [2n+n², n*C], scale [3], base [2n+n²] fp32
    -> (post_mix [M, n, 1], comb_mix [M, n, n], layer_input [M, C]).

    ``config`` overrides keys of PRE_CONFIG, ``backward_config`` keys of
    BACKWARD_CONFIG (launch parameters only; the bits do not change).
    ``logical_keys`` (int64 [M, 2 or 3], key[:, 0] < 0 for padding rows) and
    ``parameter_id`` (the session ids are ``parameter_id + ".fn"``, ``".scale"``,
    ``".base"``) are required when fn, scale or base require grad, inside
    ``canonical_backward_session``; with keys, padding rows get zero gradients."""
    if sinkhorn_repeat < 0:
        raise ValueError("sinkhorn_repeat must be non-negative")
    return _MHCPreDSV4.apply(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        int(sinkhorn_repeat),
        config,
        backward_config,
        _needs_backward(residual, fn, scale, base),
        logical_keys,
        parameter_id,
    )


# --- ops ---------------------------------------------------------------------------------------


def _check(name, tensor, dtype, shape=None):
    """Raise for an input these ops do not support, instead of running it unvalidated."""
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")


def _check_streams(residual):
    """(n, C) of a bf16 residual [..., n, C] with n = SUPPORTED_N."""
    _check("residual", residual, torch.bfloat16)
    if residual.dim() < 2 or residual.shape[-2] != SUPPORTED_N:
        raise ValueError(
            f"residual must have shape [..., {SUPPORTED_N}, C], got {tuple(residual.shape)}"
        )
    return residual.shape[-2:]


def check_pre_inputs(residual, fn, scale, base):
    """Inputs of mHC pre: bf16 residual [..., n, C]; fp32 fn, scale, base."""
    n, c = _check_streams(residual)
    rows = 2 * n + n * n
    _check("fn", fn, torch.float32, (rows, n * c))
    _check("scale", scale, torch.float32, (3,))
    _check("base", base, torch.float32, (rows,))


def _require_rocm():
    """Raise unless the current GPU is a ROCm GPU of a validated architecture (gfx942)."""
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("The Triton mHC ops require a ROCm GPU.")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != VALIDATED_ARCH:
        raise RuntimeError(
            f"The Triton mHC ops are validated on {VALIDATED_ARCH} only; this GPU is {arch}."
        )


class TritonMHCPreOp:
    """Batch-invariant mHC pre; the PyTorch reference is rl_engine/ops/mhc/training.py."""

    def __init__(self):
        _require_rocm()
        logger.info("Linked RL-Kernel's batch-invariant Triton mHC pre.")

    def __call__(
        self,
        residual,
        fn,
        scale,
        base,
        *,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        logical_keys=None,
        parameter_id=None,
    ):
        """``logical_keys`` (int64 [tokens, 2 or 3], leading dimensions flattened) and
        ``parameter_id`` are needed only to train fn, scale or base; see mhc_pre_dsv4."""
        check_pre_inputs(residual, fn, scale, base)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        post, comb, layer_input = mhc_pre_dsv4(
            residual.reshape(-1, n, c).contiguous(),
            fn.contiguous(),
            scale.contiguous(),
            base.contiguous(),
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            logical_keys=logical_keys,
            parameter_id=parameter_id,
        )
        return post.reshape(*lead, n), comb.reshape(*lead, n, n), layer_input.reshape(*lead, c)


# Kernels whose compiled variants a dynamic-shape check counts.
KERNELS = (
    _mhc_fused_split_kernel,
    _mhc_reduce_apply_kernel,
    _mhc_sinkhorn_kernel,
    _mhc_pre_backward_mix_kernel,
    _mhc_backward_dx_kernel,
)
