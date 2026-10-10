# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton port of the deterministic standard-softmax attention core (issue #147).

This is a *bitwise* re-implementation of ``csrc/cuda/attention/deterministic_attention.cu``
(exposed as ``_C.deterministic_attention_forward`` / ``_C.deterministic_attention_backward``).
Every reduction here reproduces the C++ kernel's floating-point order exactly, so the
Triton path and the native path return bit-identical ``out``/``lse``/``dQ``/``dK``/``dV``
for the same inputs on the same device.

The pipeline mirrors the native one 1:1:

    forward : QK -> masked softmax+LSE -> PV
    backward: dP -> softmax backward -> dQ -> dK -> dV

The three arithmetic contracts that have to be honoured for bitwise parity are:

1. **Dot products are sequential FMA chains.** The C++ kernels accumulate
   ``acc += (float)a[i] * (float)b[i]`` over ascending ``i`` in a single thread, which
   hipcc/nvcc contract into a chain of FMAs. Every reduction below loops over the
   contraction index one element at a time and uses :func:`tl.fma`, so no vector
   tree reduction is ever introduced. That is why the contraction index is the *loop*
   and the head dim / output tile is the *vector*: the opposite (and much faster)
   arrangement would reassociate the sum.
2. **Row softmax uses the 256-lane partial + binary-tree layout.** The C++ softmax
   assigns key ``k`` to thread ``k % 256``, sums each lane's keys in ascending order,
   and then folds the 256 partials with ``stride = 128, 64, ... 1``. :func:`_tree_sum_256`
   reproduces that fold exactly by repeatedly reshaping to ``(2, n)`` and summing axis 0.
3. **Transcendentals reproduce the vendor libm.** Every Triton exp/log intrinsic lowers
   to a bare hardware ``v_exp_f32``/``v_log_f32``, which is ~1 ULP away from the
   ``expf``/``logf`` the C++ kernel calls. :func:`_expf` and :func:`_logf` below
   re-emit the vendor argument reduction instruction for instruction instead.

Performance note: like the native reference core this materialises the full FP32
``[B, Hq, Sq, Skv]`` score matrix and runs scalar-order reductions. It is a
correctness/parity core, not a FlashAttention replacement.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_HEAD_DIM = 128

_IS_ROCM = torch.version.hip is not None

# ---------------------------------------------------------------------------
# Vendor-exact expf / logf
# ---------------------------------------------------------------------------
# The softmax is the only place this core evaluates a transcendental, and it is
# also the only place where "write the obvious Triton code" is not enough:
# ``tl.exp`` / ``tl.math.exp`` / ``libdevice.exp`` all lower to ``llvm.exp.f32``,
# which the AMDGPU backend expands to a bare ``v_exp_f32``. The HIP ``expf`` the
# C++ kernel calls does a two-term argument reduction around that same hardware
# ``v_exp_f32``, so the two differ by ~1 ULP on most inputs.
#
# The sequences below reproduce, instruction for instruction, what hipcc emits
# for ``expf`` / ``logf`` on gfx9. They are verified bitwise against the vendor
# result over 4M+ random and edge-case inputs, including subnormals, +/-inf and NaN.
#
# ``_fp32_barrier`` is load-bearing: without it LLVM folds ``x * L2E_HI`` and the
# following subtract back into a single FMA, which silently changes the reduced
# argument. Inline asm is opaque to that folding.

if _IS_ROCM:
    from triton.language.extra.hip import libdevice as _ocml

    @triton.jit
    def _fp32_barrier(x):
        """Opaque move: stops LLVM re-associating across this point."""
        return tl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _expf(x):
        """Bitwise-exact HIP ``expf`` for FP32."""
        t = _fp32_barrier(x * 1.4426950216293335)
        err = tl.fma(x, 1.4426950216293335, -t)
        n = _ocml.rint(t)
        err = tl.fma(x, 1.925962855864327e-08, err)
        r = _fp32_barrier(_fp32_barrier(t - n) + err)
        y = _ocml.ldexp(_ocml.exp2(r), n.to(tl.int32))
        # Written as "constant compared against x" so an unordered (NaN) compare
        # falls through to the NaN result, matching v_cmp_ngt / v_cmp_nlt.
        y = tl.where(-103.2789306640625 > x, 0.0, y)
        return tl.where(88.72283935546875 < x, float("inf"), y)

    @triton.jit
    def _logf(x):
        """Bitwise-exact HIP ``logf`` for FP32."""
        small = x < 1.1754943508222875e-38
        log2_x = _ocml.log2(_ocml.ldexp(x, tl.where(small, 32, 0)))
        t = _fp32_barrier(log2_x * 0.6931471228599548)
        err = tl.fma(log2_x, 0.6931471228599548, -t)
        err = tl.fma(log2_x, 5.769998878690785e-08, err)
        r = _fp32_barrier(t + err)
        r = tl.where(tl.abs(log2_x) < float("inf"), r, log2_x)
        return r - tl.where(small, 22.180709838867188, 0.0)

else:

    @triton.jit
    def _expf(x):
        return tl.math.exp(x)

    @triton.jit
    def _logf(x):
        return tl.math.log(x)


#: True when :func:`_expf` / :func:`_logf` reproduce this platform's libm bitwise.
#: The nvcc ``expf``/``logf`` sequences have not been ported, so on CUDA the ops
#: below refuse to run unless the caller opts out explicitly.
BITWISE_LIBM_PARITY = _IS_ROCM

# Mirrors kSoftmaxThreads in csrc/cuda/attention/deterministic_attention.cu. The
# value is part of the arithmetic contract, not a tuning knob: changing it changes
# which keys land in which partial sum and therefore changes the result bitwise.
_SOFTMAX_LANES = 256

# Tile shapes. These only affect scheduling, never the reduction order, because
# every reduction is a per-output-element sequential loop.
_QK_BLOCK_Q = 16
_QK_BLOCK_K = 64


@triton.jit
def _tree_sum_256(vals):
    """Fold 256 partials the way the C++ shared-memory tree reduction does.

    The C++ loop is ``for (stride = 128; stride > 0; stride >>= 1) s[i] += s[i + stride]``.
    Reshaping to ``(2, n)`` and summing axis 0 is that same pairing: row-major
    ``reshape(2, n)[0] == vals[:n]`` and ``[1] == vals[n:]``, so each step is
    ``vals[i] + vals[i + n]``. The steps are written out because a Triton loop
    cannot carry a value whose shape changes.
    """
    total = tl.sum(tl.reshape(vals, (2, 128)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 64)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 32)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 16)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 8)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 4)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 2)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 1)), axis=0)
    return tl.sum(total, axis=0)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@triton.jit
def _qk_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """scores[b, hq, q, k] = scale * sum_{d ascending} Q[b,hq,q,d] * K[b,kv,k,d]."""
    pid_k = tl.program_id(0)
    pid_q = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // Hq
    hq = bh % Hq
    kv_head = hq // (Hq // Hkv)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    q_in = offs_q < Sq
    k_in = offs_k < Skv

    q_rows = q_ptr + (b * Hq + hq).to(tl.int64) * Sq * D + offs_q.to(tl.int64) * D
    k_rows = k_ptr + (b * Hkv + kv_head).to(tl.int64) * Skv * D + offs_k.to(tl.int64) * D

    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for d in range(0, D):
        qv = tl.load(q_rows + d, mask=q_in, other=0.0).to(tl.float32)
        kv = tl.load(k_rows + d, mask=k_in, other=0.0).to(tl.float32)
        acc = tl.fma(qv[:, None], kv[None, :], acc)

    dst = (
        scores_ptr
        + (b * Hq + hq).to(tl.int64) * Sq * Skv
        + offs_q.to(tl.int64)[:, None] * Skv
        + offs_k[None, :]
    )
    tl.store(dst, scale * acc, mask=q_in[:, None] & k_in[None, :])


@triton.jit
def _masked_softmax_lse_kernel(
    scores_ptr,
    lse_ptr,
    mask_ptr,
    Hq,
    Sq,
    Skv,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LANES: tl.constexpr,
):
    """Mask, softmax and LSE one ``(b, hq, q)`` row in place, C++ reduction order."""
    row = tl.program_id(0)
    b = row // (Hq * Sq)
    q = row % Sq
    row_base = scores_ptr + row.to(tl.int64) * Skv

    if CAUSAL:
        causal_limit = Skv - Sq + q + 1
    else:
        causal_limit = Skv

    lane = tl.arange(0, LANES)
    neg_inf = float("-inf")
    minus_inf_vec = tl.full((LANES,), neg_inf, tl.float32)

    # Phase 1: write -inf over masked entries and take the row max. Max is
    # associative and commutative in IEEE-754, so the tree shape is irrelevant here.
    lane_max = minus_inf_vec
    for start in range(0, Skv, LANES):
        cols = start + lane
        in_range = cols < Skv
        valid = in_range & (cols < causal_limit)
        if HAS_MASK:
            keep = tl.load(mask_ptr + b.to(tl.int64) * Skv + cols, mask=in_range, other=0)
            valid = valid & (keep != 0)
        scores = tl.load(row_base + cols, mask=in_range, other=neg_inf)
        tl.store(row_base + cols, minus_inf_vec, mask=in_range & ~valid)
        lane_max = tl.maximum(lane_max, tl.where(valid, scores, neg_inf))
    row_max = tl.max(lane_max, axis=0)

    # Phase 2: exponentiate in place. Lane ``t`` sums keys t, t+LANES, ... ascending,
    # exactly like thread ``t`` in the C++ kernel; masked lanes contribute +0.0, which
    # is bitwise neutral for this non-negative sum.
    lane_sum = tl.zeros((LANES,), dtype=tl.float32)
    for start in range(0, Skv, LANES):
        cols = start + lane
        in_range = cols < Skv
        valid = in_range & (cols < causal_limit)
        if HAS_MASK:
            keep = tl.load(mask_ptr + b.to(tl.int64) * Skv + cols, mask=in_range, other=0)
            valid = valid & (keep != 0)
        scores = tl.load(row_base + cols, mask=in_range, other=neg_inf)
        probs = tl.where(valid, _expf(scores - row_max), 0.0)
        tl.store(row_base + cols, probs, mask=in_range)
        lane_sum += probs
    row_sum = _tree_sum_256(lane_sum)

    # Phase 3: normalise and emit the LSE. A fully masked row already holds zeros
    # from phase 2, so dividing it by 1.0 reproduces the C++ zero-fill branch.
    is_empty = row_sum == 0.0
    denom = tl.where(is_empty, 1.0, row_sum)
    for start in range(0, Skv, LANES):
        cols = start + lane
        in_range = cols < Skv
        probs = tl.load(row_base + cols, mask=in_range, other=0.0)
        tl.store(row_base + cols, probs / denom, mask=in_range)

    lse_val = tl.where(is_empty, neg_inf, row_max + _logf(row_sum))
    tl.store(lse_ptr + row, lse_val)


@triton.jit
def _pv_kernel(
    p_ptr,
    v_ptr,
    out_ptr,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out[b, hq, q, d] = sum_{k ascending} P[b,hq,q,k] * V[b,kv,k,d]."""
    row = tl.program_id(0)
    bh = row // Sq
    b = bh // Hq
    hq = bh % Hq
    kv_head = hq // (Hq // Hkv)

    offs_d = tl.arange(0, BLOCK_D)
    d_in = offs_d < D
    p_base = p_ptr + row.to(tl.int64) * Skv
    v_base = v_ptr + (b * Hkv + kv_head).to(tl.int64) * Skv * D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for col in range(0, Skv):
        p = tl.load(p_base + col)
        vv = tl.load(v_base + tl.cast(col, tl.int64) * D + offs_d, mask=d_in, other=0.0).to(
            tl.float32
        )
        acc = tl.fma(p, vv, acc)

    tl.store(out_ptr + row.to(tl.int64) * D + offs_d, acc, mask=d_in)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@triton.jit
def _dp_kernel(
    do_ptr,
    v_ptr,
    dp_ptr,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dP[b, hq, q, k] = sum_{d ascending} dO[b,hq,q,d] * V[b,kv,k,d]."""
    pid_k = tl.program_id(0)
    pid_q = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // Hq
    hq = bh % Hq
    kv_head = hq // (Hq // Hkv)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    q_in = offs_q < Sq
    k_in = offs_k < Skv

    do_rows = do_ptr + (b * Hq + hq).to(tl.int64) * Sq * D + offs_q.to(tl.int64) * D
    v_rows = v_ptr + (b * Hkv + kv_head).to(tl.int64) * Skv * D + offs_k.to(tl.int64) * D

    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for d in range(0, D):
        dov = tl.load(do_rows + d, mask=q_in, other=0.0).to(tl.float32)
        vv = tl.load(v_rows + d, mask=k_in, other=0.0).to(tl.float32)
        acc = tl.fma(dov[:, None], vv[None, :], acc)

    dst = (
        dp_ptr
        + (b * Hq + hq).to(tl.int64) * Sq * Skv
        + offs_q.to(tl.int64)[:, None] * Skv
        + offs_k[None, :]
    )
    tl.store(dst, acc, mask=q_in[:, None] & k_in[None, :])


@triton.jit
def _softmax_backward_kernel(
    ds_ptr,
    p_ptr,
    Skv,
    LANES: tl.constexpr,
):
    """delta = sum_k dP*P (C++ tree order); then dS = P * (dP - delta) in place."""
    row = tl.program_id(0)
    ds_base = ds_ptr + row.to(tl.int64) * Skv
    p_base = p_ptr + row.to(tl.int64) * Skv
    lane = tl.arange(0, LANES)

    lane_delta = tl.zeros((LANES,), dtype=tl.float32)
    for start in range(0, Skv, LANES):
        cols = start + lane
        in_range = cols < Skv
        dp = tl.load(ds_base + cols, mask=in_range, other=0.0)
        p = tl.load(p_base + cols, mask=in_range, other=0.0)
        lane_delta = tl.fma(dp, p, lane_delta)
    delta = _tree_sum_256(lane_delta)

    for start in range(0, Skv, LANES):
        cols = start + lane
        in_range = cols < Skv
        dp = tl.load(ds_base + cols, mask=in_range, other=0.0)
        p = tl.load(p_base + cols, mask=in_range, other=0.0)
        tl.store(ds_base + cols, p * (dp - delta), mask=in_range)


@triton.jit
def _dq_kernel(
    ds_ptr,
    k_ptr,
    dq_ptr,
    scale,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dQ[b, hq, q, d] = scale * sum_{k ascending} dS[b,hq,q,k] * K[b,kv,k,d]."""
    row = tl.program_id(0)
    bh = row // Sq
    b = bh // Hq
    hq = bh % Hq
    kv_head = hq // (Hq // Hkv)

    offs_d = tl.arange(0, BLOCK_D)
    d_in = offs_d < D
    ds_base = ds_ptr + row.to(tl.int64) * Skv
    k_base = k_ptr + (b * Hkv + kv_head).to(tl.int64) * Skv * D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for col in range(0, Skv):
        ds = tl.load(ds_base + col)
        kv = tl.load(k_base + tl.cast(col, tl.int64) * D + offs_d, mask=d_in, other=0.0).to(
            tl.float32
        )
        acc = tl.fma(ds, kv, acc)

    tl.store(dq_ptr + row.to(tl.int64) * D + offs_d, scale * acc, mask=d_in)


@triton.jit
def _dk_kernel(
    ds_ptr,
    q_ptr,
    dk_ptr,
    scale,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dK[b, hkv, k, d] = scale * sum_{group head, then q, both ascending} dS * Q."""
    k_idx = tl.program_id(0)
    b_hkv = tl.program_id(1)
    b = b_hkv // Hkv
    hkv = b_hkv % Hkv
    group = Hq // Hkv

    offs_d = tl.arange(0, BLOCK_D)
    d_in = offs_d < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for local in range(0, group):
        hq = hkv * group + local
        ds_head = ds_ptr + (b * Hq + hq).to(tl.int64) * Sq * Skv + k_idx
        q_head = q_ptr + (b * Hq + hq).to(tl.int64) * Sq * D
        for qi in range(0, Sq):
            ds = tl.load(ds_head + tl.cast(qi, tl.int64) * Skv)
            qv = tl.load(q_head + tl.cast(qi, tl.int64) * D + offs_d, mask=d_in, other=0.0).to(
                tl.float32
            )
            acc = tl.fma(ds, qv, acc)

    dst = dk_ptr + (b * Hkv + hkv).to(tl.int64) * Skv * D + k_idx.to(tl.int64) * D + offs_d
    tl.store(dst, scale * acc, mask=d_in)


@triton.jit
def _dv_kernel(
    p_ptr,
    do_ptr,
    dv_ptr,
    Hq,
    Hkv,
    Sq,
    Skv,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dV[b, hkv, k, d] = sum_{group head, then q, both ascending} P * dO."""
    k_idx = tl.program_id(0)
    b_hkv = tl.program_id(1)
    b = b_hkv // Hkv
    hkv = b_hkv % Hkv
    group = Hq // Hkv

    offs_d = tl.arange(0, BLOCK_D)
    d_in = offs_d < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for local in range(0, group):
        hq = hkv * group + local
        p_head = p_ptr + (b * Hq + hq).to(tl.int64) * Sq * Skv + k_idx
        do_head = do_ptr + (b * Hq + hq).to(tl.int64) * Sq * D
        for qi in range(0, Sq):
            p = tl.load(p_head + tl.cast(qi, tl.int64) * Skv)
            dov = tl.load(do_head + tl.cast(qi, tl.int64) * D + offs_d, mask=d_in, other=0.0).to(
                tl.float32
            )
            acc = tl.fma(p, dov, acc)

    dst = dv_ptr + (b * Hkv + hkv).to(tl.int64) * Skv * D + k_idx.to(tl.int64) * D + offs_d
    tl.store(dst, acc, mask=d_in)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def _dummy_mask(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_empty((1,), dtype=torch.bool)


def triton_deterministic_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: Optional[torch.Tensor],
    output_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(out, lse, P)`` matching ``_C.deterministic_attention_forward`` bitwise."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mask = key_padding_mask.contiguous() if key_padding_mask is not None else None

    b, hq, sq, d = q.shape
    hkv, skv = k.shape[1], k.shape[2]

    scores = torch.empty((b, hq, sq, skv), device=q.device, dtype=torch.float32)
    lse = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q, dtype=torch.float32 if output_fp32 else q.dtype)

    _qk_kernel[(triton.cdiv(skv, _QK_BLOCK_K), triton.cdiv(sq, _QK_BLOCK_Q), b * hq)](
        q,
        k,
        scores,
        float(scale),
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_Q=_QK_BLOCK_Q,
        BLOCK_K=_QK_BLOCK_K,
        num_warps=4,
    )
    _masked_softmax_lse_kernel[(b * hq * sq,)](
        scores,
        lse,
        mask if mask is not None else _dummy_mask(q),
        hq,
        sq,
        skv,
        CAUSAL=causal,
        HAS_MASK=mask is not None,
        LANES=_SOFTMAX_LANES,
        num_warps=4,
    )
    _pv_kernel[(b * hq * sq,)](
        scores,
        v,
        out,
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_D=d,
        num_warps=4,
    )
    return out, lse, scores


def triton_deterministic_attention_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dQ, dK, dV)`` matching ``_C.deterministic_attention_backward`` bitwise."""
    do = grad_output.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    p = p.contiguous()

    b, hq, sq, d = q.shape
    hkv, skv = k.shape[1], k.shape[2]

    # dS reuses the dP buffer exactly like the native backward does.
    ds = torch.empty((b, hq, sq, skv), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _dp_kernel[(triton.cdiv(skv, _QK_BLOCK_K), triton.cdiv(sq, _QK_BLOCK_Q), b * hq)](
        do,
        v,
        ds,
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_Q=_QK_BLOCK_Q,
        BLOCK_K=_QK_BLOCK_K,
        num_warps=4,
    )
    _softmax_backward_kernel[(b * hq * sq,)](
        ds,
        p,
        skv,
        LANES=_SOFTMAX_LANES,
        num_warps=4,
    )
    _dq_kernel[(b * hq * sq,)](
        ds,
        k,
        dq,
        float(scale),
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_D=d,
        num_warps=4,
    )
    _dk_kernel[(skv, b * hkv)](
        ds,
        q,
        dk,
        float(scale),
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_D=d,
        num_warps=4,
    )
    _dv_kernel[(skv, b * hkv)](
        p,
        do,
        dv,
        hq,
        hkv,
        sq,
        skv,
        D=d,
        BLOCK_D=d,
        num_warps=4,
    )
    return dq, dk, dv


class _TritonDeterministicAttentionFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        key_padding_mask: Optional[torch.Tensor],
        output_fp32: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        mask_c = key_padding_mask.contiguous() if key_padding_mask is not None else None

        out, lse, p = triton_deterministic_attention_forward(
            q_c, k_c, v_c, causal, float(scale), mask_c, output_fp32
        )

        ctx.save_for_backward(q_c, k_c, v_c, p, mask_c)
        ctx.causal = causal
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)

        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_c, k_c, v_c, p, _mask_c = ctx.saved_tensors

        if grad_out.dtype != q_c.dtype:
            grad_out = grad_out.to(q_c.dtype)
        dq, dk, dv = triton_deterministic_attention_backward(
            grad_out.contiguous(), q_c, k_c, v_c, p, float(ctx.scale)
        )
        return dq, dk, dv, None, None, None, None


class TritonDeterministicAttentionOp:
    """Triton twin of :class:`DeterministicAttentionOp`, bitwise identical to it.

    The public surface matches the native op so either can be dropped into the
    strict-attention harness. Validation is duplicated rather than imported so the
    Triton path stays usable when the native extension is not built.
    """

    backend_id = "rlkernel.triton.deterministic_attention"

    def __init__(self, *, require_bitwise_libm: bool = True) -> None:
        """``require_bitwise_libm=False`` trades bitwise parity for portability.

        The softmax needs a bitwise-exact ``expf``/``logf``; only the HIP sequences
        are ported (see :data:`BITWISE_LIBM_PARITY`). Opting out keeps the kernel
        deterministic and batch-invariant but no longer bit-identical to the
        native core, so it is never the default.
        """
        if require_bitwise_libm and not BITWISE_LIBM_PARITY:
            raise RuntimeError(
                "Triton deterministic attention is bitwise-identical to "
                "_C.deterministic_attention_* only on ROCm: the nvcc expf/logf "
                "argument reduction has not been ported. Construct with "
                "require_bitwise_libm=False to run the non-bitwise fallback."
            )
        self.bitwise_libm = BITWISE_LIBM_PARITY

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _lse = self.forward_with_lse(
            q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(q, k, v, key_padding_mask)
        resolved_scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        return _TritonDeterministicAttentionFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask, False
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(q, k, v, key_padding_mask)
        resolved_scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        out, _lse = _TritonDeterministicAttentionFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask, True
        )
        return out

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"q/k/v must be 4-D [B, H, S, D], got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        b, hq, sq, d = q.shape
        hkv, skv = k.shape[1], k.shape[2]
        if k.shape[0] != b or v.shape[0] != b:
            raise ValueError("batch size mismatch between q/k/v")
        if v.shape[1] != hkv or v.shape[2] != skv or k.shape[3] != d or v.shape[3] != d:
            raise ValueError(
                f"k/v shape mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected k/v [B={b}, Hkv, Skv, D={d}]"
            )
        if d != _HEAD_DIM:
            raise ValueError(f"head dim D must be {_HEAD_DIM}, got {d}")
        if hq % hkv != 0:
            raise ValueError(f"Hq={hq} not divisible by Hkv={hkv} (GQA group)")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"only FP16/BF16 supported, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, v must share the same dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("q, k, v must be GPU tensors")
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
            if key_padding_mask.shape != (b, skv):
                raise ValueError(
                    f"key_padding_mask must be [B, Skv]=[{b}, {skv}], "
                    f"got {tuple(key_padding_mask.shape)}"
                )
        if sq < 1 or skv < 1:
            raise ValueError(f"Sq and Skv must be positive, got Sq={sq}, Skv={skv}")


def triton_deterministic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return TritonDeterministicAttentionOp().forward(
        q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
    )


def triton_deterministic_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return TritonDeterministicAttentionOp().forward_with_lse(
        q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
    )


def triton_deterministic_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return TritonDeterministicAttentionOp().forward_fp32(
        q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
    )


__all__ = [
    "TritonDeterministicAttentionOp",
    "triton_deterministic_attention",
    "triton_deterministic_attention_backward",
    "triton_deterministic_attention_forward",
    "triton_deterministic_attention_fp32",
    "triton_deterministic_attention_with_lse",
]
