# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order flash-style attention for MiniMax-H3 (contract ``h3-flash-attention-v1``).

BF16 ``[B, S, 56, 128]`` non-causal MHA, first derivatives. Q/K are already RMS-normalised and
rotated by the caller.

Reduction order (the contract):

* one program owns one (batch, head, row tile) and walks the key axis in ascending ``BLOCK_N``
  tiles (no split-KV, no atomics, no autotuning); tile shapes do not depend on ``S`` or ``B``;
* a row's output therefore depends only on its own query row and the whole K/V sequence, never
  on the batch size, the position in the batch or the other rows; heads are independent, so any
  head-sharded layout (TP, Ulysses CP after the all-to-all) gives the same bytes;
* ``dK``/``dV`` walk the query axis ascending, ``dQ`` walks the key axis ascending, each in its
  own kernel with FP32 register accumulators and one final rounding;
* probabilities are never rounded to BF16 alone: ``P = P_hi + P_lo`` (two BF16 terms, relative
  error about 2^-16) feed the second MMA, so the BF16 tensor-core path keeps near-FP32
  accuracy. Exponentials use libdevice ``expf`` (no ``ex2.approx``), no TF32, no fast-math.

The in-MMA accumulation order is the architecture's, so the contract is bitwise per
architecture family (Hopper wgmma / Ampere mma.sync), the same statement as
``h3-projection-gemm-v1``.
"""

from __future__ import annotations

import math

import torch

try:  # pragma: no cover - import guard mirrors the other Triton backends
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

HEADS = 56
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)
PROFILE = "h3-flash-attention-v1"
FWD_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
BWD_CONFIG = {"BLOCK": 64, "num_warps": 4, "num_stages": 2}

if HAS_TRITON:

    @triton.jit
    def _split(p):
        hi = p.to(tl.bfloat16)
        lo = (p - hi.to(tl.float32)).to(tl.bfloat16)
        return hi, lo

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        O,
        LSE,
        stride_b,
        stride_s,
        stride_h,
        lse_b,
        lse_h,
        S,
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        base = b * stride_b + h * stride_h
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        row_ok = rows < S
        q = tl.load(
            Q + base + rows[:, None] * stride_s + dims[None, :], mask=row_ok[:, None], other=0.0
        )
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        for start in range(0, S, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            col_ok = cols < S
            k = tl.load(
                K + base + cols[None, :] * stride_s + dims[:, None], mask=col_ok[None, :], other=0.0
            )
            s = tl.dot(q, k) * scale
            s = tl.where(col_ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = libdevice.exp(m_i - m_new)
            p = libdevice.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                V + base + cols[:, None] * stride_s + dims[None, :], mask=col_ok[:, None], other=0.0
            )
            hi, lo = _split(p)
            acc = acc * alpha[:, None]
            acc = tl.dot(hi, v, acc)
            acc = tl.dot(lo, v, acc)
            m_i = m_new
        out = acc / l_i[:, None]
        tl.store(
            O + base + rows[:, None] * stride_s + dims[None, :],
            out.to(tl.bfloat16),
            mask=row_ok[:, None],
        )
        tl.store(LSE + b * lse_b + h * lse_h + rows, m_i + tl.log(l_i), mask=row_ok)

    @triton.jit
    def _delta_kernel(
        O,
        dO,
        DELTA,
        stride_b,
        stride_s,
        stride_h,
        lse_b,
        lse_h,
        S,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        base = b * stride_b + h * stride_h
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        ok = rows < S
        o = tl.load(
            O + base + rows[:, None] * stride_s + dims[None, :], mask=ok[:, None], other=0.0
        ).to(tl.float32)
        do = tl.load(
            dO + base + rows[:, None] * stride_s + dims[None, :], mask=ok[:, None], other=0.0
        ).to(tl.float32)
        tl.store(DELTA + b * lse_b + h * lse_h + rows, tl.sum(o * do, 1), mask=ok)

    @triton.jit
    def _bwd_dkdv_kernel(
        Q,
        K,
        V,
        dO,
        LSE,
        DELTA,
        dK,
        dV,
        stride_b,
        stride_s,
        stride_h,
        lse_b,
        lse_h,
        S,
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        base = b * stride_b + h * stride_h
        cols = pid_n * BLOCK + tl.arange(0, BLOCK)
        dims = tl.arange(0, D)
        col_ok = cols < S
        k = tl.load(
            K + base + cols[:, None] * stride_s + dims[None, :], mask=col_ok[:, None], other=0.0
        )
        v = tl.load(
            V + base + cols[:, None] * stride_s + dims[None, :], mask=col_ok[:, None], other=0.0
        )
        dk = tl.zeros([BLOCK, D], tl.float32)
        dv = tl.zeros([BLOCK, D], tl.float32)
        for start in range(0, S, BLOCK):
            rows = start + tl.arange(0, BLOCK)
            row_ok = rows < S
            q = tl.load(
                Q + base + rows[:, None] * stride_s + dims[None, :], mask=row_ok[:, None], other=0.0
            )
            do = tl.load(
                dO + base + rows[:, None] * stride_s + dims[None, :],
                mask=row_ok[:, None],
                other=0.0,
            )
            lse = tl.load(LSE + b * lse_b + h * lse_h + rows, mask=row_ok, other=0.0)
            delta = tl.load(DELTA + b * lse_b + h * lse_h + rows, mask=row_ok, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            p = libdevice.exp(s - lse[:, None])
            p = tl.where(row_ok[:, None] & col_ok[None, :], p, 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            p_hi, p_lo = _split(p)
            ds_hi, ds_lo = _split(ds)
            dv = tl.dot(tl.trans(p_hi), do, dv)
            dv = tl.dot(tl.trans(p_lo), do, dv)
            dk = tl.dot(tl.trans(ds_hi), q, dk)
            dk = tl.dot(tl.trans(ds_lo), q, dk)
        tl.store(
            dK + base + cols[:, None] * stride_s + dims[None, :],
            (dk * scale).to(tl.bfloat16),
            mask=col_ok[:, None],
        )
        tl.store(
            dV + base + cols[:, None] * stride_s + dims[None, :],
            dv.to(tl.bfloat16),
            mask=col_ok[:, None],
        )

    @triton.jit
    def _bwd_dq_kernel(
        Q,
        K,
        V,
        dO,
        LSE,
        DELTA,
        dQ,
        stride_b,
        stride_s,
        stride_h,
        lse_b,
        lse_h,
        S,
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        base = b * stride_b + h * stride_h
        rows = pid_m * BLOCK + tl.arange(0, BLOCK)
        dims = tl.arange(0, D)
        row_ok = rows < S
        q = tl.load(
            Q + base + rows[:, None] * stride_s + dims[None, :], mask=row_ok[:, None], other=0.0
        )
        do = tl.load(
            dO + base + rows[:, None] * stride_s + dims[None, :], mask=row_ok[:, None], other=0.0
        )
        lse = tl.load(LSE + b * lse_b + h * lse_h + rows, mask=row_ok, other=0.0)
        delta = tl.load(DELTA + b * lse_b + h * lse_h + rows, mask=row_ok, other=0.0)
        dq = tl.zeros([BLOCK, D], tl.float32)
        for start in range(0, S, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            col_ok = cols < S
            k = tl.load(
                K + base + cols[:, None] * stride_s + dims[None, :], mask=col_ok[:, None], other=0.0
            )
            v = tl.load(
                V + base + cols[:, None] * stride_s + dims[None, :], mask=col_ok[:, None], other=0.0
            )
            s = tl.dot(q, tl.trans(k)) * scale
            p = libdevice.exp(s - lse[:, None])
            p = tl.where(row_ok[:, None] & col_ok[None, :], p, 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            ds_hi, ds_lo = _split(ds)
            dq = tl.dot(ds_hi, k, dq)
            dq = tl.dot(ds_lo, k, dq)
        tl.store(
            dQ + base + rows[:, None] * stride_s + dims[None, :],
            (dq * scale).to(tl.bfloat16),
            mask=row_ok[:, None],
        )


def _validate(*tensors: torch.Tensor) -> None:
    if not HAS_TRITON:
        raise RuntimeError("h3-flash-attention-v1 requires Triton (no fallback)")
    q = tensors[0]
    for x in tensors:
        if x.ndim != 4 or x.shape != q.shape or x.shape[2:] != (HEADS, HEAD_DIM):
            raise ValueError("H3 flash attention requires equal [B,S,56,128] tensors")
        if x.dtype != torch.bfloat16:
            raise ValueError("H3 flash attention requires BF16 tensors")
        if not x.is_contiguous():
            raise ValueError("H3 flash attention requires contiguous BSHD tensors")
        if not x.is_cuda or x.device != q.device or torch.version.hip is not None:
            raise ValueError("H3 flash attention requires tensors on one NVIDIA CUDA device")
    if q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("H3 flash attention requires nonempty tensors")


def _strides(x: torch.Tensor):
    return x.stride(0), x.stride(1), x.stride(2)


def flash_forward(q, k, v):
    _validate(q, k, v)
    b, s = q.shape[:2]
    out = torch.empty_like(q)
    lse = torch.empty(b, HEADS, s, device=q.device, dtype=torch.float32)
    cfg = FWD_CONFIG
    grid = (triton.cdiv(s, cfg["BLOCK_M"]), b * HEADS)
    _fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        *_strides(q),
        lse.stride(0),
        lse.stride(1),
        s,
        SCALE,
        H=HEADS,
        D=HEAD_DIM,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out, lse


def flash_backward(grad_out, q, k, v, out, lse):
    _validate(q, k, v, out, grad_out)
    b, s = q.shape[:2]
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    delta = torch.empty_like(lse)
    cfg = BWD_CONFIG
    grid = (triton.cdiv(s, cfg["BLOCK"]), b * HEADS)
    common = dict(
        H=HEADS,
        D=HEAD_DIM,
        BLOCK=cfg["BLOCK"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    _delta_kernel[grid](
        out,
        grad_out,
        delta,
        *_strides(q),
        lse.stride(0),
        lse.stride(1),
        s,
        H=HEADS,
        D=HEAD_DIM,
        BLOCK_M=cfg["BLOCK"],
    )
    args = (q, k, v, grad_out, lse, delta)
    st = (*_strides(q), lse.stride(0), lse.stride(1), s, SCALE)
    _bwd_dkdv_kernel[grid](*args, dk, dv, *st, **common)
    _bwd_dq_kernel[grid](*args, dq, *st, **common)
    return dq, dk, dv


class _FlashFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = flash_forward(q, k, v)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        return flash_backward(grad_out.contiguous(), q, k, v, out, lse)


class H3FlashAttentionOp:
    """Drop-in for ``H3FullAttentionOp``: BF16 ``[B,S,56,128]`` -> same (Q/K pre-rotated)."""

    backend_id = "rlkernel.triton.h3_flash_attention.v1"

    def __call__(self, q, k, v):
        _validate(q, k, v)
        return _FlashFn.apply(q, k, v)

    forward = __call__

    def provenance(self):
        return {
            "backend": self.backend_id,
            "profile": PROFILE,
            "fwd_config": dict(FWD_CONFIG),
            "bwd_config": dict(BWD_CONFIG),
            "fallback": False,
            "layout": "BSHD",
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "bf16",
            "causal": False,
            "accumulator": "fp32",
            "probability_precision": "two-term BF16 split (P_hi + P_lo)",
            "reduction": (
                "ascending BLOCK_N key tiles per row tile; "
                "dK/dV ascending query tiles; dQ ascending key tiles"
            ),
            "exp": "libdevice expf",
            "supports_cp": False,
            "supports_double_backward": False,
        }
