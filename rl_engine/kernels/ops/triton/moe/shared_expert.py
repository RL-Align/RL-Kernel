# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-5 (#64) Shared Expert MLP strict Triton kernels.

Numeric profile ``oracle-fp32-serial-v1`` (see ``rl_engine/moe/oracle.py``):
FP32 accumulation, serial ascending-k, multiply and add rounded separately.
Each output element is owned by one lane and reduced serially, so there is no
cross-lane floating-point reduction and results are batch/padding invariant.

The one-round SwiGLU core (FP32 math, single BF16 round on the output) is the
shared-mode (``p_s = None``, no clamp) variant shared with P5-2 (#63).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on non-GPU installs
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _strict_gemm_kernel(
        A,
        B,
        C,
        K,
        N,
        stride_bn,
        stride_bk,
        BLOCK_N: tl.constexpr,
    ):
        # C[m, n] = sum_{k ascending} A[m, k] * B(n, k); FP32 accumulator,
        # one lane per output element, mul and add rounded separately
        # (fp32 arith in Triton is IEEE by default: no FMA contraction).
        m = tl.program_id(0)
        pn = tl.program_id(1)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        a_row = A + m * K
        b_cols = B + offs_n * stride_bn
        for k in range(0, K):
            a = tl.load(a_row + k).to(tl.float32)
            b = tl.load(b_cols + k * stride_bk, mask=mask_n, other=0.0).to(tl.float32)
            prod = a * b
            acc = acc + prod
        tl.store(C + m * N + offs_n, acc, mask=mask_n)

    @triton.jit
    def _swiglu_shared_fwd_kernel(
        Z,
        H,
        n_elem,
        width,
        BLOCK: tl.constexpr,
    ):
        # One-round SwiGLU, shared mode: h = BF16(SiLU(gate) * up), FP32 math,
        # gate = z[:, :F], up = z[:, F:] packed in one [T, 2F] FP32 tensor.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        row = offs // width
        col = offs - row * width
        gate_index = row * (2 * width) + col
        g = tl.load(Z + gate_index, mask=mask, other=0.0)
        u = tl.load(Z + gate_index + width, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sig
        h = (silu * u).to(tl.bfloat16)
        tl.store(H + offs, h, mask=mask)

    @triton.jit
    def _swiglu_shared_bwd_kernel(
        DH,
        Z,
        DZ,
        n_elem,
        width,
        BLOCK: tl.constexpr,
    ):
        # dgate = ((dh * u) * dsilu); dup = dh * silu; both round to BF16 once
        # at the operator edge (mirrors the oracle's cat(...).to(bfloat16)).
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        row = offs // width
        col = offs - row * width
        gate_index = row * (2 * width) + col
        g = tl.load(Z + gate_index, mask=mask, other=0.0)
        u = tl.load(Z + gate_index + width, mask=mask, other=0.0)
        dh = tl.load(DH + offs, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sig
        # dsilu = sig * (1 + g * (1 - sig)), each op rounded separately.
        t = 1.0 - sig
        t = g * t
        t = 1.0 + t
        dsilu = sig * t
        dgate = (dh * u) * dsilu
        dup = dh * silu
        tl.store(DZ + gate_index, dgate.to(tl.bfloat16), mask=mask)
        tl.store(DZ + gate_index + width, dup.to(tl.bfloat16), mask=mask)


def _check_cuda_2d(t: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    if not t.is_cuda:
        raise NotImplementedError(f"{name} must be a CUDA tensor for the Triton backend")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D")
    if t.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")


def strict_gemm(a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
    """``a @ b.T`` (or ``a @ b`` when ``trans_b``): BF16 in, FP32 out, strict."""
    if not TRITON_AVAILABLE:
        raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
    _check_cuda_2d(a, torch.bfloat16, "a")
    _check_cuda_2d(b, torch.bfloat16, "b")
    m, k = a.shape
    if trans_b:
        bk, n = b.shape
        stride_bn, stride_bk = 1, n
    else:
        n, bk = b.shape
        stride_bn, stride_bk = k, 1
    if bk != k:
        raise ValueError(f"K mismatch: a has K={k}, b has K={bk}")
    out = torch.empty(m, n, dtype=torch.float32, device=a.device)
    if out.numel() == 0:
        return out
    if k == 0:
        return out.zero_()
    block_n = min(triton.next_power_of_2(n), 256)
    grid = (m, triton.cdiv(n, block_n))
    _strict_gemm_kernel[grid](a, b, out, k, n, stride_bn, stride_bk, BLOCK_N=block_n)
    return out


def swiglu_shared_fwd(z: torch.Tensor) -> torch.Tensor:
    """One-round SwiGLU forward, shared mode: FP32 [T, 2F] -> BF16 [T, F]."""
    if not TRITON_AVAILABLE:
        raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
    _check_cuda_2d(z, torch.float32, "z")
    if z.shape[1] % 2 != 0:
        raise ValueError("z width must be even (packed gate|up)")
    width = z.shape[1] // 2
    h = torch.empty(z.shape[0], width, dtype=torch.bfloat16, device=z.device)
    n_elem = h.numel()
    if n_elem == 0:
        return h
    block = 1024
    grid = (triton.cdiv(n_elem, block),)
    _swiglu_shared_fwd_kernel[grid](z, h, n_elem, width, BLOCK=block)
    return h


def swiglu_shared_bwd(dh: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """One-round SwiGLU backward, shared mode: returns packed BF16 dz [T, 2F]."""
    if not TRITON_AVAILABLE:
        raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
    _check_cuda_2d(dh, torch.bfloat16, "dh")
    _check_cuda_2d(z, torch.float32, "z")
    if z.shape[1] % 2 != 0:
        raise ValueError("z width must be even (packed gate|up)")
    if dh.shape[0] != z.shape[0] or dh.shape[1] * 2 != z.shape[1]:
        raise ValueError("dh shape must match the packed gate/up halves of z")
    dz = torch.empty_like(z, dtype=torch.bfloat16)
    n_elem = dh.numel()
    if n_elem == 0:
        return dz
    block = 1024
    grid = (triton.cdiv(n_elem, block),)
    _swiglu_shared_bwd_kernel[grid](dh, z, dz, n_elem, dh.shape[1], BLOCK=block)
    return dz
