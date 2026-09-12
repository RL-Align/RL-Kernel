# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant MFMA GEMM for ROCm gfx942 with a fixed chunked-K contract.

Arithmetic contract ``rlkernel.det_gemm.triton_mfma_rocm.v1``:

* BF16 operands, FP32 accumulation on ``v_mfma_f32_16x16x16_bf16``
  (``matrix_instr_nonkdim=16``, ``kpack=2`` pinned; both change the in-tile
  K order and are therefore part of the contract).
* K is consumed in ascending ``BLOCK_K=64`` tiles inside fixed
  ``CHUNK_K``-wide chunks.  Every chunk accumulates from zero; chunk partials
  are combined in ascending order in FP32 and rounded to BF16 exactly once.
* No Split-K other than the chunk decomposition above, no autotuning.

Every output element is a pure function of its own A row, its own B column and
the contract.  Tile shape, warp count, pipelining depth, grid order, operand
strides, the row count ``M`` and whether the chunks are evaluated by one
program (monolithic schedule) or by ``num_chunks`` programs plus a fixed-order
reduction (split schedule) do not change a single bit.  The split schedule
exists only to give small-M decode GEMMs enough programs to saturate HBM.

The kernels accept arbitrary positive strides, so native ``[N, K]`` weights,
``[K, N]`` prepared weights and transposed activation views are consumed
without materializing a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _TRITON_AVAILABLE = False

MFMA_GEMM_CONTRACT_ID = "rlkernel.det_gemm.triton_mfma_rocm.v1"
BLOCK_K = 64
CHUNK_K = 1024
MATRIX_INSTR_NONKDIM = 16
KPACK = 2
# Rows at or below this count use the split schedule when K spans several
# chunks.  Purely a performance threshold; both schedules are bit-identical.
SPLIT_SCHEDULE_MAX_ROWS = 64


@dataclass(frozen=True)
class MfmaGemmConfig:
    block_m: int
    block_n: int
    num_warps: int
    waves_per_eu: int = 2
    num_stages: int = 2
    group_m: int = 8


_DECODE_CONFIG = MfmaGemmConfig(16, 32, 2, waves_per_eu=0, num_stages=2, group_m=1)
_QWEN_QKV_GATE_DECODE_CONFIG = MfmaGemmConfig(32, 64, 4, waves_per_eu=0, num_stages=2, group_m=1)
_QWEN_LM_HEAD_DECODE_CONFIG = MfmaGemmConfig(16, 128, 4, waves_per_eu=2, num_stages=2, group_m=1)
_SMALL_CONFIG = MfmaGemmConfig(64, 128, 4, waves_per_eu=2, num_stages=2, group_m=8)
_LARGE_CONFIG = MfmaGemmConfig(128, 128, 4, waves_per_eu=2, num_stages=2, group_m=8)


def select_config(m_size: int, n_size: int, k_size: int) -> MfmaGemmConfig:
    """Pick a performance configuration.  Never affects the result bits."""

    if m_size <= SPLIT_SCHEDULE_MAX_ROWS:
        # Qwen3-8B TP4 decode is bandwidth-bound and benefits from more
        # N-parallel programs on its widest projections.  Keep the one-row
        # QKV case on the lower-overhead default.
        if k_size == 4096 and (n_size == 6144 or (n_size == 1536 and m_size > 1)):
            return _QWEN_QKV_GATE_DECODE_CONFIG
        if k_size == 4096 and n_size >= 32768:
            return _QWEN_LM_HEAD_DECODE_CONFIG
        return _DECODE_CONFIG
    if m_size <= 1024:
        return _SMALL_CONFIG
    return _LARGE_CONFIG


if _TRITON_AVAILABLE:

    @triton.jit
    def _chunk_dot(
        a_ptrs,
        b_ptrs,
        offs_k,
        K,
        chunk,
        stride_ak,
        stride_bk,
        BLOCK_K: tl.constexpr,
        CHUNK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """Accumulate one K chunk from zero in the pinned MFMA order."""

        TILES_PER_CHUNK: tl.constexpr = CHUNK_K // BLOCK_K
        # A trailing chunk shorter than CHUNK_K evaluates only its live tiles;
        # tiles past K are never issued, so no exact-zero products enter the
        # accumulator.
        chunk_start = chunk * CHUNK_K
        num_tiles = min(TILES_PER_CHUNK, tl.cdiv(K - chunk_start, BLOCK_K))
        acc = tl.zeros((a_ptrs.shape[0], b_ptrs.shape[1]), dtype=tl.float32)
        for tile in range(0, num_tiles):
            k0 = chunk_start + tile * BLOCK_K
            if EVEN_K:
                a = tl.load(a_ptrs + k0 * stride_ak)
                b = tl.load(b_ptrs + k0 * stride_bk)
            else:
                kmask = offs_k < K - k0
                a = tl.load(a_ptrs + k0 * stride_ak, mask=kmask[None, :], other=0.0)
                b = tl.load(b_ptrs + k0 * stride_bk, mask=kmask[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
        return acc

    @triton.jit
    def _mfma_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CHUNK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :].to(tl.int64) * stride_bn

        num_chunks = tl.cdiv(K, CHUNK_K)
        total = _chunk_dot(
            a_ptrs, b_ptrs, offs_k, K, 0, stride_ak, stride_bk, BLOCK_K, CHUNK_K, EVEN_K
        )
        for chunk in range(1, num_chunks):
            partial = _chunk_dot(
                a_ptrs, b_ptrs, offs_k, K, chunk, stride_ak, stride_bk, BLOCK_K, CHUNK_K, EVEN_K
            )
            total = total + partial

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = (
            c_ptr
            + offs_cm[:, None].to(tl.int64) * stride_cm
            + offs_cn[None, :].to(tl.int64) * stride_cn
        )
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, total.to(c_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _mfma_gemm_split_partial_kernel(
        a_ptr,
        b_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_pc,
        stride_pm,
        stride_pn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CHUNK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        chunk = tl.program_id(2)
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :].to(tl.int64) * stride_bn
        acc = _chunk_dot(
            a_ptrs, b_ptrs, offs_k, K, chunk, stride_ak, stride_bk, BLOCK_K, CHUNK_K, EVEN_K
        )
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        p_ptrs = (
            p_ptr
            + chunk * stride_pc
            + offs_cm[:, None].to(tl.int64) * stride_pm
            + offs_cn[None, :].to(tl.int64) * stride_pn
        )
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(p_ptrs, acc, mask=mask)

    @triton.jit
    def _mfma_gemm_split_reduce_kernel(
        p_ptr,
        c_ptr,
        M,
        N,
        num_chunks,
        stride_pc,
        stride_pm,
        stride_pn,
        stride_cm,
        stride_cn,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M * N
        m = offs // N
        n = offs % N
        src = p_ptr + m.to(tl.int64) * stride_pm + n.to(tl.int64) * stride_pn
        total = tl.load(src, mask=mask, other=0.0)
        for chunk in range(1, num_chunks):
            total = total + tl.load(src + chunk * stride_pc, mask=mask, other=0.0)
        dst = c_ptr + m.to(tl.int64) * stride_cm + n.to(tl.int64) * stride_cn
        tl.store(dst, total.to(c_ptr.dtype.element_ty), mask=mask)


def _validate_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("MFMA GEMM expects two 2-D tensors")
    if a.size(1) != b.size(0):
        raise ValueError(f"MFMA GEMM K mismatch: {a.size(1)} and {b.size(0)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("MFMA GEMM requires BF16 inputs")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise RuntimeError("MFMA GEMM inputs must share one ROCm device")
    for name, tensor in (("a", a), ("b", b)):
        if min(tensor.stride()) < 0:
            raise ValueError(f"MFMA GEMM operand {name} must not use negative strides")
        if tensor.stride(0) != 1 and tensor.stride(1) != 1 and tensor.numel() > 1:
            raise ValueError(f"MFMA GEMM operand {name} needs one unit-stride dimension")


def mfma_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: MfmaGemmConfig | None = None,
    force_split: bool | None = None,
) -> torch.Tensor:
    """Return ``a @ b`` in BF16 under the pinned MFMA contract.

    ``a`` is ``[M, K]`` and ``b`` is ``[K, N]``; both may be strided views
    (for example ``weight.t()`` for a native ``[N, K]`` weight).  ``out`` may be
    a preallocated BF16 ``[M, N]`` buffer, which is also allowed to be a
    narrowed view of a larger staging allocation.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    _validate_operands(a, b)
    m_size, k_size = a.shape
    n_size = b.size(1)
    if out is None:
        out = torch.empty((m_size, n_size), dtype=torch.bfloat16, device=a.device)
    else:
        if tuple(out.shape) != (m_size, n_size):
            raise ValueError(
                f"MFMA GEMM output must have shape {(m_size, n_size)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(f"MFMA GEMM output must be BF16, got {out.dtype}")
        if out.device != a.device:
            raise RuntimeError("MFMA GEMM output must live on the input device")
        if out.stride(1) != 1:
            raise ValueError("MFMA GEMM output rows must be contiguous")
        if out.requires_grad:
            raise ValueError("MFMA GEMM output buffer must not require gradients")
    if m_size == 0 or n_size == 0:
        return out
    if k_size == 0:
        return out.zero_()
    if config is None:
        config = select_config(m_size, n_size, k_size)
    even_k = k_size % BLOCK_K == 0
    num_chunks = triton.cdiv(k_size, CHUNK_K)
    use_split = (
        num_chunks > 1 and m_size <= SPLIT_SCHEDULE_MAX_ROWS
        if force_split is None
        else bool(force_split) and num_chunks > 1
    )
    common = dict(
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=BLOCK_K,
        CHUNK_K=CHUNK_K,
        EVEN_K=even_k,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        num_stages=config.num_stages,
        matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM,
        kpack=KPACK,
    )
    if not use_split:
        grid = (triton.cdiv(m_size, config.block_m) * triton.cdiv(n_size, config.block_n),)
        _mfma_gemm_kernel[grid](
            a,
            b,
            out,
            m_size,
            n_size,
            k_size,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            GROUP_M=config.group_m,
            **common,
        )
        return out
    partial = torch.empty((num_chunks, m_size, n_size), dtype=torch.float32, device=a.device)
    grid = (
        triton.cdiv(n_size, config.block_n),
        triton.cdiv(m_size, config.block_m),
        num_chunks,
    )
    _mfma_gemm_split_partial_kernel[grid](
        a,
        b,
        partial,
        m_size,
        n_size,
        k_size,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        **common,
    )
    reduce_block = 1024
    _mfma_gemm_split_reduce_kernel[(triton.cdiv(m_size * n_size, reduce_block),)](
        partial,
        out,
        m_size,
        n_size,
        num_chunks,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK=reduce_block,
        num_warps=4,
    )
    return out


def mfma_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a @ weight.T`` for a native ``[N, K]`` weight, no transpose copy."""

    return mfma_gemm(a, weight.t(), out=out)


def mfma_linear_input_gradient(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``dX = dY @ W`` with ``W`` in native ``[N, K]`` layout."""

    return mfma_gemm(grad_output, weight)


def _reduction_major_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor.t()`` with the reduction axis contiguous.

    A column-major A operand (``dY.t()`` for the weight gradient) loads several
    times slower than a row-major one on gfx942.  The transpose copy is a
    memory-bound pass that costs a few percent of the GEMM and does not change
    any loaded value, so the result stays bitwise identical.
    """

    transposed = tensor.t()
    if transposed.stride(1) == 1:
        return transposed
    return transposed.contiguous()


def mfma_linear_weight_gradient(a: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """``dW = dY.T @ X`` returned in native ``[N, K]`` layout."""

    return mfma_gemm(_reduction_major_copy(grad_output), a)


def warmup(device: torch.device | None = None) -> None:
    """Compile every schedule/config variant so no JIT runs inside graph capture."""

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm the MFMA GEMM before HIP Graph capture")
    with torch.inference_mode():
        warmup_rows_and_configs = (
            (1, (_DECODE_CONFIG, _QWEN_LM_HEAD_DECODE_CONFIG)),
            (4, (_QWEN_QKV_GATE_DECODE_CONFIG,)),
            (SPLIT_SCHEDULE_MAX_ROWS + 1, (_SMALL_CONFIG,)),
            (1025, (_LARGE_CONFIG,)),
        )
        for k_size in (
            CHUNK_K,
            2 * CHUNK_K + BLOCK_K,
            CHUNK_K + 8,
            4 * CHUNK_K,
        ):
            b = torch.zeros((k_size, 64), dtype=torch.bfloat16, device=device)
            for rows, configs in warmup_rows_and_configs:
                a = torch.zeros((rows, k_size), dtype=torch.bfloat16, device=device)
                for config in configs:
                    mfma_gemm(a, b, config=config)
                    mfma_gemm(a, b.t().contiguous().t(), config=config)
    torch.cuda.synchronize(device)


class MfmaLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, weight)
        return mfma_linear(a, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, weight = ctx.saved_tensors
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        if grad_out.stride(1) != 1 and grad_out.stride(0) != 1:
            grad_out = grad_out.contiguous()
        da = mfma_linear_input_gradient(grad_out, weight) if ctx.needs_input_grad[0] else None
        dw = mfma_linear_weight_gradient(a, grad_out) if ctx.needs_input_grad[1] else None
        return da, dw


class MfmaGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return mfma_gemm(a, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        if grad_out.stride(1) != 1 and grad_out.stride(0) != 1:
            grad_out = grad_out.contiguous()
        da = mfma_gemm(grad_out, b.t()) if ctx.needs_input_grad[0] else None
        db = mfma_gemm(_reduction_major_copy(a), grad_out) if ctx.needs_input_grad[1] else None
        return da, db


__all__ = [
    "BLOCK_K",
    "CHUNK_K",
    "KPACK",
    "MATRIX_INSTR_NONKDIM",
    "MFMA_GEMM_CONTRACT_ID",
    "MfmaGemmConfig",
    "MfmaGemmFn",
    "MfmaLinearFn",
    "mfma_gemm",
    "mfma_linear",
    "mfma_linear_input_gradient",
    "mfma_linear_weight_gradient",
    "select_config",
    "warmup",
]
