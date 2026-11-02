# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Decode GEMM epilogue with the original BF16-rounded SwiGLU arithmetic."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rl_engine.backends.rocm.gemm import mfma_gemm as M


@triton.jit
def _reduce_swiglu(
    P, OUT, ROWS: tl.constexpr, N: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < ROWS * (N // 2)
    row, column = index // (N // 2), index % (N // 2)
    source = row * N + column
    gate = tl.load(P + source, mask, 0.0)
    up = tl.load(P + source + N // 2, mask, 0.0)
    for chunk in range(1, CHUNKS):
        gate = gate + tl.load(P + chunk * ROWS * N + source, mask, 0.0)
        up = up + tl.load(P + chunk * ROWS * N + source + N // 2, mask, 0.0)
    # Preserve the GEMM's output rounding before applying the HIP SwiGLU formula.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    tl.store(OUT + index, (gate * sigmoid) * up, mask)


def forward(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use only for the split MFMA inference schedule with packed gate/up rows."""
    b = weight.t()
    M._validate_operands(a, b)
    rows, k = a.shape
    n = weight.shape[0]
    if not (0 < rows <= M.SPLIT_SCHEDULE_MAX_ROWS and k > M.CHUNK_K and n > 0 and n % 2 == 0):
        raise ValueError("fused SwiGLU requires a nonempty split-decode packed projection")
    chunks = triton.cdiv(k, M.CHUNK_K)
    config = M.select_config(rows, n, k)
    partial = torch.empty((chunks, rows, n), device=a.device, dtype=torch.float32)
    output = torch.empty((rows, n // 2), device=a.device, dtype=a.dtype)
    M._mfma_gemm_split_partial_kernel[
        (
            triton.cdiv(n, config.block_n),
            triton.cdiv(rows, config.block_m),
            chunks,
        )
    ](
        a,
        b,
        partial,
        rows,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=M.BLOCK_K,
        CHUNK_K=M.CHUNK_K,
        EVEN_K=k % M.BLOCK_K == 0,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        num_stages=config.num_stages,
        matrix_instr_nonkdim=M.MATRIX_INSTR_NONKDIM,
        kpack=M.KPACK,
    )
    _reduce_swiglu[(triton.cdiv(rows * n // 2, 256),)](
        partial,
        output,
        rows,
        n,
        chunks,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
