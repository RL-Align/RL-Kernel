# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The H3 GEMM rows' fixed FP32-tile GEMM at tensor-parallel shard shapes (``h3-parallel-v2``).

``h3_attention_o_gemm`` / ``h3_ffn_gate_up_gemm`` (#554) and ``h3_ffn_down_gemm`` (#535) run
``_det_gemm_fp32_kernel`` with one fixed launch: BF16 operands into the tensor-core dot, FP32
accumulation over 32-wide K blocks in ascending order, 64x64 output tiles, four warps, two stages,
no FP fusion and no split-K (``FixedFP32Backend``). Those operators accept the full TP=1 shapes on
SM90 only; ``gemm`` here is the same launch for any shape and device, so the tensor-parallel
projections of those rows reduce exactly as the rows do:

* a column shard's output is bitwise those columns of the row's output (each output element is one
  program's K loop, which does not depend on N);
* ``out_dtype=torch.float32`` stores the FP32 accumulator unrounded, for the canonical-unit
  partials of the row-parallel reduction (``parallel._units_partial``).

The launch options are pinned and checked by the tests against ``FixedFP32Backend``. The down row's
backend (``h3_down_fp32_backend.FixedFP32Backend``, #535) passes the same options except
``INDEX_64``, which it leaves at the kernel's default ``False``: that only widens the kernel's index
arithmetic to 64 bits and changes no value while every offset fits in 32 bits (the down operands
are at most 32768 x 14336 elements). The tests pin that exception, and the SM90 cross-check
(``tools/validation/models/h3_gemm_rows_crosscheck.py``) finds the down row's ``dX`` and ``dW``
bitwise.
"""

from __future__ import annotations

import torch

KERNEL_ID = "rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel"
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
# The launch of FixedFP32Backend.gemm (rl_engine/backends/cuda/gemm/h3_projection_fp32.py).
LAUNCH = dict(
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    BLOCK_K=BLOCK_K,
    PROMOTE_INPUTS=False,
    HAS_BIAS=False,
    INDEX_64=True,
    COMPENSATED=False,
    CHUNK_K=0,
    EVEN_K=False,
    EVEN_N=False,
    num_warps=4,
    num_stages=2,
    enable_fp_fusion=False,
)


def _kernel():
    from rl_engine.backends.shared.triton.gemm import det_gemm

    if not det_gemm._TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for the H3 GEMM rows (no fallback)")
    return det_gemm._det_gemm_fp32_kernel, det_gemm.triton


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    a_t: bool = False,
    b_t: bool = False,
) -> torch.Tensor:
    """``C = A @ B`` on the rows' fixed FP32-tile launch; BF16 operands, ``out_dtype`` BF16 or FP32.

    ``a`` is ``(M, K)`` (``a_t``: given as ``(K, M)``), ``b`` is ``(K, N)`` (``b_t``: given as
    ``(N, K)``), as in ``h3_projection.gemm``. Operands may be strided; nothing is copied.
    """

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"the H3 GEMM rows take BF16 operands, got {a.dtype} and {b.dtype}")
    if not a.is_cuda or a.device != b.device:
        raise ValueError("the H3 GEMM rows need both operands on one CUDA device")
    out_dtype = out_dtype or torch.bfloat16
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"out_dtype must be bfloat16 or float32, got {out_dtype}")
    a = a.t() if a_t else a
    b = b.t() if b_t else b
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError(f"reduction axes differ: {k} vs {kb}")
    if m == 0 or n == 0 or k == 0:
        # an empty shard (no rows on this rank) or an empty reduction: the exact result, zeros
        return torch.zeros((m, n), dtype=out_dtype, device=a.device)
    kernel, triton = _kernel()
    c = torch.empty((m, n), dtype=out_dtype, device=a.device)
    with torch.cuda.device(a.device):  # the launch runs on the current device's stream
        kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            a, b, c, None, m, n, k, *a.stride(), *b.stride(), *c.stride(), **LAUNCH
        )
    return c
