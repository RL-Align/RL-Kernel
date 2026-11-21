# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order FP32 row sums (contract ``h3-row-mean-v1``).

``row_sums(x)`` sums each row of a ``(B, K)`` FP32 matrix in an order that depends only on
``K``. Two launches:

1. every row is cut into chunks of ``CHUNK`` elements; one program per (row, chunk) adds the
   chunk's elements lane by lane in ``BLOCK``-wide steps (lane ``j`` sums elements ``j``,
   ``j + BLOCK``, ... in ascending order), then adds the ``BLOCK`` lanes in a pairwise tree
   (lanes ``2i`` and ``2i + 1``, then pairs of those);
2. one program per row adds that row's chunk sums the same way.

Nothing depends on ``B`` or on the position of a row: a row's sum is the same bytes alone or
in any batch, and the same on every call. There are no atomics, no split that depends on the
grid and no autotuning. The tree is written out over logical lane indices instead of
``tl.sum`` over all lanes: ``tl.sum``'s tree follows the register layout, which Triton picks
from the alignment it can prove for a pointer or a stride, so a row could sum differently
when sliced from another offset. Every addition is one FP32 rounding of two terms, so the
bytes depend only on ``K`` and on ``CHUNK`` and ``BLOCK``, which are part of the contract.

It replaces the ``[B, K] x [K, 1]`` fixed-order GEMM that ``fixed_order_mean`` used: that
GEMM walks all of ``K`` in one program per row tile, which is slow for a few rows of millions
of elements (one H3 video latent per sample is about 3.1 M).
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - import guard mirrors the other Triton backends
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

CONTRACT = "h3-row-mean-v1"
# Fixed launch configuration (part of the contract; changing one changes the bytes).
CHUNK = 8192
BLOCK = 1024
NUM_WARPS = 4

if HAS_TRITON:

    @triton.jit
    def _pair_sum(x, HALF: tl.constexpr):
        # (2 * HALF,) -> (HALF,): element i is x[2i] + x[2i+1]. Logical indices, not the
        # register layout, decide the pairs, and a sum of two terms is the same either way round.
        return tl.sum(tl.reshape(x, (HALF, 2)), axis=1)

    @triton.jit
    def _tree_sum_1024(x):
        # Pairwise tree over the 1024 lanes: (0+1), (2+3), ... then the pairs of those, and so on.
        x = _pair_sum(x, 512)
        x = _pair_sum(x, 256)
        x = _pair_sum(x, 128)
        x = _pair_sum(x, 64)
        x = _pair_sum(x, 32)
        x = _pair_sum(x, 16)
        x = _pair_sum(x, 8)
        x = _pair_sum(x, 4)
        x = _pair_sum(x, 2)
        x = _pair_sum(x, 1)
        return tl.sum(x, axis=0)  # one element

    @triton.jit
    def _chunk_sums_kernel(
        x_ptr, out_ptr, K, n_chunks, stride_row, CHUNK: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        chunk = tl.program_id(1)
        start = chunk * CHUNK
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for i in tl.static_range(0, CHUNK, BLOCK):
            offs = start + i + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + row * stride_row + offs, mask=offs < K, other=0.0)
        tl.store(out_ptr + row * n_chunks + chunk, _tree_sum_1024(acc))

    @triton.jit
    def _partial_sums_kernel(p_ptr, out_ptr, n_chunks, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for i in range(0, n_chunks, BLOCK):
            offs = i + tl.arange(0, BLOCK)
            acc += tl.load(p_ptr + row * n_chunks + offs, mask=offs < n_chunks, other=0.0)
        tl.store(out_ptr + row, _tree_sum_1024(acc))


def row_sums(x: torch.Tensor) -> torch.Tensor:
    """Sum of each row of a ``(B, K)`` FP32 CUDA matrix, ``(B,)`` FP32, in the fixed order above."""

    if not HAS_TRITON:
        raise RuntimeError("h3-row-mean-v1 needs Triton")
    if x.dim() != 2 or x.dtype != torch.float32 or not x.is_cuda:
        raise TypeError("row_sums takes a (B, K) FP32 CUDA tensor")
    if x.stride(1) != 1:
        raise ValueError("row_sums needs rows with unit stride (call .contiguous())")
    rows, k = x.shape
    if k == 0:
        raise ValueError("row_sums needs at least one element per row")
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows == 0:
        return out
    n_chunks = triton.cdiv(k, CHUNK)
    if n_chunks > 65535:
        raise ValueError(f"row_sums supports up to {65535 * CHUNK} elements per row, got {k}")
    partial = torch.empty((rows, n_chunks), device=x.device, dtype=torch.float32)
    # Triton launches on the current device; ``x`` may live on another one.
    with torch.cuda.device(x.device):
        _chunk_sums_kernel[(rows, n_chunks)](
            x, partial, k, n_chunks, x.stride(0), CHUNK=CHUNK, BLOCK=BLOCK, num_warps=NUM_WARPS
        )
        _partial_sums_kernel[(rows,)](partial, out, n_chunks, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out
