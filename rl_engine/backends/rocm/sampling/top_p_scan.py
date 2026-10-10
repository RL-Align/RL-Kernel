# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm top-p scan with PyTorch's exact Sklansky addition order.

PyTorch scans consecutive tiles, adding the preceding tile's carry to the
first element *before* the Sklansky tree. A local scan plus a final carry
addition changes rounding. Instead, compute the subtrees independent of that
first element in parallel, propagate carries along the original tree's left
spine, and reconstruct each prefix using the same ordered additions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_subtrees(X, P, N: tl.constexpr, C: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    i = tl.arange(0, C)
    x = tl.load(X + row * N + chunk * C + i, chunk * C + i < N, other=0.0)
    for m in tl.static_range(LOG):
        s = 1 << m
        j = ((i >> (m + 1)) << (m + 1)) + s - 1
        left = tl.gather(x, j, 0)
        # Leave the left spine disconnected until its incoming carry is known.
        x = tl.where(((i & s) != 0) & (i >= 2 * s), x + left, x)
    tl.store(P + row * triton.cdiv(N, C) * C + chunk * C + i, x)


@triton.jit
def _scan_carries(P, A, N: tl.constexpr, C: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    carry = tl.full((), 0.0, tl.float32)
    for chunk in range(triton.cdiv(N, C)):
        tl.store(A + row * triton.cdiv(N, C) + chunk, carry)
        base = P + row * triton.cdiv(N, C) * C + chunk * C
        carry = tl.load(base) + carry
        for m in tl.static_range(LOG):
            carry = tl.load(base + (1 << (m + 1)) - 1) + carry


@triton.jit
def _scan_prefixes(P, A, Y, N: tl.constexpr, C: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    i = tl.arange(0, C)
    base = P + row * triton.cdiv(N, C) * C + chunk * C
    result = tl.load(base) + tl.load(A + row * triton.cdiv(N, C) + chunk)
    result = tl.broadcast_to(result, (C,))
    for m in tl.static_range(LOG):
        last = (1 << (m + 1)) - 1
        part = tl.load(base + tl.minimum(i, last))
        result = tl.where(i >= (1 << m), part + result, result)
    tl.store(Y + row * N + chunk * C + i, result, chunk * C + i < N)


def probability_cumsum(probabilities: torch.Tensor) -> torch.Tensor:
    """Scan contiguous FP32 probability rows (at least two) in-place.

    A single row uses a different PyTorch/CUB arithmetic contract and must
    remain on the native path. This helper is private to the guarded sampler.
    """
    rows, n = probabilities.shape
    # Match ATen's get_log_num_threads_x_inner_scan, including its bounds.
    log_threads = min(9, max(4, (9 + (n - 1).bit_length() - (rows - 1).bit_length()) // 2))
    tile = 2 << log_threads
    chunks = triton.cdiv(n, tile)
    partial = torch.empty(
        (rows, chunks * tile), dtype=probabilities.dtype, device=probabilities.device
    )
    carries = torch.empty((rows, chunks), dtype=probabilities.dtype, device=probabilities.device)
    options = {"enable_fp_fusion": False}
    _scan_subtrees[(rows, chunks)](
        probabilities, partial, n, tile, log_threads + 1, num_warps=4, **options
    )
    _scan_carries[(rows,)](partial, carries, n, tile, log_threads + 1, num_warps=1, **options)
    _scan_prefixes[(rows, chunks)](
        partial, carries, probabilities, n, tile, log_threads + 1, num_warps=4, **options
    )
    return probabilities


def apply_top_k_top_p(logits, k, p):
    """vLLM's ascending-sort filter with only its FP32 cumsum replaced."""
    sorted_logits, ids = logits.sort(dim=-1, descending=False)
    if k is not None:
        threshold = sorted_logits.gather(1, (sorted_logits.size(1) - k.to(torch.long)).unsqueeze(1))
        sorted_logits.masked_fill_(sorted_logits < threshold, float("-inf"))
    cumulative = probability_cumsum(sorted_logits.softmax(dim=-1))
    removed = cumulative <= 1 - p.unsqueeze(1)
    removed[:, -1] = False
    sorted_logits.masked_fill_(removed, float("-inf"))
    return logits.scatter_(dim=-1, index=ids, src=sorted_logits)
