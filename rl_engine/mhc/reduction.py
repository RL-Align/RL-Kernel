# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Frozen reduction primitives for the P1 mHC/RMSNorm contract (issue #2).

Every reduction inside the P1 operators goes through this module. Nothing in
`oracle.py` is allowed to call `torch.sum`, `torch.matmul`, `mean`, `einsum`
or any other library reduction: those pick their own tree and would silently
un-freeze the arithmetic.

Two trees are pinned, and only two:

- **Long reductions** (K = 4 * D controller dot, D-wide sum-of-squares,
  token-major parameter grads) use :func:`fixed_sum` / :func:`fixed_dot`:
  a *single FP32 accumulator, left-to-right in ascending index order*, with
  every multiply and add rounding separately (mul-then-add, no FMA fusion).
  This is the same order the repository's existing `reduce_rows_fp32`
  left-fold uses, so a P1 kernel and the WS1 VJP path agree by construction.
- **4-element stream reductions** (the four mHC residual streams, the 4x4
  Sinkhorn row/column sums) use :func:`stream4_sum`: the balanced tree
  ``(a0 + a1) + (a2 + a3)`` pinned by issue #2.

Banned everywhere downstream, per issue #2: Split-K, Stream-K, atomic partial
accumulation, and any reduction order that varies with batch size, token
count, SM count or any other runtime condition. A kernel either reproduces
these bytes (`__fmul_rn` / `__fadd_rn`) or registers its own numeric profile
-- never silently.
"""

from __future__ import annotations

import torch

# The pinned 4-way tree, written out so the docstring and the code cannot drift.
STREAM4_TREE = "(a0+a1)+(a2+a3)"
LONG_REDUCTION_ORDER = "serial-ascending-left-fold"

HC_MULT = 4


def fixed_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Left-fold ``x`` along ``dim`` with a single FP32 accumulator.

    Ascending index order, one add at a time. Equivalent in value to
    ``x.sum(dim)`` but with the addition order pinned.
    """
    x32 = x.to(torch.float32)
    x32 = x32.movedim(dim, 0)
    acc = torch.zeros(x32.shape[1:], dtype=torch.float32, device=x32.device)
    for i in range(x32.shape[0]):
        acc = acc + x32[i]
    return acc


def fixed_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b.T`` with FP32 mul-then-add in ascending-k order.

    ``a``: [M, K], ``b``: [N, K] (any float dtype) -> FP32 [M, N]. This is the
    P1-D6 fixed-K GEMM reference: one FP32 accumulator per output element, K
    walked left to right in a single unsplit pass, and a single cast at the
    output (performed by the caller). Split-K / Stream-K / atomic partial
    merges would all change these bytes.
    """
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    m, k = a32.shape
    n, kb = b32.shape
    if kb != k:
        raise ValueError(f"fixed_dot K mismatch: {k} vs {kb}")
    acc = torch.zeros(m, n, dtype=torch.float32, device=a32.device)
    for kk in range(k):
        acc = acc + a32[:, kk].unsqueeze(1) * b32[:, kk].unsqueeze(0)
    return acc


def fixed_sumsq(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``sum(x**2)`` along ``dim``, FP32, ascending order, mul-then-add.

    The square rounds before it is accumulated, matching a kernel that does
    ``acc = __fadd_rn(acc, __fmul_rn(v, v))``.
    """
    x32 = x.to(torch.float32)
    x32 = x32.movedim(dim, 0)
    acc = torch.zeros(x32.shape[1:], dtype=torch.float32, device=x32.device)
    for i in range(x32.shape[0]):
        acc = acc + x32[i] * x32[i]
    return acc


def stream4_sum(parts: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The pinned 4-element tree ``(a0 + a1) + (a2 + a3)`` in FP32.

    Used for every 4-way reduction in P1: the four residual streams in
    ``mhc_pre`` / ``mhc_post``, and the row/column sums of the 4x4 Sinkhorn
    matrix. Note this is deliberately *not* the ascending left fold -- a
    balanced tree is what the 4-stream kernels can actually emit, and issue #2
    pins it explicitly.
    """
    if len(parts) != HC_MULT:
        raise ValueError(f"stream4_sum expects {HC_MULT} parts, got {len(parts)}")
    a0, a1, a2, a3 = (p.to(torch.float32) for p in parts)
    return (a0 + a1) + (a2 + a3)


def stream4_sum_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """:func:`stream4_sum` over a length-4 axis of a tensor."""
    if x.shape[dim] != HC_MULT:
        raise ValueError(f"stream4_sum_dim needs a length-{HC_MULT} axis, got {x.shape[dim]}")
    moved = x.movedim(dim, 0)
    return stream4_sum([moved[0], moved[1], moved[2], moved[3]])


def stream4_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Max over a length-4 axis with the same balanced tree as :func:`stream4_sum`.

    ``max(max(a0, a1), max(a2, a3))`` -- pinned so the Sinkhorn softmax shift
    cannot drift between a tree reduction and a serial scan. (Max is
    associative in IEEE arithmetic, so this fixes the code, not the value.)
    """
    if x.shape[dim] != HC_MULT:
        raise ValueError(f"stream4_max needs a length-{HC_MULT} axis, got {x.shape[dim]}")
    m = x.movedim(dim, 0).to(torch.float32)
    return torch.maximum(torch.maximum(m[0], m[1]), torch.maximum(m[2], m[3]))


__all__ = [
    "HC_MULT",
    "LONG_REDUCTION_ORDER",
    "STREAM4_TREE",
    "fixed_dot",
    "fixed_sum",
    "fixed_sumsq",
    "stream4_max",
    "stream4_sum",
    "stream4_sum_dim",
]
