# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native PyTorch GEMM -- NON-deterministic reference baseline (WS1).

WARNING: torch.matmul (cuBLAS) does NOT guarantee batch-invariance -- cuBLAS
selects kernels by shape and may use split-K. This op exists only as a
correctness reference and benchmark target, NOT as a fallback. It is
intentionally excluded from the det_gemm registry dispatch.
"""
import torch

from rl_engine.utils.logger import logger


class NativeGemmOp:
    """Plain torch.matmul. Non-deterministic; reference / benchmark use only."""

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        logger.info("NativeGemmOp ready (non-deterministic torch.matmul reference).")

    def __call__(self, a, b):
        return torch.matmul(a, b)


_K_TREE_LEAF = 32


def _index_order_fma_leaf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate one BF16 GEMM leaf with scalar FP32 FMAs in index order."""
    acc = a.new_zeros((a.size(0), b.size(1)), dtype=torch.float32)
    a_columns = a.float().double().unsqueeze(-1).unbind(dim=1)
    b_rows = b.float().double().unsqueeze(0).unbind(dim=1)
    # BF16 products are exact in FP64. If the exponent gap makes the FP64
    # sum inexact, the smaller term is too small to change the FP32 rounding
    # of the larger term. FP32 subnormal ties are resolved exactly in FP64.
    for a_column, b_row in zip(a_columns, b_rows, strict=True):
        acc = torch.addcmul(acc.double(), a_column, b_row).float()
    return acc.to(torch.bfloat16)


class DetGemmTreeReferenceOp:
    """Deterministic leaf-space mid-split tree, the canonical gold.

    The WS1 deterministic GEMM rounds every 32-element leaf and every tree
    merge node to BF16, so comparing a candidate against the single-rounding
    torch.matmul reference fails structurally at near-cancellation outputs.
    This op evaluates the exact contract tree (32-element leaves accumulated
    with index-order FP32 FMA, BF16 RNE at every leaf and merge, splitting in
    LEAF space) and is the accuracy gold for the deterministic GEMM across
    all backend profiles. Differentiable, so the gradient-accuracy judgment
    also compares against the tree's own VJP.
    """

    def __init__(self):
        logger.info("DetGemmTreeReferenceOp ready (leaf-space mid-split tree gold).")

    def __call__(self, a, b):
        a = a.contiguous()
        b = b.contiguous()
        k = a.size(1)
        num_leaves = (k + _K_TREE_LEAF - 1) // _K_TREE_LEAF

        def reduce_range(lo: int, hi: int) -> torch.Tensor:
            # [lo, hi) is a range of LEAF indices.
            if hi - lo == 1:
                start = lo * _K_TREE_LEAF
                end = min(start + _K_TREE_LEAF, k)
                return _index_order_fma_leaf(a[:, start:end], b[start:end, :])
            midpoint = lo + (hi - lo) // 2
            return reduce_range(lo, midpoint) + reduce_range(midpoint, hi)

        return reduce_range(0, num_leaves)


def native_gemm(a, b):
    return torch.matmul(a, b)
