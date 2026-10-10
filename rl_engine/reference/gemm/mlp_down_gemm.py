# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native PyTorch reference for the Qwen-Image MLP down projection (WS1).

The operator is the MMDiT feed-forward down projection
``y = single_cast(x @ W.T + b)`` for the image and text streams
(``img_mlp.net.2`` / ``txt_mlp.net.2``, ``[12288 -> 3072]`` with bias), applied
to the already-dtype-rounded GELU output of the up projection.

This module is the **independent FP32 CPU reference** the RFC asks for, and it
is also the contract's definition. Frozen reduction order
(``mlp-down-gemm-tree``):

1. the reduction length splits into 32-wide leaves (short tail allowed, missing
   k treated as ``+0.0``);
2. each leaf is an ascending-k FP32 chain from ``+0.0`` in which every
   multiply-into-add is one correctly-rounded FP32 FMA;
3. leaves combine through a mid-split tree ``T(l, r) = T(l, m) + T(m, r)``, so the
   tree depends only on the reduction length and a contiguous half-K split
   composes;
4. bias is added once, in FP32, after the complete tree;
5. the single FP32 -> output-dtype round-to-nearest-even cast happens at the
   store. No other cast exists anywhere in the operator.

The correctly-rounded FMA is emulated in fp64 (Figueroa's double-rounding
theorem makes this exact for the normal range), which is the same arithmetic the
device kernels reach with ``__fmaf_rn``/``fma.rn``. Parameter gradients use
ascending-row left folds; ``dx`` reuses the same reduction tree over ``N``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from rl_engine.utils.logger import logger

LEAF_WIDTH = 32
MLP_DOWN_GEMM_CONTRACT = "mlp-down-gemm-tree"


def _fma_rn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """One correctly-rounded FP32 FMA, emulated exactly via fp64."""

    return (a.double() * b.double() + c.double()).float()


def tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` under the frozen reduction tree, in FP32.

    ``a`` is ``[M, R]``; ``b`` is ``[R, N]``; the reduction runs over ``R`` and
    is split into leaves of :data:`LEAF_WIDTH` in leaf space.
    """

    a = a.float().contiguous()
    b = b.float().contiguous()
    reduction = a.size(1)
    leaves = math.ceil(reduction / LEAF_WIDTH)

    def leaf(index: int) -> torch.Tensor:
        start = index * LEAF_WIDTH
        end = min(start + LEAF_WIDTH, reduction)
        acc = torch.zeros(a.size(0), b.size(1), dtype=torch.float32, device=a.device)
        for k in range(start, end):
            acc = _fma_rn(a[:, k : k + 1], b[k : k + 1, :], acc)
        return acc

    def merge(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return leaf(lo)
        mid = lo + (hi - lo) // 2
        return merge(lo, mid) + merge(mid, hi)

    if leaves == 0:
        return torch.zeros(a.size(0), b.size(1), dtype=torch.float32, device=a.device)
    return merge(0, leaves)


def left_fold_weight_gradient(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``dW``: ascending-row left fold, one correctly-rounded FMA per row."""

    grad = grad.float().contiguous()
    x = x.float().contiguous()
    acc = torch.zeros(grad.size(1), x.size(1), dtype=torch.float32, device=grad.device)
    for row in range(grad.size(0)):
        acc = _fma_rn(grad[row].unsqueeze(1), x[row].unsqueeze(0), acc)
    return acc


def left_fold_bias_gradient(grad: torch.Tensor) -> torch.Tensor:
    """``db``: ascending-row FP32 left fold."""

    grad = grad.float().contiguous()
    acc = torch.zeros(grad.size(1), dtype=torch.float32, device=grad.device)
    for row in range(grad.size(0)):
        acc = acc + grad[row]
    return acc


def mlp_down_gemm_reference_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """FP32 tree result plus the bias added once in fp32 (no output cast)."""

    out = tree_gemm(x, weight.t().contiguous())
    if bias is not None:
        out = out + bias.float()
    return out


def mlp_down_gemm_reference_backward(
    x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference VJP: ``dx`` reuses the tree, ``dW``/``db`` are left folds."""

    grad = grad_output.float().contiguous()
    x = x.float().contiguous()
    weight = weight.float().contiguous()
    grad_x = tree_gemm(grad, weight)
    grad_w = left_fold_weight_gradient(grad, x)
    grad_b = left_fold_bias_gradient(grad)
    return grad_x, grad_w, grad_b


class _MlpDownGemmTreeFunction(torch.autograd.Function):
    """Dtype path: fp32 tree internally, the single RNE cast at the store."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        out = mlp_down_gemm_reference_forward(x2d, weight, bias)
        ctx.save_for_backward(x2d, weight, bias)
        return out.to(x.dtype).reshape(*x.shape[:-1], weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x2d, weight, bias = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.size(-1))
        grad_x, grad_w, grad_b = mlp_down_gemm_reference_backward(x2d, weight, grad_2d)
        grad_x = grad_x.to(x2d.dtype)
        grad_w = grad_w.to(weight.dtype)
        return grad_x, grad_w, None if bias is None else grad_b.to(bias.dtype)


class NativeMlpDownGemmOp:
    """Independent fp32-CPU reference and PyTorch dispatch backend."""

    op_class = "reduction"
    is_batch_invariant = True

    def __init__(self) -> None:
        logger.info("NativeMlpDownGemmOp ready (contract %s).", MLP_DOWN_GEMM_CONTRACT)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dtype path: returns ``x.dtype`` with the single output cast."""

        if x.size(-1) != weight.size(-1):
            raise ValueError("x K must match weight K")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        if bias is not None and bias.numel() != weight.size(0):
            raise ValueError("bias must have N elements")
        return _MlpDownGemmTreeFunction.apply(x, weight, bias)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accuracy gold: the FP32 tree result, before the output cast."""

        x2d = x.reshape(-1, x.size(-1))
        out = mlp_down_gemm_reference_forward(x2d, weight, bias)
        return out.reshape(*x.shape[:-1], weight.size(0))


def mlp_down_gemm(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """Convenience wrapper around the PyTorch reference."""

    return NativeMlpDownGemmOp()(x, weight, bias=bias)
