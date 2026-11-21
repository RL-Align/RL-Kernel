# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict BF16 linear projection shared by Muse-Glimmer training and rollout."""

from typing import Literal

import torch

from rl_engine.backends.extension import _C
from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_tree_gemm
from rl_engine.ops.autograd.backward_runtime import record_backward


class _DenseGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, backend):
        ctx.save_for_backward(x, weight)
        ctx.backend = backend
        with torch.cuda.device(x.device):
            if backend == "cuda":
                return _C.det_gemm_fwd_rhs_transposed(x, weight, fixed_leaf_tree=True)
            return _triton_tree_gemm(x, weight.T, fixed_leaf_tree=True)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx = dw = None
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                if ctx.backend == "cuda":
                    dx = _C.det_gemm_fwd(dy, weight, fixed_leaf_tree=True)
                else:
                    dx = _triton_tree_gemm(dy, weight, fixed_leaf_tree=True)
            if ctx.needs_input_grad[1]:
                if ctx.backend == "cuda":
                    dw = _C.det_gemm_db_transposed(x, dy, fixed_leaf_tree=True)
                else:
                    dw = _triton_tree_gemm(
                        x.T,
                        dy,
                        transpose_output=True,
                        preserve_a_strides=True,
                        fixed_leaf_tree=True,
                    )
        record_backward(
            "det_gemm",
            kernel_id=f"muse_glimmer.{ctx.backend}.fixed_leaf_tree",
            impl="muse_glimmer_dense_gemm",
            family=ctx.backend,
        )
        return dx, dw, None


def dense_gemm(
    x: torch.Tensor, weight: torch.Tensor, *, backend: Literal["cuda", "triton"]
) -> torch.Tensor:
    """Compute one independent projection from X[M,K] and native weight W[N,K]."""
    if backend not in ("cuda", "triton"):
        raise ValueError(f"unknown Muse-Glimmer dense GEMM backend {backend!r}")
    if x.ndim != 2 or weight.ndim != 2 or x.size(1) != weight.size(1):
        raise ValueError("Muse-Glimmer dense GEMM requires X[M,K] and W[N,K] with the same K")
    if x.numel() == 0 or weight.numel() == 0:
        raise ValueError("Muse-Glimmer dense GEMM requires positive M, N and K")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("Muse-Glimmer dense GEMM requires BF16 inputs and weights")
    if not x.is_cuda or x.device != weight.device:
        raise RuntimeError("Muse-Glimmer dense GEMM requires inputs on the same GPU")
    if backend == "cuda" and (_C is None or torch.version.hip is not None):
        raise RuntimeError("Muse-Glimmer CUDA dense GEMM requires the native CUDA extension")
    return _DenseGemm.apply(x, weight, backend)
