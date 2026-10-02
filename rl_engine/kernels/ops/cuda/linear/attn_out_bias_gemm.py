# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA backend for the Qwen-Image attn-out bias GEMM (contract
``attn-out-bias-gemm-tree-v1``).

Wraps the correctness-anchor kernels in ``csrc/cuda/gemm/attn_out_bias_gemm.cu``:

- forward: one thread per output element evaluates the full frozen tree
  (32-wide ascending-k FMA chains + mid-split merges), adds bias once in fp32
  and performs the single RNE cast at the store -- no partials, memory O(S*N).
- ``dx`` reuses the same tree kernel on ``(dY, W.T)``; ``dW`` runs the
  ascending-row left-fold kernel (contract c-prime); ``db`` uses the shared
  ``row_local_bias_fp32`` left fold.
- On bf16 inputs FMA and separate mul-add are bit-identical (exact products),
  so this backend matches the FP32 CPU reference byte for byte; on fp32 inputs
  the CUDA and Triton backends match each other byte for byte (same FMA
  discipline) and the torch reference within a declared tolerance.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_REQUIRED_SYMBOLS = (
    "attn_out_bias_gemm_cuda_forward",
    "attn_out_tree_gemm_cuda",
    "attn_out_dw_left_fold_cuda",
)


def _cuda_backend_available() -> bool:
    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)


class _AttnOutBiasGemmCudaFunction(torch.autograd.Function):
    """CUDA forward/backward under the same contract as the reference."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        bias_f = None if bias is None else bias.float()
        out = _C.attn_out_bias_gemm_cuda_forward(
            x2d.float(), weight.float(), bias_f, x.dtype == torch.bfloat16
        )
        ctx.save_for_backward(x2d, weight, bias)
        ctx.lead_shape = x.shape[:-1]
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from rl_engine.kernels.ops.backward_runtime import record_backward
        from rl_engine.kernels.ops.vjp_fp32 import row_local_bias_fp32

        x2d, weight, bias = ctx.saved_tensors
        g_f = grad_output.reshape(-1, grad_output.size(-1)).float()
        grad_x = _C.attn_out_tree_gemm_cuda(g_f, weight.t().contiguous().float())
        grad_w = _C.attn_out_dw_left_fold_cuda(g_f, x2d.float())
        grad_x = grad_x.reshape(*ctx.lead_shape, x2d.size(-1)).to(x2d.dtype)
        grad_w = grad_w.to(weight.dtype)
        grad_b: Optional[torch.Tensor] = None
        if bias is not None:
            grad_b = row_local_bias_fp32(grad_output).to(bias.dtype)
        record_backward(
            "attn_out_bias_gemm",
            kernel_id="attn-out-bias-gemm-tree-v1",
            impl="cuda_per_thread_tree_fma",
            family="cuda",
        )
        return grad_x, grad_w, grad_b


class CudaAttnOutBiasGemmOp:
    """CUDA backend: per-thread full tree forward, tree dx, left-fold dW."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "cuda_per_thread_tree_fma"

    def __init__(self) -> None:
        if not _cuda_backend_available():
            raise RuntimeError(
                "attn_out_bias_gemm CUDA symbols are not compiled into the extension. "
                "Rebuild with the CUDA extension enabled: 'pip install -e .' on a CUDA "
                "host (csrc/cuda/gemm/attn_out_bias_gemm.cu is part of the default "
                "cuda_sources list); refusing non-strict fallback."
            )
        logger.info("CudaAttnOutBiasGemmOp ready (per-thread tree, FMA chains).")

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
        if not x.is_cuda or x.device != weight.device:
            raise ValueError("CudaAttnOutBiasGemmOp requires CUDA tensors on one device")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        return _AttnOutBiasGemmCudaFunction.apply(x, weight, bias)
