# SPDX-License-Identifier: Apache-2.0
"""Fixed-order FP32 compressor projection, independent of fused N or token M."""
import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        shape = x.shape
        output = _triton_gemm_fp32(x.reshape(-1, shape[-1]).contiguous(), weight.T)
        return output.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        dy = grad.reshape(-1, weight.shape[0]).contiguous()
        dx = (
            _triton_gemm_fp32(dy, weight).to(x.dtype).reshape_as(x)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = _triton_gemm_fp32(dy.T, flat).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw


def compressor_projection(x, weight):
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
    ):
        raise ValueError("compressor projection requires original BF16 input and weight matrices")
    return _Projection.apply(x, weight)
