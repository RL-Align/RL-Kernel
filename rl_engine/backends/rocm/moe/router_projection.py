# SPDX-License-Identifier: Apache-2.0
"""Fixed-order original router GEMM, retaining FP32 logits and leaf gradients."""

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32


class _Router(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        shape = x.shape
        result = _triton_gemm_fp32(x.reshape(-1, shape[-1]), weight.T)
        return result.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        grad = dy.reshape(-1, weight.shape[0]).contiguous()
        dx = (
            _triton_gemm_fp32(grad, weight).to(x.dtype).reshape_as(x)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = _triton_gemm_fp32(grad.T, flat).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw


def router_projection(x, weight):
    if (
        x.dtype != torch.bfloat16
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[-1] != 4096
        or weight.shape != (256, 4096)
    ):
        raise ValueError("original DSV4 BF16 activation and 256x4096 router weight required")
    if (
        not torch.version.hip
        or torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx942"
    ):
        raise ValueError("initial original router profile requires gfx942")
    return _Router.apply(x, weight)
