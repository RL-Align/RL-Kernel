# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unfused strict final modulation using native CUDA LayerNorm and fixed sums."""

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C
from rl_engine.reference.model_specific.qwenimage.final_adaln_out import EPS, HIDDEN, validate_final

from .adaln_mod_projection import (
    implementation_info,
    projection_backward,
    projection_forward,
    require_device,
    require_extension,
)


class _Final(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, temb, weight, bias):
        table, act = projection_forward(temb, weight, bias)
        scale, shift = table.chunk(2, -1)
        xf = x.float().contiguous()
        norm, mean, rstd = torch.native_layer_norm(xf, (HIDDEN,), None, None, EPS)
        norm = norm.to(x.dtype)
        gain = 1 + scale
        out = norm * gain[:, None, :] + shift[:, None, :]
        ctx.save_for_backward(xf, mean, rstd, norm, gain, temb, act, weight)
        ctx.dtype = x.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        xf, mean, rstd, norm, gain, temb, act, weight = ctx.saved_tensors
        gf = grad.float().contiguous()
        dn = (gf * gain.float()[:, None, :]).contiguous()
        dx = torch.ops.aten.native_layer_norm_backward(
            dn, xf, (HIDDEN,), mean, rstd, None, None, (True, False, False)
        )[0]
        ds = _C.qwenimage_token_sum((gf * norm.float()).contiguous())
        db = _C.qwenimage_token_sum(gf)
        dt, dw, dbias = projection_backward(torch.cat((ds, db), -1), temb, act, weight)
        return dx.to(ctx.dtype), dt, dw, dbias


class QwenImageFinalAdaLNOutCudaOp:
    op_class = "reduction"
    kernel_id = "qwenimage_det_linear+aten.native_layer_norm+aten.mul+aten.add"

    def __init__(self):
        require_extension()

    def forward(self, x, temb, weight, bias):
        validate_final(x, temb, weight, bias)
        require_device(x)
        return _Final.apply(x, temb, weight, bias)

    __call__ = forward

    def implementation_info(self, x, temb, weight, bias):
        validate_final(x, temb, weight, bias)
        info = implementation_info("qwenimage_final_adaln_out", x)
        info.update(
            norm="aten.native_layer_norm[fp32,eps=1e-6]",
            token_gradient_tree="recursive_32_token_balanced",
            cast_boundaries="silu,projection,norm,1+scale,product,output",
            fused=False,
        )
        return info
