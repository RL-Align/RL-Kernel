# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini attention O-projection contract (INK-17)."""

from __future__ import annotations

import torch


def attention_o_residual(
    attention: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply deterministic O projection, then add the attention residual.

    The projection reduction runs in FP32 with a fixed hidden-dimension left
    fold. The residual add also occurs in FP32, and the final tensor returns to
    the attention input dtype.
    """
    if attention.ndim < 2 or residual.shape != attention.shape:
        raise ValueError("attention and residual must have the same [..., hidden] shape")
    if (
        weight.ndim != 2
        or weight.shape[1] != attention.shape[-1]
        or weight.shape[0] != residual.shape[-1]
    ):
        raise ValueError("O-projection weight must have shape [hidden, hidden]")
    if attention.device != residual.device or attention.device != weight.device:
        raise ValueError("attention, residual and O-projection weight must share a device")
    if (
        not attention.is_floating_point()
        or not residual.is_floating_point()
        or not weight.is_floating_point()
    ):
        raise TypeError("attention O-projection tensors must be floating point")
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError("O-projection bias must have shape [hidden]")
        if bias.device != attention.device:
            raise ValueError("O-projection bias must share the attention device")

    attention_flat = attention.reshape(-1, attention.shape[-1]).float()
    weight_fp32 = weight.float()
    projected = torch.zeros(
        (attention_flat.shape[0], weight.shape[0]), device=attention.device, dtype=torch.float32
    )
    for hidden_index in range(attention.shape[-1]):
        projected = (
            projected
            + attention_flat[:, hidden_index : hidden_index + 1] * weight_fp32[:, hidden_index]
        )
    if bias is not None:
        projected = projected + bias.float()
    output = projected + residual.reshape(-1, residual.shape[-1]).float()
    return output.reshape_as(residual).to(dtype=attention.dtype)


class _NativeAttentionOResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attention, residual, weight, bias):
        from rl_engine.backends.extension import _C

        output = _C.inkling2_attention_o_residual_forward(
            attention.contiguous(), residual.contiguous(), weight.contiguous(), bias.contiguous()
        )
        ctx.save_for_backward(attention.contiguous(), residual.contiguous(), weight.contiguous())
        ctx.has_bias = bias.numel() != 0
        ctx.need_bias = ctx.needs_input_grad[3]
        return output

    @staticmethod
    def backward(ctx, dy):
        from rl_engine.backends.extension import _C

        attention, residual, weight = ctx.saved_tensors
        d_attention, d_residual, d_weight, d_bias = _C.inkling2_attention_o_residual_backward(
            dy.contiguous(), attention, residual, weight, bool(ctx.need_bias)
        )
        if not ctx.has_bias or not ctx.need_bias:
            d_bias = None
        return d_attention, d_residual, d_weight, d_bias


def attention_o_residual_cuda(
    attention: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the native CUDA O projection and residual kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_attention_o_residual_forward"):
        raise RuntimeError("Inkling 2 attention O residual CUDA extension is unavailable")
    if not attention.is_cuda or attention.ndim != 2 or residual.shape != attention.shape:
        raise ValueError("native attention and residual must be matching 2-D CUDA tensors")
    if (
        weight.ndim != 2
        or weight.shape[1] != attention.shape[1]
        or weight.shape[0] != residual.shape[1]
    ):
        raise ValueError("native O-projection weight must have shape [output, input]")
    if attention.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native O residual supports float16, bfloat16, and float32")
    if residual.dtype != attention.dtype or weight.dtype != attention.dtype:
        raise TypeError("native O residual tensors must share dtype")
    if bias is None:
        bias = torch.empty(0, device=attention.device, dtype=attention.dtype)
    elif bias.shape != (weight.shape[0],):
        raise ValueError("bias must have shape [output]")
    if bias.device != attention.device or bias.dtype != attention.dtype:
        raise ValueError("native O residual tensors must share device and dtype")
    return _NativeAttentionOResidualFunction.apply(attention, residual, weight, bias)
