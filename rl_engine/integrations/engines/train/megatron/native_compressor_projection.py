# SPDX-License-Identifier: Apache-2.0
"""Preserve the real compressor FP32 GEMM output while retaining its Parameters."""

import torch


class _Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.mm(x, weight.T, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        # The pinned mm(out_dtype=FP32) overload has no autograd derivative.
        # Preserve the FP32 upstream gradient before each original BF16 leaf.
        dx = torch.mm(dy.float(), weight.float()).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy.float().T, x.float()).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw


class CompressorProjectionFP32(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        bias = getattr(original, "bias", None)
        # The pinned TE represents bias=False with an empty Tensor.
        has_bias = bias is not None and (not isinstance(bias, torch.Tensor) or bias.numel() != 0)
        if original.weight.dtype != torch.bfloat16 or has_bias:
            raise ValueError("expected original bias-free BF16 compressor projection")
        self.weight = original.weight

    def forward(self, x, **kwargs):
        if kwargs or x.dtype != torch.bfloat16:
            raise ValueError("compressor GEMM requires original BF16 operands")
        shape = x.shape
        y = _Gemm.apply(x.reshape(-1, shape[-1]).contiguous(), self.weight)
        if not hasattr(self, "_executed"):
            print(
                "RLK_NATIVE_COMPRESSOR_FP32_EXECUTED",
                __file__,
                "torch.mm BF16 operands, FP32 output",
                flush=True,
            )
            self._executed = True
        return y.reshape(*shape[:-1], self.weight.shape[0]), None

    def backward_dw(self):
        pass


def bind_compressor_projection(compressor):
    for name in ("linear_wkv", "linear_wgate"):
        original = getattr(compressor, name)
        replacement = CompressorProjectionFP32(original)
        assert replacement.weight is original.weight
        setattr(compressor, name, replacement)
    return compressor
