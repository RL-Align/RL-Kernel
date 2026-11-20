# SPDX-License-Identifier: Apache-2.0
"""Differentiable correctness baselines following the pinned official equations.

These are explicitly references; model execution never selects them as a ROCm
fallback. Reduction order may differ from the production kernel profile.
"""

import torch


def engram_gated_residual_reference(
    x, key, value, q_weight, k_weight, token_mask=None, *, eps=1e-20
):
    h, k = x.float(), key.float()
    w = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * w * k).sum(-1) * rstd * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def compressor_pool_reference(kv, score):
    """Pre-norm, pre-RoPE compressor pooling over [...,ratio,D]."""
    return (kv.float() * score.float().softmax(dim=-2)).sum(dim=-2)


def dequantize_fp4(codes, scales):
    codes = codes.view(torch.uint8)
    nibble = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=codes.device
    )
    return values[nibble] * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)


def quant_dequant_reference(x):
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, 32)
    scale = 2.0 ** torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    quant = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (quant * scale.unsqueeze(-1)).reshape(shape).to(x.dtype)


class _QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return quant_dequant_reference(x)

    @staticmethod
    def backward(ctx, dy):
        return dy


def quantized_linear_reference(x, dequantized_weight):
    a = _QuantSTE.apply(x).float()
    result = torch.zeros((*x.shape[:-1], dequantized_weight.shape[0]), device=x.device)
    for start in range(0, x.shape[-1], 32):
        result = result + a[..., start : start + 32] @ dequantized_weight[:, start : start + 32].t()
    return result.bfloat16()
