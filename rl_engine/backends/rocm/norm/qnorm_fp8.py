# SPDX-License-Identifier: Apache-2.0
"""DSV4 FP32-accumulator QNorm -> FP8 boundary with identity quantization STE.

The AITER production kernel quantizes before its optional BF16 output store.
Both wq_b consumers reuse those codes/scales; neither requantizes the BF16 view.
This does not alter packed checkpoint weights or replace model execution.
"""

import torch


def rmsnorm_vjp(x, weight, dy, epsilon):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf, wf, df = x.to(dtype), weight.to(dtype), dy.to(dtype)
    inv = (xf.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    unit = xf * inv
    grad_unit = df * wf
    dx = inv * (grad_unit - unit * (grad_unit * unit).mean(-1, keepdim=True))
    dw = (df * unit).reshape(-1, weight.numel()).sum(0)
    return dx.to(x.dtype), dw.to(weight.dtype)


class _FusedQNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon):
        from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

        if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or x.device.type != "cuda":
            raise ValueError("DSV4 QNorm requires original BF16 GPU inputs/weight")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if flat.shape[-1] != 1024:
            raise ValueError("initial gfx942 QNorm profile supports original DSV4 q_lora_rank=1024")
        if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx942":
            raise ValueError("initial QNorm profile requires gfx942")
        codes = torch.empty_like(flat, dtype=torch.float8_e4m3fnuz)
        scales = torch.empty(
            (flat.shape[0], flat.shape[1] // 128), device=x.device, dtype=torch.float32
        )
        output = torch.empty_like(flat)
        fused_qk_rmsnorm_group_quant(
            q=flat,
            q_weight=weight,
            q_epsilon=epsilon,
            q_out_quantized=codes,
            q_out_scale=scales,
            q_out_unquantized=output,
            group_size=128,
            transpose_scale=False,
        )
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        ctx.mark_non_differentiable(codes, scales)
        return output.reshape(shape), codes, scales

    @staticmethod
    def backward(ctx, dy, _dcodes, _dscales):
        x, weight = ctx.saved_tensors
        dx, dw = rmsnorm_vjp(x, weight, dy, ctx.epsilon)
        return dx, dw, None


class FusedQNorm(torch.nn.Module):
    profile = "dsv4-qnorm-fp32-before-fp8-aiter-gfx942-v1"

    def __init__(self, norm, epsilon):
        super().__init__()
        if getattr(norm, "zero_centered_gamma", False):
            raise ValueError("DSV4 QNorm expects ordinary original gamma")
        self.weight = norm.weight  # Preserve the actual trainable Parameter identity.
        self.epsilon = epsilon
        self._activation = None

    def forward(self, x):
        y, codes, scales = _FusedQNorm.apply(x, self.weight, self.epsilon)
        self._activation = (y, codes, scales)
        if not hasattr(self, "_executed"):
            print("RLK_QNORM_FP8_EXECUTED", __file__, self.profile, flush=True)
            self._executed = True
        return y

    def quantized_input(self, x):
        if self._activation is None:
            raise RuntimeError("QNorm activation requested before forward")
        y, codes, scales = self._activation
        # Native indexer detaches qr, retaining its storage and row ordering.
        # Fail rather than silently attach a cached activation to another sample.
        if x.data_ptr() != y.data_ptr() or x.shape != y.shape or x.stride() != y.stride():
            raise ValueError("FP8 consumer does not reference the current normalized activation")
        return codes, scales
