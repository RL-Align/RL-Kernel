# SPDX-License-Identifier: Apache-2.0
"""Eager original block-FP8 projection with frozen packed weights and input VJP.

Production AITER group128 FNUZ activation quantization and fixed CK no-split
forward; original block128 checkpoint weights and an explicit Triton input VJP. Original
checkpoint weight bytes/scales remain resident; no BF16 weight cache is built.
"""

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.triton.moe.fused_mlp import _decode_e4m3, _e8m0_scale


@triton.jit
def _vjp(DY, W, WS, DX, M, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    k = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32)
    nn = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(triton.cdiv(N, 32)):
        n = start * 32 + nn
        dy = tl.load(
            DY + m[:, None] * N + n[None, :], (m[:, None] < M) & (n[None, :] < N), other=0
        ).to(tl.bfloat16)
        w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), other=0)
        sw = _e8m0_scale(
            tl.load(
                WS + (n[:, None] // 128) * triton.cdiv(K, 128) + k[None, :] // 128,
                (n[:, None] < N) & (k[None, :] < K),
                other=127,
            )
        )
        weights = (_decode_e4m3(w) * sw).to(tl.bfloat16)
        acc = tl.dot(dy, weights, acc)
    tl.store(
        DX + m[:, None] * K + k[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (k[None, :] < K)
    )


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales):
        from aiter import QuantType, get_hip_quant
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

        from rl_engine.backends.rocm.gemm.shared_down_fp8 import KERNEL

        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        m, k = x.shape
        n = codes.shape[0]
        # Reuse the production ROCm activation quantizer selected by vLLM.
        # FNUZ changes the interpretation of the original OCP weight bytes by
        # exactly 1/2; double their original E8M0 scale to preserve the values.
        q, activation_scale = get_hip_quant(QuantType.per_1x128)(
            x, quant_dtype=torch.float8_e4m3fnuz, transpose_scale=False
        )
        fnuz_codes = torch.where(codes == 128, 0, codes).to(torch.uint8)
        fnuz_weight = fnuz_codes.view(torch.float8_e4m3fnuz)
        weight_scale = scales.view(torch.float8_e8m0fnu).float() * 2.0
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        with torch.cuda.device(x.device):
            gemm_a8w8_blockscale_ck(
                q, fnuz_weight, activation_scale, weight_scale, y, splitK=0, kernelName=KERNEL
            )
        ctx.save_for_backward(codes, scales)
        ctx.input_shape = shape
        return y.reshape(*shape[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        w, s = ctx.saved_tensors
        n, k = w.shape
        dy = dy.reshape(-1, n).contiguous()
        dx = torch.empty((dy.shape[0], k), device=dy.device, dtype=torch.bfloat16)
        with torch.cuda.device(dy.device):
            _vjp[(triton.cdiv(dy.shape[0], 16), triton.cdiv(k, 32))](
                dy, w, s, dx, dy.shape[0], n, k, enable_fp_fusion=False
            )
        return dx.reshape(ctx.input_shape), None, None


class CheckpointFP8Linear(torch.nn.Module):
    profile = "dsv4-block128-checkpoint-aiter-ck-nosplit-gfx942-v1"

    def __init__(self, weight, scales):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
            raise TypeError("expected original E4M3/E8M0 checkpoint tensors")
        if tuple(scales.shape) != (
            triton.cdiv(weight.shape[0], 128),
            triton.cdiv(weight.shape[1], 128),
        ):
            raise ValueError("expected original block128 weight scales")
        self.register_buffer("codes", weight.view(torch.uint8).contiguous())
        self.register_buffer("scales", scales.view(torch.uint8).contiguous())

    def forward(self, x, **kwargs):
        return _Projection.apply(x, self.codes, self.scales), None

    def backward_dw(self):
        pass  # Frozen checkpoint weights; the autograd input VJP still executes.


class CheckpointFP8Group(torch.nn.Module):
    def __init__(self, weight, scales, groups):
        super().__init__()
        rows = weight.shape[0] // groups
        if rows % 128:
            raise ValueError("group rows must align with checkpoint block128")
        self.projections = torch.nn.ModuleList(
            [
                CheckpointFP8Linear(
                    weight[g * rows : (g + 1) * rows].contiguous(),
                    scales[g * (rows // 128) : (g + 1) * (rows // 128)].contiguous(),
                )
                for g in range(groups)
            ]
        )

    def forward(self, x):
        return torch.stack(
            [linear(x[..., g, :])[0] for g, linear in enumerate(self.projections)], dim=-2
        )
