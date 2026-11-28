# SPDX-License-Identifier: Apache-2.0
"""Eager original block-FP8 projection with frozen packed weights and input VJP.

Production AITER group128 native FP8 activation quantization and fixed CK no-split
forward; original block128 checkpoint weights and an explicit Triton input VJP. Original
checkpoint weight bytes/scales remain resident; no BF16 weight cache is built.
"""

import torch
import triton
import triton.language as tl

from rl_engine.backends.rocm.fp8 import fp8_dtype
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
            DY + m[:, None] * N + n[None, :],
            (m[:, None] < M) & (n[None, :] < N),
            other=0,
        ).to(tl.bfloat16)
        w = tl.load(
            W + n[:, None] * K + k[None, :],
            (n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
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
        DX + m[:, None] * K + k[None, :],
        acc.to(tl.bfloat16),
        (m[:, None] < M) & (k[None, :] < K),
    )


def original_block_fp8_gemm(q, activation_scale, codes, scales):
    """One declared CK no-split contract for actual prequantized operands."""
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

    from rl_engine.backends.rocm.gemm.shared_down_fp8 import KERNEL

    dtype = fp8_dtype(q.device)
    if q.ndim != 2 or q.dtype != dtype or not q.is_contiguous():
        raise ValueError("expected contiguous original FP8 activation codes")
    m, k = q.shape
    if (
        codes.ndim != 2
        or codes.shape[1] != k
        or codes.dtype != torch.uint8
        or scales.dtype != torch.uint8
        or tuple(scales.shape)
        != (triton.cdiv(codes.shape[0], 128), triton.cdiv(k, 128))
        or activation_scale.shape != (m, k // 128)
        or activation_scale.dtype != torch.float32
        or k % 128
    ):
        raise ValueError("invalid original block128 FP8 projection operands")
    weight_scale = scales.view(torch.float8_e8m0fnu).float()
    if dtype == torch.float8_e4m3fnuz:
        weight = torch.where(codes == 128, 0, codes).to(torch.uint8).view(dtype)
        weight_scale = weight_scale * 2.0
    else:
        weight = codes.view(dtype)
    y = torch.empty((m, codes.shape[0]), device=q.device, dtype=torch.bfloat16)
    with torch.cuda.device(q.device):
        gemm_a8w8_blockscale_ck(
            q, weight, activation_scale, weight_scale, y, splitK=0, kernelName=KERNEL
        )
    return y


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales, activation_codes=None, activation_scales=None):
        from aiter import QuantType, get_hip_quant

        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        m, k = x.shape
        n = codes.shape[0]
        dtype = fp8_dtype(x.device)
        # Reuse the production ROCm activation quantizer selected by vLLM.
        # FNUZ changes the interpretation of the original OCP weight bytes by
        # exactly 1/2; double their original E8M0 scale to preserve the values.
        if activation_codes is None:
            q, activation_scale = get_hip_quant(QuantType.per_1x128)(
                x, quant_dtype=dtype, transpose_scale=False
            )
        else:
            q, activation_scale = activation_codes, activation_scales
            if q.shape != x.shape or q.dtype != dtype:
                raise ValueError("invalid prequantized QNorm codes")
            if (
                activation_scale.shape != (m, k // 128)
                or activation_scale.dtype != torch.float32
            ):
                raise ValueError("invalid prequantized QNorm scales")
        y = original_block_fp8_gemm(q, activation_scale, codes, scales)
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
        return (dx.reshape(ctx.input_shape), None, None) + (None,) * (
            len(ctx.needs_input_grad) - 3
        )


class CheckpointFP8Linear(torch.nn.Module):
    profile = "dsv4-block128-checkpoint-aiter-ck-nosplit-gfx942-v1"

    def __init__(self, weight, scales):
        super().__init__()
        if not torch.version.hip or weight.device.type != "cuda":
            raise RuntimeError("checkpoint FP8 projection requires a ROCm GPU")
        fp8_dtype(weight.device)
        arch = torch.cuda.get_device_properties(weight.device).gcnArchName.split(":")[0]
        self.profile = f"dsv4-block128-checkpoint-aiter-ck-nosplit-{arch}-v1"
        if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
            raise TypeError("expected original E4M3/E8M0 checkpoint tensors")
        if tuple(scales.shape) != (
            triton.cdiv(weight.shape[0], 128),
            triton.cdiv(weight.shape[1], 128),
        ):
            raise ValueError("expected original block128 weight scales")
        self.register_buffer("codes", weight.view(torch.uint8).contiguous())
        self.register_buffer("scales", scales.view(torch.uint8).contiguous())

    def bind_qnorm(self, norm):
        object.__setattr__(self, "_qnorm_provider", norm)

    def forward(self, x, **kwargs):
        provider = getattr(self, "_qnorm_provider", None)
        if provider is None:
            return _Projection.apply(x, self.codes, self.scales), None
        codes, scales = provider.quantized_input(x)
        return _Projection.apply(x, self.codes, self.scales, codes, scales), None

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
            [linear(x[..., g, :])[0] for g, linear in enumerate(self.projections)],
            dim=-2,
        )
