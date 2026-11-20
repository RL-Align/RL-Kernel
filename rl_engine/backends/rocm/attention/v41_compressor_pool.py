# SPDX-License-Identifier: Apache-2.0
"""V4.1 C1/C2 pre-norm compressor pooling; no V4 overlap or position bias."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pool(KV, S, Y, P, D: tl.constexpr, R: tl.constexpr):
    group = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    r = tl.arange(0, R)
    off = (group * R + r[:, None]) * D + d[None, :]
    s = tl.load(S + off, d[None, :] < D, other=0)
    k = tl.load(KV + off, d[None, :] < D, other=0)
    p = tl.exp(s - tl.max(s, 0)[None, :])
    p = p / tl.sum(p, 0)[None, :]
    y = tl.sum(k * p, 0)
    tl.store(Y + group * D + d, y, d < D)
    tl.store(P + off, p, d[None, :] < D)


@triton.jit
def _pool_bwd(DY, KV, P, Y, DK, DS, D: tl.constexpr, R: tl.constexpr):
    group = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    r = tl.arange(0, R)
    off = (group * R + r[:, None]) * D + d[None, :]
    k = tl.load(KV + off, d[None, :] < D, other=0)
    p = tl.load(P + off, d[None, :] < D, other=0)
    dy = tl.load(DY + group * D + d, d < D, other=0)
    y = tl.load(Y + group * D + d, d < D, other=0)
    tl.store(DK + off, p * dy[None, :], d[None, :] < D)
    tl.store(DS + off, (p * (k - y[None, :])) * dy[None, :], d[None, :] < D)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score):
        if not torch.version.hip or kv.device.type != "cuda":
            raise RuntimeError("V4.1 compressor pooling requires an AMD ROCm GPU")
        if kv.shape != score.shape or kv.ndim < 3 or kv.shape[-2] not in (1, 2):
            raise ValueError("expected matching [...,ratio,D] with ratio 1 or 2")
        if kv.dtype != torch.float32 or score.dtype != torch.float32 or score.device != kv.device:
            raise ValueError("compressor pooling uses colocated FP32 projection outputs")
        if kv.shape[-1] == 0:
            raise ValueError("head dimension must be positive")
        kv, score = kv.contiguous(), score.contiguous()
        r, d = kv.shape[-2:]
        groups = kv.numel() // (r * d)
        out = torch.empty((*kv.shape[:-2], d), device=kv.device, dtype=torch.float32)
        p = torch.empty_like(kv)
        with torch.cuda.device(kv.device):
            if groups:
                _pool[(groups, triton.cdiv(d, 256))](
                    kv, score, out, p, d, r, enable_fp_fusion=False
                )
        ctx.save_for_backward(kv, p, out)
        return out

    @staticmethod
    def backward(ctx, dy):
        kv, p, out = ctx.saved_tensors
        r, d = kv.shape[-2:]
        groups = kv.numel() // (r * d)
        dk, ds = torch.empty_like(kv), torch.empty_like(kv)
        with torch.cuda.device(kv.device):
            if groups:
                _pool_bwd[(groups, triton.cdiv(d, 256))](
                    dy.contiguous(), kv, p, out, dk, ds, d, r, enable_fp_fusion=False
                )
        return dk, ds


def compressor_pool(kv, score):
    """Pool projected C2 KV; callers retain partial groups and apply BF16 cast/norm.

    C1's actual model path is a BF16 projection followed by norm and bypasses
    this pooling primitive. C1 here is only the degenerate pooling contract.
    """
    return _Pool.apply(kv, score)
