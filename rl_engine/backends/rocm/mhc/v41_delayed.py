# SPDX-License-Identifier: Apache-2.0
"""V4.1 delayed-pre mHC primitives with fixed reductions and full parameter VJP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inv(X, R, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + t * D + d, d < D, other=0)
    tl.store(R + t, tl.rsqrt(tl.sum(x * x, 0) / D + EPS))


@triton.jit
def _inv_bwd(DR, X, R, DX, D: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + t * D + d, d < D, other=0)
    r = tl.load(R + t)
    dr = tl.load(DR + t)
    tl.store(DX + t * D + d, x * ((-dr / D) * r * r * r), d < D)


class _InverseRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        if x.dtype != torch.float32 or x.device.type != "cuda" or not torch.version.hip:
            raise ValueError("mHC controller inverse RMS requires ROCm FP32 rows")
        x = x.contiguous()
        d = x.shape[-1]
        if not 0 < d <= 32768:
            raise ValueError("controller width outside validated kernel profile")
        rows = x.numel() // d
        r = torch.empty((*x.shape[:-1], 1), device=x.device)
        with torch.cuda.device(x.device):
            _inv[(rows,)](x, r, d, eps, triton.next_power_of_2(d), enable_fp_fusion=False)
        ctx.save_for_backward(x, r)
        return r

    @staticmethod
    def backward(ctx, dr):
        x, r = ctx.saved_tensors
        d = x.shape[-1]
        dx = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _inv_bwd[(x.numel() // d,)](
                dr.contiguous(), x, r, dx, d, triton.next_power_of_2(d), enable_fp_fusion=False
            )
        return dx, None


@triton.jit
def _split(M, S, B, PRE, POST, COMB, SIG, TAPE, EPS: tl.constexpr, ITERS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, 4)
    scale0 = tl.load(S)
    scale1 = tl.load(S + 1)
    scale2 = tl.load(S + 2)
    a = tl.load(M + t * 24 + h) * scale0 + tl.load(B + h)
    b = tl.load(M + t * 24 + 4 + h) * scale1 + tl.load(B + 4 + h)
    p = 1.0 / (1.0 + tl.exp(-a))
    q = 1.0 / (1.0 + tl.exp(-b))
    tl.store(PRE + t * 4 + h, p + EPS)
    tl.store(POST + t * 4 + h, 2.0 * q)
    tl.store(SIG + t * 8 + h, p)
    tl.store(SIG + t * 8 + 4 + h, q)
    c = h[:, None] * 4 + h[None, :]
    z = tl.load(M + t * 24 + 8 + c) * scale2 + tl.load(B + 8 + c)
    exp = tl.exp(z - tl.max(z, 1)[:, None])
    z = exp / tl.sum(exp, 1)[:, None] + EPS
    tl.store(TAPE + t * (2 * ITERS) * 16 + c, z)
    for step in range(2 * ITERS - 1):
        if step % 2 == 0:
            z = z / (tl.sum(z, 0)[None, :] + EPS)
        else:
            z = z / (tl.sum(z, 1)[:, None] + EPS)
        tl.store(TAPE + (t * (2 * ITERS) + step + 1) * 16 + c, z)
    tl.store(COMB + t * 16 + c, z)


@triton.jit
def _split_bwd(
    DPRE, DPOST, DCOMB, M, S, SIG, TAPE, DM, DBP, DSP, EPS: tl.constexpr, ITERS: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.arange(0, 4)
    p = tl.load(SIG + t * 8 + h)
    q = tl.load(SIG + t * 8 + 4 + h)
    da = tl.load(DPRE + t * 4 + h) * p * (1.0 - p)
    db = tl.load(DPOST + t * 4 + h) * 2.0 * q * (1.0 - q)
    c = h[:, None] * 4 + h[None, :]
    grad = tl.load(DCOMB + t * 16 + c)
    for index in range(2 * ITERS - 1):
        step = 2 * ITERS - 2 - index
        prev = tl.load(TAPE + (t * (2 * ITERS) + step) * 16 + c)
        after = tl.load(TAPE + (t * (2 * ITERS) + step + 1) * 16 + c)
        if step % 2 == 0:
            grad = (grad - tl.sum(grad * after, 0)[None, :]) / (tl.sum(prev, 0)[None, :] + EPS)
        else:
            grad = (grad - tl.sum(grad * after, 1)[:, None]) / (tl.sum(prev, 1)[:, None] + EPS)
    soft = tl.load(TAPE + t * (2 * ITERS) * 16 + c) - EPS
    dc = soft * (grad - tl.sum(grad * soft, 1)[:, None])
    tl.store(DM + t * 24 + h, da * tl.load(S))
    tl.store(DM + t * 24 + 4 + h, db * tl.load(S + 1))
    tl.store(DM + t * 24 + 8 + c, dc * tl.load(S + 2))
    tl.store(DBP + t * 24 + h, da)
    tl.store(DBP + t * 24 + 4 + h, db)
    tl.store(DBP + t * 24 + 8 + c, dc)
    tl.store(DSP + t * 3, tl.sum(da * tl.load(M + t * 24 + h), 0))
    tl.store(DSP + t * 3 + 1, tl.sum(db * tl.load(M + t * 24 + 4 + h), 0))
    tl.store(DSP + t * 3 + 2, tl.sum(tl.sum(dc * tl.load(M + t * 24 + 8 + c), 0), 0))


@triton.jit
def _fold(P, Y, T: tl.constexpr, N: tl.constexpr):
    n = tl.arange(0, 32)
    acc = tl.full((32,), 0, tl.float32)
    for t in range(T):
        acc = acc + tl.load(P + t * N + n, n < N, other=0)
    tl.store(Y + n, acc, n < N)


class _Split(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, eps, iters):
        if mixes.shape[-1] != 24 or scale.shape != (3,) or base.shape != (24,):
            raise ValueError("delayed mHC profile requires HC4 controller shapes")
        if any(t.dtype != torch.float32 or t.device != mixes.device for t in (mixes, scale, base)):
            raise ValueError("controller operands must share FP32 device")
        if mixes.device.type != "cuda" or not torch.version.hip or not 1 <= iters <= 32:
            raise ValueError("requires ROCm and 1..32 Sinkhorn iterations")
        mixes, scale, base = (t.contiguous() for t in (mixes, scale, base))
        lead = mixes.shape[:-1]
        rows = mixes.numel() // 24
        pre = torch.empty((*lead, 4), device=mixes.device)
        post = torch.empty_like(pre)
        comb = torch.empty((*lead, 4, 4), device=mixes.device)
        sig = torch.empty((rows, 8), device=mixes.device)
        tape = torch.empty((rows, 2 * iters, 16), device=mixes.device)
        with torch.cuda.device(mixes.device):
            _split[(rows,)](
                mixes, scale, base, pre, post, comb, sig, tape, eps, iters, enable_fp_fusion=False
            )
        ctx.save_for_backward(mixes, scale, sig, tape)
        ctx.eps, ctx.iters = eps, iters
        return pre, post, comb

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        mixes, scale, sig, tape = ctx.saved_tensors
        rows = mixes.numel() // 24
        dm = torch.empty_like(mixes)
        dbp = torch.empty((rows, 24), device=mixes.device)
        dsp = torch.empty((rows, 3), device=mixes.device)
        db = torch.empty(24, device=mixes.device)
        ds = torch.empty(3, device=mixes.device)
        with torch.cuda.device(mixes.device):
            _split_bwd[(rows,)](
                dpre.contiguous(),
                dpost.contiguous(),
                dcomb.contiguous(),
                mixes,
                scale,
                sig,
                tape,
                dm,
                dbp,
                dsp,
                ctx.eps,
                ctx.iters,
                enable_fp_fusion=False,
            )
            _fold[(1,)](dbp, db, rows, 24, enable_fp_fusion=False)
            _fold[(1,)](dsp, ds, rows, 3, enable_fp_fusion=False)
        return dm, ds, db, None, None


@triton.jit
def _collapse(X, P, Y, D: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    acc = tl.full((256,), 0, tl.float32)
    for h in tl.static_range(4):
        x = tl.load(X + (t * 4 + h) * D + d, d < D, other=0).to(tl.float32)
        acc = acc + x * tl.load(P + t * 4 + h)
    tl.store(Y + t * D + d, acc, d < D)


@triton.jit
def _collapse_bwd(DY, X, P, DX, DP, D: tl.constexpr, B: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    dy = tl.load(DY + t * D + d, d < D, other=0).to(tl.float32)
    x = tl.load(X + (t * 4 + h) * D + d, d < D, other=0).to(tl.float32)
    tl.store(DX + (t * 4 + h) * D + d, dy * tl.load(P + t * 4 + h), d < D)
    tl.store(DP + t * 4 + h, tl.sum(dy * x, 0))


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre):
        if (
            x.shape[-2] != 4
            or pre.shape != x.shape[:-1]
            or x.dtype != torch.bfloat16
            or pre.dtype != torch.float32
        ):
            raise ValueError("expected BF16 [...,4,D] and FP32 [...,4] delayed pre mix")
        x, pre = x.contiguous(), pre.contiguous()
        d = x.shape[-1]
        y = torch.empty((*x.shape[:-2], d), device=x.device, dtype=x.dtype)
        with torch.cuda.device(x.device):
            _collapse[(x.numel() // (4 * d), triton.cdiv(d, 256))](
                x, pre, y, d, enable_fp_fusion=False
            )
        ctx.save_for_backward(x, pre)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, pre = ctx.saved_tensors
        d = x.shape[-1]
        dx, dp = torch.empty_like(x), torch.empty_like(pre)
        with torch.cuda.device(x.device):
            _collapse_bwd[(x.numel() // (4 * d), 4)](
                dy.contiguous(),
                x,
                pre,
                dx,
                dp,
                d,
                triton.next_power_of_2(d),
                enable_fp_fusion=False,
            )
        return dx, dp


def inverse_rms(x, *, eps):
    return _InverseRMS.apply(x, float(eps))


def split_sinkhorn(mixes, scale, base, *, eps=1e-6, iters=20):
    return _Split.apply(mixes, scale, base, float(eps), int(iters))


def collapse(x, pre):
    return _Collapse.apply(x, pre)
