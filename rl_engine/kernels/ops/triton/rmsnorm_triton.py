import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y, RSTD, T: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_H: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(ss / H + EPS)
    y = x * rstd * w

    tl.store(Y + row * H + offs, y, mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rmsnorm_bwd_dx_kernel(
    DY, X, W, RSTD, DX, PARTIAL_DW, T: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    dy = tl.load(DY + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row).to(tl.float32)

    gw = dy * w
    dot = tl.sum(gw * x, axis=0)
    dx = rstd * gw - x * rstd * rstd * rstd * dot / H

    pdw = dy * x * rstd

    tl.store(DX + row * H + offs, dx, mask=mask)
    tl.store(PARTIAL_DW + row * H + offs, pdw, mask=mask)


@triton.jit
def _rmsnorm_bwd_dw_kernel(PARTIAL_DW, DW, T: tl.constexpr, H: tl.constexpr, BLOCK_T: tl.constexpr):
    col = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < T

    vals = tl.load(PARTIAL_DW + offs_t * H + col, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(DW + col, acc)


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps: float = 1e-6):
        assert x.is_cuda and weight.is_cuda
        assert x.dim() == 2 and weight.dim() == 1
        T, H = x.shape
        assert weight.numel() == H

        y = torch.empty_like(x)
        rstd = torch.empty((T,), device=x.device, dtype=torch.float32)

        block_h = triton.next_power_of_2(H)
        assert block_h <= 131072, "H too large for this simple Triton kernel"

        _rmsnorm_fwd_kernel[(T,)](x, weight, y, rstd, T, H, eps, BLOCK_H=block_h)
        ctx.save_for_backward(x, weight, rstd)
        ctx.H = H
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        T, H = x.shape
        dx = torch.empty_like(x)
        partial_dw = torch.empty((T, H), device=x.device, dtype=torch.float32)
        dw = torch.empty((H,), device=x.device, dtype=torch.float32)

        block_h = triton.next_power_of_2(H)
        block_t = triton.next_power_of_2(T)
        assert block_t <= 131072, "T too large for this simple single-program dw reduction"

        _rmsnorm_bwd_dx_kernel[(T,)](
            grad_out, x, weight, rstd, dx, partial_dw, T, H, BLOCK_H=block_h
        )
        _rmsnorm_bwd_dw_kernel[(H,)](partial_dw, dw, T, H, BLOCK_T=block_t)
        return dx, dw.to(weight.dtype), None


def rmsnorm_triton(x, weight, eps: float = 1e-6):
    return RMSNormTriton.apply(x, weight, eps)
