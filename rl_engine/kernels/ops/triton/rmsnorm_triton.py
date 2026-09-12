import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32

if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X,
        W,
        Y,
        RSTD,
        T: tl.constexpr,
        H: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_H: tl.constexpr,
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
        DY,
        X,
        W,
        RSTD,
        DX,
        PARTIAL_DW,
        T: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
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


def _require_triton():
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for RMSNorm")


def rmsnorm_triton_forward_with_rstd(x, weight, eps: float = 1e-6):
    _require_triton()
    assert x.device.type in ("cuda", "hip", "xpu", "musa")
    assert weight.device.type in ("cuda", "hip", "xpu", "musa")
    assert x.dim() == 2 and weight.dim() == 1
    rows, hidden = x.shape
    assert weight.numel() == hidden

    output = torch.empty_like(x)
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block_hidden = triton.next_power_of_2(hidden)
    assert block_hidden <= 131072, "H too large for this simple Triton kernel"
    _rmsnorm_fwd_kernel[(rows,)](x, weight, output, rstd, rows, hidden, eps, BLOCK_H=block_hidden)
    return output, rstd


def rmsnorm_triton_backward_rows(grad_out, x, weight, rstd):
    _require_triton()
    rows, hidden = x.shape
    dx = torch.empty_like(x)
    partial_dw = torch.empty((rows, hidden), device=x.device, dtype=torch.float32)
    block_hidden = triton.next_power_of_2(hidden)
    _rmsnorm_bwd_dx_kernel[(rows,)](
        grad_out,
        x,
        weight,
        rstd,
        dx,
        partial_dw,
        rows,
        hidden,
        BLOCK_H=block_hidden,
    )
    return dx, partial_dw


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps: float = 1e-6):
        assert x.device.type in ("cuda", "hip", "xpu", "musa")
        assert weight.device.type in ("cuda", "hip", "xpu", "musa")
        assert x.dim() == 2 and weight.dim() == 1
        T, H = x.shape
        assert weight.numel() == H

        y, rstd = rmsnorm_triton_forward_with_rstd(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        dx, partial_dw = rmsnorm_triton_backward_rows(grad_out, x, weight, rstd)
        dw = reduce_rows_fp32(partial_dw)
        record_backward(
            "rms_norm",
            kernel_id=(
                "rl_engine.kernels.ops.triton.rmsnorm_triton._rmsnorm_bwd_dx_kernel"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="triton_rmsnorm_dx_declared_fp32_rowfold_dw",
            family="triton",
        )
        return dx, dw.to(weight.dtype), None


def rmsnorm_triton(x, weight, eps: float = 1e-6):
    return RMSNormTriton.apply(x, weight, eps)


class RMSNormTritonOp:
    """Triton RMSNorm wrapper compatible with the shared operator harness."""

    def __init__(self):
        _require_triton()

    backward_impl = "triton_rmsnorm_dx_declared_fp32_rowfold_dw"

    def __call__(self, x, weight, *, eps: float = 1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps: float = 1e-6):
        hidden = x.shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        y_2d = rmsnorm_triton(x_2d, weight.contiguous(), eps=eps)
        return y_2d.view_as(x)

    def parameter_vjp_contributions_fp32(self, *, x, weight, grad_output, eps: float = 1e-6):
        del weight
        x32 = x.float()
        rstd = torch.rsqrt(x32.square().mean(dim=-1) + float(eps))
        rows = grad_output.float() * x32 * rstd.unsqueeze(-1)
        return {"weight": rows}
