from __future__ import annotations
import importlib.util
from typing import TypedDict, cast
import torch
from torch.autograd.function import once_differentiable
from torch.version import hip as torch_hip

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

_EPS = 1.0e-6
_VALID_D = (128, 4096)
_MAX_T = 2**31 - 1
_DGAMMA_BLOCK = 128
_LAUNCH_OPTIONS = {
    "num_warps": 4,
    "num_stages": 1,
    "enable_fp_fusion": False,
}


class RMSNormSaved(TypedDict):
    r: torch.Tensor
    d: int


def _require_nvidia_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("Triton candidate requires NVIDIA CUDA")


def _check_tensor(value, name, *, dtype, shape=None, device=None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")


def _check_forward(x, gamma, eps) -> tuple[int, int]:
    _require_nvidia_triton()
    _check_tensor(x, "x", dtype=torch.bfloat16)
    if x.ndim != 2:
        raise ValueError("x must have shape [T, D]")
    t, d = x.shape
    if t <= 0 or t > _MAX_T:
        raise ValueError(f"T must be in [1, {_MAX_T}]")
    if d not in _VALID_D:
        raise ValueError(f"D must be one of {_VALID_D}")
    _check_tensor(gamma, "gamma", dtype=torch.float32, shape=(d,), device=x.device)
    if float(eps) != _EPS:
        raise ValueError(f"eps must be exactly {_EPS}")
    return t, d


if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    @triton.jit
    def _rmsnorm_residual_fwd_kernel(
        x_ptr,
        gamma_ptr,
        y_ptr,
        residual_ptr,
        r_ptr,
        D: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row.to(tl.int64) * D
        acc = tl.zeros((1,), tl.float32)

        for col in tl.range(0, D, loop_unroll_factor=1):
            value = tl.load(x_ptr + base + col).to(tl.float32)
            square = libdevice.mul_rn(value, value)
            acc = libdevice.add_rn(acc, square)

        d32 = tl.full((1,), D, tl.float32)
        eps32 = tl.full((1,), EPS, tl.float32)
        mean = libdevice.div_rn(acc, d32)
        r = libdevice.rsqrt(libdevice.add_rn(mean, eps32))

        cols = tl.arange(0, D)
        x_bf16 = tl.load(x_ptr + base + cols)
        x32 = x_bf16.to(tl.float32)
        gamma32 = tl.load(gamma_ptr + cols).to(tl.float32)
        y32 = libdevice.mul_rn(libdevice.mul_rn(x32, r), gamma32)

        tl.store(y_ptr + base + cols, y32.to(tl.bfloat16, fp_downcast_rounding="rtne"))
        tl.store(residual_ptr + base + cols, x_bf16)
        tl.store(r_ptr + row + tl.arange(0, 1), r)

    @triton.jit
    def _rmsnorm_residual_dx_kernel(
        dy_ptr, d_residual_ptr, x_ptr, gamma_ptr, r_ptr, dx_ptr, D: tl.constexpr
    ):
        row = tl.program_id(0)
        base = row.to(tl.int64) * D
        r = tl.load(r_ptr + row)
        q = tl.zeros((1,), tl.float32)

        for col in tl.range(0, D, loop_unroll_factor=1):
            dy32 = tl.load(dy_ptr + base + col).to(tl.float32)
            gamma32 = tl.load(gamma_ptr + col).to(tl.float32)
            x32 = tl.load(x_ptr + base + col).to(tl.float32)
            u = libdevice.mul_rn(dy32, gamma32)
            q = libdevice.add_rn(q, libdevice.mul_rn(u, x32))

        r3 = libdevice.mul_rn(libdevice.mul_rn(r, r), r)
        d32 = tl.full((1,), D, tl.float32)
        cols = tl.arange(0, D)
        dy32 = tl.load(dy_ptr + base + cols).to(tl.float32)
        dr32 = tl.load(d_residual_ptr + base + cols).to(tl.float32)
        x32 = tl.load(x_ptr + base + cols).to(tl.float32)
        gamma32 = tl.load(gamma_ptr + cols).to(tl.float32)
        u = libdevice.mul_rn(dy32, gamma32)
        numerator = libdevice.mul_rn(libdevice.mul_rn(x32, r3), q)
        rhs = libdevice.div_rn(numerator, d32)
        dx_norm = libdevice.sub_rn(libdevice.mul_rn(r, u), rhs)

        tl.store(dx_ptr + base + cols, libdevice.add_rn(dx_norm, dr32))

    @triton.jit
    def _rmsnorm_residual_dgamma_kernel(
        dy_ptr, x_ptr, r_ptr, dgamma_ptr, T, D: tl.constexpr, BLOCK: tl.constexpr
    ):
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < D
        acc = tl.zeros((BLOCK,), tl.float32)

        for row in tl.range(0, T, loop_unroll_factor=1):
            base = row.to(tl.int64) * D
            dy32 = tl.load(dy_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
            x32 = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + row)
            term = libdevice.mul_rn(libdevice.mul_rn(dy32, x32), r)
            acc = libdevice.add_rn(acc, term)

        tl.store(dgamma_ptr + cols, acc, mask=mask)


def triton_rmsnorm_residual_fwd(
    x, gamma, eps=1.0e-6
) -> tuple[torch.Tensor, torch.Tensor, RMSNormSaved]:
    t, d = _check_forward(x, gamma, eps)
    x_c = x.contiguous()
    gamma_c = gamma.contiguous()
    y = torch.empty_like(x_c)
    residual = torch.empty_like(x_c)
    r = torch.empty((t,), dtype=torch.float32, device=x.device)

    with torch.cuda.device(x.device):
        _rmsnorm_residual_fwd_kernel[(t,)](
            x_c,
            gamma_c,
            y,
            residual,
            r,
            D=d,
            EPS=float(eps),
            **_LAUNCH_OPTIONS,
        )

    return y, residual, {"r": r, "d": d}


def triton_rmsnorm_residual_bwd(
    dy, d_residual, x, gamma, saved
) -> tuple[torch.Tensor, torch.Tensor]:
    t, d = _check_forward(x, gamma, _EPS)
    if not isinstance(saved, dict) or saved.get("d") != d or "r" not in saved:
        raise ValueError("saved stats does not match the forward input")
    _check_tensor(dy, "dy", dtype=torch.bfloat16, shape=x.shape, device=x.device)
    _check_tensor(
        d_residual, "d_residual", dtype=torch.bfloat16, shape=x.shape, device=x.device
    )
    _check_tensor(
        saved["r"],
        "saved.r",
        dtype=torch.float32,
        shape=(t,),
        device=x.device,
    )
    dy_c = dy.contiguous()
    dr_c = d_residual.contiguous()
    x_c = x.contiguous()
    gamma_c = gamma.contiguous()
    r_c = saved["r"].contiguous()
    dx = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    dgamma = torch.empty(gamma.shape, dtype=torch.float32, device=x.device)

    with torch.cuda.device(x.device):
        _rmsnorm_residual_dx_kernel[(t,)](
            dy_c,
            dr_c,
            x_c,
            gamma_c,
            r_c,
            dx,
            D=d,
            **_LAUNCH_OPTIONS,
        )
        grid = ((d + _DGAMMA_BLOCK - 1) // _DGAMMA_BLOCK,)
        _rmsnorm_residual_dgamma_kernel[grid](
            dy_c, x_c, r_c, dgamma, t, D=d, BLOCK=_DGAMMA_BLOCK, **_LAUNCH_OPTIONS
        )

    return dx, dgamma


class RMSNormResidualTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):
        y, residual, saved = triton_rmsnorm_residual_fwd(x, gamma, eps)
        ctx.save_for_backward(x, gamma, saved["r"])
        ctx.set_materialize_grads(False)
        return y, residual

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, d_residual):
        x, gamma, r = ctx.saved_tensors
        dy = torch.zeros_like(x) if dy is None else dy
        d_residual = torch.zeros_like(x) if d_residual is None else d_residual
        dx, dgamma = triton_rmsnorm_residual_bwd(
            dy, d_residual, x, gamma, {"r": r, "d": x.shape[1]}
        )
        return dx.to(x.dtype), dgamma.to(gamma.dtype), None


def rmsnorm_residual(x, gamma, eps=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor], RMSNormResidualTriton.apply(x, gamma, eps)
    )


class RMSNormResidualTritonOp:

    def __init__(self) -> None:
        _require_nvidia_triton()

    def __call__(self, x, gamma, *, eps=1.0e-6):
        return self.forward(x, gamma, eps=eps)

    def forward(self, x, gamma, *, eps=1.0e-6):
        return rmsnorm_residual(x, gamma, eps)
