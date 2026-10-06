import importlib
from typing import cast
import torch
from torch.autograd.function import once_differentiable
from torch.version import hip as torch_hip


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is required")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    for name in ("mhc_rmsnorm_residual_forward", "mhc_rmsnorm_residual_backward"):
        if not hasattr(ext, name):
            raise RuntimeError(f"Missing extension symbol: {name}")
    return ext


def cuda_rmsnorm_residual_fwd(x, gamma, eps=1e-6):
    y, residual, r = _extension().mhc_rmsnorm_residual_forward(
        x.contiguous(), gamma.contiguous(), float(eps)
    )
    return y, residual, {"r": r, "d": x.shape[1]}


def cuda_rmsnorm_residual_bwd(dy, dr, x, gamma, saved):
    if x.ndim != 2:
        raise ValueError("x must be [T, D]")
    if not isinstance(saved, dict) or "r" not in saved or saved.get("d") != x.shape[1]:
        raise ValueError("saved state does not match the forward input")
    return tuple(
        _extension().mhc_rmsnorm_residual_backward(
            dy.contiguous(),
            dr.contiguous(),
            x.contiguous(),
            gamma.contiguous(),
            saved["r"],
        )
    )


class RMSNormResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, eps):
        y, residual, saved = cuda_rmsnorm_residual_fwd(x, gamma, eps)
        ctx.save_for_backward(x, gamma, saved["r"])
        ctx.set_materialize_grads(False)
        return y, residual

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dr):
        x, gamma, r = ctx.saved_tensors
        dy = torch.zeros_like(x) if dy is None else dy
        dr = torch.zeros_like(x) if dr is None else dr
        dx, dg = cuda_rmsnorm_residual_bwd(dy, dr, x, gamma, {"r": r, "d": x.shape[1]})
        return dx.to(x.dtype), dg.to(gamma.dtype), None


def rmsnorm_residual(x, gamma, eps=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(tuple[torch.Tensor, torch.Tensor], RMSNormResidual.apply(x, gamma, eps))


class RMSNormResidualCudaOp:

    def __init__(self) -> None:
        _extension()

    def __call__(self, x, gamma, *, eps=1e-6):
        return self.forward(x, gamma, eps=eps)

    def forward(self, x, gamma, *, eps=1e-6):
        return rmsnorm_residual(x, gamma, eps)
