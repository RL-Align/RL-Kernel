from __future__ import annotations
from typing import Any, cast

import torch
from torch.autograd.function import once_differentiable

from rl_engine.mhc import oracle


class _NativeRMSNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):
        y, residual, saved = oracle.rmsnorm_residual_fwd(x, gamma, float(eps))
        ctx.save_for_backward(x, gamma, saved["r"])
        ctx.set_materialize_grads(False)
        return y, residual

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, d_residual):
        x, gamma, r = ctx.saved_tensors
        dy = torch.zeros_like(x) if dy is None else dy
        d_residual = torch.zeros_like(x) if d_residual is None else d_residual
        saved = {"x32": x.to(torch.float32), "r": r, "d": x.shape[1]}
        dx, dgamma = oracle.rmsnorm_residual_bwd(dy, d_residual, x, gamma, saved)
        return dx.to(x.dtype), dgamma.to(gamma.dtype), None


def rmsnorm_residual(x, gamma, eps=1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor], _NativeRMSNormResidual.apply(x, gamma, eps)
    )


class NativeRMSNormResidualOp:

    def __call__(self, x, gamma, *, eps=1.0e-6):
        return self.forward(x, gamma, eps=eps)

    def forward(self, x, gamma, *, eps=1.0e-6):
        return rmsnorm_residual(x, gamma, eps)
