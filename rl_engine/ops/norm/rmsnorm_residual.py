# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Task 5 public entry: RMSNorm plus the original-row residual branch, with autograd.

The caller names the backend. ``"reference"`` is the serial FP32 reference
(any device); ``"rocm"`` is the MI300X/MI355X Triton candidate. An unavailable
backend raises; nothing switches backend behind the caller's back.

Autograd exposes ``(y, residual)``. The residual branch is the *input* row, so
its upstream gradient folds straight into ``dx``: ``dx = dx_norm + d_residual``
in FP32, rounded to the input dtype once at the autograd edge. ``gamma.grad``
is the FP32 token fold.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.contracts.operators.rmsnorm_residual import EPS, PROFILE, RMSNormResidualSaved
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.reference.norm.rmsnorm_residual import RMSNormResidualReference

Backend = Literal["reference", "rocm"]


def select_backend(backend: Backend):
    if backend == "reference":
        return RMSNormResidualReference
    if backend == "rocm":
        from rl_engine.backends.rocm.norm.rmsnorm_residual import RocmRMSNormResidual

        return RocmRMSNormResidual
    raise ValueError(f"unknown rmsnorm_residual backend {backend!r}; want 'reference' or 'rocm'")


class _RMSNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps, backend, recompute):
        impl = select_backend(backend)
        y, residual, saved = impl.forward(x, gamma, eps)
        ctx.backend = backend
        ctx.recompute = recompute
        ctx.eps = float(eps)
        if recompute:
            ctx.save_for_backward(x, gamma)
        else:
            ctx.save_for_backward(x, gamma, saved.r)
        ctx.set_materialize_grads(False)
        return y, residual

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, d_residual):
        impl = select_backend(ctx.backend)
        x, gamma, *rest = ctx.saved_tensors
        if ctx.recompute:
            _, _, saved = impl.forward(x, gamma, ctx.eps)
        else:
            saved = RMSNormResidualSaved(r=rest[0], hidden=x.shape[1])
        dy = torch.zeros_like(x) if dy is None else dy
        d_residual = torch.zeros_like(x) if d_residual is None else d_residual
        dx, dgamma = impl.backward(dy, d_residual, x, gamma, saved)
        record_backward(
            "rmsnorm_residual",
            kernel_id=f"{impl.__module__}.{impl.__name__}.backward",
            impl=f"{PROFILE}:{impl.name}:{'recompute' if ctx.recompute else 'saved'}",
            family=impl.name,
        )
        return (
            dx.to(x.dtype) if ctx.needs_input_grad[0] else None,
            dgamma.to(gamma.dtype) if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
        )


def rmsnorm_residual(
    x: Tensor,
    gamma: Tensor,
    eps: float = EPS,
    *,
    backend: Backend,
    recompute: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return ``(y, residual)``: the BF16 normalized row and a byte copy of ``x``.

    ``x`` is BF16 ``[T, D]``, ``gamma`` FP32 ``[D]``, ``eps`` must be ``1e-6``.
    ``recompute=True`` saves only the inputs and re-derives the inverse RMS in
    backward with the same kernel; the result is byte-identical to the saved
    path and tested as such. Double backward is not supported.
    """
    return _RMSNormResidual.apply(x, gamma, eps, backend, recompute)


__all__ = ["Backend", "rmsnorm_residual", "select_backend"]
