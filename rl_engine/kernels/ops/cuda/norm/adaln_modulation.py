# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Autograd wrapper for the native CUDA AdaLN modulation kernels."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import _validate


class _AdaLNCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulation, eps):
        y, gate = _C.adaln_modulation_forward(x, modulation, eps)
        ctx.save_for_backward(x, modulation)
        ctx.eps = eps
        return y, gate

    @staticmethod
    def backward(ctx, grad_y, grad_gate):
        x, modulation = ctx.saved_tensors
        dx, dm = _C.adaln_modulation_backward(
            grad_y.contiguous(), grad_gate.contiguous(), x, modulation, ctx.eps
        )
        return dx, dm, None


class CudaAdaLNModulationOp:
    def __init__(self):
        if not _EXT_AVAILABLE or not all(
            hasattr(_C, name)
            for name in ("adaln_modulation_forward", "adaln_modulation_backward")
        ):
            raise RuntimeError("CUDA AdaLN extension symbols are unavailable")

    def __call__(self, x, modulation, *, eps=1e-6):
        return self.forward(x, modulation, eps=eps)

    def forward(self, x, modulation, *, eps=1e-6):
        _validate(x, modulation, eps)
        if not x.is_cuda or torch.version.hip is not None or x.shape[-1] > 4096:
            raise RuntimeError("CUDA AdaLN requires NVIDIA CUDA tensors with H <= 4096")
        return _AdaLNCuda.apply(x.contiguous(), modulation.contiguous(), float(eps))
