# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Autograd wrapper for the native CUDA AdaLN modulation kernels."""

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.norm.adaln_modulation import (
    _indexed_domain_reason,
    _int32_addressable,
    _validate,
    _validate_select01,
    _validate_select01_metadata,
)


class _AdaLNIndexedCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulation, index, eps, threads):
        ctx.set_materialize_grads(False)
        y, gate = _C.adaln_indexed_forward(x, modulation, index, eps, threads)
        ctx.save_for_backward(x, modulation, index)
        ctx.eps, ctx.threads = eps, threads
        return y, gate

    @staticmethod
    def backward(ctx, grad_y, grad_gate):
        x, modulation, index = ctx.saved_tensors
        for grad in (grad_y, grad_gate):
            if grad is not None and not _int32_addressable(grad):
                raise ValueError("indexed CUDA upstream requires int32 indexing")
        dx, dm = _C.adaln_indexed_backward(
            None if grad_y is None else grad_y.contiguous(),
            None if grad_gate is None else grad_gate.contiguous(),
            x,
            modulation,
            index,
            ctx.eps,
            ctx.threads,
        )
        return None if grad_y is None else dx, dm, None, None, None


class _AdaLNCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulation, eps, threads):
        y, gate = _C.adaln_modulation_forward(x, modulation, eps, threads)
        ctx.save_for_backward(x, modulation)
        ctx.eps = eps
        ctx.threads = threads
        return y, gate

    @staticmethod
    def backward(ctx, grad_y, grad_gate):
        x, modulation = ctx.saved_tensors
        dx, dm = _C.adaln_modulation_backward(
            grad_y.contiguous(), grad_gate.contiguous(), x, modulation, ctx.eps, ctx.threads
        )
        return dx, dm, None, None


class CudaAdaLNModulationOp:
    def __init__(self, *, threads=256):
        if threads not in (128, 256, 512):
            raise ValueError("AdaLN threads must be 128, 256, or 512")
        self.threads = threads
        if not _EXT_AVAILABLE or not all(
            hasattr(_C, name) for name in ("adaln_modulation_forward", "adaln_modulation_backward")
        ):
            raise RuntimeError("CUDA AdaLN extension symbols are unavailable")
        self.indexed_available = all(
            hasattr(_C, name) for name in ("adaln_indexed_forward", "adaln_indexed_backward")
        )

    def __call__(self, x, modulation, *, eps=1e-6, modulate_index=None):
        return self.forward(x, modulation, eps=eps, modulate_index=modulate_index)

    def forward(self, x, modulation, *, eps=1e-6, modulate_index=None):
        if modulate_index is not None:
            _validate_select01_metadata(x, modulation, modulate_index, eps)
            reason = _indexed_domain_reason(x, modulation, modulate_index, eps)
            if reason in {"int32 indexing", "B <= 65535"}:
                raise ValueError(f"indexed CUDA requires {reason}")
            if reason is not None:
                raise RuntimeError(f"indexed CUDA requires {reason}")
            _validate_select01(x, modulation, modulate_index, eps)
            if not self.indexed_available:
                raise RuntimeError("indexed CUDA extension symbols are unavailable")
            return _AdaLNIndexedCuda.apply(
                x.contiguous(),
                modulation.contiguous(),
                modulate_index.to(dtype=torch.int32).contiguous(),
                float(eps),
                self.threads,
            )
        _validate(x, modulation, eps)
        if x.shape[0] > 65535:
            raise ValueError("CUDA AdaLN requires B <= 65535")
        if max(x.numel(), modulation.numel()) > 2**31 - 1:
            raise ValueError("CUDA AdaLN requires tensors to fit int32 indexing")
        if not x.is_cuda or torch.version.hip is not None or x.shape[-1] > 4096:
            raise RuntimeError("CUDA AdaLN requires NVIDIA CUDA tensors with H <= 4096")
        return _AdaLNCuda.apply(x.contiguous(), modulation.contiguous(), float(eps), self.threads)
