# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Explicit ROCm task 5 candidate. Selected by the caller; never a silent fallback.

The pinned AITER revision (326cffa) has ``rms_norm``, the in-place
``fused_add_rms_norm_opus`` and ``rmsnorm2d_fwd_with_add``, plus Triton
RMSNorm backward kernels. None of them keeps the unnormalized row for the
residual branch, exposes the per-row FP32 inverse RMS to a caller, or pins the
sum-of-squares and ``dgamma`` reduction order; the in-place variants also
overwrite the input. This module therefore carries its own kernels and does
not modify AITER.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.contracts.operators.rmsnorm_residual import (
    EPS,
    PROFILE,
    RMSNormResidualSaved,
    validate_backward,
    validate_forward,
)

_BLOCK = 256
_DGAMMA_BLOCK = 128
_ARCHES = ("gfx942", "gfx950")
_LAUNCH = {"num_warps": 4, "enable_fp_fusion": False}


def supported_device(device: torch.device) -> bool:
    """True on a ROCm build with an MI300X (gfx942) or MI355X (gfx950) device."""
    if not torch.version.hip or device.type != "cuda" or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return arch in _ARCHES


def _require_rocm(x: Tensor) -> None:
    if not torch.version.hip or x.device.type != "cuda":
        raise RuntimeError(f"{PROFILE} ROCm candidate requires ROCm; no CPU path is registered")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch not in _ARCHES:
        raise RuntimeError(f"{PROFILE} has no kernel candidate for {arch}")


def _row_strides(value: Tensor) -> tuple[int, int]:
    # A [T, D] view with any strides is read in place; outputs are contiguous.
    return int(value.stride(0)), int(value.stride(1))


class RocmRMSNormResidual:
    """MI300X/MI355X candidate for task 5 with explicit forward/backward entry points."""

    name = "rocm-triton"
    profile = PROFILE

    @staticmethod
    @torch.no_grad()
    def forward(
        x: Tensor, gamma: Tensor, eps: float = EPS
    ) -> tuple[Tensor, Tensor, RMSNormResidualSaved]:
        tokens, hidden = validate_forward(x, gamma, eps)
        _require_rocm(x)
        from .rmsnorm_residual_kernels import _forward

        gamma_c = gamma.contiguous()
        y = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
        residual = torch.empty_like(y)
        r = torch.empty((tokens,), device=x.device, dtype=torch.float32)
        if tokens:
            with torch.cuda.device(x.device):
                _forward[(tokens,)](
                    x,
                    gamma_c,
                    y,
                    residual,
                    r,
                    hidden,
                    *_row_strides(x),
                    float(eps),
                    _BLOCK,
                    **_LAUNCH,
                )
        return y, residual, RMSNormResidualSaved(r=r, hidden=hidden)

    @staticmethod
    @torch.no_grad()
    def backward(
        dy: Tensor, d_residual: Tensor, x: Tensor, gamma: Tensor, saved: RMSNormResidualSaved
    ) -> tuple[Tensor, Tensor]:
        tokens, hidden = validate_backward(dy, d_residual, x, gamma, saved)
        _require_rocm(x)
        from .rmsnorm_residual_kernels import _backward_dgamma, _backward_dx

        gamma_c = gamma.contiguous()
        r = saved.r.contiguous()
        dx = torch.empty((tokens, hidden), device=x.device, dtype=torch.float32)
        dgamma = torch.empty((hidden,), device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            if tokens:
                _backward_dx[(tokens,)](
                    dy,
                    d_residual,
                    x,
                    gamma_c,
                    r,
                    dx,
                    hidden,
                    *_row_strides(dy),
                    *_row_strides(d_residual),
                    *_row_strides(x),
                    _BLOCK,
                    **_LAUNCH,
                )
            # T == 0 still runs: the fold over zero rows stores exact zeros.
            _backward_dgamma[((hidden + _DGAMMA_BLOCK - 1) // _DGAMMA_BLOCK,)](
                dy,
                x,
                r,
                dgamma,
                tokens,
                hidden,
                *_row_strides(dy),
                *_row_strides(x),
                _DGAMMA_BLOCK,
                **_LAUNCH,
            )
        return dx, dgamma


__all__ = ["RocmRMSNormResidual", "supported_device"]
