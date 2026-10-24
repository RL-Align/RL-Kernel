# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch reference for conditioning noise mix."""

from __future__ import annotations

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_inputs(sample: Tensor, timestep: float | Tensor, noise: Tensor) -> None:
    """Validate without reading CUDA values or synchronizing the device.

    Time is fixed conditioning metadata, scalar or one value per batch sample.
    The caller supplies finite time values in [0, 1], as in H3 scale_noise.
    """
    if sample.dtype not in _SUPPORTED_DTYPES or noise.dtype != sample.dtype:
        raise TypeError("sample and noise must share an fp16, bf16, or fp32 dtype")
    if sample.device != noise.device:
        raise ValueError("sample and noise must share a device")
    if sample.shape != noise.shape:
        raise ValueError("sample and noise must share a shape")
    if sample.ndim < 1 or sample.numel() == 0:
        raise ValueError("sample and noise must be non-empty tensors with a batch dimension")
    if not sample.is_contiguous() or not noise.is_contiguous():
        raise ValueError("sample and noise must be contiguous")
    if isinstance(timestep, Tensor):
        if timestep.requires_grad:
            raise ValueError("timestep is fixed conditioning metadata; gradients are unsupported")
        if timestep.device != sample.device:
            raise ValueError("timestep must be on the same device as sample")
        if timestep.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("timestep must have an fp16, bf16, or fp32 dtype")
        if not timestep.is_contiguous():
            raise ValueError("timestep must be contiguous")
        scalar = timestep.ndim == 0 or timestep.numel() == 1
        per_sample = (
            1 <= timestep.ndim <= sample.ndim
            and timestep.shape[0] == sample.shape[0]
            and all(size == 1 for size in timestep.shape[1:])
        )
        if not (scalar or per_sample):
            raise ValueError("timestep must be scalar or [B] with optional trailing singleton axes")
    elif not isinstance(timestep, float):
        raise TypeError("timestep must be a float or floating-point tensor")


def _prepare_timestep(sample: Tensor, timestep: float | Tensor) -> Tensor:
    """Normalize a validated timestep to a flat tensor in the execution dtype."""
    if isinstance(timestep, Tensor):
        return timestep.to(dtype=sample.dtype).reshape(-1)
    return torch.tensor([timestep], device=sample.device, dtype=sample.dtype)


class NativeConditioningNoiseMixOp:
    """Pure PyTorch conditioning noise mix with eager dtype rounding."""

    op_class = "elementwise"

    def __call__(self, sample: Tensor, timestep: float | Tensor, noise: Tensor) -> Tensor:
        return self.forward(sample, timestep, noise)

    def forward(self, sample: Tensor, timestep: float | Tensor, noise: Tensor) -> Tensor:
        return self._mix(sample, timestep, noise, dtype=sample.dtype)

    def forward_fp32(self, sample: Tensor, timestep: float | Tensor, noise: Tensor) -> Tensor:
        return self._mix(sample, timestep, noise, dtype=torch.float32)

    @staticmethod
    def _mix(
        sample: Tensor, timestep: float | Tensor, noise: Tensor, *, dtype: torch.dtype
    ) -> Tensor:
        _validate_inputs(sample, timestep, noise)
        sample, noise = sample.to(dtype=dtype), noise.to(dtype=dtype)
        time = _prepare_timestep(sample, timestep)
        time = time.reshape((-1,) + (1,) * (sample.ndim - 1))
        # Keep eager operation boundaries: low-precision products round before
        # their addition, exactly as MiniMaxH3Scheduler.scale_noise.
        return time * sample + (1.0 - time) * noise
