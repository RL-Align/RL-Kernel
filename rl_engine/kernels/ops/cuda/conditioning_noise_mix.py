# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA conditioning noise mix with first-order backward."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.conditioning_noise_mix import _prepare_timestep, _validate_inputs


class _ConditioningNoiseMixCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        ctx.save_for_backward(timestep)
        return _C.conditioning_noise_mix_forward(sample, timestep, noise)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        (timestep,) = ctx.saved_tensors
        d_sample, d_noise = _C.conditioning_noise_mix_backward(
            grad_out.resolve_neg().contiguous(), timestep
        )
        return (
            d_sample if ctx.needs_input_grad[0] else None,
            None,
            d_noise if ctx.needs_input_grad[2] else None,
        )


class ConditioningNoiseMixCudaOp:
    """CUDA conditioning noise mix matching the eager PyTorch reference."""

    op_class = "elementwise"

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not all(
            hasattr(_C, name)
            for name in ("conditioning_noise_mix_forward", "conditioning_noise_mix_backward")
        ):
            raise RuntimeError("conditioning_noise_mix requires the compiled CUDA extension")

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
        if sample.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("conditioning_noise_mix requires NVIDIA CUDA tensors")
        _validate_inputs(sample, timestep, noise)
        sample, noise = sample.to(dtype=dtype), noise.to(dtype=dtype)
        timestep = _prepare_timestep(sample, timestep)
        return _ConditioningNoiseMixCudaFunction.apply(
            sample.resolve_neg(), timestep.resolve_neg(), noise.resolve_neg()
        )
