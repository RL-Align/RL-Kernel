# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 sinusoidal timestep features (RFC #420 ``timestep_sinusoid_h3``)."""

from __future__ import annotations

import torch

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.minimax_h3 import H3_FREQ_DIM, H3_MAX_PERIOD
from rl_engine.reference.minimax_h3.fixed_order import tree_sum_lastdim_fp32
from rl_engine.reference.minimax_h3.timestep_sinusoid import (
    NativeH3TimestepSinusoidOp,
    validate_h3_timesteps,
)

KERNEL_ID = "rl_engine._C.h3_timestep_sinusoid_forward"
BACKWARD_IMPL = "row_local_fp32_analytic_tree_sum"


def h3_sinusoid_cuda_available() -> bool:
    return bool(_EXT_AVAILABLE and hasattr(_C, "h3_timestep_sinusoid_forward"))


class _H3TimestepSinusoidCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, timestep: torch.Tensor, num_channels: int, check_range: bool):
        t32 = timestep.detach().float().contiguous()
        out = _C.h3_timestep_sinusoid_forward(
            t32, int(num_channels), float(H3_MAX_PERIOD), check_range
        )
        ctx.save_for_backward(out)
        ctx.num_channels = int(num_channels)
        ctx.timestep_dtype = timestep.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (out,) = ctx.saved_tensors
        half = ctx.num_channels // 2
        freq = NativeH3TimestepSinusoidOp.frequencies_fp32(ctx.num_channels, out.device)
        cos_part, sin_part = out[:, :half], out[:, half:]
        g = grad_out.float()
        # d cos(t f)/dt = -f sin(t f); d sin(t f)/dt = f cos(t f). Row-local,
        # summed over channels with a fixed pairwise tree.
        contrib = torch.cat(
            [-(g[:, :half] * sin_part) * freq, (g[:, half:] * cos_part) * freq], dim=-1
        )
        grad_t = tree_sum_lastdim_fp32(contrib).to(ctx.timestep_dtype)
        record_backward(
            "timestep_sinusoid_h3",
            kernel_id="rl_engine.reference.minimax_h3.fixed_order.tree_sum_lastdim_fp32",
            impl=BACKWARD_IMPL,
            family="cuda",
        )
        return grad_t, None, None


class H3TimestepSinusoidCudaOp:
    """CUDA candidate: one thread per (timestep, frequency), FP32 output.

    Construction fails when the extension lacks the symbol, so the registry
    falls back to the PyTorch reference instead of silently mis-dispatching.
    """

    op_class = "elementwise"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not h3_sinusoid_cuda_available():
            raise RuntimeError(
                "rl_engine._C.h3_timestep_sinusoid_forward is unavailable; rebuild the "
                "CUDA extension with csrc/cuda/h3/timestep_sinusoid.cu"
            )

    def __call__(self, timestep: torch.Tensor, *, num_channels: int = H3_FREQ_DIM):
        return self.forward(timestep, num_channels=num_channels)

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        num_channels: int = H3_FREQ_DIM,
        check_range: bool = True,
    ) -> torch.Tensor:
        # Native code owns value validation, avoiding a second host sync here.
        validate_h3_timesteps(timestep, num_channels, check_range=False)
        if not timestep.is_cuda:
            raise ValueError("H3TimestepSinusoidCudaOp needs a CUDA timestep tensor")
        return _H3TimestepSinusoidCuda.apply(timestep, num_channels, check_range)

    def forward_fp32(self, timestep: torch.Tensor, *, num_channels: int = H3_FREQ_DIM):
        # The op is FP32 end to end; the FP32 path is the op itself.
        return self.forward(timestep, num_channels=num_channels)
