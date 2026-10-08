# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA WS1 Flow-GRPO transition with row-local deterministic FP32 VJP."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.diffusion.flow_sde_step_logp import (
    FlowSDEResult,
    execution_trace,
    prepare_inputs,
)


class _FlowSDEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sample, velocity, params, auxiliary, replay):
        target, logp, mean, std, coeff = _C.flow_sde_step_logp_forward(
            sample, velocity, params, auxiliary, replay
        )
        ctx.save_for_backward(target, mean, coeff)
        ctx.replay = replay
        ctx.mark_non_differentiable(std)
        if replay:
            ctx.mark_non_differentiable(target)
        return target, logp, mean, std

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_target, grad_logp, grad_mean, grad_std):
        target, mean, coeff = ctx.saved_tensors

        def gradient_or_zero(gradient, like):
            if gradient is None:
                return torch.zeros_like(like)
            return gradient.float().contiguous()

        grad_target = gradient_or_zero(grad_target, target)
        grad_mean = gradient_or_zero(grad_mean, mean)
        if grad_logp is None:
            grad_logp = mean.new_zeros((mean.shape[0],))
        dx, dv = _C.flow_sde_step_logp_backward(
            target,
            mean,
            coeff,
            grad_target,
            grad_logp.float().contiguous(),
            grad_mean,
            ctx.replay,
        )
        return dx, dv, None, None, None


class CUDAFlowSDEStepLogpOp:
    """Native CUDA forward/backward. Unsupported configurations fail closed."""

    def __init__(self):
        self._last_trace = None
        required = (
            "flow_sde_step_logp_forward",
            "flow_sde_step_logp_backward",
            "flow_sde_strict_math",
        )
        if not _EXT_AVAILABLE or any(not hasattr(_C, name) for name in required):
            raise RuntimeError("Flow SDE CUDA symbols missing; rebuild rl_engine._C")
        if not _C.flow_sde_strict_math():
            raise RuntimeError("Flow SDE strict CUDA rejects KERNEL_ALIGN_USE_FAST_MATH")

    def __call__(
        self,
        sample,
        model_output,
        sigma,
        sigma_next,
        *,
        sigma_max,
        noise_level=0.7,
        noise=None,
        prev_sample=None,
    ):
        if not sample.is_cuda or torch.version.hip is not None:
            raise RuntimeError("Flow SDE CUDA requires NVIDIA CUDA; no fallback")
        params, auxiliary = prepare_inputs(
            sample,
            model_output,
            sigma,
            sigma_next,
            sigma_max=sigma_max,
            noise_level=noise_level,
            noise=noise,
            prev_sample=prev_sample,
        )
        result = FlowSDEResult(
            *_FlowSDEFunction.apply(
                sample.float().contiguous(),
                model_output.float().contiguous(),
                params,
                auxiliary,
                prev_sample is not None,
            )
        )
        self._last_trace = execution_trace("cuda", sample)
        self._last_trace["execution_recorded"] = True
        self._last_trace["mode"] = "sampling" if noise is not None else "replay"
        return result

    apply = __call__
    forward = __call__
    forward_fp32 = __call__

    def execution_trace(self, sample=None):
        return dict(self._last_trace or execution_trace("cuda", sample))
