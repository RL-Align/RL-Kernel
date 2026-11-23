# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini dynamic K/V convolution contract (INK-16)."""

from __future__ import annotations

import torch

KERNEL_SIZE = 4
KV_STREAMS = 2


def dynamic_kv_conv(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply token-dependent causal K/V convolution with explicit state.

    ``x`` is ``[batch, time, 2, hidden]`` and coefficients are
    ``[batch, time, 2, hidden, 4]``. The previous three K/V frames are passed
    in ``state`` as ``[batch, 2, hidden, 3]``. Tap accumulation is FP32 and
    proceeds from oldest to newest before casting back to the input dtype.
    """
    if x.ndim != 4 or x.shape[2] != KV_STREAMS:
        raise ValueError("K/V input must have shape [batch, time, 2, hidden]")
    batch, time, _, hidden = x.shape
    expected_coefficients = (batch, time, KV_STREAMS, hidden, KERNEL_SIZE)
    if coefficients.shape != expected_coefficients:
        raise ValueError(f"coefficients must have shape {expected_coefficients}")
    if not x.is_floating_point() or not coefficients.is_floating_point():
        raise TypeError("dynamic K/V convolution tensors must be floating point")
    if x.device != coefficients.device or x.dtype != coefficients.dtype:
        raise ValueError("input and coefficients must share device and dtype")

    if state is None:
        state = torch.zeros(
            (batch, KV_STREAMS, hidden, KERNEL_SIZE - 1),
            device=x.device,
            dtype=x.dtype,
        )
    expected_state = (batch, KV_STREAMS, hidden, KERNEL_SIZE - 1)
    if state.shape != expected_state:
        raise ValueError(f"state must have shape {expected_state}")
    if state.device != x.device or state.dtype != x.dtype:
        raise ValueError("state must share device and dtype with input")

    history = torch.cat((state, x.permute(0, 2, 3, 1)), dim=-1)
    output = torch.zeros((batch, KV_STREAMS, hidden, time), device=x.device, dtype=torch.float32)
    coefficients_fp32 = coefficients.float().permute(0, 2, 3, 1, 4)
    for tap in range(KERNEL_SIZE):
        output = output + history[..., tap : tap + time].float() * coefficients_fp32[..., tap]
    next_state = history[..., -(KERNEL_SIZE - 1) :].contiguous()
    return output.permute(0, 3, 1, 2).to(dtype=x.dtype), next_state


class _NativeDynamicKVConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficients: torch.Tensor, state: torch.Tensor):
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_dynamic_kv_conv_forward_with_state"):
            raise RuntimeError("Inkling 2 CUDA dynamic K/V convolution extension is unavailable")
        output, next_state = _C.inkling2_dynamic_kv_conv_forward_with_state(
            x.contiguous(), coefficients.contiguous(), state.contiguous()
        )
        ctx.save_for_backward(x, coefficients, state)
        return output, next_state

    @staticmethod
    def backward(ctx, grad_output, grad_next_state):
        from rl_engine.backends.extension import _C

        x, coefficients, state = ctx.saved_tensors
        if grad_next_state is not None and torch.any(grad_next_state != 0):
            raise RuntimeError("native dynamic K/V convolution state gradients are not supported")
        return _C.inkling2_dynamic_kv_conv_backward(
            grad_output.contiguous(), x, coefficients, state
        )


def dynamic_kv_conv_cuda(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the native CUDA INK-16 dynamic K/V convolution kernel."""
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("native dynamic K/V convolution requires CUDA tensors")
    if state is None:
        state = torch.zeros((x.shape[0], 2, x.shape[3], 3), device=x.device, dtype=x.dtype)
    if not state.is_cuda:
        raise ValueError("native dynamic K/V convolution state must be CUDA")
    return _NativeDynamicKVConvFunction.apply(x, coefficients, state)
