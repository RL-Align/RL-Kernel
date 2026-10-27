# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Task 28: FP32 gate logits with one arithmetic path for all batch sizes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.gemm.aiter_fixed_gemm import (
    aiter_fixed_linear_fp32,
    fixed_gemm_execution,
)


def _validate(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("gate expects hidden [M,K] and weight [experts,K]")
    if x.shape[1] != weight.shape[1] or x.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("gate requires matching positive K and a positive expert count")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("gate requires unquantized BF16 inputs and weights; no implicit casts")
    if any(s <= 0 for t in (x, weight) for s in t.stride()):
        raise ValueError("gate requires positive strides (expanded tensors are unsupported)")
    if not x.is_cuda or weight.device != x.device or torch.version.hip is None:
        raise RuntimeError("gate requires both tensors on the same ROCm GPU")
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("task-28 gate does not support Graph capture")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.partition(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"unsupported gate architecture: {arch}")


class _GateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        if x.shape[0] == 0:
            return torch.empty((0, weight.shape[0]), device=x.device, dtype=torch.float32)
        return aiter_fixed_linear_fp32(x, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_logits):
        # Keep upstream FP32 logits gradients intact. The existing fixed-order
        # FP32 GEMM is an explicit backward implementation, not a forward fallback.
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        x, weight = ctx.saved_tensors
        if x.shape[0] == 0:
            return (
                torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None,
            )
        dx = dw = None
        grad = grad_logits.float().contiguous()
        if ctx.needs_input_grad[0]:
            dx = _triton_gemm_fp32(grad, weight.float()).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = _triton_gemm_fp32(grad.t().contiguous(), x.float()).to(weight.dtype)
        return dx, dw


@dataclass(frozen=True)
class GateProjectionResult:
    logits: torch.Tensor
    execution: dict


def router_gate_projection(
    hidden: torch.Tensor, weight: torch.Tensor, *, model_version: str
) -> GateProjectionResult:
    """Project all experts, retaining FP32 scores and per-call execution provenance.

    The caller supplies a synchronized version covering checkpoint, LoRA and
    scales. No weight transpose/derived cache exists; reloads are read directly.
    Positive strides, including row padding and transposes, are accepted.
    Score transforms, selection, capacity, and routing derivatives are tasks 29–34.
    """
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("gate requires a nonempty synchronized model_version")
    _validate(hidden, weight)
    execution = fixed_gemm_execution(hidden)
    execution.update(
        {
            "task": 28,
            "model_version": model_version,
            "input_shape": list(hidden.shape),
            "weight_shape": list(weight.shape),
            "input_stride": list(hidden.stride()),
            "weight_stride": list(weight.stride()),
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "grad_enabled": torch.is_grad_enabled(),
            "input_requires_grad": hidden.requires_grad,
            "weight_requires_grad": weight.requires_grad,
            "kernel_launched": hidden.shape[0] != 0,
            "backward_backend": "rlk.triton.fixed_fp32_dot",
        }
    )
    return GateProjectionResult(_GateFn.apply(hidden, weight), execution)


class RouterGateProjectionOp:
    """Semantic-catalog entry; all eligibility checks run before kernel launch."""

    def __call__(self, hidden, weight, *, model_version):
        return router_gate_projection(hidden, weight, model_version=model_version)
