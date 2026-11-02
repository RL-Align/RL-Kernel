# SPDX-License-Identifier: Apache-2.0
"""Task 29: standalone sqrt-softplus scores and a fixed first-order VJP."""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.router_sqrt_softplus import execution_record, implementation
from rl_engine.contracts.operators.router_sqrt_softplus import SUPPORTED_ARCHS


def _validate(logits):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("router sqrt-softplus expects a tensor")
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("router sqrt-softplus expects [M,E] with positive E")
    if logits.dtype != torch.float32:
        raise TypeError("router sqrt-softplus requires FP32 logits; no implicit casts")
    if any(stride <= 0 for stride in logits.stride()):
        raise ValueError("router sqrt-softplus requires positive input strides")
    if not logits.is_cuda or torch.version.hip is None:
        raise RuntimeError("router sqrt-softplus requires a ROCm GPU")
    with torch.cuda.device(logits.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("task-29 router sqrt-softplus does not support Graph capture")
    arch = torch.cuda.get_device_properties(logits.device).gcnArchName.partition(":")[0]
    if arch not in SUPPORTED_ARCHS:
        raise RuntimeError(f"unsupported router sqrt-softplus architecture: {arch}")
    if not torch.isfinite(logits).all().item():
        raise ValueError("router sqrt-softplus requires finite FP32 logits")


class _RouterScoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            scores = api.router_sqrt_softplus_fwd(logits)
        ctx.save_for_backward(logits, scores)
        return scores

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, scores = ctx.saved_tensors
        if not torch.isfinite(grad_output).all().item():
            raise ValueError("router sqrt-softplus requires finite FP32 upstream gradients")
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            return api.router_sqrt_softplus_bwd(logits, scores, grad_output)


@dataclass(frozen=True)
class RouterScoreResult:
    scores: torch.Tensor
    execution: dict


def router_sqrt_softplus(logits, *, model_version: str) -> RouterScoreResult:
    """FP32 [M,E] → FP32 [M,E]; no selection, normalization, bias or scale.

    All finite FP32 logits are admitted. Hardware subnormal/underflow behavior is
    part of the recorded platform contract. NaN/Inf and Graph capture fail closed.
    """
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("router sqrt-softplus requires a synchronized model_version")
    _validate(logits)
    record = execution_record(logits)
    record.update(
        {
            "model_version": model_version,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": logits.requires_grad,
        }
    )
    return RouterScoreResult(_RouterScoreFn.apply(logits), record)


class RouterSqrtSoftplusOp:
    def __call__(self, logits, *, model_version):
        return router_sqrt_softplus(logits, model_version=model_version)
