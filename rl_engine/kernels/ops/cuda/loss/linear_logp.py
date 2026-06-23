# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.loss.linear_logp import chunked_linear_logp_backward
from rl_engine.utils.logger import logger

# Hidden-dim slice the SM90 forward streams per TMA load; D must be a multiple of
# it (mirrors `constexpr int BK` in csrc/cuda/fused_linear_logp_sm90.cu).
_SM90_BK = 32


def _sm90_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
    """Whether the bf16 TMA+MMA forward can run these inputs directly."""
    return (
        hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and lm_head_weight.dtype == torch.bfloat16
        and hidden.size(-1) % _SM90_BK == 0
    )


def _fallback_op():
    """Portable op for inputs the SM90 forward cannot take (fp32/fp16, or a hidden
    dim not divisible by the kernel's K slice). Prefers Triton, else native."""
    try:
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    except Exception:  # pragma: no cover - Triton missing
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()


class _FusedLinearLogpSM90Function(torch.autograd.Function):
    """SM90 TMA+WGMMA fused forward + Liger-style chunked backward.

    The forward calls the compiled ``_C.fused_linear_logp_sm90`` kernel (logits
    never materialized). The backward reuses the deterministic chunked cuBLAS
    path so gradients flow into ``hidden``, ``lm_head_weight`` and ``bias``.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        logp, _lse = _C.fused_linear_logp_sm90(hidden_2d, weight, target_1d, bias)

        ctx.save_for_backward(hidden_2d, weight, bias if bias is not None else hidden_2d, target_1d)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = chunked_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            target_1d,
            bias_t,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
        )
        # Inputs: hidden, lm_head_weight, bias, target_ids.
        return grad_hidden, grad_weight, grad_bias, None


class FusedLinearLogpSM90Op:
    """SM90 (Hopper) TMA+WGMMA fused linear log-prob.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing the
    ``[N, V]`` logits. Requires the extension built with ``KERNEL_ALIGN_FORCE_SM90=1``
    on an SM90 device; bfloat16 hidden/weight only.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_linear_logp_sm90"):
            raise RuntimeError(
                "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.fused_linear_logp_sm90 kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(hidden, lm_head_weight, target_ids, bias)

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if lm_head_weight.device != hidden.device:
            raise ValueError(
                f"lm_head_weight device {lm_head_weight.device} must match hidden "
                f"device {hidden.device}"
            )
        n_tokens = hidden.numel() // hidden.size(-1)
        if target_ids.numel() != n_tokens:
            raise ValueError(
                f"target_ids must have one id per token: expected {n_tokens}, "
                f"got {target_ids.numel()}"
            )
        if bias is not None:
            if bias.numel() != lm_head_weight.size(0):
                raise ValueError(
                    f"bias must have V={lm_head_weight.size(0)} elements, got {bias.numel()}"
                )
            if bias.device != hidden.device:
                raise ValueError(
                    f"bias device {bias.device} must match hidden device {hidden.device}"
                )
        if not _sm90_supported(hidden, lm_head_weight):
            return _fallback_op()(hidden, lm_head_weight, target_ids, bias)
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        return _FusedLinearLogpSM90Function.apply(hidden, lm_head_weight, bias, target_ids)
