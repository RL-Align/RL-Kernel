# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

import torch

from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False


def _ascend_supported(logits: torch.Tensor) -> bool:
    """Whether the Ascend C forward can run these logits directly.

    NPU tensors only, bf16/fp32 only (mirrors the SM90 kernel's dtype gate).
    """
    return logits.device.type == "npu" and logits.dtype in (
        torch.bfloat16,
        torch.float32,
    )


def _fallback_op():
    """Portable op for inputs the Ascend forward cannot take.

    Triton rejects non-CUDA devices, so on NPU the only fallback is native.
    """
    from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp

    return NativeBatchInvariantLogpOp()


def _batch_invariant_logp_backward(
    logits_2d: torch.Tensor,
    target_1d: torch.Tensor,
    lse: torch.Tensor,
    grad_flat: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Compute onehot - softmax using one full-size FP32 workspace."""
    valid = target_1d != ignore_index
    safe_target = torch.where(valid, target_1d, torch.zeros_like(target_1d))

    grad = torch.empty_like(logits_2d, dtype=torch.float32)
    grad.copy_(logits_2d)
    grad.sub_(lse.unsqueeze(1)).exp_().neg_()
    grad.scatter_add_(
        1,
        safe_target.unsqueeze(1),
        valid.to(dtype=grad.dtype).unsqueeze(1),
    )
    grad.mul_((grad_flat * valid).unsqueeze(1))
    return grad.to(logits_2d.dtype)


class _BatchInvariantLogpAscendFunction(torch.autograd.Function):
    # Autograd wrapper: Ascend C forward + PyTorch-formula backward.
    # The SM90 op reuses Triton's tile-wise backward; Triton is unavailable on
    # NPU, so the backward uses the same onehot - softmax formula as the SM90
    # wrapper's portable branch, reusing the forward-saved lse.

    @staticmethod
    def forward(ctx, logits, target_ids, ignore_index):
        lead_shape = logits.shape[:-1]
        vocab_size = logits.size(-1)

        logits_2d = logits.reshape(-1, vocab_size).contiguous()
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()

        logp, lse = _C_npu.batch_invariant_logp_ascend(logits_2d, target_1d, int(ignore_index))

        ctx.save_for_backward(logits_2d, target_1d, lse)
        ctx.ignore_index = ignore_index
        ctx.lead_shape = lead_shape
        ctx.vocab_size = vocab_size
        return logp.reshape(lead_shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, target_1d, lse = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        vocab_size = ctx.vocab_size

        grad_flat = grad_output.reshape(-1).contiguous().to(torch.float32)

        grad_logits = _batch_invariant_logp_backward(
            logits_2d,
            target_1d,
            lse,
            grad_flat,
            ignore_index,
        )

        grad_logits = grad_logits.reshape(ctx.lead_shape + (vocab_size,))
        return grad_logits, None, None


class BatchInvariantLogpAscendOp:
    # Ascend C batch-invariant selected-token log-probability (forward kernel).

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "batch_invariant_logp_ascend"):
            raise RuntimeError(
                "batch_invariant_logp_ascend is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C_npu.batch_invariant_logp_ascend kernel.")

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        return self.apply(logits, target_ids, ignore_index=ignore_index, validate=validate)

    def apply(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError(
                f"logits must be at least 2-D ([*lead, V]), got shape {tuple(logits.shape)}"
            )
        if logits.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"logits leading shape {tuple(logits.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )

        if not _ascend_supported(logits):
            return _fallback_op()(logits, target_ids, ignore_index=ignore_index, validate=validate)

        if validate:
            vocab_size = logits.size(-1)
            target_flat = target_ids.reshape(-1)
            valid_targets = target_flat[target_flat != ignore_index]
            if valid_targets.numel() > 0 and (
                (valid_targets < 0).any() or (valid_targets >= vocab_size).any()
            ):
                bad = valid_targets[(valid_targets < 0) | (valid_targets >= vocab_size)]
                raise ValueError(
                    f"target_ids contains values outside [0, {vocab_size}): {bad.tolist()}"
                )

        return _BatchInvariantLogpAscendFunction.apply(logits, target_ids, ignore_index)
