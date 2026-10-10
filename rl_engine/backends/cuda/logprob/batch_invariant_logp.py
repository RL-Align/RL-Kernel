# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


def _sm90_supported(logits: torch.Tensor) -> bool:
    """Whether the TMA forward can run these logits directly.

    Hopper (SM90) only, bf16/fp32 only, and the TMA descriptor needs the vocab
    row stride (``V * element_size``) to be a multiple of 16 bytes.

    The device capability is checked per input (not just at registry init). The
    caller raises when this returns false; silent fallback is forbidden.
    """
    if not logits.is_cuda or logits.dtype not in (torch.bfloat16, torch.float32):
        return False
    if torch.cuda.get_device_capability(logits.device)[0] != 9:
        return False
    return (logits.size(-1) * logits.element_size()) % 16 == 0


class _BatchInvariantLogpSM90Function(torch.autograd.Function):
    # Autograd wrapper: SM90 TMA forward + tile-wise softmax backward.

    @staticmethod
    def forward(ctx, logits, target_ids, ignore_index):
        lead_shape = logits.shape[:-1]
        vocab_size = logits.size(-1)

        logits_2d = logits.reshape(-1, vocab_size).contiguous()
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()

        logp, lse = _C.batch_invariant_logp_sm90(logits_2d, target_1d, int(ignore_index))

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
        num_tokens = logits_2d.shape[0]

        grad_flat = grad_output.reshape(-1).contiguous().to(torch.float32)
        grad_logits = torch.empty_like(logits_2d)

        # Reuse Triton's tile-wise backward
        try:
            import triton

            from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
                _BLOCK_V,
                _batch_invariant_logp_bwd_kernel,
            )

            grid = (num_tokens, triton.cdiv(vocab_size, _BLOCK_V))
            _batch_invariant_logp_bwd_kernel[grid](
                logits_2d,
                target_1d,
                lse,
                grad_flat,
                grad_logits,
                num_tokens,
                vocab_size,
                logits_2d.stride(0),
                ignore_index=ignore_index,
                BLOCK_V=_BLOCK_V,
            )
        except Exception:  # pragma: no cover - Triton missing
            valid = target_1d != ignore_index
            safe_target = torch.where(valid, target_1d, torch.zeros_like(target_1d))
            probs = torch.exp(logits_2d.float() - lse.unsqueeze(1))
            onehot = torch.zeros_like(probs)
            onehot.scatter_(1, safe_target.unsqueeze(1), 1.0)
            grad = grad_flat.unsqueeze(1) * (onehot - probs)
            grad = torch.where(valid.unsqueeze(1), grad, torch.zeros_like(grad))
            grad_logits = grad.to(logits_2d.dtype)

        grad_logits = grad_logits.reshape(ctx.lead_shape + (vocab_size,))
        return grad_logits, None, None


class BatchInvariantLogpSM90Op:
    # SM90 (Hopper) TMA batch-invariant selected-token log-probability.

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "batch_invariant_logp_sm90"):
            raise RuntimeError(
                "batch_invariant_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.batch_invariant_logp_sm90 kernel.")

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

        if not _sm90_supported(logits):
            raise RuntimeError(
                "BatchInvariantLogpSM90Op requires Hopper CUDA with a 16-byte-aligned "
                "vocab stride; Triton/Native fallback is forbidden"
            )

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

        return _BatchInvariantLogpSM90Function.apply(logits, target_ids, ignore_index)

    def forward_with_lse(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the exact SM90 path and return its direct FP32 logprob/LSE outputs.

        Unlike the production ``apply`` method, this diagnostic entry point never
        falls back to Triton or PyTorch, so comparison provenance stays truthful.
        """
        if logits.dim() < 2:
            raise ValueError(
                f"logits must be at least 2-D ([*lead, V]), got shape {tuple(logits.shape)}"
            )
        if logits.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"logits leading shape {tuple(logits.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        if not _sm90_supported(logits):
            raise RuntimeError(
                "exact cuda-sm90 logprob diagnostics require Hopper, CUDA BF16/FP32 logits, "
                "and a 16-byte-aligned vocab row stride; fallback is disabled"
            )
        if validate:
            vocab_size = logits.size(-1)
            valid_targets = target_ids.reshape(-1)
            valid_targets = valid_targets[valid_targets != ignore_index]
            if valid_targets.numel() and (
                (valid_targets < 0).any() or (valid_targets >= vocab_size).any()
            ):
                bad = valid_targets[(valid_targets < 0) | (valid_targets >= vocab_size)]
                raise ValueError(
                    f"target_ids contains values outside [0, {vocab_size}): {bad.tolist()}"
                )

        lead_shape = logits.shape[:-1]
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()
        logp, lse = _C.batch_invariant_logp_sm90(logits_2d, target_1d, int(ignore_index))
        return logp.reshape(lead_shape), lse.reshape(lead_shape)
