# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA adapter for the frozen joint-attention softmax contract."""

from __future__ import annotations

from math import prod

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.joint_attn_softmax_layout import (
    LogicalKeyMapping,
    validate_key_padding_mask,
)

# ---------------------------------------------------------------------------
# Key-layout preparation
# ---------------------------------------------------------------------------


def _build_key_mapping(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> LogicalKeyMapping:
    """Build one compact logical-key map per batch item on the GPU."""
    validate_key_padding_mask(scores, key_padding_mask)
    logical_to_physical, valid_key_counts = _C.joint_attn_softmax_build_key_mapping(
        key_padding_mask.contiguous()
    )
    return LogicalKeyMapping(
        logical_to_physical=logical_to_physical,
        valid_key_counts=valid_key_counts,
        rows_per_batch=prod(scores.shape[1:-1]),
    )


# ---------------------------------------------------------------------------
# Autograd bridge
# ---------------------------------------------------------------------------


class _JointAttnSoftmaxCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        output_fp32: bool,
        logical_to_physical: torch.Tensor | None,
        valid_key_counts: torch.Tensor | None,
        rows_per_batch: int,
    ) -> torch.Tensor:
        scores_contiguous = scores.contiguous()
        if output_fp32:
            probabilities = (
                _C.joint_attn_softmax_forward_fp32(scores_contiguous)
                if logical_to_physical is None
                else _C.joint_attn_softmax_forward_fp32(
                    scores_contiguous,
                    logical_to_physical,
                    valid_key_counts,
                    rows_per_batch,
                )
            )
            saved_probabilities_fp32 = probabilities
        else:
            probabilities, saved_probabilities_fp32 = (
                _C.joint_attn_softmax_forward_with_state(scores_contiguous)
                if logical_to_physical is None
                else _C.joint_attn_softmax_forward_with_state(
                    scores_contiguous,
                    logical_to_physical,
                    valid_key_counts,
                    rows_per_batch,
                )
            )
        if logical_to_physical is None or valid_key_counts is None:
            ctx.save_for_backward(saved_probabilities_fp32)
        else:
            ctx.save_for_backward(
                saved_probabilities_fp32,
                logical_to_physical,
                valid_key_counts,
            )
        ctx.has_key_layout = logical_to_physical is not None
        ctx.input_bf16 = scores.dtype == torch.bfloat16
        ctx.rows_per_batch = rows_per_batch
        return probabilities

    @staticmethod
    def backward(
        ctx,
        grad_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities_fp32 = ctx.saved_tensors[0]
        logical_to_physical = ctx.saved_tensors[1] if ctx.has_key_layout else None
        valid_key_counts = ctx.saved_tensors[2] if ctx.has_key_layout else None
        grad_scores = _C.joint_attn_softmax_backward(
            probabilities_fp32,
            grad_probabilities.contiguous(),
            ctx.input_bf16,
            logical_to_physical,
            valid_key_counts,
            ctx.rows_per_batch,
        )
        return grad_scores, None, None, None, None


# ---------------------------------------------------------------------------
# Public operator
# ---------------------------------------------------------------------------


class JointAttnSoftmaxCudaOp:
    """Online tiled CUDA softmax over the final key dimension."""

    backend_id = "rlkernel.cuda.joint_attn_softmax"
    provenance = {
        "selected_backend": "cuda",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v2-logical-mask-tile256-exp7",
        "fallback": False,
    }

    def __init__(self) -> None:
        required_cuda_symbols = (
            "joint_attn_softmax_build_key_mapping",
            "joint_attn_softmax_forward",
            "joint_attn_softmax_forward_fp32",
            "joint_attn_softmax_forward_with_state",
            "joint_attn_softmax_backward",
        )
        if not _EXT_AVAILABLE or not all(hasattr(_C, name) for name in required_cuda_symbols):
            raise RuntimeError(
                "joint-attention softmax CUDA symbols are unavailable; rebuild rl_engine._C"
            )

    def __call__(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`, matching the native-op interface."""
        return self.forward(scores, key_padding_mask=key_padding_mask)

    def forward(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute in FP32 and cast once at the final CUDA write."""
        return self._forward_impl(scores, False, key_padding_mask)

    def forward_fp32(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return FP32 probabilities using fixed 256-key online tiles."""
        return self._forward_impl(scores, True, key_padding_mask)

    def _forward_impl(
        self,
        scores: torch.Tensor,
        output_fp32: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_scores(scores)
        key_mapping = (
            None if key_padding_mask is None else _build_key_mapping(scores, key_padding_mask)
        )
        return self._forward_core(scores, output_fp32, key_mapping)

    @staticmethod
    def _forward_core(
        scores: torch.Tensor,
        output_fp32: bool,
        key_mapping: LogicalKeyMapping | None,
    ) -> torch.Tensor:
        logical_to_physical = None if key_mapping is None else key_mapping.logical_to_physical
        valid_key_counts = None if key_mapping is None else key_mapping.valid_key_counts
        rows_per_batch = 1 if key_mapping is None else key_mapping.rows_per_batch
        if not torch.is_grad_enabled() or not scores.requires_grad:
            forward = (
                _C.joint_attn_softmax_forward_fp32 if output_fp32 else _C.joint_attn_softmax_forward
            )
            return (
                forward(scores.contiguous())
                if key_mapping is None
                else forward(
                    scores.contiguous(),
                    logical_to_physical,
                    valid_key_counts,
                    rows_per_batch,
                )
            )
        return _JointAttnSoftmaxCudaFunction.apply(
            scores,
            output_fp32,
            logical_to_physical,
            valid_key_counts,
            rows_per_batch,
        )

    @staticmethod
    def _validate_scores(scores: torch.Tensor) -> None:
        if not scores.is_cuda:
            raise RuntimeError("JointAttnSoftmaxCudaOp requires a CUDA tensor")
        if scores.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"scores must use BF16 or FP32, got {scores.dtype}")
        if scores.dim() < 1:
            raise ValueError("scores must be at least 1-D with shape [..., K]")
        if scores.size(-1) == 0:
            raise ValueError("scores key dimension must be non-empty")
