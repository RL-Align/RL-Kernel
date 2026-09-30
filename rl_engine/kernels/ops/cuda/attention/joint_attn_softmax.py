# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA adapter for the frozen joint-attention softmax contract."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


class _JointAttnSoftmaxCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, output_fp32: bool) -> torch.Tensor:
        scores_contiguous = scores.contiguous()
        if output_fp32:
            probabilities = _C.joint_attn_softmax_forward_fp32(scores_contiguous)
            saved_probabilities_fp32 = probabilities
        else:
            probabilities, saved_probabilities_fp32 = _C.joint_attn_softmax_forward_with_state(
                scores_contiguous
            )
        ctx.save_for_backward(saved_probabilities_fp32)
        ctx.input_bf16 = scores.dtype == torch.bfloat16
        return probabilities

    @staticmethod
    def backward(ctx, grad_probabilities: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probabilities_fp32,) = ctx.saved_tensors
        grad_scores = _C.joint_attn_softmax_backward(
            probabilities_fp32,
            grad_probabilities.contiguous(),
            ctx.input_bf16,
        )
        return grad_scores, None


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
        "kernel_fingerprint": "joint-attn-softmax-v1-tile256-exp7",
        "fallback": False,
    }

    def __init__(self) -> None:
        required_cuda_symbols = (
            "joint_attn_softmax_forward",
            "joint_attn_softmax_forward_fp32",
            "joint_attn_softmax_forward_with_state",
            "joint_attn_softmax_backward",
        )
        if not _EXT_AVAILABLE or not all(hasattr(_C, name) for name in required_cuda_symbols):
            raise RuntimeError(
                "joint-attention softmax CUDA symbols are unavailable; rebuild rl_engine._C"
            )

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`, matching the native-op interface."""
        return self.forward(scores)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute in FP32 and cast once at the final CUDA write."""
        self._validate_scores(scores)
        if not torch.is_grad_enabled() or not scores.requires_grad:
            return _C.joint_attn_softmax_forward(scores.contiguous())
        return _JointAttnSoftmaxCudaFunction.apply(scores, False)

    def forward_fp32(self, scores: torch.Tensor) -> torch.Tensor:
        """Return FP32 probabilities using fixed 256-key online tiles."""
        self._validate_scores(scores)
        if not torch.is_grad_enabled() or not scores.requires_grad:
            return _C.joint_attn_softmax_forward_fp32(scores.contiguous())
        return _JointAttnSoftmaxCudaFunction.apply(scores, True)

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
