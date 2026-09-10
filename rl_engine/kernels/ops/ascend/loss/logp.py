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


class _FusedLogpAscendAutograd(torch.autograd.Function):
    """Autograd bridge for the Ascend fused selected-logprob forward.

    Mirrors the CUDA ``_FusedLogpAutograd``: the VJP is row-local
    (``dlogits = grad * (one_hot(target) - softmax)``), computed in FP32 and
    cast only the final input VJP back to the input dtype. There is no
    cross-token reduction, so Batch/Chunk layout cannot change the result.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_ids: torch.Tensor):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        labels = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = _C_npu.fused_logp_ascend(logits_2d, labels)
        ctx.save_for_backward(logits_2d, labels)
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, labels = ctx.saved_tensors
        probs = torch.softmax(logits.float(), dim=-1)
        rows = torch.arange(logits.size(0), device=logits.device)
        probs[rows, labels] -= 1.0
        grad = -grad_output.reshape(-1, 1).float() * probs
        return grad.to(ctx.input_dtype).reshape(ctx.input_shape), None


class FusedLogpAscendOp:
    """Batch-invariant fused LogP for Ascend NPU.

    The Ascend C forward mirrors the deterministic CUDA kernel's two-pass
    (row max, then sum-exp) fp32 reduction with a fixed tile order; the
    output is fp32, matching ``DeterministicLogpCUDAOp``'s contract.
    """

    is_fused_logp = True
    is_batch_invariant = True

    def __init__(self):
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "fused_logp_ascend"):
            raise RuntimeError(
                "fused_logp_ascend is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        self.op = _C_npu.fused_logp_ascend
        logger.info("Successfully linked to precompiled _C_npu.fused_logp_ascend kernel.")

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, token_ids)

    def _ascend_supported(self, logits: torch.Tensor) -> bool:
        """NPU tensors only; bf16/fp16/fp32 (mirrors the CUDA kernel's gate)."""
        return (
            logits.device.type == "npu"
            and logits.is_contiguous()
            and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        )

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if not self._ascend_supported(logits):
            from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp

            return NativeLogpOp()(logits, token_ids)
        return _FusedLogpAscendAutograd.apply(logits, token_ids)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if not self._ascend_supported(logits):
            from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp

            return NativeLogpOp().forward_fp32(logits, token_ids)
        return _FusedLogpAscendAutograd.apply(logits, token_ids)
