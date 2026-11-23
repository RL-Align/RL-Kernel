# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MUSA-owned fused selected-logprob operator."""

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.ops.logprob.fused_logp import (
    FusedLogpAutogradForward,
    FusedLogpOp,
    fused_logp_backward_chunked,
)
from rl_engine.utils.logger import logger


class _MusaFusedLogpAutograd(FusedLogpAutogradForward):
    """MUSA bridge preserving the native backward dtype and fallback contract."""

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, labels = ctx.saved_tensors
        if logits.device.type == "musa" and hasattr(ctx.backend, "fused_logp_backward"):
            grad = ctx.backend.fused_logp_backward(
                logits,
                labels,
                grad_output.reshape(-1).contiguous(),
            )
        else:
            grad = fused_logp_backward_chunked(logits, labels, grad_output, ctx.input_dtype)
        return grad.reshape(ctx.input_shape), None, None


class MusaFusedLogpOp(FusedLogpOp):
    """Use the native MUSA forward and backward functions exposed by ``rl_engine._C``."""

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp"):
            raise RuntimeError("Base custom kernel 'fused_logp' is unavailable.")
        self._backend = _C
        self.op = self._backend.fused_logp
        logger.info("Successfully linked to precompiled _C.fused_logp fallback kernel.")

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return _MusaFusedLogpAutograd.apply(logits, token_ids, self._backend)
