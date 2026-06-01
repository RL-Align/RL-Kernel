# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
from rl_engine.utils.logger import logger
from rl_engine.kernels.ops.base import _EXT_AVAILABLE, _C


class FlashAttentionOp:
    """
    Standard FlashAttention wrapper for CUDA.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if not _EXT_AVAILABLE:
            raise RuntimeError(
                "Core binary extension is unavailable. FlashAttention cannot be initialized."
            )

        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library.")
        except ImportError:
            if hasattr(_C, "flash_attn_forward"):
                self.op = _C.flash_attn_forward
                logger.info("Successfully linked to RL-Kernel _C.flash_attn_forward.")
            else:
                raise RuntimeError(
                    "Neither external flash_attn nor _C.flash_attn_forward is available."
                ) from None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Standard attention forward pass.
        Args:
            q: (batch, seqlen, nheads, headdim)
            k: (batch, seqlen, nheads_k, headdim)
            v: (batch, seqlen, nheads_k, headdim)
        """
        assert q.dtype in [torch.float16, torch.bfloat16], "FlashAttention requires FP16 or BF16"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
