# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib.util
import os

import torch

from rl_engine.utils.logger import logger


def _select_flash_attn_backend() -> str:
    """Select the installed FlashAttention ROCm backend before importing flash_attn."""
    if os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "").upper() == "TRUE":
        return "triton"
    if importlib.util.find_spec("flash_attn_2_cuda") is None:
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        return "triton"
    return "ck"


class RocmFlashAttentionOp:
    """
    Standard FlashAttention wrapper for ROCm.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if torch.version.hip is None:
            raise RuntimeError("RocmFlashAttentionOp requires a ROCm PyTorch build.")

        backend = _select_flash_attn_backend()
        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library (%s backend).", backend)
        except ImportError as exc:
            raise RuntimeError(
                "ROCm FlashAttention requires a ROCm-compatible flash-attn installation. "
                "See docs/getting_started/installation.md#rocm-backend."
            ) from exc

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
        valid_dtypes = (torch.float16, torch.bfloat16)
        if (
            q.dtype not in valid_dtypes
            or k.dtype not in valid_dtypes
            or v.dtype not in valid_dtypes
        ):
            raise TypeError("FlashAttention requires FP16 or BF16 for q/k/v")
        # PyTorch uses the CUDA device API for both CUDA and ROCm tensors.
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("Inputs must be on a CUDA/ROCm GPU device")
        if not (q.device == k.device == v.device):
            raise ValueError("q, k, and v must be on the same device")

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
