# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os

import torch

from rl_engine.utils.logger import logger

_MAX_TESTED_ROCM_TRITON_HEAD_DIM = 512


def _select_flash_attn_backend() -> str:
    """Select the installed FlashAttention ROCm backend."""
    return "triton"


class RocmFlashAttentionOp:
    """
    Standard FlashAttention wrapper for ROCm.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if torch.version.hip is None:
            raise RuntimeError("RocmFlashAttentionOp requires a ROCm PyTorch build.")

        backend = _select_flash_attn_backend()
        if backend == "triton":
            # flash-attn selects the ROCm CK/Triton backend at import time.
            os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library (%s backend).", backend)
        except (ImportError, OSError, RuntimeError) as exc:
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
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "q, k, and v must be rank-4 tensors: (batch, seqlen, nheads, head_dim)"
            )

        head_dim = q.shape[-1]
        if head_dim == 0:
            raise ValueError("head_dim must be positive")
        if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
            raise ValueError("q, k, and v must have the same head_dim")
        if head_dim > _MAX_TESTED_ROCM_TRITON_HEAD_DIM:
            raise NotImplementedError(
                "RL-Kernel's ROCm FlashAttention wrapper currently supports "
                f"head_dim <= {_MAX_TESTED_ROCM_TRITON_HEAD_DIM}; got {head_dim}"
            )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
