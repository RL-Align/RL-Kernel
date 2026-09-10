# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Ascend NPU prefix-shared fused attention (GRPO decode workload).

Port of the CUDA op in rl_engine/kernels/ops/cuda/attention/prefix_shared_attn.py:
in GRPO, the G generated responses share the exact same prompt-prefix KV cache,
so K/V are stored once per batch and broadcast across all G query groups.

Forward: softmax(Q K^T * scale) @ V on an Ascend C kernel
(`_C_npu.prefix_shared_attention_ascend`) with fp32 online-softmax
accumulation. bf16 in/out, non-causal, no key-padding mask, head dim fixed at
128 -- the same surface as the CUDA `PrefixSharedAttentionOp`, which is
forward-only and so is this port.
"""

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

_HEAD_DIM = 128


class PrefixSharedAttentionAscendOp:
    """Prefix-shared softmax attention on Ascend NPU.

    q [bs, G, len_q, D] attends over a single shared k/v sequence
    [bs, len_kv, D] that every G group reuses. Mirrors the CUDA
    ``PrefixSharedAttentionOp`` surface (``op(q, k, v) -> out``).
    """

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "prefix_shared_attention_ascend"):
            raise RuntimeError(
                "prefix_shared_attention_ascend is not compiled into the extension. "
                "Rebuild with KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: "
                "'pip install -e .'"
            )
        logger.info(
            "Successfully linked to precompiled _C_npu.prefix_shared_attention_ascend kernel."
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefix-shared attention forward pass.

        Args:
            q: Query tensor of shape [bs, G, len_q, head_dim]
            k: Shared Key tensor of shape [bs, len_kv, head_dim]
            v: Shared Value tensor of shape [bs, len_kv, head_dim]

        Returns:
            Output tensor of shape [bs, G, len_q, head_dim]
        """
        return self.forward(q, k, v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(q, k, v)
        return _C_npu.prefix_shared_attention_ascend(
            q.contiguous(), k.contiguous(), v.contiguous()
        )

    @staticmethod
    def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if q.dim() != 4 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"q must be 4-D [B, G, Sq, D] and k/v 3-D [B, Skv, D], got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        b, g, sq, d = q.shape
        skv = k.shape[1]
        if k.shape[0] != b or v.shape[0] != b:
            raise ValueError("batch size mismatch between q/k/v")
        if k.shape[2] != d or v.shape[2] != d:
            raise ValueError(
                f"k/v head dim mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected D={d}"
            )
        if v.shape[1] != skv:
            raise ValueError(
                f"k/v key length mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if d != _HEAD_DIM:
            raise ValueError(f"head dim D must be {_HEAD_DIM}, got {d}")
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError(
                f"only BF16 is supported (matches the CUDA op), got "
                f"q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )
        if not (
            q.device.type == "npu" and k.device.type == "npu" and v.device.type == "npu"
        ):
            raise ValueError("q, k, v must be NPU tensors")
        if sq < 1 or skv < 1:
            raise ValueError(f"Sq and Skv must be positive, got Sq={sq}, Skv={skv}")
        if g < 1:
            raise ValueError(f"G must be positive, got G={g}")
