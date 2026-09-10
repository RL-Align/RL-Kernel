# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Ascend NPU deterministic standard-softmax attention (issue #147).

Forward: QK -> masked softmax+LSE -> PV (all FP32 intermediate) on an
Ascend C kernel (`_C_npu.deterministic_attention_ascend`). Every
(b, q_head, row) is reduced end-to-end by one AI-core block with a fixed
64-key tile order and no split-K merge, so per-row numerics are
batch-invariant (the same algorithm as the Triton reference and the CUDA
deterministic op).

Backward: Triton is unavailable on NPU, so the backward recomputes the
fp32 reference forward (`NativeAttentionOp.forward_fp32`, the same golden
path the forward kernel accumulates in) under autograd and VJPs the
upstream gradient through it, reusing the forward-saved q/k/v/mask.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False

_HEAD_DIM = 128


class _DeterministicAttentionAscendFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        key_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        mask_c = key_padding_mask.contiguous() if key_padding_mask is not None else None

        out, lse = _C_npu.deterministic_attention_ascend(
            q_c, k_c, v_c, causal, float(scale), mask_c
        )

        ctx.save_for_backward(q_c, k_c, v_c, mask_c)
        ctx.causal = causal
        ctx.scale = scale
        ctx.has_mask = mask_c is not None
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        del grad_lse  # lse is non-differentiable; always None upstream
        q, k, v, mask = ctx.saved_tensors
        # VJP of the fp32 reference forward: the Ascend C forward accumulates in
        # fp32 (like the CUDA deterministic op), so the backward must match the
        # fp32 golden path, not the low-precision dtype path.
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            out = NativeAttentionOp().forward_fp32(
                q_ref,
                k_ref,
                v_ref,
                causal=ctx.causal,
                scale=ctx.scale,
                key_padding_mask=mask if ctx.has_mask else None,
            )
        dq, dk, dv = torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_out)
        return dq, dk, dv, None, None, None


class DeterministicAttentionAscendOp:
    """Batch-invariant standard softmax attention on Ascend NPU.

    Public surface matches ``NativeAttentionOp`` / ``DeterministicAttentionOp``
    so the #108 harness can call ``forward(**inputs)`` with ``key_padding_mask``.
    Out-of-domain inputs are rejected up front (the registry-level
    ``PYTORCH_NATIVE_ATTENTION`` entry covers unavailable-kernel fallback).
    """

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "deterministic_attention_ascend"):
            raise RuntimeError(
                "deterministic_attention_ascend is not compiled into the extension. "
                "Rebuild with KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: "
                "'pip install -e .'"
            )
        logger.info(
            "Successfully linked to precompiled _C_npu.deterministic_attention_ascend kernel."
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Harness / registry main path: return out only. Differentiable."""
        out, _lse = self.forward_with_lse(
            q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (out, lse) with FP32 LSE for debug / handoff hooks."""
        self._validate_inputs(q, k, v, key_padding_mask)
        resolved_scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        out, lse = _DeterministicAttentionAscendFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask
        )
        return out, lse

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"q/k/v must be 4-D [B, H, S, D], got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        b, hq, sq, d = q.shape
        hkv, skv = k.shape[1], k.shape[2]
        if k.shape[0] != b or v.shape[0] != b:
            raise ValueError("batch size mismatch between q/k/v")
        if v.shape[1] != hkv or v.shape[2] != skv or k.shape[3] != d or v.shape[3] != d:
            raise ValueError(
                f"k/v shape mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected k/v [B={b}, Hkv, Skv, D={d}]"
            )
        if d != _HEAD_DIM:
            raise ValueError(f"head dim D must be {_HEAD_DIM}, got {d}")
        if hq % hkv != 0:
            raise ValueError(f"Hq={hq} not divisible by Hkv={hkv} (GQA group)")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"only FP16/BF16 supported, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, v must share the same dtype")
        if not (q.device.type == "npu" and k.device.type == "npu" and v.device.type == "npu"):
            raise ValueError("q, k, v must be NPU tensors")
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
            if key_padding_mask.shape != (b, skv):
                raise ValueError(
                    f"key_padding_mask must be [B, Skv]=[{b}, {skv}], "
                    f"got {tuple(key_padding_mask.shape)}"
                )
        if sq < 1 or skv < 1:
            raise ValueError(f"Sq and Skv must be positive, got Sq={sq}, Skv={skv}")
