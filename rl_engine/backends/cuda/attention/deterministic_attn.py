# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic standard-softmax attention for CUDA and ROCm (issue #147).

Forward: QK → masked softmax+LSE → PV (all FP32 intermediate).
Backward: dP → softmax_bwd → dQ/dK/dV with §4.1 fixed GQA order.
Wrapped in autograd.Function so #108 harness can .backward() through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    SplitKVMode,
    SplitKVSpec,
)
from rl_engine.utils.logger import logger

_IS_ROCM = torch.version.hip is not None
_HEAD_DIMS = (128,) if _IS_ROCM else (128, 256)
_GPU_PLATFORM = "ROCm" if _IS_ROCM else "CUDA"


@dataclass(frozen=True)
class DeterministicAttentionCoreResult:
    """Output and auditable arithmetic identity of the shared strict core."""

    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, object]


class _DeterministicAttentionFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        key_padding_mask: Optional[torch.Tensor],
        output_fp32: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        mask_c = key_padding_mask.contiguous() if key_padding_mask is not None else None

        results = (
            _C.deterministic_attention_forward_fp32(q_c, k_c, v_c, causal, float(scale), mask_c)
            if output_fp32
            else _C.deterministic_attention_forward(q_c, k_c, v_c, causal, float(scale), mask_c)
        )
        out, lse, P = results[0], results[1], results[2]

        ctx.save_for_backward(q_c, k_c, v_c, P, mask_c)
        ctx.causal = causal
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)

        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_c, k_c, v_c, P, mask_c = ctx.saved_tensors

        if grad_out.dtype != q_c.dtype:
            grad_out = grad_out.to(q_c.dtype)
        dQ, dK, dV = _C.deterministic_attention_backward(
            grad_out.contiguous(),
            q_c,
            k_c,
            v_c,
            P,
            ctx.causal,
            float(ctx.scale),
            mask_c,
        )

        return dQ, dK, dV, None, None, None, None


class DeterministicAttentionOp:
    """Batch-invariant standard softmax attention on a CUDA or ROCm GPU.

    Materializes full FP32 scores/P. Public surface matches NativeAttentionOp
    so #108 harness can call forward(**inputs) with key_padding_mask.
    """

    core_id = STRICT_ATTENTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_SCHEDULE_ID
    backend_id = "rlkernel.cuda.deterministic_attention"

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_attention_forward"):
            raise RuntimeError(
                f"Deterministic {_GPU_PLATFORM} attention kernel is unavailable. "
                "Rebuild the native extension for the active GPU platform."
            )
        if not hasattr(_C, "deterministic_attention_backward"):
            raise RuntimeError(
                f"Deterministic {_GPU_PLATFORM} attention backward kernel is unavailable. "
                "Rebuild the native extension for the active GPU platform."
            )
        logger.info("Successfully linked to _C.deterministic_attention_forward/backward.")

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
        out, lse = _DeterministicAttentionFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask, False
        )
        return out, lse

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(q, k, v, key_padding_mask)
        resolved_scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        out, _lse = _DeterministicAttentionFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask, True
        )
        return out

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
        if d not in _HEAD_DIMS:
            raise ValueError(f"head dim D must be one of {_HEAD_DIMS}, got {d}")
        if b < 1 or hq < 1 or hkv < 1:
            raise ValueError("batch size and query/key head counts must be positive")
        if hq % hkv != 0:
            raise ValueError(f"Hq={hq} not divisible by Hkv={hkv} (GQA group)")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"only FP16/BF16 supported, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, v must share the same dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("q, k, v must be GPU tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q, k, v must be on the same device")
        if key_padding_mask is not None:
            if key_padding_mask.device != q.device:
                raise ValueError("key_padding_mask must be on the input device")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
            if key_padding_mask.shape != (b, skv):
                raise ValueError(
                    f"key_padding_mask must be [B, Skv]=[{b}, {skv}], "
                    f"got {tuple(key_padding_mask.shape)}"
                )
        if sq < 1 or skv < 1:
            raise ValueError(f"Sq and Skv must be positive, got Sq={sq}, Skv={skv}")


class RLKernelDeterministicAttentionCore:
    """Materializing GPU reference core shared by training and rollout.

    Production uses FA4 CuTe on CUDA and AITER CK dense MHA on ROCm. This core
    remains useful for correctness and capability-gap diagnosis.
    """

    core_id = STRICT_ATTENTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_SCHEDULE_ID
    backend_id = (
        "rlkernel.rocm.deterministic_attention"
        if _IS_ROCM
        else "rlkernel.cuda.deterministic_attention"
    )
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"
    fallback = False
    native_attention_arithmetic = False
    production_ready = False
    reference_only = True

    def __init__(
        self,
        *,
        split_kv: SplitKVSpec | None = None,
    ) -> None:
        requested = SplitKVSpec.disabled() if split_kv is None else split_kv
        if not isinstance(requested, SplitKVSpec):
            raise TypeError("split_kv must be a SplitKVSpec")
        if requested.mode is not SplitKVMode.DISABLED:
            raise ValueError("the strict GPU Attention core requires Split-KV to be disabled")
        self.split_kv = requested
        self._op = DeterministicAttentionOp()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> DeterministicAttentionCoreResult:
        return self.forward_with_lse(q, k, v, **kwargs)

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        key_position_ids: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        self._validate_positions(
            q,
            k,
            causal=causal,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")
        out, lse = self._op.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
        )
        return DeterministicAttentionCoreResult(
            out=out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "attention_backend": self.backend_id,
                "split_kv": self.split_kv.resolve(k.size(2), backend=self.backend_id).to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "fallback": self.fallback,
                "fallback_reason": None,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "production_ready": self.production_ready,
                "reference_only": self.reference_only,
                "strict_schedule": self.strict_schedule,
            },
        )

    @staticmethod
    def _validate_positions(
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        causal: bool,
        query_position_ids: torch.Tensor | None,
        key_position_ids: torch.Tensor | None,
    ) -> None:
        if not causal:
            return
        if query_position_ids is None or key_position_ids is None:
            raise ValueError(
                "strict GPU Attention requires query_position_ids and " "key_position_ids"
            )
        expected_q_shape = (q.size(0), q.size(2))
        expected_k_shape = (k.size(0), k.size(2))
        if tuple(query_position_ids.shape) != expected_q_shape:
            raise ValueError(f"query_position_ids must have shape {expected_q_shape}")
        if tuple(key_position_ids.shape) != expected_k_shape:
            raise ValueError(f"key_position_ids must have shape {expected_k_shape}")
        if query_position_ids.device != q.device or key_position_ids.device != k.device:
            raise ValueError("strict Attention position IDs must be on the Q/K device")
        integer_dtypes = (torch.int32, torch.int64)
        if query_position_ids.dtype not in integer_dtypes or (
            key_position_ids.dtype not in integer_dtypes
        ):
            raise ValueError("strict Attention position IDs must contain integers")
        if q.size(2) > k.size(2):
            raise ValueError("causal strict Attention requires Sq <= Skv")
        if q.size(2) > 1 and bool(
            (query_position_ids[:, 1:] - query_position_ids[:, :-1] != 1).any()
        ):
            raise ValueError("query_position_ids must be contiguous and increasing")
        if k.size(2) > 1 and bool((key_position_ids[:, 1:] - key_position_ids[:, :-1] != 1).any()):
            raise ValueError("key_position_ids must be contiguous and increasing")
        if not torch.equal(query_position_ids, key_position_ids[:, -q.size(2) :]):
            raise ValueError(
                "strict GPU Attention requires queries to be the trailing "
                "contiguous positions of the logical KV sequence"
            )
