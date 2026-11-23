# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""H3 non-causal MHA with query-chunk recomputation, retaining #240 arithmetic."""
from __future__ import annotations

import importlib
import math
from types import ModuleType

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_C: ModuleType | None
try:
    _C = importlib.import_module("rl_engine._H3_C")
except ImportError:
    _C = None

_SCALE = 1.0 / math.sqrt(128)


def _extension():
    required = (
        "h3_attention_forward",
        "h3_attention_build_info",
        "h3_attention_backward_accumulate",
        "h3_attention_probabilities",
    )
    if _C is None or any(not hasattr(_C, name) for name in required):
        raise RuntimeError("H3 CUDA extension unavailable; rebuild RL-Kernel (no fallback)")
    if _C.h3_attention_build_info().get("profile") != "h3-cuda-strict-v1":
        raise RuntimeError("H3 requires a strict CUDA build; rebuild with RL_KERNEL_H3_STRICT=1")
    return _C


class _H3AttentionFn(Function):
    @staticmethod
    def forward(ctx, q, k, v, chunk_rows):
        ext = _extension()
        q, k, v = (x.contiguous() for x in (q, k, v))
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        for start in range(0, q.shape[2], chunk_rows):
            end = min(start + chunk_rows, q.shape[2])
            block_out, block_lse, p = ext.h3_attention_forward(
                q[:, :, start:end].contiguous(), k, v
            )
            out[:, :, start:end] = block_out
            lse[:, :, start:end] = block_lse
            del block_out, block_lse, p
        ctx.save_for_backward(q, k, v)
        ctx.chunk_rows = chunk_rows
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out, grad_lse):
        ext = _extension()
        q, k, v = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        for start in range(0, q.shape[2], ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, q.shape[2])
            qt = q[:, :, start:end].contiguous()
            # Deliberately retain the original softmax (not an LSE-based alternative).
            p = ext.h3_attention_probabilities(qt, k, v)
            dqt, _, _ = ext.h3_attention_backward_accumulate(
                grad_out[:, :, start:end].contiguous(),
                qt,
                k,
                v,
                p,
                dk,
                dv,
            )
            dq[:, :, start:end] = dqt
            del qt, p, dqt
        # No per-chunk rounding or partial-sum merge: scale/cast exactly once.
        dk.mul_(_SCALE)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


class H3FullAttentionOp:
    """BF16 [B,S,56,128] -> [B,S,56,128], Q/K already normalized/rotated.

    Physical sequence order defines the fixed reduction order.
    """

    _function: type[Function] = _H3AttentionFn

    backend_id = "rlkernel.cuda.h3_full_attention.query_chunk.v1"

    def __init__(self, chunk_rows=128):
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        self.chunk_rows = chunk_rows

    @staticmethod
    def _validate(q, k, v):
        if any(x.ndim != 4 for x in (q, k, v)):
            raise ValueError("H3 Q/K/V must have shape [B,S,56,128]")
        if q.shape != k.shape or q.shape != v.shape or q.shape[2:] != (56, 128):
            raise ValueError("H3 requires equal Q/K/V shapes [B,S,56,128]")
        if not 0 < q.shape[0] <= 65535 // 56 or q.shape[1] <= 0:
            raise ValueError("H3 requires nonempty tensors and batch <= 1170")
        # Native H3 staging launches eight query/key rows per grid block.
        if q.shape[1] > 65535 * 8:
            raise ValueError("H3 sequence exceeds CUDA grid bounds")
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise ValueError("H3 strict checkpoint profile requires BF16 Q/K/V")
        if any(not x.is_contiguous() for x in (q, k, v)):
            raise ValueError("H3 requires contiguous BSHD Q/K/V; call contiguous() explicitly")
        if any(x.device != q.device for x in (k, v)):
            raise ValueError("H3 Q/K/V must share a device")
        if not q.is_cuda or torch.version.hip is not None:
            raise ValueError("H3 CUDA implementation requires NVIDIA CUDA")

    def forward_with_lse(self, q, k, v):
        self._validate(q, k, v)
        out, lse = self._function.apply(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.chunk_rows
        )
        return out.transpose(1, 2), lse.transpose(1, 2)

    def forward(self, q, k, v):
        return self.forward_with_lse(q, k, v)[0]

    __call__ = forward

    def provenance(self):
        return {
            "backend": self.backend_id,
            "build": dict(_extension().h3_attention_build_info()),
            "fallback": False,
            "layout": "BSHD",
            "heads": 56,
            "head_dim": 128,
            "dtype": "bf16",
            "causal": False,
            "qk_preprocessing": "caller-owned RMSNorm + partial 3-axis RoPE",
            "chunk_rows": self.chunk_rows,
            "accumulator": "fp32",
            "reduction": "#240: ascending D/K/Q, softmax 256-thread tree",
            "backward": "recompute P; continuous FP32 dK/dV accumulators; final cast",
            "supports_cp": False,
            "supports_double_backward": False,
        }
