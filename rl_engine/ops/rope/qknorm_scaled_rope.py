# SPDX-License-Identifier: Apache-2.0
"""Explicit F2 kernel dispatch and first-order autograd; no automatic runtime binding."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .rope_table import table_identity


class QKNormRoPEForward(NamedTuple):
    query: Tensor
    key: Tensor
    query_rstd: Tensor
    key_rstd: Tensor


def _validate_inputs(q, k, positions, cos, sin, is_local, *, check_position_bounds=True):
    if not isinstance(is_local, bool):
        raise ValueError("is_local must be bool")
    if q.ndim != 3 or k.ndim != 3 or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("Q/K must have shape [T, H, 128]")
    if q.shape[0] != k.shape[0] or min(q.shape[1], k.shape[1]) < 1:
        raise ValueError("Q/K require equal token counts and positive head counts")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q/K must be BF16")
    if q.device != k.device or not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q/K must be contiguous on the same device")
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        raise RuntimeError("F2 raw forward has no backward; use qknorm_scaled_rope for autograd")
    if q.device.type != "cuda":
        raise ValueError("F2 kernels require a GPU; no CPU fallback")
    if torch.version.hip is None and torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError("F2 requires SM80 or newer on NVIDIA")
    if q.shape[0] * (q.shape[1] + k.shape[1]) > 2**31 - 1:
        raise ValueError("F2 row count exceeds the supported launch geometry")
    if not is_local:
        return
    if positions is None or positions.dtype != torch.int64 or positions.shape != (q.shape[0],):
        raise ValueError("local positions must be int64 [T]")
    if positions.device != q.device or not positions.is_contiguous():
        raise ValueError("positions must be contiguous on the Q/K device")
    if cos is None or sin is None:
        raise ValueError("local layers require cosine and sine tables")
    if cos.ndim != 2 or cos.shape[1] != 64 or sin.shape != cos.shape:
        raise ValueError("tables must have matching shape [P, 64]")
    if not 0 <= cos.shape[0] <= 131072:
        raise ValueError("table position coverage exceeds the pinned context")
    for table in (cos, sin):
        if table.dtype != torch.bfloat16 or table.device != q.device or not table.is_contiguous():
            raise ValueError("tables must be contiguous BF16 on the Q/K device")
    table_identity(cos, sin)
    # CUDA checks bounds in its native entry point; Triton needs this host check.
    # No validation is cached. CUDA graph capture is not supported yet.
    if check_position_bounds and positions.numel():
        from rl_engine.backends.shared.triton.rope.qknorm_scaled_rope import (
            _positions_outside_table,
        )

        if _positions_outside_table(positions, cos.shape[0]):
            raise ValueError("positions outside the shared table")


def qknorm_scaled_rope_forward(
    q: Tensor,
    k: Tensor,
    positions: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    *,
    is_local: bool,
    backend: str,
) -> QKNormRoPEForward:
    """Return BF16 Q/K and FP32 [T, H] rstd; use qknorm_scaled_rope for autograd.

    Requires explicit backend selection, contiguous token-major inputs, and a shared
    externally built BF16 RoPE table for local layers. Global layers ignore table
    arguments. No fallback or implicit detach of a requested training computation.
    Local tables must be a verified pair from build_shared_rope_table or
    load_shared_rope_table; table identity and mutation are checked before dispatch.
    """
    if backend not in ("cuda", "triton"):
        raise ValueError("backend must be 'cuda' or 'triton'")
    _validate_inputs(q, k, positions, cos, sin, is_local, check_position_bounds=backend != "cuda")
    if not is_local:
        positions = cos = sin = None
    if backend == "cuda":
        if torch.version.hip is not None:
            raise RuntimeError("F2 CUDA is NVIDIA-only; select Triton for ROCm")
        from rl_engine.backends.extension import _C

        if _C is None or not hasattr(_C, "qknorm_scaled_rope_forward"):
            raise RuntimeError("F2 CUDA extension is unavailable; rebuild rl_engine._C")
        try:
            outputs = _C.qknorm_scaled_rope_forward(q, k, positions, cos, sin, is_local)
        except RuntimeError as exc:
            # Keep the Python API's bounds error while native callers retain
            # their own checked entry point. Never hide unrelated CUDA errors.
            if str(exc).partition("\n")[0] == "F2 positions outside shared table":
                raise ValueError("positions outside the shared table") from exc
            raise
    else:
        from rl_engine.backends.shared.triton.rope.qknorm_scaled_rope import _forward

        outputs = _forward(q, k, positions, cos, sin, is_local=is_local)
    return QKNormRoPEForward(*outputs)


def _dispatch_backward(q, k, gq, gk, rq, rk, positions, cos, sin, *, is_local, backend):
    if backend == "cuda":
        from rl_engine.backends.extension import _C

        if _C is None or not hasattr(_C, "qknorm_scaled_rope_backward"):
            raise RuntimeError("F2 CUDA backward extension is unavailable; rebuild rl_engine._C")
        return _C.qknorm_scaled_rope_backward(q, k, gq, gk, rq, rk, positions, cos, sin, is_local)
    from rl_engine.backends.shared.triton.rope.qknorm_scaled_rope import _backward

    return _backward(q, k, gq, gk, rq, rk, positions, cos, sin, is_local=is_local)


class _QKNormScaledRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, positions, cos, sin, is_local, backend):
        if not is_local:
            positions = cos = sin = None
        result = qknorm_scaled_rope_forward(
            q, k, positions, cos, sin, is_local=is_local, backend=backend
        )
        ctx.save_for_backward(q, k, result.query_rstd, result.key_rstd, positions, cos, sin)
        ctx.is_local = is_local
        ctx.backend = backend
        ctx.set_materialize_grads(False)
        return result.query, result.key

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        q, k, rq, rk, positions, cos, sin = ctx.saved_tensors
        # Missing outputs are independent of the corresponding input branch.
        missing_q, missing_k = grad_q is None, grad_k is None
        gq = torch.zeros_like(q) if missing_q else grad_q.contiguous()
        gk = torch.zeros_like(k) if missing_k else grad_k.contiguous()
        dq, dk = _dispatch_backward(
            q, k, gq, gk, rq, rk, positions, cos, sin, is_local=ctx.is_local, backend=ctx.backend
        )
        return (
            None if missing_q or not ctx.needs_input_grad[0] else dq,
            None if missing_k or not ctx.needs_input_grad[1] else dk,
            None,
            None,
            None,
            None,
            None,
        )


def qknorm_scaled_rope(
    q: Tensor,
    k: Tensor,
    positions: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    *,
    is_local: bool,
    backend: str,
) -> tuple[Tensor, Tensor]:
    """Differentiable F2 Q/K outputs with the declared first-order VJP.

    Position/table tensors are constants (no gradients), and higher derivatives
    are unsupported. Autograd saves Q/K, FP32 rstd and local table/position tensors;
    version counters reject mutation between forward and backward.
    """
    return _QKNormScaledRoPE.apply(q, k, positions, cos, sin, is_local, backend)
