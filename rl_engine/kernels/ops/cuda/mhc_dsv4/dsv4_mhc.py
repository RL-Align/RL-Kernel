# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 mHC (profile ``dsv4-mhc-v1``): mhc_pre / mhc_post / hc_head and dX backward.

- **Controller GEMM**: DeepGEMM ``tf32_hc_prenorm_gemm`` with 16 fixed splits, which is
  batch-invariant.
- **Everything after the GEMM**: ``csrc/cuda/mhc_dsv4/dsv4_mhc.cu``, byte-equal to the golden
  ``rl_engine.kernels.ops.pytorch.mhc_dsv4`` given the same GEMM partials.
- **Fused RMSNorm**: with ``norm_weight`` (BF16 [H]), ``mhc_pre`` / ``hc_head`` return
  ``RMSNorm(y) * w`` (the layer's attn_norm / ffn_norm / final norm) instead of ``y``.
- **Autograd**: the functions treat the mHC parameters (``fn``, ``scale``, ``base``) and the
  norm weight as frozen, so backward returns dX only.
- **Fail-closed**: raises when DeepGEMM or the compiled symbols are missing.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.mhc_dsv4 import (
    HC_EPS,
    NORM_EPS,
    POST_MULT,
    RMS_EPS,
    SINKHORN_ITERS,
    gemm_partials,
)

PROFILE = "dsv4-mhc-v1"
_SYMBOLS = (
    "dsv4_mhc_pre",
    "dsv4_mhc_head",
    "dsv4_mhc_post",
    "dsv4_mhc_post_backward",
    "dsv4_mhc_pre_backward",
    "dsv4_mhc_head_backward",
)


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("DSv4 mHC requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _f32(t: Tensor) -> Tensor:
    return t.float().contiguous()


def _nw(w: Tensor | None) -> Tensor | None:
    return None if w is None else w.contiguous()


def _grad(g: Tensor) -> Tensor:
    """Gradients enter as BF16 or FP32 (BF16 -> FP32 is exact inside the kernels)."""
    return (g if g.dtype in (torch.bfloat16, torch.float32) else g.float()).contiguous()


def _bf16_out(out_dtype: torch.dtype) -> bool:
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be BF16 or FP32, got {out_dtype}")
    return out_dtype == torch.bfloat16


def mhc_pre_cuda(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
):
    """residual BF16 [T, 4, H] -> (out BF16 [T, H], post [T, 4], comb [T, 4, 4], raw [T, 24],
    ss [T]); out is y, or RMSNorm(y) * norm_weight."""
    ext = _extension()
    residual = residual.contiguous()
    pm, ps = gemm_partials(residual, fn)
    return tuple(
        ext.dsv4_mhc_pre(
            residual,
            pm,
            ps,
            _f32(scale),
            _f32(base),
            RMS_EPS,
            HC_EPS,
            SINKHORN_ITERS,
            POST_MULT,
            _nw(norm_weight),
            NORM_EPS,
        )
    )


def hc_head_cuda(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
):
    """residual BF16 [T, 4, H] -> (out BF16 [T, H], raw [T, 4], ss [T])."""
    ext = _extension()
    residual = residual.contiguous()
    pm, ps = gemm_partials(residual, fn)
    return tuple(
        ext.dsv4_mhc_head(
            residual,
            pm,
            ps,
            _f32(scale).reshape(1),
            _f32(base),
            RMS_EPS,
            HC_EPS,
            _nw(norm_weight),
            NORM_EPS,
        )
    )


def mhc_post_cuda(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """BF16 [T, 4, H]: post_j * x + sum_i comb[i, j] * residual_i."""
    return _extension().dsv4_mhc_post(x.contiguous(), residual.contiguous(), _f32(post), _f32(comb))


def mhc_post_backward_cuda(
    d_new: Tensor,
    x: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
    out_dtype: torch.dtype = torch.float32,
):
    """-> (dx [T, H], d_residual [T, 4, H], d_post [T, 4], d_comb [T, 4, 4]); dx and
    d_residual in ``out_dtype`` (BF16 = the FP32 values rounded once), the rest FP32."""
    return tuple(
        _extension().dsv4_mhc_post_backward(
            _grad(d_new),
            x.contiguous(),
            residual.contiguous(),
            _f32(post),
            _f32(comb),
            _bf16_out(out_dtype),
        )
    )


def mhc_pre_backward_cuda(
    dy: Tensor,
    d_post: Tensor,
    d_comb: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    norm_weight: Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """d_residual [T, 4, H] (``out_dtype``) of mhc_pre (frozen parameters); ``dy`` is the
    gradient of the op's output (normed when ``norm_weight`` is given)."""
    return _extension().dsv4_mhc_pre_backward(
        _grad(dy),
        _f32(d_post),
        _f32(d_comb),
        residual.contiguous(),
        _f32(fn),
        _f32(scale),
        _f32(base),
        _f32(raw),
        _f32(ss),
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
        POST_MULT,
        _nw(norm_weight),
        NORM_EPS,
        _bf16_out(out_dtype),
    )


def hc_head_backward_cuda(
    dy: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    norm_weight: Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """d_residual [T, 4, H] (``out_dtype``) of hc_head (frozen parameters)."""
    return _extension().dsv4_mhc_head_backward(
        _grad(dy),
        residual.contiguous(),
        _f32(fn),
        _f32(scale).reshape(1),
        _f32(base),
        _f32(raw),
        _f32(ss),
        RMS_EPS,
        HC_EPS,
        _nw(norm_weight),
        NORM_EPS,
        _bf16_out(out_dtype),
    )


def _zeros_if_none(g: Tensor | None, like: Tensor) -> Tensor:
    return torch.zeros_like(like, dtype=torch.float32) if g is None else g


class MhcPreFunction(torch.autograd.Function):
    """(residual, fn, scale, base, norm_weight) -> (out, post, comb); dX only."""

    @staticmethod
    def forward(ctx, residual, fn, scale, base, norm_weight):
        out, post, comb, raw, ss = mhc_pre_cuda(residual, fn, scale, base, norm_weight)
        ctx.has_norm = norm_weight is not None
        ctx.save_for_backward(residual, fn, scale, base, raw, ss, norm_weight)
        return out, post, comb

    @staticmethod
    def backward(ctx, dy, dpost, dcomb):
        residual, fn, scale, base, raw, ss, norm_weight = ctx.saved_tensors
        t, h = residual.shape[0], residual.shape[2]
        dy = _zeros_if_none(dy, residual.new_empty(t, h))
        dpost = _zeros_if_none(dpost, raw.new_empty(t, 4))
        dcomb = _zeros_if_none(dcomb, raw.new_empty(t, 4, 4))
        dres = mhc_pre_backward_cuda(
            dy,
            dpost,
            dcomb,
            residual,
            fn,
            scale,
            base,
            raw,
            ss,
            norm_weight if ctx.has_norm else None,
            out_dtype=residual.dtype,
        )
        return dres, None, None, None, None


class MhcPostFunction(torch.autograd.Function):
    """(x, residual, post, comb) -> new residual; gradients for all four inputs."""

    @staticmethod
    def forward(ctx, x, residual, post, comb):
        ctx.save_for_backward(x, residual, post, comb)
        return mhc_post_cuda(x, residual, post, comb)

    @staticmethod
    def backward(ctx, d_new):
        x, residual, post, comb = ctx.saved_tensors
        dx, dres, dpost, dcomb = mhc_post_backward_cuda(
            d_new, x, residual, post, comb, out_dtype=residual.dtype
        )
        return dx, dres, dpost, dcomb


class HcHeadFunction(torch.autograd.Function):
    """(residual, fn, scale, base, norm_weight) -> out; dX only."""

    @staticmethod
    def forward(ctx, residual, fn, scale, base, norm_weight):
        out, raw, ss = hc_head_cuda(residual, fn, scale, base, norm_weight)
        ctx.has_norm = norm_weight is not None
        ctx.save_for_backward(residual, fn, scale, base, raw, ss, norm_weight)
        return out

    @staticmethod
    def backward(ctx, dy):
        residual, fn, scale, base, raw, ss, norm_weight = ctx.saved_tensors
        dres = hc_head_backward_cuda(
            dy,
            residual,
            fn,
            scale,
            base,
            raw,
            ss,
            norm_weight if ctx.has_norm else None,
            out_dtype=residual.dtype,
        )
        return dres, None, None, None, None


def mhc_pre(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
):
    """Autograd mhc_pre: (out BF16 [T, H], post [T, 4], comb [T, 4, 4])."""
    return MhcPreFunction.apply(residual, fn, scale, base, norm_weight)


def mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Autograd mhc_post: BF16 [T, 4, H]."""
    return MhcPostFunction.apply(x, residual, post, comb)


def hc_head(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
) -> Tensor:
    """Autograd hc_head: BF16 [T, H]."""
    return HcHeadFunction.apply(residual, fn, scale, base, norm_weight)
