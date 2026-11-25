# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4.1 mHC (profile ``dsv41-mhc-v1``): the delayed mhc_pre and the final collapse.

- **Semantics**: the golden ``rl_engine.kernels.ops.pytorch.mhc.dsv41_mhc``. A sublayer
  mixes its streams with the pre carried in from the previous sublayer (stream 0 at the
  model entry) and returns its controller's pre; the model ends with a collapse by the
  last pre (DSv4.1 has no learned hc_head). ``mhc_post`` is DSv4's.
- **Kernels**: ``csrc/cuda/mhc/dsv4_mhc.cu`` (shared with ``dsv4-mhc-v1``) after DeepGEMM's
  ``tf32_hc_prenorm_gemm`` with 16 fixed splits; H up to 5120.
- **eps**: the model's ``norm_eps`` (DSv4.1: 1e-20) for the controller's rms and the fused
  norm; every function takes it.
- **Autograd**: dX only (frozen ``fn`` / ``scale`` / ``base`` / norm weight); the carried
  pre gets a gradient, which flows to the previous sublayer's controller.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.mhc import HC_EPS, POST_MULT, SINKHORN_ITERS, gemm_partials
from rl_engine.kernels.ops.pytorch.mhc.dsv41_mhc import NORM_EPS

from .dsv4_mhc import _bf16_out, _f32, _grad, _nw, _zeros_if_none

PROFILE = "dsv41-mhc-v1"
_SYMBOLS = (
    "dsv41_mhc_pre",
    "dsv41_mhc_collapse",
    "dsv41_mhc_pre_backward",
    "dsv41_mhc_collapse_backward",
)


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("DSv4.1 mHC requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _pre(p: Tensor | None) -> Tensor | None:
    return None if p is None else _f32(p)


def mhc_pre_delayed_cuda(
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
):
    """residual BF16 [T, 4, H] -> (out BF16 [T, H], post [T, 4], comb [T, 4, 4],
    pre_out [T, 4], raw [T, 24], ss [T]); out mixes with pre_in (stream 0 when None)."""
    ext = _extension()
    residual = residual.contiguous()
    pm, ps = gemm_partials(residual, fn)
    return tuple(
        ext.dsv41_mhc_pre(
            residual,
            pm,
            ps,
            _f32(scale),
            _f32(base),
            float(eps),
            HC_EPS,
            SINKHORN_ITERS,
            POST_MULT,
            _pre(pre_in),
            _nw(norm_weight),
            float(eps),
        )
    )


def hc_collapse_cuda(
    residual: Tensor, pre: Tensor, norm_weight: Tensor | None = None, eps: float = NORM_EPS
) -> Tensor:
    """BF16 [T, H]: sum_i pre_i * residual_i, normed when a weight is given."""
    return _extension().dsv41_mhc_collapse(
        residual.contiguous(), _f32(pre), _nw(norm_weight), float(eps)
    )


def mhc_pre_delayed_backward_cuda(
    dy: Tensor,
    d_post: Tensor,
    d_comb: Tensor,
    d_pre_out: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor | None]:
    """-> (d_residual [T, 4, H] in ``out_dtype``, d_pre_in [T, 4] FP32 or None)."""
    dres, dpre_in = _extension().dsv41_mhc_pre_backward(
        _grad(dy),
        _f32(d_post),
        _f32(d_comb),
        _f32(d_pre_out),
        residual.contiguous(),
        _f32(fn),
        _f32(scale),
        _f32(base),
        _f32(raw),
        _f32(ss),
        _pre(pre_in),
        float(eps),
        HC_EPS,
        SINKHORN_ITERS,
        POST_MULT,
        _nw(norm_weight),
        float(eps),
        _bf16_out(out_dtype),
    )
    return dres, (dpre_in if pre_in is not None else None)


def hc_collapse_backward_cuda(
    dy: Tensor,
    residual: Tensor,
    pre: Tensor,
    norm_weight: Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor]:
    """-> (d_residual [T, 4, H] in ``out_dtype``, d_pre [T, 4] FP32)."""
    dres, dpre = _extension().dsv41_mhc_collapse_backward(
        _grad(dy),
        residual.contiguous(),
        _f32(pre),
        _nw(norm_weight),
        float(eps),
        _bf16_out(out_dtype),
    )
    return dres, dpre


class MhcPreDelayedFunction(torch.autograd.Function):
    """(residual, fn, scale, base, pre_in, norm_weight, eps) -> (out, post, comb, pre_out);
    gradients for residual and pre_in."""

    @staticmethod
    def forward(ctx, residual, fn, scale, base, pre_in, norm_weight, eps):
        out, post, comb, pre_out, raw, ss = mhc_pre_delayed_cuda(
            residual, fn, scale, base, pre_in, norm_weight, eps
        )
        ctx.has_pre_in = pre_in is not None
        ctx.has_norm = norm_weight is not None
        ctx.eps = eps
        ctx.save_for_backward(residual, fn, scale, base, raw, ss, pre_in, norm_weight)
        return out, post, comb, pre_out

    @staticmethod
    def backward(ctx, dy, dpost, dcomb, dpre_out):
        residual, fn, scale, base, raw, ss, pre_in, norm_weight = ctx.saved_tensors
        t, h = residual.shape[0], residual.shape[2]
        dy = _zeros_if_none(dy, residual.new_empty(t, h))
        dpost = _zeros_if_none(dpost, raw.new_empty(t, 4))
        dcomb = _zeros_if_none(dcomb, raw.new_empty(t, 4, 4))
        dpre_out = _zeros_if_none(dpre_out, raw.new_empty(t, 4))
        dres, dpre_in = mhc_pre_delayed_backward_cuda(
            dy,
            dpost,
            dcomb,
            dpre_out,
            residual,
            fn,
            scale,
            base,
            raw,
            ss,
            pre_in if ctx.has_pre_in else None,
            norm_weight if ctx.has_norm else None,
            out_dtype=residual.dtype,
            eps=ctx.eps,
        )
        return dres, None, None, None, dpre_in, None, None


class HcCollapseFunction(torch.autograd.Function):
    """(residual, pre, norm_weight, eps) -> out; gradients for residual and pre."""

    @staticmethod
    def forward(ctx, residual, pre, norm_weight, eps):
        ctx.has_norm = norm_weight is not None
        ctx.eps = eps
        ctx.save_for_backward(residual, pre, norm_weight)
        return hc_collapse_cuda(residual, pre, norm_weight, eps)

    @staticmethod
    def backward(ctx, dy):
        residual, pre, norm_weight = ctx.saved_tensors
        dres, dpre = hc_collapse_backward_cuda(
            dy,
            residual,
            pre,
            norm_weight if ctx.has_norm else None,
            out_dtype=residual.dtype,
            eps=ctx.eps,
        )
        return dres, dpre, None, None


def mhc_pre_delayed(
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
):
    """Autograd delayed mhc_pre: (out BF16 [T, H], post [T, 4], comb [T, 4, 4],
    pre_out [T, 4])."""
    return MhcPreDelayedFunction.apply(residual, fn, scale, base, pre_in, norm_weight, eps)


def hc_collapse(
    residual: Tensor, pre: Tensor, norm_weight: Tensor | None = None, eps: float = NORM_EPS
) -> Tensor:
    """Autograd final collapse: BF16 [T, H]."""
    return HcCollapseFunction.apply(residual, pre, norm_weight, eps)
