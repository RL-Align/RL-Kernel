# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA [F1] wrapper: post-norm, residual add, pre-norm.

Same interface as ``TritonFusedAddNormOp``; the frozen semantics live in
``rl_engine.reference.norm.fused_add_norm``. The kernels are in
``csrc/cuda/norm/fused_add_norm.cu``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32
from rl_engine.reference.norm.fused_add_norm import POST_NORM_EPS, RMS_NORM_EPS

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_REQUIRED_SYMBOLS = ("fused_add_norm_forward", "fused_add_norm_backward")


def _require_cuda_fused_add_norm() -> None:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError(
            "CUDA fused add-norm kernels require the compiled rl_engine._C extension."
        )
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            "rl_engine._C is missing "
            + ", ".join(missing)
            + "; rebuild the extension with csrc/cuda/norm/fused_add_norm.cu."
        )


def cuda_fused_add_norm_available() -> bool:
    """Whether the CUDA kernel is built, present, and the one this row claims.

    False on ROCm even when the symbols exist. The source is in the shared
    ``cuda_sources`` list, so a ROCm build hipifies it and ``_C`` does carry
    ``fused_add_norm_forward``/``_backward`` -- but ``rsqrt_approx_ftz`` falls back
    to ``rsqrtf`` there, because hipcc cannot assemble the one line of NVIDIA PTX
    that makes the kernel bitwise equal to the Triton one. Reporting the kernel as
    available on ROCm would let an availability-gated caller, and this operator's
    own bitwise test, select a path that does not meet the claim the test asserts.
    The registry's rocm table never dispatches this backend anyway.
    """
    if getattr(torch.version, "hip", None) is not None:
        return False
    return bool(
        _EXT_AVAILABLE and _C is not None and all(hasattr(_C, n) for n in _REQUIRED_SYMBOLS)
    )


def _validate(hidden: Tensor, residual: Tensor, post_w: Tensor, pre_w: Tensor) -> None:
    for name, t in (("hidden", hidden), ("residual", residual)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {t.dtype}.")
        if not t.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor, got device '{t.device}'.")
    if hidden.shape != residual.shape:
        raise ValueError(
            f"hidden and residual must share shape, got {tuple(hidden.shape)} "
            f"and {tuple(residual.shape)}."
        )
    if hidden.dtype != residual.dtype:
        raise ValueError(
            f"hidden and residual must share dtype, got {hidden.dtype} and {residual.dtype}."
        )
    width = hidden.shape[-1]
    if width == 0:
        # See the Triton wrapper: .view(-1, 0) fails first and explains nothing.
        raise ValueError("hidden must have a non-zero last dimension, got width 0.")
    for name, w in (("post_weight", post_w), ("pre_weight", pre_w)):
        if w.dim() != 1 or w.shape[0] != width:
            raise ValueError(
                f"{name} must be 1-D of size x.shape[-1]={width}, "
                f"got tuple({name}.shape)={tuple(w.shape)}"
            )


def fused_add_norm_cuda_forward(
    hidden: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    *,
    post_eps: float,
    pre_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(residual_out, pre_normed, rstd_post, rstd_pre)``."""
    _require_cuda_fused_add_norm()
    _validate(hidden, residual, post_weight, pre_weight)
    width = hidden.shape[-1]
    res_out, pre_out, rstd_post, rstd_pre = _C.fused_add_norm_forward(
        hidden.contiguous().view(-1, width),
        residual.contiguous().view(-1, width),
        post_weight.contiguous(),
        pre_weight.contiguous(),
        float(post_eps),
        float(pre_eps),
    )
    return res_out.view_as(hidden), pre_out.view_as(hidden), rstd_post, rstd_pre


class _FusedAddNormCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, residual, post_weight, pre_weight, post_eps, pre_eps):
        res_out, pre_out, rstd_post, rstd_pre = fused_add_norm_cuda_forward(
            hidden, residual, post_weight, pre_weight, post_eps=post_eps, pre_eps=pre_eps
        )
        ctx.save_for_backward(hidden, res_out, post_weight, pre_weight, rstd_post, rstd_pre)
        return res_out, pre_out

    @staticmethod
    def backward(ctx, grad_residual_out, grad_pre_normed):
        # Both cotangents are always tensors, never None. This Function does not
        # call ctx.set_materialize_grads(False), so autograd materialises zeros for
        # an output the caller did not consume. Measured: pre_normed.sum().backward()
        # succeeds on both backends and fills every input gradient. The None guard
        # in logprob/linear_logp.py exists because that Function does opt out, at
        # its line 1362. If this one ever opts out for the allocation, the two
        # .contiguous() calls below become the crash site.
        hidden, res_out, post_w, pre_w, rstd_post, rstd_pre = ctx.saved_tensors
        width = hidden.shape[-1]
        d_hidden, d_residual, partial_post, partial_pre = _C.fused_add_norm_backward(
            grad_residual_out.contiguous().view(-1, width),
            grad_pre_normed.contiguous().view(-1, width),
            hidden.contiguous().view(-1, width),
            res_out.contiguous().view(-1, width),
            post_w.contiguous(),
            pre_w.contiguous(),
            rstd_post,
            rstd_pre,
        )
        record_backward(
            "fused_post_norm_residual_pre_norm",
            kernel_id=(
                "rl_engine._C.fused_add_norm_backward"
                "+rl_engine.ops.autograd.vjp_fp32.reduce_rows_fp32"
            ),
            impl="cuda_fused_add_norm_declared_fp32_rowfold_dw",
            family="cuda",
        )
        return (
            d_hidden.view_as(hidden).to(hidden.dtype),
            d_residual.view_as(hidden).to(hidden.dtype),
            reduce_rows_fp32(partial_post).to(post_w.dtype),
            reduce_rows_fp32(partial_pre).to(pre_w.dtype),
            None,
            None,
        )


class CudaFusedAddNormOp:
    """CUDA [F1] wrapper shaped like the other operator-harness ops."""

    backward_impl = "cuda_fused_add_norm_declared_fp32_rowfold_dw"

    def __call__(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return self.forward(
            hidden, residual, post_weight, pre_weight, post_eps=post_eps, pre_eps=pre_eps
        )

    def forward(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return _FusedAddNormCuda.apply(hidden, residual, post_weight, pre_weight, post_eps, pre_eps)
