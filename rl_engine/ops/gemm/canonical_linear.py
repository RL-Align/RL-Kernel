# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical-row-aware deterministic linear autograd for WS1 full-model use."""

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C
from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session


def _gemm_fp32(
    a: torch.Tensor, b: torch.Tensor, family: str, *, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if family == "triton_tree":
        from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
            projection_gemm_fp32,
        )

        return projection_gemm_fp32(a, b, bias=bias)
    if bias is not None:
        return _gemm_fp32(a, b, family) + bias
    if family == "cuda":
        return _C.det_gemm_rowwise_fwd_fp32(a.contiguous(), b.contiguous())
    if family == "triton":
        return _triton_gemm(a, b, output_dtype=torch.float32)
    if family == "ascend":
        from rl_engine.backends.ascend.gemm.det_gemm import _rowwise_fp32

        return _rowwise_fp32(a, b)
    if family == "pytorch":
        # LOCAL-DEVIATION: the gold topology resolves family='pytorch'
        # on the bf16-reference path only.
        return a @ b
    raise ValueError(f"unsupported canonical linear family {family!r}")


class _CanonicalLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a,
        weight,
        logical_keys,
        parameter_id,
        family,
        bias,
        bias_parameter_id,
    ):
        session = active_session()
        parameter_grad = ctx.needs_input_grad[1] or ctx.needs_input_grad[5]
        if parameter_grad and session is None:
            raise RuntimeError("canonical linear requires an active backward session")
        if parameter_grad and logical_keys is None:
            raise ValueError("canonical parameter gradients require logical keys")
        if bias is not None and str(parameter_id) == str(bias_parameter_id):
            raise ValueError("weight and bias parameter identities must differ")
        ctx.save_for_backward(a, weight)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.family = str(family)
        ctx.bias_parameter_id = str(bias_parameter_id)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        contribution = object() if ctx.needs_input_grad[1] and ctx.needs_input_grad[5] else None
        ctx.slot = (
            session.register(
                ctx.parameter_id,
                logical_keys,
                contribution=contribution,
                reuse_canonical_order=ctx.family == "triton_tree",
            )
            if ctx.needs_input_grad[1]
            else None
        )
        ctx.bias_slot = (
            session.register(
                ctx.bias_parameter_id,
                logical_keys,
                contribution=contribution,
                reuse_canonical_order=ctx.family == "triton_tree",
            )
            if ctx.needs_input_grad[5]
            else None
        )
        rhs = weight.float().t()
        rhs = rhs.contiguous()
        return _gemm_fp32(
            a.float(), rhs, ctx.family, bias=bias.float() if bias is not None else None
        )

    @staticmethod
    def backward(ctx, grad_out):
        a, weight = ctx.saved_tensors
        grad32 = grad_out.contiguous().float()
        da = None
        if ctx.needs_input_grad[0]:
            da = _gemm_fp32(grad32, weight.float(), ctx.family).to(a.dtype)

        dweight = None
        if ctx.needs_input_grad[1]:

            def reducer(rows, grads):
                lhs = grads.float().t()
                if ctx.family != "triton_tree":
                    lhs = lhs.contiguous()
                return _gemm_fp32(lhs, rows.float(), ctx.family).to(weight.dtype)

            dweight = ctx.session.submit_linear(ctx.parameter_id, ctx.slot, a, grad32, reducer)
        dbias = None
        if ctx.needs_input_grad[5]:

            def bias_reducer(grads):
                if grads.shape[0] == 0:
                    raise RuntimeError(f"no active logical rows for {ctx.bias_parameter_id}")
                ones = torch.ones((1, grads.shape[0]), device=grads.device, dtype=torch.float32)
                # Reuse the existing one-row Triton leaves for long bias VJPs.
                return _gemm_fp32(ones, grads.float(), ctx.family).squeeze(0).to(ctx.bias_dtype)

            dbias = ctx.session.submit_rows(
                ctx.bias_parameter_id, ctx.bias_slot, grad32, bias_reducer
            )
        if ctx.family in ("triton", "triton_tree"):
            record_backward(
                "det_gemm",
                kernel_id=(
                    "rl_engine.backends.shared.triton.gemm.det_gemm."
                    "_triton_chunked_gemm_fp32 / _triton_tree_gemm"
                    if ctx.family == "triton_tree"
                    else "rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm"
                ),
                impl="triton_det_gemm_canonical_rowfold",
                family="triton",
            )
        elif ctx.family == "ascend":
            record_backward(
                "det_gemm",
                kernel_id="rl_engine._C_npu.det_gemm_rowwise_ascend_fwd_fp32",
                impl="ascend_det_gemm_canonical_rowfold",
                family="ascend",
            )
        return da, dweight, None, None, None, dbias, None


def canonical_linear_fp32(
    a: torch.Tensor,
    weight: torch.Tensor,
    logical_keys: torch.Tensor | None,
    *,
    parameter_id: str,
    family: str,
    bias: torch.Tensor | None = None,
    bias_parameter_id: str | None = None,
) -> torch.Tensor:
    return _CanonicalLinearFn.apply(
        a,
        weight,
        logical_keys,
        parameter_id,
        family,
        bias,
        bias_parameter_id if bias_parameter_id is not None else f"{parameter_id}.bias",
    )


__all__ = ["canonical_linear_fp32"]
