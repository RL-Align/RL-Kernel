# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical-row-aware deterministic linear autograd for WS1 full-model use."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.canonical_backward import active_session
from rl_engine.kernels.ops.triton.matmul.det_gemm import _triton_gemm


def _gemm_fp32(a: torch.Tensor, b: torch.Tensor, family: str) -> torch.Tensor:
    if family == "cuda":
        return _C.det_gemm_rowwise_fwd_fp32(a.contiguous(), b.contiguous())
    if family == "triton":
        return _triton_gemm(a, b, output_dtype=torch.float32)
    if family == "ascend":
        from rl_engine.kernels.ops.ascend.matmul.det_gemm import _rowwise_fp32

        return _rowwise_fp32(a, b)
    raise ValueError(f"unsupported canonical linear family {family!r}")


class _CanonicalLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight, logical_keys, parameter_id, family):
        session = active_session()
        if session is None:
            raise RuntimeError("canonical linear requires an active backward session")
        ctx.save_for_backward(a, weight)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.family = str(family)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        return _gemm_fp32(a.float(), weight.float().t().contiguous(), ctx.family)

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
                return _gemm_fp32(grads.float().t().contiguous(), rows.float(), ctx.family).to(
                    weight.dtype
                )

            dweight = ctx.session.submit_linear(ctx.parameter_id, ctx.slot, a, grad_out, reducer)
        if ctx.family == "triton":
            record_backward(
                "det_gemm",
                kernel_id="rl_engine.kernels.ops.triton.matmul.det_gemm._triton_gemm",
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
        return da, dweight, None, None, None


def canonical_linear_fp32(
    a: torch.Tensor,
    weight: torch.Tensor,
    logical_keys: torch.Tensor,
    *,
    parameter_id: str,
    family: str,
) -> torch.Tensor:
    return _CanonicalLinearFn.apply(a, weight, logical_keys, parameter_id, family)


__all__ = ["canonical_linear_fp32"]
