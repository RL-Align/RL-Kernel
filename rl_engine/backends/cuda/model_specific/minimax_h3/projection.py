# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 linear projection rows (RFC #420): audio in/out, text context, Q/K/V.

Every projection runs on one fixed-order Triton GEMM (contract ``h3-projection-gemm-v1``,
``rl_engine/backends/shared/triton/gemm/h3_projection.py``): one program per output tile,
the reduction axis walked in ascending ``BLOCK_K`` steps into an FP32 accumulator, no
split-K / stream-K / atomics / autotuning, one rounding at the store. So

* a token's output bytes do not depend on how many tokens share the call or where it sits;
* the fused ``[Q; K; V]`` projection equals the three separate ones column for column,
  forward and backward;
* a tensor-parallel output shard (whole rows of the weight) equals the same columns of
  the full product.

FP32 rows use ``input_precision="ieee"`` (no TF32); BF16 rows use the tensor-core dot.
``d_input`` reduces the output axis ascending, ``d_weight`` / ``d_bias`` reduce tokens in
logical row order. Nothing falls back to cuBLAS: a missing Triton or an unsupported
dtype/shape raises. (An earlier warp-per-column GEMV kernel, ``h3-det-linear-v1``, is
exact too but 50-250x slower than cuBLAS at thousands of tokens, so it is not used here.)
"""

from __future__ import annotations

import torch

from rl_engine.backends.shared.triton.gemm import h3_projection as gemm
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.reference.minimax_h3 import projection as ref

CONTRACT = gemm.CONTRACT
FORWARD_KERNEL_ID = "rl_engine.backends.shared.triton.gemm.h3_projection._projection_kernel"
BACKWARD_IMPL = "h3_projection_gemm_v1_ascending_n_dinput_ascending_t_dweight"
BACKWARD_KERNEL_ID = FORWARD_KERNEL_ID


def _require_triton() -> None:
    if not gemm.available():
        raise RuntimeError(
            "triton + CUDA are required for the H3 projection rows (no cuBLAS fallback is allowed)"
        )


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, row, x, weight, bias):
        x = x.contiguous()
        ctx.row = row
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return gemm.linear_forward(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[1:]
        dx = dw = db = None
        if need_x:
            dx = gemm.linear_backward_input(grad_out, weight)
        if need_w or need_b:
            dw, db = gemm.linear_backward_weight(grad_out, x, with_bias=ctx.has_bias and need_b)
            dw = dw if need_w else None
        record_backward(ctx.row, kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda")
        return None, dx, dw, db


class _QKV(torch.autograd.Function):
    """Q, K and V of one block; backward is the fused ``[Q; K; V]`` backward.

    ``d_input`` reduces over the concatenated ``3 * N`` output rows in ascending order, which
    is what the fused projection does. ``N`` is a multiple of ``BLOCK_K`` so concatenating the
    three operands does not move any block boundary: separate and fused layouts have the same
    backward bytes.
    """

    @staticmethod
    def forward(ctx, x, wq, wk, wv):
        x = x.contiguous()
        ctx.save_for_backward(x, wq, wk, wv)
        return tuple(gemm.linear_forward(x, w, None) for w in (wq, wk, wv))

    @staticmethod
    def backward(ctx, gq, gk, gv):
        x, wq, wk, wv = ctx.saved_tensors
        need_x = ctx.needs_input_grad[0]
        weights = (wq, wk, wv)
        grads = [
            torch.zeros(x.shape[0], w.shape[0], device=x.device, dtype=x.dtype) if g is None else g
            for g, w in zip((gq, gk, gv), weights)
        ]
        dx = None
        if need_x:
            dx = gemm.linear_backward_input(
                torch.cat(grads, dim=1).contiguous(), torch.cat(weights, dim=0)
            )
        dws = [
            gemm.linear_backward_weight(g, x, with_bias=False)[0] if need else None
            for g, need in zip(grads, ctx.needs_input_grad[1:])
        ]
        record_backward(
            "h3_qkv_gemm", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return (dx, *dws)


class _CudaProjection:
    """Shared CUDA operator for one projection row."""

    op_class = "reduction"
    kernel_id = FORWARD_KERNEL_ID
    backward_impl = BACKWARD_IMPL
    contract = CONTRACT
    spec: ref.ProjectionSpec
    reference_cls: type

    def __init__(self) -> None:
        _require_triton()

    def __call__(self, x, weight, bias=None):
        return self.forward(x, weight, bias)

    def forward(self, x, weight, bias=None) -> torch.Tensor:
        ref.validate_h3_projection(self.spec, x, weight, bias)
        if not x.is_cuda:
            raise ValueError(f"{self.spec.row}: needs CUDA tensors")
        return _Projection.apply(self.spec.row, x, weight, bias)

    def forward_fp32(self, x, weight, bias=None) -> torch.Tensor:
        """The CUDA projection already accumulates in FP32 and rounds once."""

        return self.forward(x, weight, bias)


class H3AudioInputProjectionCudaOp(_CudaProjection):
    spec = ref.AUDIO_INPUT


class H3AudioOutputProjectionCudaOp(_CudaProjection):
    spec = ref.AUDIO_OUTPUT


class H3TextContextProjectionCudaOp(_CudaProjection):
    spec = ref.TEXT_CONTEXT


class H3QKVGemmCudaOp:
    """Q/K/V projections of a block (BF16, no bias); separate and fused layouts agree bitwise."""

    spec = ref.QKV
    op_class = "reduction"
    kernel_id = FORWARD_KERNEL_ID
    backward_impl = BACKWARD_IMPL
    contract = CONTRACT

    def __init__(self) -> None:
        _require_triton()

    def __call__(self, x, weight_q, weight_k, weight_v):
        return self.forward(x, weight_q, weight_k, weight_v)

    def forward(self, x, weight_q, weight_k, weight_v):
        widths = {
            ref.validate_h3_projection(self.spec, x, w, None)
            for w in (weight_q, weight_k, weight_v)
        }
        if len(widths) != 1:
            raise ValueError(f"h3_qkv_gemm: Q/K/V widths differ: {sorted(widths)}")
        if not x.is_cuda:
            raise ValueError("h3_qkv_gemm: needs CUDA tensors")
        return _QKV.apply(x, weight_q, weight_k, weight_v)

    def project(self, x, weight) -> torch.Tensor:
        """One of the three projections (what a graph node that owns only Q, K or V runs)."""

        ref.validate_h3_projection(self.spec, x, weight, None)
        if not x.is_cuda:
            raise ValueError("h3_qkv_gemm: needs CUDA tensors")
        return _Projection.apply(self.spec.row, x, weight, None)

    def forward_fused(self, x, weight_qkv) -> torch.Tensor:
        """``(T, 3 * N)`` product of the concatenated ``[Q; K; V]`` weight."""

        n = (
            weight_qkv.shape[0] // 3
            if weight_qkv.dim() == 2 and weight_qkv.shape[0] % 3 == 0
            else 0
        )
        if n == 0:
            raise ValueError("h3_qkv_gemm: fused weight must be (3 * N, K)")
        ref.validate_h3_projection(self.spec, x, weight_qkv[:n], None)
        if not x.is_cuda:
            raise ValueError("h3_qkv_gemm: needs CUDA tensors")
        return _Projection.apply(self.spec.row, x, weight_qkv, None)

    def forward_fp32(self, x, weight_q, weight_k, weight_v):
        return self.forward(x, weight_q, weight_k, weight_v)
