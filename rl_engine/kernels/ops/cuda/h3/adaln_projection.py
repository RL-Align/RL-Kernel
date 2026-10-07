# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 three-modality AdaLN projection (RFC #420 ``adaln_projection_3mod``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.cuda.h3.det_linear import (
    ACT_NONE,
    CONTRACT,
    det_linear_available,
    det_linear_backward_input,
    det_linear_backward_weight,
    det_linear_forward,
    silu_backward_fp32,
)
from rl_engine.kernels.ops.pytorch.h3.adaln_projection import (
    split_adaln_table,
    validate_h3_adaln_projection,
)

KERNEL_ID = "torch.silu[fp32]+cast[weight_dtype]+rl_engine._C.h3_det_linear_forward"
BACKWARD_IMPL = "h3_det_linear_v1_chunked_dinput_ascending_row_dweight"
BACKWARD_KERNEL_ID = (
    "rl_engine._C.h3_det_linear_backward_weight"
    "+rl_engine._C.h3_det_linear_backward_input"
    "+rl_engine.kernels.ops.cuda.h3.det_linear.silu_backward_fp32"
)


class _H3AdaLNProjectionCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, temb, weight, bias):
        """Project FP32 ``(T, D)`` embeddings to a weight-dtype ``(T, 18H)`` table.

        Save the embedding, rounded SiLU activation and weight for the VJP.
        """

        temb = temb.contiguous()
        # Declared mixed-precision boundary: SiLU at temb's FP32 precision,
        # then exactly one rounding to the projection dtype. These are the
        # provider's own elementwise ops, so the activation is bitwise equal.
        act = F.silu(temb).to(weight.dtype)
        (table,) = det_linear_forward(act, weight, bias, activation=ACT_NONE)
        ctx.save_for_backward(temb, act, weight)
        return table

    @staticmethod
    def backward(ctx, grad_table):
        """Return requested embedding, weight and bias gradients in their input dtypes.

        Accumulate projection gradients deterministically and evaluate the
        straight-through cast and SiLU VJP in FP32 for the embedding gradient.
        """

        temb, act, weight = ctx.saved_tensors
        need_temb, need_weight, need_bias = ctx.needs_input_grad
        grad_table = grad_table.float().contiguous()
        d_temb = dw = db = None
        if need_weight or need_bias:
            dw, db = det_linear_backward_weight(grad_table, act, weight.dtype)
        if need_temb:
            # The cast's VJP is the identity; the SiLU VJP runs in FP32.
            d_act = det_linear_backward_input(grad_table, weight, torch.float32)
            d_temb = silu_backward_fp32(d_act, temb)
        record_backward(
            "adaln_projection_3mod",
            kernel_id=BACKWARD_KERNEL_ID,
            impl=BACKWARD_IMPL,
            family="cuda",
        )
        return d_temb, dw if need_weight else None, db if need_bias else None


class H3AdaLNProjectionCudaOp:
    """CUDA candidate: FP32 SiLU, one BF16 cast, deterministic GEMV, six views.

    The projection weight (96768 x 2688 BF16, 520 MB per block) is streamed
    once per call; every output element follows contract ``h3-det-linear-v1``,
    so a timestep's 3 x 6 modulation rows do not depend on the other
    timesteps in the call. The six outputs are views of one ``(T, 6H*3)``
    table, laid out exactly like diffusers' ``view(-1, 6H).chunk(6)``.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL
    contract = CONTRACT

    def __init__(self) -> None:
        """Require all deterministic linear symbols in the CUDA extension."""

        if not det_linear_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_det_linear_*; rebuild with csrc/cuda/h3/det_linear.cu"
            )

    def __call__(self, temb, weight, bias):
        """Return six weight-dtype ``(3T, H)`` modulation views for CUDA inputs."""

        return self.forward(temb, weight, bias)

    def forward(self, temb, weight, bias) -> tuple[torch.Tensor, ...]:
        """Return six ``(3T, H)`` views after FP32 SiLU and a deterministic projection.

        Require CUDA FP32 ``temb`` of shape ``(T, D)`` with ``T > 0`` and
        same-device FP32 or BF16 weight/bias of shapes ``(18H, D)``/``(18H,)``.
        Outputs share one table in the weight dtype; autograd covers all inputs.
        """

        hidden = validate_h3_adaln_projection(temb, weight, bias)
        if not temb.is_cuda:
            raise ValueError("H3AdaLNProjectionCudaOp needs CUDA tensors")
        table = _H3AdaLNProjectionCuda.apply(temb, weight, bias)
        return split_adaln_table(table, hidden)

    def forward_table(self, temb, weight, bias) -> torch.Tensor:
        """The raw ``(T, 6H*3)`` projection, for consumers that gather it directly."""

        validate_h3_adaln_projection(temb, weight, bias)
        if not temb.is_cuda:
            raise ValueError("H3AdaLNProjectionCudaOp needs CUDA tensors")
        return _H3AdaLNProjectionCuda.apply(temb, weight, bias)

    def forward_fp32(self, temb, weight, bias):
        """Run the CUDA projection with its declared weight-dtype output boundary."""

        return self.forward(temb, weight, bias)
