# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent CPU FP32 Linear reference for the H3 video output head."""

import torch

from rl_engine.ops.gemm.video_output_projection_fp32 import validate_projection_inputs


def _reference_matmul_fp32(a, b):
    """Evaluate each CPU dot with independent FP32 products and torch.sum."""
    a, bt = a.cpu().contiguous(), b.t().cpu().contiguous()
    output = torch.empty((a.shape[0], bt.shape[0]), dtype=torch.float32)
    # Tile only the output. Every torch.sum consumes the full logical K.
    # This bounds temporary memory without copying a candidate's K partition.
    for row in range(0, a.shape[0], 16):
        for column in range(0, bt.shape[0], 16):
            products = a[row : row + 16, None, :] * bt[None, column : column + 16, :]
            output[row : row + 16, column : column + 16] = products.sum(dim=-1)
    return output


class _ReferenceLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        output = _reference_matmul_fp32(x.reshape(-1, weight.shape[1]), weight.t())
        output = output.reshape(*x.shape[:-1], weight.shape[0])
        return output if bias is None else output + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, weight.shape[0]).contiguous().float()
        rows = x.reshape(-1, weight.shape[1])
        dx = _reference_matmul_fp32(grad, weight).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw = _reference_matmul_fp32(grad.t(), rows) if ctx.needs_input_grad[1] else None
        dbias = grad.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, dbias


class NativeVideoOutputProjectionFP32Op:
    reference_id = "cpu-fp32-output-products-sum-v1"

    def forward_fp32(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        validate_projection_inputs(x, weight, bias)
        # CPU avoids a GPU TF32 or batch-dependent cuBLAS reference path.
        # Long CPU BLAS dots can exceed the shared FP32 accuracy threshold.
        # Use independent FP32 products and full-K sums for forward and VJPs.
        with torch.autocast(device_type="cpu", enabled=False):
            return _ReferenceLinearFn.apply(
                x.cpu(), weight.cpu(), None if bias is None else bias.cpu()
            ).to(x.device)

    __call__ = forward_fp32
