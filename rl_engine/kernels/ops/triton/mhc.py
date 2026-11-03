# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from rl_engine.kernels.ops.pytorch.mhc import NativeMHCPreHAggregateOp


@triton.jit
def _h_aggregate_forward(Residual, Pre, Output, D: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    hidden = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = token * 4 * D + hidden
    r0 = tl.load(Residual + offset, hidden < D, 0).to(tl.float32)
    r1 = tl.load(Residual + offset + D, hidden < D, 0).to(tl.float32)
    r2 = tl.load(Residual + offset + 2 * D, hidden < D, 0).to(tl.float32)
    r3 = tl.load(Residual + offset + 3 * D, hidden < D, 0).to(tl.float32)
    p0 = libdevice.mul_rn(tl.load(Pre + token * 4), r0)
    p1 = libdevice.mul_rn(tl.load(Pre + token * 4 + 1), r1)
    p2 = libdevice.mul_rn(tl.load(Pre + token * 4 + 2), r2)
    p3 = libdevice.mul_rn(tl.load(Pre + token * 4 + 3), r3)
    result = libdevice.add_rn(libdevice.add_rn(p0, p1), libdevice.add_rn(p2, p3))
    tl.store(Output + token * D + hidden, result.to(tl.bfloat16), hidden < D)


@triton.jit
def _h_aggregate_grad_residual(GradOutput, Pre, GradResidual, D: tl.constexpr,
                             BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    hidden = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    grad = tl.load(GradOutput + token * D + hidden, hidden < D, 0).to(tl.float32)
    for stream in tl.static_range(4):
        pre = tl.load(Pre + token * 4 + stream)
        result = libdevice.mul_rn(grad, pre)
        tl.store(GradResidual + (token * 4 + stream) * D + hidden, result, hidden < D)


@triton.jit
def _h_aggregate_grad_pre(GradOutput, Residual, GradPre, D: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.arange(0, 4)
    acc = tl.full((4,), 0, tl.float32)
    for hidden in tl.range(0, D, loop_unroll_factor=1):
        grad = tl.load(GradOutput + token * D + hidden).to(tl.float32)
        residual = tl.load(Residual + (token * 4 + stream) * D + hidden).to(tl.float32)
        acc = libdevice.add_rn(acc, libdevice.mul_rn(grad, residual))
    tl.store(GradPre + token * 4 + stream, acc)


class MHCPreHAggregateTritonOp:
    """Four-stream aggregation with an explicit FP32 composite backward."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "triton_fixed_sum_mhc_pre_h_aggregate_backward"

    def __init__(self):
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("MHC H Aggregate Triton requires NVIDIA CUDA")

    @staticmethod
    def _validate_inputs(residual, pre):
        NativeMHCPreHAggregateOp._validate_inputs(residual, pre)
        if residual.device.type != "cuda":
            raise ValueError("residual and pre must be CUDA tensors")
        if torch.cuda.get_device_capability(residual.device)[0] < 8:
            raise RuntimeError("MHC H Aggregate requires SM80 or newer")

    def __call__(self, residual, pre):
        return self.forward(residual, pre)

    def forward(self, residual, pre):
        self._validate_inputs(residual, pre)
        if torch.is_grad_enabled() and (residual.requires_grad or pre.requires_grad):
            raise RuntimeError("MHC H Aggregate does not expose standalone autograd; "
                               "use the explicit FP32 composite backward")
        tokens, _, hidden = residual.shape
        output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=residual.device)
        if tokens:
            with torch.cuda.device(residual.device):
                _h_aggregate_forward[(tokens, triton.cdiv(hidden, 256))](
                    residual, pre, output, hidden, 256, enable_fp_fusion=False)
        return output

    def backward_fp32(self, grad_output, residual, pre):
        self._validate_inputs(residual, pre)
        tokens, _, hidden = residual.shape
        if grad_output.shape != (tokens, hidden):
            raise ValueError("grad_output must have shape [num_tokens, hidden_size]")
        if grad_output.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("grad_output must be bfloat16 or float32")
        if grad_output.device != residual.device:
            raise RuntimeError("grad_output, residual, and pre must be on the same device")
        if not grad_output.is_contiguous():
            raise ValueError("grad_output must be contiguous")
        grad_residual = torch.empty_like(residual, dtype=torch.float32)
        grad_pre = torch.empty_like(pre)
        if tokens:
            with torch.cuda.device(residual.device):
                _h_aggregate_grad_residual[(tokens, triton.cdiv(hidden, 256))](
                    grad_output, pre, grad_residual, hidden, 256, enable_fp_fusion=False)
                _h_aggregate_grad_pre[(tokens,)](
                    grad_output, residual, grad_pre, hidden, num_warps=1, enable_fp_fusion=False)
        return grad_residual, grad_pre
