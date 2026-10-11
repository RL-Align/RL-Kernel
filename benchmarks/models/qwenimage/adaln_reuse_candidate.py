# SPDX-License-Identifier: Apache-2.0
"""Audit-only composition of the repository's vendored vLLM persistent GEMM.

This adapter supplies autograd; the upstream GEMM itself is forward-only.
No operator registration or global dispatch override is performed. FP32
Triton products require TRITON_F32_DEFAULT=ieee in the audit subprocess.
"""
import os

import torch
import torch.nn.functional as F

from benchmarks.layers.vllm_batch_invariant_matmul import matmul_persistent


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, temb, weight, bias):
        act = F.silu(temb).contiguous()
        ctx.save_for_backward(temb, act, weight)
        return matmul_persistent(act, weight.T, bias)

    @staticmethod
    def backward(ctx, grad):
        temb, act, weight = ctx.saved_tensors
        # Match the fixed CUDA operator's declared FP32 VJP intermediates.
        grad = grad.float().contiguous()
        da = matmul_persistent(grad, weight.float())
        dw = matmul_persistent(grad.T, act.float()).to(weight.dtype)
        db = grad.sum(0).to(weight.dtype)
        z = temb.float()
        sigmoid = torch.sigmoid(z)
        dt = da * (sigmoid * (1 + z * (1 - sigmoid)))
        return dt.to(temb.dtype), dw, db


class Projection:
    def __call__(self, temb, weight, bias):
        assert os.environ.get("TRITON_F32_DEFAULT") == "ieee"
        return _Linear.apply(temb, weight, bias)
