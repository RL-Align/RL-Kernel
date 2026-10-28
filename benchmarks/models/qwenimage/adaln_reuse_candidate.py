# SPDX-License-Identifier: Apache-2.0
"""Audit-only composition of the repository's vendored vLLM persistent GEMM.

This adapter supplies autograd; the upstream GEMM itself is forward-only.
No operator registration or global dispatch override is performed. FP32
Triton products require TRITON_F32_DEFAULT=ieee in the audit subprocess.
"""
import importlib
import os
import sys
import types

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


class Final:
    def __call__(self, x, temb, weight, bias):
        scale, shift = Projection()(temb, weight, bias).chunk(2, -1)
        # Native modulation autograd: explicitly a candidate, not a certified
        # replacement for the existing fixed FP32 token-reduction backward.
        norm = F.layer_norm(x.float(), (3072,), eps=1e-6).to(x.dtype)
        return norm * (1 + scale)[:, None, :] + shift[:, None, :]


class FinalLiger:
    def __call__(self, x, temb, weight, bias):
        from liger_kernel.ops.layer_norm import LigerLayerNormFunction

        scale, shift = Projection()(temb, weight, bias).chunk(2, -1)
        gain = torch.ones(3072, device=x.device, dtype=torch.float32)
        offset = torch.zeros_like(gain)
        norm = LigerLayerNormFunction.apply(x.float(), gain, offset, 1e-6).to(x.dtype)
        return norm * (1 + scale)[:, None, :] + shift[:, None, :]


class FinalFlashAttn:
    """Final modulation with flash-attention's Triton LayerNorm (forward and backward).

    Only ``flash_attn.ops.triton.layer_norm`` is loaded from the checkout; the
    package ``__init__`` (which needs the compiled attention extension) is skipped.
    """

    def __init__(self, source):
        if "flash_attn" not in sys.modules:
            package = types.ModuleType("flash_attn")
            package.__path__ = [str(source / "flash_attn")]
            sys.modules["flash_attn"] = package
        self.layer_norm_fn = importlib.import_module(
            "flash_attn.ops.triton.layer_norm"
        ).layer_norm_fn

    def __call__(self, x, temb, weight, bias):
        scale, shift = Projection()(temb, weight, bias).chunk(2, -1)
        # Parameter-free LayerNorm: constant unit gain and zero offset, as for Liger.
        gain = torch.ones(3072, device=x.device, dtype=torch.float32)
        offset = torch.zeros_like(gain)
        norm = self.layer_norm_fn(x.float(), gain, offset, eps=1e-6).to(x.dtype)
        return norm * (1 + scale)[:, None, :] + shift[:, None, :]
