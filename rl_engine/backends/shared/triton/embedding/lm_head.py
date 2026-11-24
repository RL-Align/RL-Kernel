# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton deterministic LM head built on the pinned no-split-K GEMM."""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm
from rl_engine.ops.autograd.backward_runtime import record_backward


class _TritonLMHeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, output_fp32=False):
        ctx.save_for_backward(hidden, weight, bias if bias is not None else hidden.new_empty(0))
        ctx.has_bias = bias is not None
        flat = hidden.reshape(-1, hidden.size(-1)).contiguous()
        out = _triton_gemm(
            flat,
            weight.t().contiguous(),
            output_dtype=torch.float32 if output_fp32 else None,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*hidden.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, weight.size(0)).contiguous()
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight_c = weight.contiguous()
        if grad_2d.dtype != torch.bfloat16:
            grad_2d = grad_2d.to(torch.bfloat16)
        if hidden_2d.dtype != torch.bfloat16:
            hidden_2d = hidden_2d.to(torch.bfloat16)
        if weight_c.dtype != torch.bfloat16:
            weight_c = weight_c.to(torch.bfloat16)
        grad_hidden = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # Keep the reduction in FP32 before restoring the execution dtype.
            # BF16-input dot rounding can otherwise exceed the shared gradient
            # accuracy contract for the full-vocabulary projection.
            grad_hidden = (
                _triton_gemm(grad_2d, weight_c, output_dtype=torch.float32)
                .reshape_as(hidden)
                .to(hidden.dtype)
            )
        if ctx.needs_input_grad[1]:
            grad_weight = _triton_gemm(grad_2d.t().contiguous(), hidden_2d).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            rows = grad_output.reshape(-1, weight.size(0)).float()
            acc = torch.zeros((rows.shape[1],), device=rows.device, dtype=torch.float32)
            for index in range(rows.shape[0]):
                acc = acc + rows[index]
            grad_bias = acc.to(bias.dtype)
        record_backward(
            "lm_head",
            kernel_id="rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm",
            impl="triton_det_gemm",
            family="triton",
        )
        return grad_hidden, grad_weight, grad_bias, None


class TritonLMHeadOp:
    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "triton_det_gemm"

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(hidden, weight, bias=bias)

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("TritonLMHeadOp requires BF16 hidden and weight")
        return _TritonLMHeadFn.apply(hidden, weight, bias, False)

    def forward_fp32(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("TritonLMHeadOp requires BF16 hidden and weight")
        return _TritonLMHeadFn.apply(hidden, weight, bias, True)

    def forward_fp32_prepared(self, hidden: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        """Inference with a lifecycle-managed contiguous BF16 ``[K,V]`` weight.

        Execute the same FP32 GEMM as ``forward_fp32`` without materializing
        the vocabulary-weight transpose on each call. The caller must refresh
        the prepared weight after every load; training uses the original API.
        """
        if hidden.dtype != torch.bfloat16 or weight_t.dtype != torch.bfloat16:
            raise TypeError("TritonLMHeadOp requires BF16 hidden and weight")
        if hidden.ndim < 1 or weight_t.ndim != 2 or hidden.size(-1) != weight_t.size(0):
            raise ValueError("prepared LM head requires aligned hidden [...,K] and weight [K,V]")
        if hidden.device != weight_t.device or not weight_t.is_contiguous():
            raise ValueError("prepared LM-head weight must be contiguous on the hidden device")
        if torch.is_grad_enabled() and (hidden.requires_grad or weight_t.requires_grad):
            raise RuntimeError("prepared LM head is inference-only; use forward_fp32 for gradients")
        flat = hidden.reshape(-1, hidden.size(-1)).contiguous()
        result = _triton_gemm(flat, weight_t, output_dtype=torch.float32)
        return result.reshape(*hidden.shape[:-1], weight_t.size(1))

    def parameter_vjp_contributions_fp32(self, *, hidden, weight, grad_output, bias=None):
        del weight, bias
        rows_h = hidden.reshape(-1, hidden.size(-1)).float()
        rows_g = grad_output.reshape(-1, grad_output.size(-1)).float()
        return {"weight": rows_g[:, :, None] * rows_h[:, None, :]}
