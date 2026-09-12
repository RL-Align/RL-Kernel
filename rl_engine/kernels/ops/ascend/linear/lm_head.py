# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


class _AscendLMHeadFunction(torch.autograd.Function):
    """Autograd bridge for the Ascend batch-invariant LM-head forward.

    The VJP is the standard linear formula computed in fp32 on the NPU
    (``grad_hidden = grad @ W``, ``grad_weight = grad^T @ H``, bias = the
    fixed-order row sum), then cast back to the input dtypes. The CUDA op
    uses its declared deterministic GEMM for the backward; on NPU the
    plain torch matmuls are the deterministic-in-practice equivalent, and
    the gtest compares gradients at the reduction contract tolerance.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, output_fp32: bool):
        bias_to_save = (
            bias if bias is not None else torch.empty(0, device=hidden.device, dtype=hidden.dtype)
        )
        ctx.save_for_backward(hidden, weight, bias_to_save)
        ctx.has_bias = bias is not None
        ctx.output_fp32 = bool(output_fp32)
        return _C_npu.lm_head_ascend(hidden, weight.contiguous(), bias, bool(output_fp32))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, bias = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, weight.size(0)).contiguous().float()
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous().float()
        weight_f = weight.contiguous().float()
        grad_hidden = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_hidden = grad_2d @ weight_f
            grad_hidden = grad_hidden.reshape_as(hidden).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_2d.t() @ hidden_2d).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            rows = grad_output.reshape(-1, weight.size(0)).float()
            acc = torch.zeros((rows.shape[1],), device=rows.device, dtype=torch.float32)
            for index in range(rows.shape[0]):
                acc = acc + rows[index]
            grad_bias = acc.to(bias.dtype)
        record_backward(
            "lm_head",
            kernel_id="rl_engine.kernels.ops.ascend.linear.lm_head._AscendLMHeadFunction",
            impl="ascend_fp32_matmul_vjp",
            family="ascend",
        )
        return grad_hidden, grad_weight, grad_bias, None


class AscendLMHeadOp:
    """Single-card batch-invariant Ascend LM-head op.

    The Ascend C forward mirrors the SM90 CUDA kernel's structure: one
    output element per block iteration, full hidden-dimension fp32 reduction
    inside that block over a fixed tile order, bias added in fp32, final
    cast to the output dtype. There is no Split-K and no algorithm
    selection, so a row's logits do not depend on batch layout.
    """

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "ascend_fp32_matmul_vjp"

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "lm_head_ascend"):
            raise RuntimeError(
                "lm_head_ascend is not compiled into the extension. "
                "Rebuild on an Ascend NPU host with KERNEL_ALIGN_FORCE_ASCEND=1."
            )
        logger.info("Successfully linked to precompiled _C_npu.lm_head_ascend kernel.")

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
        if not self._can_use_ascend(hidden, weight, bias):
            raise RuntimeError(
                "AscendLMHeadOp requires Ascend NPU bf16/fp16/fp32 inputs; "
                "Native/Triton fallback is forbidden"
            )
        return _AscendLMHeadFunction.apply(hidden, weight, bias, False)

    def forward_fp32(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._can_use_ascend(hidden, weight, bias):
            raise RuntimeError(
                "AscendLMHeadOp requires Ascend NPU bf16/fp16/fp32 inputs; "
                "Native/Triton fallback is forbidden"
            )
        return _AscendLMHeadFunction.apply(hidden, weight, bias, True)

    def parameter_vjp_contributions_fp32(self, *, hidden, weight, grad_output, bias=None):
        del weight, bias
        rows_h = hidden.reshape(-1, hidden.size(-1)).float()
        rows_g = grad_output.reshape(-1, grad_output.size(-1)).float()
        return {"weight": rows_g[:, :, None] * rows_h[:, None, :]}

    @staticmethod
    def _can_use_ascend(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> bool:
        bias_ok = bias is None or (
            bias.device.type == "npu"
            and bias.device == hidden.device
            and bias.dim() == 1
            and bias.dtype in _SUPPORTED_DTYPES
        )
        return (
            hidden.device.type == "npu"
            and weight.device.type == "npu"
            and hidden.device == weight.device
            and hidden.dim() >= 2
            and weight.dim() == 2
            and hidden.size(-1) == weight.size(1)
            and hidden.dtype in _SUPPORTED_DTYPES
            and weight.dtype in _SUPPORTED_DTYPES
            and bias_ok
        )
