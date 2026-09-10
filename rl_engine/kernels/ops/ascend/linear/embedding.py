# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

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


def _deterministic_embedding_grad_weight(
    ids: torch.Tensor,
    grad_rows: torch.Tensor,
    *,
    weight_shape: tuple[int, ...],
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    # Bitwise-identical backward by construction: the SM90 CUDA op's backward
    # is itself pure PyTorch (sorted-segment dweight), so the Ascend op reuses
    # the exact same function. Every op in it (mask, stable argsort,
    # unique_consecutive, fixed-order accumulation) is deterministic on NPU,
    # hence grad_weight matches the CUDA op bit for bit on identical inputs.
    from rl_engine.kernels.ops.cuda.linear.embedding import (
        _deterministic_embedding_grad_weight as _cuda_grad_weight,
    )

    return _cuda_grad_weight(
        ids,
        grad_rows,
        weight_shape=weight_shape,
        weight_dtype=weight_dtype,
    )


class _AscendEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids: torch.Tensor, weight: torch.Tensor, output_fp32: bool):
        ctx.save_for_backward(token_ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_fp32 = bool(output_fp32)
        return _C_npu.embedding_ascend(token_ids, weight.contiguous(), bool(output_fp32))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (token_ids,) = ctx.saved_tensors
        grad_weight = None
        if ctx.needs_input_grad[1]:
            ids = token_ids.reshape(-1).to(device=grad_output.device, dtype=torch.long)
            hidden_size = int(ctx.weight_shape[1])
            grad_rows = grad_output.reshape(ids.numel(), hidden_size)
            grad_weight = _deterministic_embedding_grad_weight(
                ids,
                grad_rows,
                weight_shape=ctx.weight_shape,
                weight_dtype=ctx.weight_dtype,
            )
        record_backward(
            "embedding",
            kernel_id=(
                "rl_engine.kernels.ops.ascend.linear.embedding."
                "_deterministic_embedding_grad_weight"
            ),
            impl="ascend_sorted_segment_dweight",
            family="ascend",
        )
        return None, grad_weight, None


class AscendEmbeddingOp(torch.nn.Module):
    """Single-card batch-invariant Ascend C embedding op.

    Forward is a pure row gather (a byte copy of weight rows), so it is
    bitwise identical to the SM90 CUDA embedding kernel on identical inputs;
    backward reuses the same sorted-segment dweight formula as the CUDA op.
    """

    op_class = "elementwise"
    is_batch_invariant = True

    def __init__(self) -> None:
        super().__init__()
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "embedding_ascend"):
            raise RuntimeError(
                "embedding_ascend is not compiled into the extension. "
                "Rebuild on an Ascend NPU host with KERNEL_ALIGN_FORCE_ASCEND=1."
            )
        logger.info("Successfully linked to precompiled _C_npu.embedding_ascend kernel.")

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self._can_use_ascend(token_ids, weight):
            raise RuntimeError(
                "AscendEmbeddingOp requires Ascend NPU bf16/fp16/fp32 inputs; "
                "Native/Triton fallback is forbidden"
            )
        return _AscendEmbeddingFunction.apply(token_ids, weight, False)

    def forward_fp32(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self._can_use_ascend(token_ids, weight):
            raise RuntimeError(
                "AscendEmbeddingOp requires Ascend NPU bf16/fp16/fp32 inputs; "
                "Native/Triton fallback is forbidden"
            )
        return _AscendEmbeddingFunction.apply(token_ids, weight, True)

    @staticmethod
    def _can_use_ascend(token_ids: torch.Tensor, weight: torch.Tensor) -> bool:
        return (
            token_ids.device.type == "npu"
            and weight.device.type == "npu"
            and token_ids.device == weight.device
            and weight.dim() == 2
            and weight.dtype in _SUPPORTED_DTYPES
        )
