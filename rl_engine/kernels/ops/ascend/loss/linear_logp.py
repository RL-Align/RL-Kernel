# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


class _FusedLinearLogpAscendFunction(torch.autograd.Function):
    """Autograd bridge for the Ascend fused linear log-prob forward.

    The backward is the shared Liger-style chunked formula from
    ``rl_engine.kernels.ops.pytorch.loss.linear_logp.chunked_linear_logp_backward``
    (the same formula the CUDA SM90 op falls back to), so gradients follow
    the CUDA op's portable backward exactly.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        output = _C_npu.fused_linear_logp_ascend(hidden_2d, weight, None, target_1d)
        ctx.save_for_backward(hidden_2d, weight, target_1d)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        return output.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import chunked_linear_logp_backward

        hidden_2d, weight, target_1d = ctx.saved_tensors
        grad_hidden, grad_weight, _ = chunked_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            target_1d,
            hidden_2d,  # bias placeholder; has_bias=False
            has_bias=False,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=None,
        )
        return grad_hidden, grad_weight, None


class FusedLinearLogpAscendOp:
    """Batch-invariant fused linear log-prob for Ascend NPU.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing
    the ``[N, V]`` logits. The Ascend C forward mirrors the SM90 kernel's
    reduction contract: ascending vocab-row scan with the online rescale
    chain, per-row fp32 dots over a fixed D-tile order, final
    ``min(zt - lse, 0)`` clamp; the output is fp32.
    """

    is_fused_logp = True
    is_batch_invariant = True

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "fused_linear_logp_ascend"):
            raise RuntimeError(
                "fused_linear_logp_ascend is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        self.op = _C_npu.fused_linear_logp_ascend
        logger.info("Successfully linked to precompiled _C_npu.fused_linear_logp_ascend kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
            NativeLinearLogpOp,
            should_use_tensor_parallel_linear_logp,
        )

        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if lm_head_weight.device != hidden.device:
            raise ValueError(
                f"lm_head_weight device {lm_head_weight.device} must match hidden "
                f"device {hidden.device}"
            )
        if hidden.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        # Tensor-parallel and bias paths are not covered by the Ascend forward;
        # delegate to the native reference (same fallback as the CUDA op).
        if (
            should_use_tensor_parallel_linear_logp(
                tp_group,
                int(vocab_start_index),
                global_vocab_size,
                lm_head_weight.size(0),
            )
            or bias is not None
        ):
            return NativeLinearLogpOp().apply(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        if not self._ascend_supported(hidden, lm_head_weight):
            return NativeLinearLogpOp().apply(hidden, lm_head_weight, target_ids)
        return _FusedLinearLogpAscendFunction.apply(hidden, lm_head_weight, target_ids)

    @staticmethod
    def _ascend_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
        return (
            hidden.device.type == "npu"
            and lm_head_weight.device.type == "npu"
            and hidden.is_contiguous()
            and lm_head_weight.is_contiguous()
            and hidden.dtype in _SUPPORTED_DTYPES
            and lm_head_weight.dtype == hidden.dtype
        )
