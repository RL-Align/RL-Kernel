# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Production SM80 fused ``linear_logp`` backend.

The native CUDA path is deliberately narrow: A100/SM80, BF16, D=4096,
V=128256, no bias or autograd, and at most 1024 tokens. All other inputs are
delegated to the existing Triton (or PyTorch) implementation, preserving the
public operator contract without extending the native kernel's validated scope.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

SM80_LINEAR_LOGP_D = 4096
SM80_LINEAR_LOGP_V = 128256
SM80_LINEAR_LOGP_MAX_TOKENS = 1024
SM80_LINEAR_LOGP_BACKEND_ID = "cuda-fused-linear-logp-sm80-v1"


def sm80_linear_logp_available() -> bool:
    return bool(_EXT_AVAILABLE) and _C is not None and hasattr(
        _C, "fused_linear_logp_sm80"
    )


def _fallback_op(hidden: torch.Tensor):
    if hidden.device.type == "cuda":
        try:
            from rl_engine.kernels.ops.triton.loss.linear_logp import (
                TritonLinearLogpOp,
            )

            return TritonLinearLogpOp()
        except Exception:  # pragma: no cover - Triton is optional
            pass
    from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

    return NativeLinearLogpOp()


def _is_native_supported(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: Optional[int],
) -> bool:
    if not sm80_linear_logp_available() or hidden.device.type != "cuda":
        return False
    if weight.device != hidden.device:
        return False
    if torch.cuda.get_device_capability(hidden.device) != (8, 0):
        return False
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if hidden.ndim < 1 or weight.ndim != 2:
        return False
    if hidden.size(-1) != SM80_LINEAR_LOGP_D:
        return False
    if tuple(weight.shape) != (SM80_LINEAR_LOGP_V, SM80_LINEAR_LOGP_D):
        return False
    n_tokens = hidden.numel() // hidden.size(-1)
    if not 0 < n_tokens <= SM80_LINEAR_LOGP_MAX_TOKENS:
        return False
    if target_ids.shape != hidden.shape[:-1]:
        return False
    if bias is not None or hidden.requires_grad or weight.requires_grad:
        return False
    if tp_group is not None or vocab_start_index != 0:
        return False
    if global_vocab_size not in (None, SM80_LINEAR_LOGP_V):
        return False
    return True


class FusedLinearLogpSM80Op:
    """A100 forward-only fast path with transparent fallback.

    Computes ``log_softmax(hidden @ weight.T + bias)[target_ids]`` and returns
    FP32 output with shape ``hidden.shape[:-1]``. The native path never
    materializes ``[N, V]``; unsupported configurations retain existing
    Triton/PyTorch semantics.
    """

    backend_id = SM80_LINEAR_LOGP_BACKEND_ID

    def __init__(self) -> None:
        if not sm80_linear_logp_available():
            raise RuntimeError(
                "fused_linear_logp_sm80 is not compiled into the CUDA extension"
            )

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

    def selected_backend(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> str:
        native = _is_native_supported(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )
        return "native_sm80" if native else "fallback"

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
        if not _is_native_supported(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        ):
            return _fallback_op(hidden)(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )

        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], "
                f"got [{t_min}, {t_max}]"
            )

        lead_shape = hidden.shape[:-1]
        n_tokens = hidden.numel() // hidden.size(-1)
        hidden_2d = hidden.reshape(n_tokens, hidden.size(-1)).contiguous()
        weight_2d = lm_head_weight.contiguous()
        target_1d = target_ids.reshape(n_tokens).to(
            device=hidden.device, dtype=torch.int32
        ).contiguous()
        return _C.fused_linear_logp_sm80(
            hidden_2d, weight_2d, target_1d
        ).reshape(lead_shape)
