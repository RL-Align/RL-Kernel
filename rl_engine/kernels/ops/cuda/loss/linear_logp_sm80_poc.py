# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""SM80 (Ampere / A100) native fused ``linear_logp`` -- stage-1 PoC.

This module is **explicit-call only**: it is deliberately not registered in the
kernel registry (``rl_engine.kernels.registry``) and carries no autograd
``Function``. It is the thin Python entry point for the stage-1 native kernel
defined in ``csrc/cuda/fused_linear_logp_sm80.cu``.

Contract (matches the SM90/Triton ``linear_logp`` data flow, forward-only):

    hidden     : [..., D] BF16 CUDA        (lead dims are flattened to [N, D])
    weight     : [V, D]   BF16 CUDA         logit[n, v] = hidden[n] . weight[v]
    target_ids : [...]    int CUDA/CPU      one target vocab id per token
    return     : [...]    FP32 CUDA         log p(target) = z_target - logsumexp

The [N, V] logit tensor is never materialized. The PoC kernel only requires
``D % 16 == 0`` and ``V % 64 == 0``; it has been validated for D=4096,
V=128256, BF16, N in [128, 4096] on A100 (SM80).
"""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def sm80_linear_logp_available() -> bool:
    """Whether the compiled extension exposes the SM80 PoC entry point."""
    return bool(_EXT_AVAILABLE) and _C is not None and hasattr(_C, "fused_linear_logp_sm80")


def sm80_linear_logp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward-only fused selected-token log-prob on SM80 (no autograd).

    Args:
        hidden: ``[..., D]`` BF16 CUDA tensor.
        weight: ``[V, D]`` BF16 CUDA tensor (row-major LM head weights).
        target_ids: ``[...]`` integer tensor with one vocab id per token.

    Returns:
        ``[...]`` FP32 CUDA tensor of selected log-probs, shaped like
        ``target_ids`` / ``hidden.shape[:-1]``.
    """
    if not sm80_linear_logp_available():
        raise RuntimeError(
            "SM80 fused linear_logp PoC is not available: the _C extension was "
            "not built with csrc/cuda/fused_linear_logp_sm80.cu."
        )
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("sm80_linear_logp: hidden and weight must be CUDA tensors")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("sm80_linear_logp: hidden and weight must be bfloat16")
    if weight.dim() != 2:
        raise ValueError(f"sm80_linear_logp: weight must be [V, D], got {tuple(weight.shape)}")
    if hidden.size(-1) != weight.size(1):
        raise ValueError(
            f"sm80_linear_logp: hidden dim {hidden.size(-1)} must match weight "
            f"dim {weight.size(1)}"
        )
    if weight.device != hidden.device:
        raise ValueError("sm80_linear_logp: weight must be on the same device as hidden")

    lead_shape = hidden.shape[:-1]
    n_tokens = hidden.numel() // hidden.size(-1)
    if target_ids.numel() != n_tokens:
        raise ValueError(
            f"sm80_linear_logp: target_ids must have one id per token: expected "
            f"{n_tokens}, got {target_ids.numel()}"
        )

    hidden_2d = hidden.contiguous().reshape(n_tokens, hidden.size(-1))
    weight_2d = weight.contiguous()
    target_1d = target_ids.reshape(n_tokens)

    logp = _C.fused_linear_logp_sm80(hidden_2d, weight_2d, target_1d)
    return logp.reshape(lead_shape)
