# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public entry for the DeepSeek-V4 mHC ops: mHC pre, mHC post and mHC head.

Each function looks up its op in ``kernel_registry`` for the device of its
first tensor argument and calls it: aiter on ROCm, the fp32 PyTorch reference
on CPU. Other devices have no mHC backend and raise ``RuntimeError``. The
arithmetic is recorded in rl_engine/contracts/operators/mhc.py.
"""

from __future__ import annotations

from torch import Tensor

from rl_engine.contracts.operators.mhc import (
    HC_POST_MULT_VALUE,
    HC_PRE_EPS,
    HC_SINKHORN_EPS,
    RMS_EPS,
    SINKHORN_REPEAT,
)
from rl_engine.runtime.registry import kernel_registry


def mhc_pre(
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    *,
    rms_eps: float = RMS_EPS,
    hc_pre_eps: float = HC_PRE_EPS,
    hc_sinkhorn_eps: float = HC_SINKHORN_EPS,
    hc_post_mult_value: float = HC_POST_MULT_VALUE,
    sinkhorn_repeat: int = SINKHORN_REPEAT,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(post_mix [..., n], comb_mix [..., n, n], layer_input [..., C])``
    for ``residual`` [..., n, C]."""
    op = kernel_registry.get_op("mhc_pre", device=residual.device)
    return op(
        residual,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


def mhc_post(y: Tensor, residual: Tensor, post_mix: Tensor, comb_mix: Tensor) -> Tensor:
    """Return the new residual [..., n, C] from the sublayer output ``y`` [..., C]."""
    op = kernel_registry.get_op("mhc_post", device=residual.device)
    return op(y, residual, post_mix, comb_mix)


def mhc_head(
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    *,
    rms_eps: float = RMS_EPS,
    hc_pre_eps: float = HC_PRE_EPS,
) -> Tensor:
    """Return the hidden state [..., C] contracted from ``residual`` [..., n, C]."""
    op = kernel_registry.get_op("mhc_head", device=residual.device)
    return op(residual, fn, scale, base, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps)


__all__ = ["mhc_head", "mhc_post", "mhc_pre"]
