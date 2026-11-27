# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden DSv4.1 manifold hyper-connections, numeric profile ``dsv41-mhc-v1``.

DSv4.1 keeps DSv4's controller (``dsv4_mhc.controller``: the same GEMM, sigmoid, softmax
and Sinkhorn) and changes the wiring, as vLLM's ``mhc_pre_delayed_torch`` and DeepSeek-V4.1
do: a sublayer collapses its streams with the pre computed by the *previous* sublayer and
returns its own pre for the next one.

    pre_out, post, comb = controller(residual)                  # as dsv4-mhc-v1
    y      = residual_0                    if pre_in is None    # the model's first sublayer
           = sum_i pre_in_i * residual_i   otherwise            # -> BF16
    out    = y, or rms_norm(y, norm_weight)
    final  = rms_norm(sum_i pre_i * residual_i, norm)           # collapse: no learned hc_head

Arithmetic, beyond ``dsv4-mhc-v1`` (whose docstring is the contract for every shared step):

- H may be 5120: the slice scheme has five runs of 1024 columns per slice; a slice's FMA
  chain runs over its chunks in ascending order, as for H <= 4096.
- The first sublayer's ``y`` is a copy of stream 0, not a mix with ``(1, 0, 0, 0)``.
- In the backward, the aggregate leg of ``d_residual`` uses the pre the mix used
  (``e_0`` for the first sublayer), and the controller's pre gradient is ``d_pre_out``, the
  gradient of the pre this op returned. ``d_pre_in = slice_dot(residual_i, dy)``.

DSv4.1 runs the first layer's controller on the embedding broadcast to four streams, so the
caller passes the expanded residual (as DeepSeek's reference does; vLLM's summed-``fn``
shortcut is not used).

``eps`` is the model's ``norm_eps`` (1e-20 in the release; DSv4 uses 1e-6), used for the
controller's rms and the fused RMSNorm, as in DeepSeek's reference (``Block.hc_mixes`` and
``attn_norm`` / ``ffn_norm`` / ``norm``). The Sinkhorn / pre ``hc_eps`` stays 1e-6.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .dsv4_mhc import (
    HC,
    _controller_backward,
    _gemm_backward,
    aggregate,
    controller,
    gemm_partials,
    rms_norm,
    rms_norm_backward,
    slice_dot,
    split_sum,
)

NORM_EPS = 1e-20  # DSv4.1's norm_eps


def _maybe_norm(y: Tensor, norm_weight: Tensor | None, eps: float) -> Tensor:
    return y if norm_weight is None else rms_norm(y, norm_weight, eps)[0]


def _mix(residual: Tensor, pre_in: Tensor | None) -> Tensor:
    return residual[:, 0].contiguous() if pre_in is None else aggregate(pre_in, residual)


def _leg_pre(residual: Tensor, pre_in: Tensor | None) -> Tensor:
    """The pre of the aggregate leg: ``pre_in``, or ``e_0`` for the stream-0 mix."""
    if pre_in is not None:
        return pre_in.float()
    e0 = torch.zeros(residual.shape[0], HC, dtype=torch.float32, device=residual.device)
    e0[:, 0] = 1.0
    return e0


def mhc_pre_delayed_from_partials(
    residual: Tensor,
    pmix: Tensor,
    pss: Tensor,
    scale: Tensor,
    base: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """-> ``(out BF16 [T, H], post [T, 4], comb [T, 4, 4], pre_out [T, 4], raw [T, 24],
    ss [T])``; ``out`` is the mix with ``pre_in`` (stream 0 when ``None``), normed when a
    weight is given."""
    raw = split_sum(pmix)[:, :24]
    ss = split_sum(pss)
    ctl = controller(raw, ss, scale, base, residual.shape[1] * residual.shape[2], rms_eps=eps)
    y = _mix(residual, pre_in)
    return _maybe_norm(y, norm_weight, eps), ctl["post"], ctl["comb"], ctl["pre"], raw, ss


def mhc_pre_delayed(
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``mhc_pre_delayed_from_partials`` on DeepGEMM's partials (CUDA only)."""
    pm, ps = gemm_partials(residual, fn)
    return mhc_pre_delayed_from_partials(residual, pm, ps, scale, base, pre_in, norm_weight, eps)


def hc_collapse(
    residual: Tensor, pre: Tensor, norm_weight: Tensor | None = None, eps: float = NORM_EPS
) -> Tensor:
    """DSv4.1's final collapse: ``sum_i pre_i * residual_i`` -> BF16 [T, H], normed when a
    weight is given."""
    return _maybe_norm(aggregate(pre.float(), residual), norm_weight, eps)


def _unnorm(dy: Tensor, y: Tensor, norm_weight: Tensor | None, eps: float) -> Tensor:
    if norm_weight is None:
        return dy.float()
    _, r = rms_norm(y, norm_weight, eps)
    return rms_norm_backward(dy, y, norm_weight, r)


def mhc_pre_delayed_backward(
    dy: Tensor,
    d_post: Tensor,
    d_comb: Tensor,
    d_pre_out: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    pre_in: Tensor | None = None,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor | None]:
    """``(d_residual [T, 4, H] FP32, d_pre_in [T, 4] FP32 or None)`` of
    ``mhc_pre_delayed`` (frozen parameters). ``dy`` is the gradient of ``out`` (normed when
    a weight is given); ``d_post`` / ``d_comb`` / ``d_pre_out`` those of the returned
    coefficients."""
    k = residual.shape[1] * residual.shape[2]
    ctl = controller(raw, ss, scale, base, k, rms_eps=eps)
    dyf = _unnorm(dy, _mix(residual, pre_in), norm_weight, eps)
    leg = _leg_pre(residual, pre_in)
    dres_agg = torch.stack([leg[:, i, None] * dyf for i in range(HC)], dim=1)
    d_pre_in = None if pre_in is None else slice_dot(residual.float(), dyf[:, None, :])
    draw, dss = _controller_backward(
        raw, ctl, scale, k, ss, d_pre_out.float(), d_post.float(), d_comb.float()
    )
    return dres_agg + _gemm_backward(residual, fn, draw, dss), d_pre_in


def hc_collapse_backward(
    dy: Tensor,
    residual: Tensor,
    pre: Tensor,
    norm_weight: Tensor | None = None,
    eps: float = NORM_EPS,
) -> tuple[Tensor, Tensor]:
    """``(d_residual [T, 4, H] FP32, d_pre [T, 4] FP32)`` of ``hc_collapse``."""
    p = pre.float()
    dyf = _unnorm(dy, aggregate(p, residual), norm_weight, eps)
    dres = torch.stack([p[:, i, None] * dyf for i in range(HC)], dim=1)
    return dres, slice_dot(residual.float(), dyf[:, None, :])
