# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE router (golden): sqrt-softplus scores, top-6 / hash routing, weights.

The model math follows vLLM's DSv4 router (``ISEEKYAN/vllm@1a6542c``,
``fused_moe/router/dsv4_topk.py`` for scored layers and
``topk_softplus_sqrt_kernels.cu`` ``dsv4HashTopkSoftplusSqrt`` for the first
``num_hash_layers`` layers). vLLM writes softplus two different ways there and
uses fast-math ``exp`` / ``log`` / ``sqrt``; the arithmetic below is our own and
fixed (design doc P3), so the CUDA kernel matches it byte for byte and the
rollout and the actor recompute run the same kernel.

Forward, per token, from FP32 gate logits ``x`` ``[T, E]``::

    s_e   = sqrt(max(x_e, 0) + log1pf(expf(-|x_e|)))   libdevice (run on CUDA),
                                                        add and sqrt correctly rounded
    key_e = s_e + bias_e,   NaN -> -1e30
    ids   = the 6 largest keys, ties to the smallest expert id, in that order
            (hash layers: ids = tid2eid[input_ids], the table's order)
    w_k   = s_{ids_k},   NaN -> 0
    S     = (((((w_0 + w_1) + w_2) + w_3) + w_4) + w_5)
    f     = c / (S if S > 0 else 1)                     c = routed_scaling_factor (FP32)
    out_k = w_k * f

Backward (the gate weight and the bias are frozen: only dX), given ``d = dL/dout``::

    g      = (((d_0 * out_0) + d_1 * out_1) + ...)        ascending k, no FMA
    dw_k   = (c * d_k - g) / S           if S > 0, else c * d_k
    sig(x) = 1 / (1 + expf(-x)) for x >= 0, else e / (1 + e) with e = expf(x)
    dx_k   = (dw_k * sig(x_k)) / (2 * s_k)   if s_k > 0 and finite, else 0
    dX[t]  = ((dx_0 * W[ids_0]) + dx_1 * W[ids_1]) + ...  FP32, ascending k, no FMA

``dX`` is the gate GEMM's dgrad written as a six-row gather, since only six
logits per token carry a gradient; it is exact in the order above.
"""

from __future__ import annotations

import torch
from torch import Tensor

TOP_K = 6
NAN_KEY = -1e30


def sqrt_softplus(x: Tensor) -> Tensor:
    """``sqrt(softplus(x))`` in the fixed form above; FP32 in, FP32 out."""
    if x.dtype != torch.float32:
        raise TypeError("router logits must be FP32")
    return torch.sqrt(torch.clamp_min(x, 0.0) + torch.log1p(torch.exp(-x.abs())))


def _sum_slots(v: Tensor) -> Tensor:
    acc = v[:, 0]
    for k in range(1, v.shape[1]):
        acc = acc + v[:, k]
    return acc


def route_topk(
    logits: Tensor,
    bias: Tensor | None,
    routed_scaling_factor: float,
    *,
    input_ids: Tensor | None = None,
    tid2eid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(weights FP32 [T, 6], ids int32 [T, 6])``.

    Hash routing when ``tid2eid`` is given (``input_ids`` required, bias unused);
    otherwise the top-6 of ``s + bias`` (``bias`` required).
    """
    if logits.dtype != torch.float32 or logits.dim() != 2:
        raise TypeError("logits must be FP32 [T, E]")
    s = sqrt_softplus(logits)
    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        if tid2eid.dim() != 2 or tid2eid.shape[1] != TOP_K:
            raise ValueError("tid2eid must be [vocab, 6]")
        ids = tid2eid[input_ids.long()].long()
    else:
        if bias is None:
            raise ValueError("scored routing requires the correction bias")
        key = s + bias.to(torch.float32)
        key = torch.where(torch.isnan(key), torch.full_like(key, NAN_KEY), key)
        # Stable descending sort keeps equal keys in ascending id order.
        ids = torch.sort(key, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    w = s.gather(1, ids)
    w = torch.where(torch.isnan(w), torch.zeros_like(w), w)
    total = _sum_slots(w)
    c = torch.tensor(routed_scaling_factor, dtype=torch.float32, device=logits.device)
    f = c / torch.where(total > 0, total, torch.ones_like(total))
    return w * f[:, None], ids.to(torch.int32)


def _sigmoid(x: Tensor) -> Tensor:
    one = torch.ones_like(x)
    pos = one / (one + torch.exp(-x))
    e = torch.exp(x)
    neg = e / (one + e)
    return torch.where(x >= 0, pos, neg)


def route_backward_logits(
    logits: Tensor, ids: Tensor, d_weights: Tensor, routed_scaling_factor: float
) -> Tensor:
    """``dL/dx`` at the six selected logits, FP32 ``[T, 6]`` (slot order)."""
    x = logits.gather(1, ids.long())
    s = sqrt_softplus(x)
    finite = ~torch.isnan(s)
    w = torch.where(finite, s, torch.zeros_like(s))
    total = _sum_slots(w)
    positive = total > 0
    c = torch.tensor(routed_scaling_factor, dtype=torch.float32, device=logits.device)
    f = c / torch.where(positive, total, torch.ones_like(total))
    out = w * f[:, None]
    d = d_weights.to(torch.float32)
    g = _sum_slots(d * out)
    cd = c * d
    dw = torch.where(
        positive[:, None],
        (cd - g[:, None]) / torch.where(positive, total, torch.ones_like(total))[:, None],
        cd,
    )
    live = finite & (s > 0) & ~torch.isinf(s)
    two_s = torch.where(live, s * 2.0, torch.ones_like(s))
    dx = (dw * _sigmoid(x)) / two_s
    return torch.where(live, dx, torch.zeros_like(dx))


def route_backward_dx(
    logits: Tensor,
    ids: Tensor,
    d_weights: Tensor,
    gate_weight: Tensor,
    routed_scaling_factor: float,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``dL/dX`` through the frozen gate ``W`` ``[E, H]``: ``[T, H]`` in ``out_dtype``."""
    dsel = route_backward_logits(logits, ids, d_weights, routed_scaling_factor)
    w = gate_weight.to(torch.float32)
    idx = ids.long()
    acc = torch.zeros(logits.shape[0], w.shape[1], dtype=torch.float32, device=logits.device)
    for k in range(ids.shape[1]):
        acc = acc + dsel[:, k : k + 1] * w[idx[:, k]]
    return acc.to(out_dtype)
