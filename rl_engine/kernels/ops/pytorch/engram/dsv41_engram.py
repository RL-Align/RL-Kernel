# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden DSv4.1 Engram gate, numeric profile ``dsv41-engram-v1``.

Semantics follow vLLM's ``_fused_engram_post_wkv_kernel``
(``vllm/models/deepseek_v41/common/engram.py``): for each token and stream ``h`` of the
four mHC streams, with ``kv = wkv(rows)`` ``[T, 5 H]`` (four keys, then one value)::

    key   = kv[:, h H : (h + 1) H]
    rx    = 1 / sqrt(sum(x_h^2) / H + eps)
    rk    = 1 / sqrt(sum(key^2) / H + eps)
    d     = ((sum(((x_h * q_h) * k_h) * key) * rx) * rk) * (1 / sqrt(H))
    g     = sigmoid(sign(d) * sqrt(max(|d|, clamp)))           0 where the token is masked
    out_h = BF16(x_h + g * value)

Arithmetic (fixed so the CUDA kernel matches byte for byte):

- The three H-long sums use the mHC slice scheme (``mhc.slice_dot``: 128 slices of runs
  of 8, an ``fmaf`` chain per slice over ascending chunks, then the fixed slice tree), so
  H = 5120 is five chunks.
- Everything else is FP32 with separately rounded operations; ``sqrt`` and division are
  correctly rounded, ``sigmoid(z) = 1 / (1 + expf(-z))`` with CUDA's ``expf`` (run on CUDA).

Backward: dX of the streams only (the tables, ``wkv``, ``q`` and ``k`` are frozen); no byte
contract, see :func:`engram_gate_backward_reference`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.pytorch.mhc import HC, slice_dot

CLAMP = 1e-6


def _c(value: float, like: Tensor) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=like.device)


def _rms(ss: Tensor, h: int, eps: float) -> Tensor:
    return _c(1.0, ss) / torch.sqrt(ss / _c(float(h), ss) + _c(eps, ss))


def engram_gate(
    x: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    mask: Tensor | None = None,
    eps: float = 1e-6,
    clamp: float = CLAMP,
) -> tuple[Tensor, Tensor]:
    """``x`` BF16 [T, 4, H], ``kv`` FP32 [T, 5 H] -> ``(out BF16 [T, 4, H], gate [T, 4])``."""
    if not x.is_cuda:
        raise RuntimeError("the Engram golden is defined on CUDA (libdevice expf)")
    t, hc, h = x.shape
    assert hc == HC and kv.shape == (t, (HC + 1) * h)
    xf = x.float()
    key = kv[:, : HC * h].float().reshape(t, HC, h)
    value = kv[:, HC * h :].float()
    u = (xf * q_weight.float()[None]) * k_weight.float()[None]
    rx = _rms(slice_dot(xf, xf), h, eps)
    rk = _rms(slice_dot(key, key), h, eps)
    dot = slice_dot(u, key)
    ch = _c(1.0, xf) / torch.sqrt(_c(float(h), xf))
    d = ((dot * rx) * rk) * ch
    gi = torch.sqrt(torch.maximum(d.abs(), _c(clamp, d)))
    gi = torch.where(d < 0, -gi, gi)
    one = _c(1.0, d)
    g = one / (one + torch.exp(-gi))
    if mask is not None:
        g = torch.where(mask[:, None], g, torch.zeros_like(g))
    out = (xf + g[..., None] * value[:, None, :]).to(torch.bfloat16)
    return out, g


def engram_gate_backward_reference(
    dout: Tensor,
    x: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    mask: Tensor | None = None,
    eps: float = 1e-6,
    clamp: float = CLAMP,
) -> Tensor:
    """FP64 autograd dX of the gate's math (the reference the CUDA backward is checked
    against within a tolerance)."""
    t, hc, h = x.shape
    xd = x.double().requires_grad_(True)
    key = kv[:, : HC * h].double().reshape(t, HC, h)
    value = kv[:, HC * h :].double()
    rx = torch.rsqrt(xd.square().mean(-1) + eps)
    rk = torch.rsqrt(key.square().mean(-1) + eps)
    dot = (xd * q_weight.double()[None] * k_weight.double()[None] * key).sum(-1)
    d = dot * rx * rk / h**0.5
    gi = torch.sign(d) * torch.sqrt(torch.clamp(d.abs(), min=clamp))
    g = torch.sigmoid(gi)
    if mask is not None:
        g = g * mask[:, None].double()
    out = xd + g[..., None] * value[:, None, :]
    (out * dout.double()).sum().backward()
    return xd.grad
