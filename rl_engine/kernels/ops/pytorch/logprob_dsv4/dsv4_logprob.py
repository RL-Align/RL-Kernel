# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 selected-token log-probability, numeric profile ``dsv4-logprob-v1`` (golden).

This module is the arithmetic contract; the CUDA kernels in
``csrc/cuda/logprob_dsv4/dsv4_logprob.cu`` are byte-equal to it. Run it on CUDA:
``torch.exp`` / ``torch.log`` there are libdevice ``expf`` / ``logf``, the functions
the kernels call (CPU differs in the last bit).

Inputs: FP32 logits ``z [n, V]`` (the LM-head GEMM's FP32 accumulator, never rounded
to BF16), target ids ``[n]``, a temperature ``T > 0`` (Python float). ``V`` is a
multiple of 64 (DSv4: 129280).

Every FP32 step is correctly rounded; nothing is contracted into an FMA::

    z'   = z                         if T == 1
         = z / T                     otherwise (division, not a reciprocal multiply)
    tiles: 64 fixed tiles of L = V / 64 columns, tile t = [t L, (t + 1) L)
    per (row, tile):
        m_t  = max over the tile (exact)
        lane l in 0..31 owns columns t L + l + 32 k, k ascending:
            s_l = sum_k expf(z'_c - m_t)            left to right
        s_t  = halving over the 32 lanes: s_l += s_{l + o}, o = 16, 8, 4, 2, 1
    per row:
        M    = max_t m_t (exact)
        e_t  = s_t * expf(m_t - M)
        S    = (e_i + e_{i + 32}) for i < 32, then halving o = 16, 8, 4, 2, 1
        lse  = M + logf(S)
        logp = z'_target - lse

The 64-tile split is the #241 vocab-TP structure: with the vocabulary sharded over
TP ranks on tile boundaries (TP = 1, 2, 4, 8 for DSv4), each rank computes its tiles'
``(m_t, s_t)``, the partials are gathered, and the merge is the same, so the bytes do
not depend on the TP degree. Each row is computed alone, so they do not depend on the
batch either.

Backward (``dL/dz`` given ``g = dL/dlogp``; the LSE output carries no gradient)::

    p_c   = expf(z'_c - lse)
    d_c   = g * (1 - p_c)  if c == target  else  g * (-p_c)
    dz_c  = d_c             if T == 1      else  d_c / T
    returned as BF16 (rounded once)

With ``g = 1`` this is ``P = (onehot - p) [/ T]``; the CUDA autograd forms
``dL/dhidden = BF16(g * (P @ W))`` from it (see ``rl_engine.kernels.ops.cuda.logprob_dsv4``).
"""

from __future__ import annotations

import torch
from torch import Tensor

PROFILE = "dsv4-logprob-v1"
NUM_TILES = 64
LANES = 32


def _scaled(z: Tensor, temperature: float) -> Tensor:
    z = z.float()
    if temperature == 1.0:
        return z
    return z / torch.tensor(temperature, dtype=torch.float32, device=z.device)


def _halve(x: Tensor) -> Tensor:
    """Halving over the last dimension (a power of two): x_l += x_{l + o}."""
    while x.shape[-1] > 1:
        o = x.shape[-1] // 2
        x = x[..., :o] + x[..., o:]
    return x[..., 0]


def tile_stats(z: Tensor, temperature: float = 1.0, num_tiles: int = NUM_TILES):
    """``(m, s)`` FP32 ``[n, num_tiles]`` for logits covering ``num_tiles`` whole tiles."""
    n, v = z.shape
    if v % num_tiles:
        raise ValueError(f"{v} columns do not split into {num_tiles} tiles")
    tile = v // num_tiles
    zp = _scaled(z, temperature).view(n, num_tiles, tile)
    m = zp.amax(dim=-1)
    steps = -(-tile // LANES)
    pad = torch.full(
        (n, num_tiles, steps * LANES), float("-inf"), dtype=torch.float32, device=z.device
    )
    pad[..., :tile] = zp
    e = torch.exp(pad - m.unsqueeze(-1)).view(n, num_tiles, steps, LANES)
    s = e[:, :, 0]
    for k in range(1, steps):
        s = s + e[:, :, k]
    return m, _halve(s)


def merge(m: Tensor, s: Tensor) -> Tensor:
    """LSE FP32 ``[n]`` from the 64 tile partials ``[n, 64]``."""
    if m.shape[-1] != NUM_TILES:
        raise ValueError(f"expected {NUM_TILES} tile partials, got {m.shape[-1]}")
    big = m.amax(dim=-1)
    e = s * torch.exp(m - big.unsqueeze(-1))
    return big + torch.log(_halve(e[:, :32] + e[:, 32:]))


def selected_logprob(z: Tensor, targets: Tensor, temperature: float = 1.0):
    """``(logp, lse)`` FP32 ``[n]`` from FP32 logits ``[n, V]``."""
    m, s = tile_stats(z, temperature)
    lse = merge(m, s)
    zt = _scaled(z.gather(1, targets.long().view(-1, 1)).view(-1), temperature)
    return zt - lse, lse


def selected_logprob_backward(
    z: Tensor, targets: Tensor, lse: Tensor, grad: Tensor, temperature: float = 1.0
) -> Tensor:
    """``dL/dz`` BF16 ``[n, V]`` from ``g = dL/dlogp`` FP32 ``[n]``."""
    zp = _scaled(z, temperature)
    p = torch.exp(zp - lse.unsqueeze(-1))
    d = -p
    idx = targets.long().view(-1, 1)
    d.scatter_(1, idx, 1 - p.gather(1, idx))
    d = grad.float().unsqueeze(-1) * d
    if temperature != 1.0:
        d = d / torch.tensor(temperature, dtype=torch.float32, device=z.device)
    return d.to(torch.bfloat16)
