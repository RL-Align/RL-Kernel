# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""LoRA adapters on frozen DSv4 FP8 linears, numeric profile ``dsv4-lora-v1``.

Unmerged (user decision 2026-11-11): the rollout must run the same forward, so the FP8
base weights are never touched. With ``A`` BF16 ``[r, in]``, ``B`` BF16 ``[out, r]`` and
``scale = alpha / r``::

    h = BF16(x @ A^T)                     DeepGEMM bf16_gemm_nt, FP32 accumulate, one round
    d = h @ B^T                           DeepGEMM bf16_gemm_nt, FP32
    y = BF16(FP32(y_base) + scale * d)    two correctly rounded FP32 steps, one BF16 round

The rank is padded with zeros to a multiple of 64 for DeepGEMM: the zero rows of ``A`` and
columns of ``B`` add exact zeros, so the padding changes no byte. Each row depends only
on its own input, so the forward is batch-invariant
(``tests/models/test_dsv4_lora.py``).

Backward (``g = dL/dy``; the base weight is frozen), in FP32::

    dd = scale * g
    dB = dd^T h        dh = dd B        dA = dh^T x        dx = dh A

``dA`` / ``dB`` reduce over tokens with ``torch.mm``. That is deterministic run to run
for fixed shapes, not batch-invariant (as for any weight gradient).

DSv4 attention targets (user decision): ``wq_a``, ``wkv``, ``wq_b`` and ``wo_b``.

MoE (:class:`MoeLora`, Megatron-Bridge's default ``share_expert_adapters=True``): one
``fc1`` / ``fc2`` adapter pair shared by all routed experts, and one pair for the shared
expert. Routed ``fc1``: ``z = FP32 z_base + lora_delta(x)`` before the clamp-SwiGLU (the
delta depends only on the token, so it is computed once per token and gathered to its 6
rows); routed ``fc2``: ``y = BF16(FP32(y_base) + lora_delta(h))`` with ``h`` the BF16
SwiGLU output (``p_s`` already applied).

Expert parallelism: as in Megatron-Bridge (the routed adapter is broadcast from EP rank 0
and its gradients are all-reduced over the EP group), the routed adapter is one global
adapter. The token's rank computes ``u1 = BF16(x A1^T)`` and sends it with the MXFP8 row;
the expert's rank applies ``B1``. That gives the EP = 1 bytes (the GEMMs are row-invariant).
The routed adapters' gradients are summed over the EP group inside the backward; the
shared-expert adapter is an ordinary data-parallel parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PROFILE = "dsv4-lora-v1"
TARGETS = ("wq_a", "wkv", "wq_b", "wo_b")
_RANK_ALIGN = 64


@dataclass
class LoraAdapter:
    a: Tensor  # BF16 [r, in], trainable
    b: Tensor  # BF16 [out, r], trainable
    alpha: float

    def __post_init__(self) -> None:
        if self.a.dtype != torch.bfloat16 or self.b.dtype != torch.bfloat16:
            raise TypeError("LoRA A and B must be BF16")
        if self.a.dim() != 2 or self.b.dim() != 2 or self.b.shape[1] != self.a.shape[0]:
            raise ValueError(f"A {tuple(self.a.shape)} and B {tuple(self.b.shape)} do not match")

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])

    @property
    def scale(self) -> float:
        return float(self.alpha) / self.rank

    @property
    def in_features(self) -> int:
        return int(self.a.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.b.shape[0])

    @classmethod
    def init(cls, in_features, out_features, rank, alpha, device="cuda", seed=0, zero_b=True):
        """Kaiming-style ``A``, zero ``B`` (the usual start: the adapter adds nothing)."""
        g = torch.Generator(device=device).manual_seed(seed)
        a = (torch.randn(rank, in_features, generator=g, device=device) / in_features**0.5).to(
            torch.bfloat16
        )
        if zero_b:
            b = torch.zeros(out_features, rank, dtype=torch.bfloat16, device=device)
        else:
            b = (torch.randn(out_features, rank, generator=g, device=device) / rank**0.5).to(
                torch.bfloat16
            )
        return cls(a, b, alpha)


def _padded(ad: LoraAdapter) -> tuple[Tensor, Tensor]:
    r = ad.rank
    rp = -(-r // _RANK_ALIGN) * _RANK_ALIGN
    if rp == r:
        return ad.a.detach().contiguous(), ad.b.detach().contiguous()
    a = torch.zeros(rp, ad.in_features, dtype=torch.bfloat16, device=ad.a.device)
    b = torch.zeros(ad.out_features, rp, dtype=torch.bfloat16, device=ad.b.device)
    a[:r] = ad.a.detach()
    b[:, :r] = ad.b.detach()
    return a, b


def _gemm_nt(x: Tensor, w: Tensor) -> Tensor:
    from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm

    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        dg, _ = require_deepgemm(x.device)
        dg.bf16_gemm_nt(x.contiguous(), w, out)
    return out


def lora_hidden(x: Tensor, ad: LoraAdapter) -> Tensor:
    """``h = BF16(x @ A^T)`` ``[T, r]`` (the unpadded rank)."""
    a, _ = _padded(ad)
    return _gemm_nt(x, a)[:, : ad.rank].to(torch.bfloat16)


def lora_delta(x: Tensor, ad: LoraAdapter) -> Tensor:
    """``scale * (BF16(x A^T) B^T)`` FP32 ``[T, out]``; ``x`` BF16 ``[T, in]``."""
    if x.dtype != torch.bfloat16 or x.shape[-1] != ad.in_features:
        raise ValueError(f"x must be BF16 [T, {ad.in_features}]")
    a, b = _padded(ad)
    h = _gemm_nt(x, a).to(torch.bfloat16)
    return _gemm_nt(h, b) * ad.scale


def lora_up(h: Tensor, ad: LoraAdapter) -> Tensor:
    """``scale * (h B^T)`` FP32 for ``h = lora_hidden(x)`` (BF16 ``[n, r]``): the second half
    of :func:`lora_delta`, byte-equal to it row for row."""
    _, b = _padded(ad)
    rp = b.shape[1]
    if h.shape[1] != rp:
        hp = torch.zeros(h.shape[0], rp, dtype=torch.bfloat16, device=h.device)
        hp[:, : h.shape[1]] = h
        h = hp
    return _gemm_nt(h, b) * ad.scale


def lora_apply(y_base: Tensor, x: Tensor, ad: LoraAdapter) -> Tensor:
    """``BF16(FP32(y_base) + scale * (BF16(x A^T) B^T))``; ``x`` BF16 ``[T, in]``."""
    return (y_base.float() + lora_delta(x, ad)).to(torch.bfloat16)


def lora_backward(dy: Tensor, x: Tensor, ad: LoraAdapter) -> tuple[Tensor, Tensor, Tensor]:
    """``(dx FP32 [T, in], dA FP32 [r, in], dB FP32 [out, r])`` from ``dy`` ``[T, out]``."""
    h = lora_hidden(x, ad).float()
    dd = dy.float() * ad.scale
    db = dd.t() @ h
    dh = dd @ ad.b.detach().float()
    da = dh.t() @ x.float()
    dx = dh @ ad.a.detach().float()
    return dx, da, db


@dataclass
class MoeLora:
    """DSv4 MoE LoRA: routed ``fc1`` (H -> 2F, gate rows then up rows) and ``fc2``
    (F -> H) shared by every routed expert; optional shared-expert ``fc1`` / ``fc2``."""

    fc1: LoraAdapter | None = None
    fc2: LoraAdapter | None = None
    shared_fc1: LoraAdapter | None = None
    shared_fc2: LoraAdapter | None = None

    def adapters(self) -> list[tuple[str, LoraAdapter]]:
        names = ("fc1", "fc2", "shared_fc1", "shared_fc2")
        return [(n, getattr(self, n)) for n in names if getattr(self, n) is not None]
