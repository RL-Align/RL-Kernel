# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""BF16 dense linear on DeepGEMM (DSv4.1 design decision D12).

DSv4.1's dense weights are MXFP8 with 32 x 32 block scales, which DeepGEMM's SM90 FP8 GEMM
(128 x 128 scales) cannot take. D12 dequantizes them once at load to BF16 (exact) and runs
``bf16_gemm_nt``, which is batch-invariant at every DSv4.1 shape
(``scripts/probe_dsv41_deepgemm_bf16_bi.py``). The surface mirrors :class:`Fp8Linear`
(``forward`` / ``backward_dx``), so model code takes either.

``act_quant=True`` rounds the activation to MXFP8 first (:func:`mxfp8_fake_quant`, the
release's training arithmetic: DeepSeek's ``linear`` quantizes every input of an MXFP8
weight per 32 values). An E4M3 value times a power of two is a BF16 value, so the BF16
GEMM then computes what an MXFP8 x MXFP8 GEMM would, up to its accumulation order.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm

_FP8_MAX_INV = torch.tensor(1.0 / 448.0, dtype=torch.float32)


def mxfp8_fake_quant(x: Tensor, block: int = 32) -> Tensor:
    """BF16 ``x`` rounded to MXFP8 along its last dim and back (exact ops, row-local).

    DeepSeek's ``act_quant(x, 32, "ue8m0", inplace=True)``: per block of ``block`` values,
    ``amax = max(|x|, 1e-4)``, ``s = 2^ceil(log2(amax * fp32(1/448)))`` from the bits,
    ``q = e4m3(clamp(x / s, -448, 448))`` (round to nearest even), result ``BF16(q * s)``.
    """
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {block}")
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // block, block)
    amax = xf.abs().amax(-1).clamp(min=1e-4)
    s = amax * _FP8_MAX_INV.to(x.device)
    e = ((s.view(torch.int32) + 0x7FFFFF) >> 23) & 0xFF  # biased exponent of the scale
    scale = (e << 23).view(torch.float32)
    inv = ((254 - e) << 23).view(torch.float32)  # 1 / scale, exact
    q = (xf * inv[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale[..., None]).to(torch.bfloat16).reshape(shape)


class Bf16Linear:
    """Bias-free ``y = x @ W.T`` with ``W`` BF16 ``[N, K]``; dX on DeepGEMM too."""

    def __init__(self, weight: Tensor, act_quant: bool = False) -> None:
        self.act_quant = act_quant
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise TypeError(f"weight must be BF16 [N, K], got {weight.dtype} {tuple(weight.shape)}")
        self._dg, self.capability = require_deepgemm(weight.device)
        self.weight = weight.contiguous()

    @classmethod
    def from_mxfp8(cls, q: Tensor, scale: Tensor, act_quant: bool = True) -> "Bf16Linear":
        """Checkpoint tensors: E4M3 ``[N, K]`` and E8M0 ``[N/32, K/32]`` (dequantized exactly);
        the activation is rounded to MXFP8 by default, as the release was trained."""
        from rl_engine.models.deepseek_v41.loader import mxfp8_dequant

        return cls(mxfp8_dequant(q, scale), act_quant)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    def _check_rows(self, x: Tensor, width: int, name: str) -> Tensor:
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(f"{name} must be [M, {width}], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16, got {x.dtype}")
        return x.contiguous()

    def forward(self, x: Tensor, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """``x`` BF16 ``[M, K]`` -> ``y`` ``[M, N]`` in ``out_dtype`` (BF16 or FP32; FP32
        accumulation either way)."""
        x = self._check_rows(x, self.in_features, "x")
        if self.act_quant:
            x = mxfp8_fake_quant(x)
        y = torch.empty(x.shape[0], self.out_features, dtype=out_dtype, device=x.device)
        if x.shape[0]:
            self._dg.bf16_gemm_nt(x, self.weight, y)
        return y

    __call__ = forward

    def backward_dx(self, dy: Tensor) -> Tensor:
        """``dy`` BF16 ``[M, N]`` -> ``dX`` BF16 ``[M, K]`` (``dy @ W``, DeepGEMM NN)."""
        dy = self._check_rows(dy, self.out_features, "dy")
        dx = torch.empty(dy.shape[0], self.in_features, dtype=torch.bfloat16, device=dy.device)
        if dy.shape[0]:
            self._dg.bf16_gemm_nn(dy, self.weight, dx)
        return dx
