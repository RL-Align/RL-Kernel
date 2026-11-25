# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""BF16 dense linear on DeepGEMM (DSv4.1 design decision D12).

DSv4.1's dense weights are MXFP8 with 32 x 32 block scales, which DeepGEMM's SM90 FP8 GEMM
(128 x 128 scales) cannot take. D12 dequantizes them once at load to BF16 (exact) and runs
``bf16_gemm_nt``, which is batch-invariant at every DSv4.1 shape
(``scripts/probe_dsv41_deepgemm_bf16_bi.py``). The surface mirrors :class:`Fp8Linear`
(``forward`` / ``backward_dx``), so model code takes either.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm


class Bf16Linear:
    """Bias-free ``y = x @ W.T`` with ``W`` BF16 ``[N, K]``; dX on DeepGEMM too."""

    def __init__(self, weight: Tensor) -> None:
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise TypeError(f"weight must be BF16 [N, K], got {weight.dtype} {tuple(weight.shape)}")
        self._dg, self.capability = require_deepgemm(weight.device)
        self.weight = weight.contiguous()

    @classmethod
    def from_mxfp8(cls, q: Tensor, scale: Tensor) -> "Bf16Linear":
        """Checkpoint tensors: E4M3 ``[N, K]`` and E8M0 ``[N/32, K/32]`` (dequantized exactly)."""
        from rl_engine.models.deepseek_v41.loader import mxfp8_dequant

        return cls(mxfp8_dequant(q, scale))

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
