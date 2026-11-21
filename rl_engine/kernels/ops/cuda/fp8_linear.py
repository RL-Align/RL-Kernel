# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 dense FP8 linear on DeepGEMM (SM90), matching the vLLM rollout.

``y = x @ W.T`` for a weight ``W [N, K]`` stored as FP8 E4M3 with 128x128
blocks and UE8M0-valued FP32 scales (the DSv4 checkpoint format), as vLLM runs
the attention projections (``fused_wqa_wkv``, ``wq_b``, ``wo_b``, indexer
``wq_b``) and the shared expert::

    x (BF16 [M, K]) -> fp8_group_quant (1x128, UE8M0) -> DeepGEMM fp8_gemm_nt -> y (BF16 [M, N])

Backward, on the kernels ``scripts/probe_dsv4_deepgemm_bi.py`` verified (D3):

* :meth:`Fp8Linear.backward_dx` -- ``dX = dY @ W``: the same NT kernel with
  ``W.T`` precomputed once (block-quantizing ``W.T`` gives the same bytes as
  transposing the block-quantized ``W``); batch-invariant like the forward.
* :meth:`Fp8Linear.backward_dw` -- ``dW = dY.T @ X``: the 1D1D wgrad kernel
  (``recipe=(1, 1, 128)``) on token-direction quantized operands, FP32 output,
  optionally accumulated into an existing gradient; run-to-run deterministic.
  Tokens are zero-padded to a multiple of 128 by the quantizer.

Every step is row-local or one of those GEMMs, so forward and dX are
batch-invariant. Fail-closed (rule 7): construction raises when DeepGEMM is not
the pinned build on SM90 with a CUDA 12.8+ JIT, or ``rl_engine._C`` lacks the
quantizer; there is no fallback to another GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.quant import Fp8GroupQuantCudaOp

BLOCK = 128
WGRAD_RECIPE = (1, 1, 128)


@dataclass(frozen=True)
class Fp8BlockWeight:
    """``[N, K]`` FP8 E4M3 weight with FP32 scales ``[N/128, K/128]``."""

    q: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if self.q.dtype != torch.float8_e4m3fn or self.scale.dtype != torch.float32:
            raise TypeError(
                f"expected (FP8 E4M3, FP32 scale), got ({self.q.dtype}, {self.scale.dtype})"
            )
        if self.q.ndim != 2 or self.q.shape[0] % BLOCK or self.q.shape[1] % BLOCK:
            raise ValueError(f"weight shape {tuple(self.q.shape)} must be multiples of {BLOCK}")
        blocks = (self.q.shape[0] // BLOCK, self.q.shape[1] // BLOCK)
        if tuple(self.scale.shape) != blocks:
            raise ValueError(f"scale shape {tuple(self.scale.shape)} != {blocks}")

    @property
    def shape(self) -> torch.Size:
        return self.q.shape

    def pair(self) -> tuple[Tensor, Tensor]:
        return self.q, self.scale

    def t(self) -> "Fp8BlockWeight":
        return Fp8BlockWeight(self.q.t().contiguous(), self.scale.t().contiguous())

    def dequantize(self) -> Tensor:
        """BF16 ``W`` (exact: an E4M3 value times a power of two fits BF16)."""
        n, k = self.q.shape
        blocks = self.q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        return (blocks * self.scale[:, None, :, None]).view(n, k).to(torch.bfloat16)


def quantize_weight(w: Tensor) -> Fp8BlockWeight:
    """128x128-block FP8 with UE8M0-valued scales, the DSv4 checkpoint format."""
    from deep_gemm.utils import per_block_cast_to_fp8

    if w.ndim != 2 or w.shape[0] % BLOCK or w.shape[1] % BLOCK:
        raise ValueError(f"weight shape {tuple(w.shape)} must be multiples of {BLOCK}")
    q, scale = per_block_cast_to_fp8(w.contiguous(), use_ue8m0=True)
    return Fp8BlockWeight(q, scale.contiguous())


class Fp8Linear:
    """Bias-free DSv4 FP8 linear ``y = x @ W.T`` with dX and dW on DeepGEMM."""

    def __init__(self, weight: Fp8BlockWeight) -> None:
        self._dg, self.capability = require_deepgemm(weight.q.device)
        self._quant = Fp8GroupQuantCudaOp()
        self.weight = weight
        self._weight_t: Fp8BlockWeight | None = None

    @classmethod
    def from_bf16(cls, w: Tensor) -> "Fp8Linear":
        return cls(quantize_weight(w))

    @classmethod
    def from_fp8(cls, q: Tensor, scale: Tensor) -> "Fp8Linear":
        """Checkpoint tensors: FP8 ``[N, K]`` and FP32 scales ``[N/128, K/128]``."""
        return cls(Fp8BlockWeight(q.contiguous(), scale.contiguous()))

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def weight_t(self) -> Fp8BlockWeight:
        """``W.T`` for dgrad, built on first use and kept with the weight."""
        if self._weight_t is None:
            self._weight_t = self.weight.t()
        return self._weight_t

    # Kernels ---------------------------------------------------------------
    def _check_rows(self, x: Tensor, width: int, name: str) -> Tensor:
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(f"{name} must be [M, {width}], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16, got {x.dtype}")
        return x.contiguous()

    def _gemm_nt(self, a: tuple[Tensor, Tensor], w: Fp8BlockWeight) -> Tensor:
        out = torch.empty(a[0].shape[0], w.shape[0], device=a[0].device, dtype=torch.bfloat16)
        if a[0].shape[0]:
            self._dg.fp8_gemm_nt(a, w.pair(), out, disable_ue8m0_cast=True)
        return out

    def _linear(self, x: Tensor, w: Fp8BlockWeight) -> Tensor:
        if not x.shape[0]:
            return torch.empty(0, w.shape[0], device=x.device, dtype=torch.bfloat16)
        return self._gemm_nt(self._quant(x), w)

    # Public surface ----------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """``x`` BF16 ``[M, K]`` -> ``y`` BF16 ``[M, N]``."""
        return self._linear(self._check_rows(x, self.in_features, "x"), self.weight)

    __call__ = forward

    def forward_quantized(self, a: tuple[Tensor, Tensor]) -> Tensor:
        """Already-quantized activations ``(FP8 [M, K], FP32 scales)`` from a fused kernel."""
        if a[0].ndim != 2 or a[0].shape[1] != self.in_features:
            raise ValueError(f"activation must be [M, {self.in_features}], got {tuple(a[0].shape)}")
        return self._gemm_nt(a, self.weight)

    def backward_dx(self, dy: Tensor) -> Tensor:
        """``dy`` BF16 ``[M, N]`` -> ``dX`` BF16 ``[M, K]`` (FP8 dgrad, batch-invariant)."""
        return self._linear(self._check_rows(dy, self.out_features, "dy"), self.weight_t)

    def backward_dw(self, dy: Tensor, x: Tensor, out: Tensor | None = None) -> Tensor:
        """``dW = dy.T @ x`` as FP32 ``[N, K]``; accumulates into ``out`` when given.

        Deterministic run to run, not batch-invariant: it reduces over tokens.
        """
        dy = self._check_rows(dy, self.out_features, "dy")
        x = self._check_rows(x, self.in_features, "x")
        if dy.shape[0] != x.shape[0]:
            raise ValueError(f"dy has {dy.shape[0]} tokens, x has {x.shape[0]}")
        shape = (self.out_features, self.in_features)
        if out is None:
            out = torch.zeros(shape, device=x.device, dtype=torch.float32)
        elif out.dtype != torch.float32 or tuple(out.shape) != shape or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous FP32 {shape} tensor")
        if not x.shape[0]:
            return out
        a = self._quant.forward_tokens(dy)  # [N, T_pad]
        b = self._quant.forward_tokens(x)  # [K, T_pad]
        self._dg.fp8_gemm_nt(a, b, out, c=out, recipe=WGRAD_RECIPE, disable_ue8m0_cast=True)
        return out
