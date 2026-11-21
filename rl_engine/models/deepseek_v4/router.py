# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant DSv4 MoE router (P3): gate GEMM, top-6 / hash routing, weights.

``logits = x @ W.T`` runs DeepGEMM ``bf16_gemm_nt`` with FP32 output (BF16 ``x``
``[T, 4096]`` and gate ``W`` ``[256, 4096]``); DeepGEMM picks one tile and no
split-K for every M, so the logits are batch-invariant (design doc D4; vLLM's BI
mode uses cuBLAS ``torch.mm`` here, which we do not rely on). Routing and the
weights follow the fixed contract in ``rl_engine.kernels.ops.pytorch.router``.

The gate and the correction bias are frozen; autograd returns ``dL/dx`` only,
from ``dL/dweights`` (the expert ids carry no gradient).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.router import route_backward_dx_cuda, route_topk_cuda


def gate_logits(x: Tensor, gate_weight: Tensor) -> Tensor:
    """FP32 ``[T, E]`` gate logits on DeepGEMM (batch-invariant)."""
    if x.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        raise TypeError("the DSv4 gate GEMM takes BF16 x and BF16 weights")
    dg, _ = require_deepgemm(x.device)
    out = torch.empty(x.shape[0], gate_weight.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        dg.bf16_gemm_nt(x.contiguous(), gate_weight, out)
    return out


class _RouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, bias, input_ids, tid2eid, scale):
        logits = gate_logits(x, gate_weight)
        weights, ids = route_topk_cuda(logits, bias, scale, input_ids=input_ids, tid2eid=tid2eid)
        ctx.save_for_backward(logits, ids, gate_weight)
        ctx.scale = scale
        ctx.x_dtype = x.dtype
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, d_weights, _d_ids):
        logits, ids, gate_weight = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            if d_weights is None:
                d_weights = torch.zeros_like(ids, dtype=torch.float32)
            dx = route_backward_dx_cuda(
                logits, ids, d_weights, gate_weight, ctx.scale, out_dtype=ctx.x_dtype
            )
        return dx, None, None, None, None, None


@dataclass
class Dsv4Router:
    """One layer's router. ``tid2eid`` set means a hash layer (``input_ids`` required)."""

    gate_weight: Tensor  # BF16 [E, H]
    bias: Tensor | None = None  # FP32 [E], scored layers
    tid2eid: Tensor | None = None  # int32/int64 [vocab, 6], hash layers
    routed_scaling_factor: float = 1.5

    def __call__(self, x: Tensor, input_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``(weights FP32 [T, 6], ids int32 [T, 6])``; differentiable in ``x``."""
        if self.tid2eid is not None and input_ids is None:
            raise ValueError("DSv4 hash routing requires input_ids")
        if self.tid2eid is None and self.bias is None:
            raise ValueError("a scored DSv4 router needs e_score_correction_bias")
        return _RouterFunction.apply(
            x,
            self.gate_weight,
            self.bias,
            input_ids if self.tid2eid is not None else None,
            self.tid2eid,
            float(self.routed_scaling_factor),
        )
