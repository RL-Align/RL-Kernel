# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant DSv4 MoE block, single GPU (EP = 1), forward.

    router (P3)  ->  route plan  ->  dispatch  ->  routed experts (P5, SM90)
                 ->  combine (slot order) + shared expert (DeepGEMM FP8) + residual

* Router: :class:`~rl_engine.models.deepseek_v4.router.Dsv4Router`.
* Plan / dispatch / combine: ``rl_engine.kernels.ops.pytorch.moe.route_plan``
  (rows sorted by (expert, token, slot); combine in slot order, P6 policy) with
  the CUDA slot-sum kernel.
* Routed experts: P5 ``Sm90FusedMoeMlp`` (MXFP8 activations x MXFP4 frozen
  weights on FP8 WGMMA, clamp-SwiGLU, ``* p_s`` once); needs the
  ``KERNEL_ALIGN_MOE_SM90=1`` build. Each row's bytes depend only on that row.
* Shared expert: ``DeepGemmSharedExpertProvider`` (FP8 DeepGEMM, clamped SwiGLU).

Activations are MXFP8-quantized once per token and the codes / scales gathered
per row; the quantization is row-local, so this equals quantizing the rows.
Expert parallelism (rows exchanged between ranks in the same plan order) and
the backward are the next steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.moe import combine_cuda, mxfp8_act_quant_fwd_cuda
from rl_engine.kernels.ops.pytorch.moe.route_plan import RoutePlan, build_plan, dispatch
from rl_engine.models.deepseek_v4.router import Dsv4Router
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor


def _routed_backend():
    from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

    return Sm90FusedMoeMlp()


def _shared_backend():
    from rl_engine.moe.backends.deepgemm_shared_expert import DeepGemmSharedExpertProvider

    return DeepGemmSharedExpertProvider()


@dataclass
class Dsv4MoE:
    router: Dsv4Router
    w1: MXTensor  # e2m1 [E, 2F, H] (gate rows then up rows), frozen
    w2: MXTensor  # e2m1 [E, H, F], frozen
    shared_fc1: Tensor | None = None  # BF16 [2F_s, H]
    shared_fc2: Tensor | None = None  # BF16 [H, F_s]
    path: str = "fused"  # P5 path: "fused" | "two_launch"
    check_finite: bool = True  # MXFP8 quant fail-closed check (one host sync; off for graphs)
    _routed: object = field(default=None, init=False, repr=False)
    _shared: object = field(default=None, init=False, repr=False)
    _shared_w: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._routed = _routed_backend()
        if self.shared_fc1 is not None:
            self._shared = _shared_backend()
            self._shared_w = self._shared.prepare(self.shared_fc1, self.shared_fc2)

    @property
    def num_experts(self) -> int:
        return int(self.w1.shape[0])

    def routed_rows(self, x: Tensor, weights: Tensor, plan: RoutePlan) -> Tensor:
        """Weighted expert outputs BF16 ``[P, H]`` in plan order."""
        from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

        rows, p_s, output_slot = dispatch(x, weights, plan)
        x_q = mxfp8_act_quant_fwd_cuda(x, check_finite=self.check_finite)
        token = plan.token
        rows_q = MXTensor(
            codes=x_q.codes.index_select(0, token),
            scales=x_q.scales.index_select(0, token),
            elem_format="e4m3",
            shape=(plan.rows, x.shape[1]),
        )
        batch = ExpertBatch(
            x=rows,
            expert_offsets=plan.expert_offsets,
            p_s=p_s,
            w1=self.w1,
            w2=self.w2,
            lora=None,
            output_slot=output_slot,
            numeric_profile=PROFILE,
        )
        return self._routed.forward(batch, rows_q, path=self.path)

    def forward(
        self, x: Tensor, residual: Tensor | None = None, input_ids: Tensor | None = None
    ) -> Tensor:
        """MoE block output BF16 ``[T, H]`` (``+ residual`` once when given)."""
        x = x.contiguous()
        with torch.no_grad():
            weights, ids = self.router(x, input_ids)
            plan = build_plan(ids, self.num_experts)
            y_rows = self.routed_rows(x, weights, plan)
            shared = None
            if self._shared is not None:
                shared, _ = self._shared.forward(x, self._shared_w)
            return combine_cuda(y_rows, plan, shared, residual)
