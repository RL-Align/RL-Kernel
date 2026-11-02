# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Routed-expert forward over expert parallelism (tasks 37 and 38).

The raw BF16 input is what travels. The receiver keeps those exact bytes for the
LoRA path and derives the MXFP8 activation locally for the base-weight path, so
quantization runs on the expert rank and its row-local scales cannot depend on the
EP size. Failures in between go through ``EpExchange.guard`` so peers do not block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_engine.distributed.ep import EpCombineMapping, EpExchange, EpPlacement, EpRoute
from rl_engine.moe.contract import ExpertBatch, LoRAParams
from rl_engine.moe.mx_format import MXTensor, mx_quantize

_INT32_MAX = 2**31 - 1


@dataclass(frozen=True, eq=False)
class EpExpertInput:
    """Received rows in expert order. ``output_slot`` is each row's arrival position."""

    x: torch.Tensor  # raw BF16 [M, H], byte-identical to the source; for LoRA
    x_q: MXTensor  # e4m3 [M, H], derived from x on this rank; for the base weights
    p_s: torch.Tensor  # FP32 [M]
    expert_offsets: torch.Tensor  # int32 [E_local + 1]
    output_slot: torch.Tensor  # int32 [M]


def select_local_weights(w: MXTensor, placement: EpPlacement, rank: int) -> MXTensor:
    """Global ``[E, ...]`` weights to this rank's local expert order."""
    ids = placement.local_expert_ids(rank).to(w.codes.device)
    return MXTensor(
        w.codes.index_select(0, ids),
        w.scales.index_select(0, ids),
        w.elem_format,
        (ids.numel(), *w.shape[1:]),
        w.packing,
    )


def receive_expert_input(
    exchange: EpExchange, route: EpRoute, x: torch.Tensor, p_s: torch.Tensor
) -> EpExpertInput:
    """Send ``x`` BF16 ``[T, H]`` and ``p_s`` FP32 ``[T, top_k]``; quantize on arrival."""
    with exchange.guard():
        if x.dtype != torch.bfloat16:
            raise TypeError(f"x must be BF16, got {x.dtype}")
        if p_s.dtype != torch.float32:
            raise TypeError(f"p_s must be FP32, got {p_s.dtype}")
    rows = exchange.dispatch_tokens(route, x)
    weights = exchange.dispatch_slots(route, p_s)
    with exchange.guard():
        recv = route.recv
        if recv.num_rows > _INT32_MAX:
            raise OverflowError(f"{recv.num_rows} rows do not fit the kernels' int32 offsets")
        x_q = mx_quantize(rows, "e4m3")
        return EpExpertInput(
            rows,
            x_q,
            weights,
            recv.expert_offsets.to(torch.int32),
            recv.to_expert.to(torch.int32),
        )


def routed_expert_forward_ep(
    exchange: EpExchange,
    route: EpRoute,
    provider,
    x: torch.Tensor,
    p_s: torch.Tensor,
    w1: MXTensor,
    w2: MXTensor,
    *,
    lora: LoRAParams | None = None,
    weight_fingerprint: str = "",
) -> EpCombineMapping:
    """Dispatch, run the provider on local experts, return one row per slot for P6."""
    received = receive_expert_input(exchange, route, x, p_s)
    with exchange.guard():
        batch = ExpertBatch(
            x=received.x,
            expert_offsets=received.expert_offsets,
            p_s=received.p_s,
            w1=w1,
            w2=w2,
            lora=lora,
            output_slot=received.output_slot,
            numeric_profile=provider.numeric_profile,
            weight_fingerprint=weight_fingerprint,
        )
        outputs = provider.forward(batch, received.x_q)
    return exchange.return_outputs(route, outputs)


__all__ = [
    "EpExpertInput",
    "receive_expert_input",
    "routed_expert_forward_ep",
    "select_local_weights",
]
