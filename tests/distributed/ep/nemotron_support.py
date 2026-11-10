# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Stand-ins shared by the Nemotron EP tests, on the threaded transport and real groups.

The expert and combine use only IEEE add and multiply on single elements, so a row's
bits cannot depend on batching. Neither is the real Nemotron kernel.
"""

from __future__ import annotations

import torch

from rl_engine.distributed.ep import EpPlacement
from rl_engine.distributed.ep.nemotron import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    EpInvocation,
    NemotronEpExchange,
    ep_backward,
)

H = HIDDEN_SIZE


class LoopbackTransport:
    """EP size 1 on any device: the same code path as a group, with no peers."""

    rank, size, backend_id = 0, 1, "loopback"

    def __init__(self, device) -> None:
        self.device = torch.device(device)

    def gather_signature(self, signature: torch.Tensor) -> torch.Tensor:
        return signature[None]

    def all_to_all(self, output, send, recv_counts, send_counts) -> None:
        assert recv_counts == send_counts
        output.copy_(send)


def contiguous_placement(ep: int) -> EpPlacement:
    return EpPlacement.from_owners([e * ep // NUM_EXPERTS for e in range(NUM_EXPERTS)], ep)


def shuffled_placement(ep: int, seed: int = 7) -> EpPlacement:
    order = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed))
    return EpPlacement.from_owners((order % ep).tolist(), ep)


def make_exchange(placement: EpPlacement, transport=None, group=None, strict=True):
    return NemotronEpExchange(
        placement,
        model_version="model-v1",
        placement_version="place-v1",
        transport=transport,
        group=group,
        strict=strict,
    )


def global_inputs(total: int, seed: int = 0):
    """BF16 ``x``, ascending ``ids``, FP32 ``weights`` and BF16 ``target``, all per token."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(total, H, generator=g) * 2).to(torch.bfloat16)
    ids = torch.rand(total, NUM_EXPERTS, generator=g).argsort(dim=1)[:, :TOP_K].sort(dim=1).values
    weights = torch.rand(total, TOP_K, generator=g)
    target = (torch.randn(total, H, generator=g)).to(torch.bfloat16)
    return x, ids, weights, target


def expert_scales(seed: int = 11) -> torch.Tensor:
    return torch.rand(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)) + 0.5


def split_sizes(total: int, ep: int, empty_rank: int | None = 1) -> list[int]:
    """Uneven token counts per rank; ``empty_rank`` sources nothing when there are peers."""
    if ep == 1:
        return [total]
    active = [r for r in range(ep) if r != empty_rank]
    sizes = [0] * ep
    for index, rank in enumerate(active):
        sizes[rank] = total // len(active) + (1 if index < total % len(active) else 0)
    return sizes


def rank_slice(tensor: torch.Tensor, sizes: list[int], rank: int) -> torch.Tensor:
    start = sum(sizes[:rank])
    return tensor[start : start + sizes[rank]]


def row_experts(call) -> torch.Tensor:
    """Global expert id of each ``x_rows`` row, recovered from the local offsets."""
    counts = call.expert_offsets[1:] - call.expert_offsets[:-1]
    return torch.repeat_interleave(call.local_experts.to(counts.device), counts)


def stand_in_expert(x_rows: torch.Tensor, experts: torch.Tensor, scales: torch.Tensor):
    f = x_rows.to(torch.float32)
    return (f * scales.to(f.device)[experts][:, None] + f * f * 0.25).to(torch.bfloat16)


def fixed_order_combine(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Ascending-slot FP32 weighted sum of ``[T * 6, H]`` rows, cast once to BF16."""
    lanes = rows.view(-1, TOP_K, H).to(torch.float32)
    total = torch.zeros_like(lanes[:, 0])
    for slot in range(TOP_K):
        total = total + weights[:, slot : slot + 1] * lanes[:, slot]
    return total.to(torch.bfloat16)


def run_layer(ep_exchange, x, ids, weights, target, scales, invocation=None):
    """One forward through the exchange and combine; returns ``(call, out, loss)``."""
    call = ep_exchange.dispatch(x, ids, invocation or EpInvocation(0, 0, 0))
    with ep_exchange.exchange.guard():
        y_rows = stand_in_expert(call.x_rows, row_experts(call), scales)
    rows = call.return_outputs(y_rows)
    out = fixed_order_combine(rows, weights)
    loss = (out.to(torch.float32) * target.to(torch.float32)).sum()
    return call, out, loss


def train_rank(ep_exchange, x, ids, weights, target, scales):
    """Forward and backward for one rank; returns BF16 ``out`` and ``dx``."""
    x = x.clone().requires_grad_(True)
    call, out, loss = run_layer(ep_exchange, x, ids, weights, target, scales)
    ep_backward(ep_exchange, loss, [call])
    return out.detach(), x.grad


def reference_forward(x, ids, weights, scales) -> torch.Tensor:
    """No exchange: the per-slot expert and combine evaluated token by token."""
    total = []
    for token in range(x.shape[0]):
        row = x[token : token + 1].repeat(TOP_K, 1)
        y = stand_in_expert(row, ids[token], scales)
        total.append(fixed_order_combine(y, weights[token : token + 1]))
    return torch.cat(total) if total else x[:0]


def reference_dx_fp64(x, ids, weights, target, scales) -> torch.Tensor:
    """Closed form in FP64: ``sum_s w_s * (scale_e + x / 2) * target``, sharing no code."""
    f = x.to(torch.float64)
    t = target.to(torch.float64)
    a = scales.to(torch.float64)[ids]
    w = weights.to(torch.float64)
    return (w[:, :, None] * (a[:, :, None] + f[:, None] * 0.5)).sum(dim=1) * t


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Compare raw bit patterns; NaN and signed zero make float equality the wrong test."""
    return tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
