# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Nemotron Nano grouped expert MLP + fixed-order combine specification.

One differentiable reference for the `moe_experts_combine` row of RFC #434:
grouped up-projection GEMM with a relu-square epilogue, grouped down-projection,
a fixed-order weighted combine over the six route slots and the optional shared
expert.  This module fixes the *semantics*; it does not reproduce the kernels'
mid-split BF16 arithmetic.

FP64 inputs stay FP64 end to end, so `gradcheck` and the scalar oracle see the
idealized composition.  BF16/FP32 inputs follow the operator cast table: the
up-projection leaf rounds to BF16, the relu-square gate rounds to BF16, the
down-projection and the combine stay FP32, and the operator has exactly one
exit cast.  Accumulation is whatever `torch.matmul` chooses, so this is a
numerical baseline, not a bitwise contract.  The operator's own pairing - a BF16
payload with FP32 route weights - is accepted; other combinations must share the
payload dtype, so the oracle's precision is unambiguous.

Ordering, matching the kernel contract:

* experts ascend, i.e. the router `offsets` segments;
* combine slots ascend per token, in FP32;
* no reduction crosses token rows.
"""

from __future__ import annotations

import torch

from rl_engine.reference.moe.nemotron_router import dispatch_tokens

CONTRACT_VERSION = "nemotron-experts-sm90-v1"
# Bound every 32-bit index product: 6 * MAX_TOKENS packed rows, 2688 columns.
MAX_TOKENS = 65536


def _check_stacked(w_up: torch.Tensor, w_down: torch.Tensor, width: int) -> int:
    if w_up.ndim != 3 or w_down.ndim != 3:
        raise ValueError("stacked expert weights must be rank three")
    if w_up.shape[0] != w_down.shape[0]:
        raise ValueError("up and down weights must cover the same expert set")
    if w_up.shape[2] != width or w_down.shape[1] != width:
        raise ValueError("hidden widths must agree")
    if w_up.shape[1] != w_down.shape[2]:
        raise ValueError("intermediate widths must agree")
    if w_up.dtype != w_down.dtype or w_up.device != w_down.device:
        raise ValueError("stacked weights must share dtype and device")
    return w_up.shape[0]


def _check_shared(
    width: int, w_sh1: torch.Tensor | None, w_sh2: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if (w_sh1 is None) != (w_sh2 is None):
        raise ValueError("shared expert weights must be supplied as a pair")
    if w_sh1 is None or w_sh2 is None:
        raise ValueError("shared expert weights are absent")
    if w_sh1.ndim != 2 or w_sh2.ndim != 2:
        raise ValueError("shared expert weights must be rank two")
    if w_sh1.shape[1] != width or w_sh2.shape[0] != width:
        raise ValueError("shared expert hidden widths must agree")
    if w_sh1.shape[0] != w_sh2.shape[1]:
        raise ValueError("shared expert intermediate widths must agree")
    return w_sh1, w_sh2


def _leaf_round(value: torch.Tensor, leaf: torch.dtype | None) -> torch.Tensor:
    return value if leaf is None else value.to(leaf).to(value.dtype)


def routed_experts(
    packed: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Wide FP32/FP64 `[tokens, hidden]` grouped expert output before the exit cast.

    `packed` is the router payload, so this is the split form of
    `experts_reference` that a provider takes: the combine sums slots ascending
    and never reduces across token rows.
    """
    tokens, slots = weights.shape
    width = packed.shape[1]
    experts, intermediate = w_up.shape[0], w_up.shape[1]
    if packed.shape[0] != tokens * slots:
        raise ValueError("packed must hold one row per (token, slot)")
    compute = packed.dtype
    leaf = None if compute == torch.float64 else torch.bfloat16
    if offsets.numel() != experts + 1:
        raise ValueError("offsets must hold experts+1 exclusive prefix sums")
    if permutation.numel() != packed.shape[0]:
        raise ValueError("permutation must cover every packed row")
    up = torch.zeros((packed.shape[0], intermediate), dtype=compute, device=packed.device)
    for expert in range(experts):
        lo = int(offsets[expert])
        hi = int(offsets[expert + 1])
        if lo == hi:
            continue  # An empty expert contributes no rows and no reduction.
        up[lo:hi] = packed[lo:hi] @ w_up[expert].transpose(0, 1)
    gate = _leaf_round(up, leaf).relu()
    gate = _leaf_round(gate * gate, leaf)
    down = torch.zeros((packed.shape[0], width), dtype=compute, device=packed.device)
    for expert in range(experts):
        lo = int(offsets[expert])
        hi = int(offsets[expert + 1])
        if lo == hi:
            continue
        down[lo:hi] = gate[lo:hi] @ w_down[expert].transpose(0, 1)
    combined = down.index_select(0, permutation.argsort()).reshape(tokens, slots, width)
    routed = torch.zeros((tokens, width), dtype=compute, device=packed.device)
    for slot in range(slots):
        routed = routed + combined[:, slot] * weights[:, slot : slot + 1]
    return routed


def shared_experts(x: torch.Tensor, w_sh1: torch.Tensor, w_sh2: torch.Tensor) -> torch.Tensor:
    """Wide FP32/FP64 `[tokens, hidden]` shared expert output before the exit cast."""
    compute = x.dtype
    leaf = None if compute == torch.float64 else torch.bfloat16
    up = x @ w_sh1.transpose(0, 1)
    gate = _leaf_round(up, leaf).relu()
    gate = _leaf_round(gate * gate, leaf)
    return gate @ w_sh2.transpose(0, 1)


def experts_reference(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_sh1: torch.Tensor | None = None,
    w_sh2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `[tokens, hidden]` BF16 (or FP64 for FP64 inputs) expert output.

    `hidden_states` is the unweighted token payload, `expert_ids` the ascending
    selected experts and `weights` the router weights for those slots.  Dispatch
    reuses the router reference `dispatch_tokens`, so the packed order and the
    `offsets` prefix sums are identical to the operator contract.
    """
    if hidden_states.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("hidden_states and expert_ids must be rank two")
    if weights.shape != expert_ids.shape:
        raise ValueError("weights must match the selected expert shape")
    if expert_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("token counts must agree")
    if not hidden_states.is_floating_point() or not weights.is_floating_point():
        raise ValueError("hidden states and weights must be floating tensors")
    if expert_ids.dtype != torch.int64 or expert_ids.device != hidden_states.device:
        raise ValueError("expert_ids must be int64 on the input device")
    if weights.device != hidden_states.device:
        raise ValueError("weights must live on the input device")

    width = hidden_states.shape[1]
    experts = _check_stacked(w_up, w_down, width)
    if w_up.device != hidden_states.device:
        raise ValueError("stacked weights must live on the input device")
    shared = None
    if w_sh1 is not None or w_sh2 is not None:
        shared = _check_shared(width, w_sh1, w_sh2)
    if weights.dtype != hidden_states.dtype and not (
        hidden_states.dtype == torch.bfloat16 and weights.dtype == torch.float32
    ):
        # The operator's canonical pairing is a BF16 payload with FP32 route
        # weights; anything else must match the payload so the oracle's precision
        # is unambiguous (FP64 in, FP64 out).
        raise ValueError(
            "weights must be float32 with a bfloat16 payload, or match the payload dtype"
        )
    if w_up.dtype != hidden_states.dtype:
        raise ValueError("hidden states and stacked weights must share a dtype")

    exact = hidden_states.dtype == torch.float64
    compute = torch.float64 if exact else torch.float32
    permutation, offsets, packed = dispatch_tokens(hidden_states, expert_ids, experts)
    wide = routed_experts(
        packed.to(compute),
        weights.to(compute),
        permutation,
        offsets,
        w_up.to(compute),
        w_down.to(compute),
    )
    if shared is not None:
        wide = wide + shared_experts(
            hidden_states.to(compute), shared[0].to(compute), shared[1].to(compute)
        )
    return wide if exact else wide.to(torch.bfloat16)
