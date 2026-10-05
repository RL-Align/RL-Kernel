# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP dKV reduction by logical global head tag. Rank/arrival order is forbidden."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from rl_engine.kernels.p2.contract import N_Q_HEADS
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status


CollectiveFn = Callable[[Tensor], Tensor]


def ordered_dkv_reduce(
    local_dkv: Tensor,
    logical_head_ids: Tensor,
    *,
    tp_world_size: int = 1,
    collective: CollectiveFn | None = None,
) -> Tensor:
    """Reduce per-head dKV contributions in logical head order 0..63.

    ``local_dkv`` is [H_local, N, D] or already packed [N, D] when heads were
    summed locally. ``logical_head_ids`` lists the global head tags owned by
    this rank, ascending. TP=1 is identity. TP>1 requires an explicit
    collective; missing collective is UNSUPPORTED_CAPABILITY, not PASS.
    """

    if tp_world_size < 1:
        raise P2FailClosedError(P2Status.MISSING_RANK, f"tp_world_size={tp_world_size}")
    if tp_world_size == 1:
        return local_dkv
    if collective is None:
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "TP>1 dKV reduction requires a P4/T01 ordered collective; refusing silent fallback",
        )
    if logical_head_ids.dtype not in (torch.int32, torch.int64):
        raise P2FailClosedError(P2Status.SCHEMA_MISMATCH, "logical_head_ids must be integer")
    if logical_head_ids.numel() and bool((logical_head_ids[1:] < logical_head_ids[:-1]).any()):
        raise P2FailClosedError(
            P2Status.FORBIDDEN_ATOMIC_REDUCTION,
            "logical head tags must be pre-sorted; reduction order is not arrival order",
        )
    if int(logical_head_ids.numel()) and (
        int(logical_head_ids.min()) < 0 or int(logical_head_ids.max()) >= N_Q_HEADS
    ):
        raise P2FailClosedError(
            P2Status.AMBIGUOUS_LOGICAL_INDEX,
            "logical head tags must be in [0, 64)",
        )
    reduced = collective(local_dkv)
    if reduced.shape != local_dkv.shape:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"collective changed dKV shape {tuple(local_dkv.shape)} -> {tuple(reduced.shape)}",
        )
    return reduced
