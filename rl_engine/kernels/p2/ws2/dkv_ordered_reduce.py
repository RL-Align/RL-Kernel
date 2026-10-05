# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP dKV reduction by logical global head tag. Rank/arrival order is forbidden."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from rl_engine.kernels.p2.contract import N_Q_HEADS
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status

CollectiveFn = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


def _validate_head_ids(ids: Tensor, *, count: int, device: torch.device) -> None:
    if (
        not isinstance(ids, Tensor)
        or ids.ndim != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or ids.numel() != count
        or ids.device != device
    ):
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"head tags must be a [{count}] integer tensor on {device}",
        )
    if bool(((ids < 0) | (ids >= N_Q_HEADS)).any()):
        raise P2FailClosedError(
            P2Status.AMBIGUOUS_LOGICAL_INDEX, "logical head tags must be in [0, 64)"
        )
    if ids.unique().numel() != ids.numel():
        raise P2FailClosedError(
            P2Status.DUPLICATE_LOGICAL_OWNER, "each logical head must have one contribution"
        )


def ordered_dkv_reduce(
    local_dkv: Tensor,
    logical_head_ids: Tensor,
    *,
    tp_world_size: int = 1,
    collective: CollectiveFn | None = None,
) -> Tensor:
    """Fold FP32 per-head [H_local,N,D] contributions by global head 0..63.

    Local tags must be ascending. TP>1's collective takes contributions and
    tags and gathers (contributions, tags) for all 64 heads, in any arrival
    order, without reducing them. The fold returns [N,D]. TP=1 also accepts
    an already packed [N,D] covering all heads and returns it by identity.
    """

    if type(tp_world_size) is not int or tp_world_size < 1:
        raise P2FailClosedError(P2Status.MISSING_RANK, f"tp_world_size={tp_world_size}")
    if (
        not isinstance(local_dkv, Tensor)
        or local_dkv.ndim not in (2, 3)
        or local_dkv.dtype != torch.float32
        or local_dkv.shape[-1] == 0
    ):
        raise P2FailClosedError(P2Status.SCHEMA_MISMATCH, "dKV must be FP32 [H_local,N,D] or [N,D]")
    if tp_world_size > 1 and local_dkv.ndim == 2:
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "TP>1 packed dKV has lost global head reduction order",
        )
    local_count = local_dkv.shape[0] if local_dkv.ndim == 3 else N_Q_HEADS
    _validate_head_ids(logical_head_ids, count=local_count, device=local_dkv.device)
    if bool((logical_head_ids[1:] < logical_head_ids[:-1]).any()):
        raise P2FailClosedError(
            P2Status.FORBIDDEN_ATOMIC_REDUCTION, "local logical head tags must be ascending"
        )
    if tp_world_size == 1:
        if local_count != N_Q_HEADS:
            raise P2FailClosedError(
                P2Status.MISSING_GLOBAL_VISIBILITY, "TP=1 must own all 64 logical heads"
            )
        if local_dkv.ndim == 2:
            return local_dkv
        contributions, head_ids = local_dkv, logical_head_ids
    else:
        if collective is None:
            raise P2FailClosedError(
                P2Status.UNSUPPORTED_CAPABILITY, "TP>1 requires a tagged per-head gather"
            )
        gathered = collective(local_dkv, logical_head_ids)
        if not isinstance(gathered, tuple) or len(gathered) != 2:
            raise P2FailClosedError(
                P2Status.SCHEMA_MISMATCH, "collective must return (contributions, head tags)"
            )
        contributions, head_ids = gathered
        if (
            not isinstance(contributions, Tensor)
            or contributions.ndim != 3
            or contributions.shape[1:] != local_dkv.shape[1:]
            or contributions.dtype != local_dkv.dtype
            or contributions.device != local_dkv.device
        ):
            raise P2FailClosedError(
                P2Status.SCHEMA_MISMATCH,
                "gathered dKV shape, dtype and device must match local dKV",
            )
        if contributions.shape[0] != N_Q_HEADS:
            raise P2FailClosedError(
                P2Status.MISSING_GLOBAL_VISIBILITY, "gather must contain all 64 logical heads"
            )
    _validate_head_ids(head_ids, count=N_Q_HEADS, device=local_dkv.device)
    order = torch.argsort(head_ids)
    ordered = contributions.index_select(0, order)
    if tp_world_size > 1 and not torch.equal(
        ordered.index_select(0, logical_head_ids.long()), local_dkv
    ):
        raise P2FailClosedError(
            P2Status.CORRUPT_ARTIFACT, "gather changed the contributions owned by this rank"
        )
    reduced = local_dkv.new_zeros(local_dkv.shape[-2:])
    for head in range(N_Q_HEADS):
        reduced = reduced + ordered[head]
    return reduced
