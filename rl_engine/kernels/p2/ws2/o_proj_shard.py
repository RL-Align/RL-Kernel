# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP sharding contract for grouped wo_a / wo_b. Collectives are T07/P4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.kernels.p2.contract import N_O_PROJ_GROUPS
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status


@dataclass(frozen=True)
class OProjShardPlan:
    tp_rank: int
    tp_world_size: int
    group_ids: tuple[int, ...]
    wo_b_split: str
    merge_order: str
    actual_collective: str | None
    mqa_kv_placement: str = "replicated"

    def __post_init__(self) -> None:
        if self.tp_world_size not in {1, 2, 4, 8}:
            raise P2FailClosedError(
                P2Status.UNSUPPORTED_CAPABILITY,
                f"o-proj TP world size must be 1/2/4/8, got {self.tp_world_size}",
            )
        if not (0 <= self.tp_rank < self.tp_world_size):
            raise P2FailClosedError(P2Status.MISSING_RANK, f"tp_rank={self.tp_rank}")
        if any(g < 0 or g >= N_O_PROJ_GROUPS for g in self.group_ids):
            raise P2FailClosedError(P2Status.SCHEMA_MISMATCH, f"group_ids={self.group_ids}")
        if self.merge_order != "group_index_ascending":
            raise P2FailClosedError(
                P2Status.INVALID_CANDIDATE_ORDER,
                "o-proj merge order must be group_index_ascending",
            )
        if self.tp_world_size > 1 and not self.actual_collective:
            raise P2FailClosedError(
                P2Status.MISSING_PROVENANCE,
                "TP>1 o-proj requires actual collective provenance",
            )
        if self.mqa_kv_placement not in {"replicated", "sharded"}:
            raise P2FailClosedError(
                P2Status.SCHEMA_MISMATCH,
                f"mqa_kv_placement must be replicated|sharded, got {self.mqa_kv_placement!r}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "group_ids": list(self.group_ids),
            "wo_b_split": self.wo_b_split,
            "merge_order": self.merge_order,
            "actual_collective": self.actual_collective,
            "mqa_kv_placement": self.mqa_kv_placement,
        }


def tp1_plan() -> OProjShardPlan:
    return OProjShardPlan(
        tp_rank=0,
        tp_world_size=1,
        group_ids=tuple(range(N_O_PROJ_GROUPS)),
        wo_b_split="none",
        merge_order="group_index_ascending",
        actual_collective=None,
        mqa_kv_placement="replicated",
    )
