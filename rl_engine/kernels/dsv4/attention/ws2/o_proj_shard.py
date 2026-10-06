# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Whole-group wo_a / row-parallel wo_b ownership; execution is T07/P4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.kernels.dsv4.attention.contract import N_O_PROJ_GROUPS
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status


@dataclass(frozen=True)
class OProjShardPlan:
    """Validate local ownership; cross-rank coverage is checked by T07/P4."""

    tp_rank: int
    tp_world_size: int
    group_ids: tuple[int, ...]
    wo_b_split: str
    merge_order: str
    actual_collective: str | None
    mqa_kv_placement: str = "replicated"

    def __post_init__(self) -> None:
        if type(self.tp_world_size) is not int or self.tp_world_size not in {1, 2, 4, 8}:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                f"o-proj TP world size must be 1/2/4/8, got {self.tp_world_size}",
            )
        if type(self.tp_rank) is not int or not (0 <= self.tp_rank < self.tp_world_size):
            raise DSv4FailClosedError(DSv4Status.MISSING_RANK, f"tp_rank={self.tp_rank}")
        if not isinstance(self.group_ids, tuple) or any(
            type(g) is not int or g < 0 or g >= N_O_PROJ_GROUPS for g in self.group_ids
        ):
            raise DSv4FailClosedError(DSv4Status.SCHEMA_MISMATCH, f"group_ids={self.group_ids}")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise DSv4FailClosedError(
                DSv4Status.DUPLICATE_LOGICAL_OWNER, f"duplicate group_ids={self.group_ids}"
            )
        if self.group_ids != tuple(sorted(self.group_ids)):
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER, "group_ids must be ascending"
            )
        if not self.group_ids:
            raise DSv4FailClosedError(
                DSv4Status.MISSING_GLOBAL_VISIBILITY, "o-proj shard must own at least one group"
            )
        if self.tp_world_size == 1 and self.group_ids != tuple(range(N_O_PROJ_GROUPS)):
            raise DSv4FailClosedError(
                DSv4Status.MISSING_GLOBAL_VISIBILITY, "TP=1 must own all eight o-proj groups"
            )
        if self.wo_b_split not in {"none", "row"}:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY, f"unsupported wo_b_split={self.wo_b_split!r}"
            )
        expected_split = "none" if self.tp_world_size == 1 else "row"
        if self.wo_b_split != expected_split:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
                f"TP={self.tp_world_size} requires wo_b_split={expected_split!r}",
            )
        if self.merge_order != "group_index_ascending":
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                "o-proj merge order must be group_index_ascending",
            )
        if self.tp_world_size > 1 and not self.actual_collective:
            raise DSv4FailClosedError(
                DSv4Status.MISSING_PROVENANCE,
                "TP>1 o-proj requires actual collective provenance",
            )
        if self.mqa_kv_placement not in {"replicated", "sharded"}:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
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
