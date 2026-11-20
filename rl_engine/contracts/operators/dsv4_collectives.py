# SPDX-License-Identifier: Apache-2.0
"""Sealed rank/order contract for DSV4 TP, SP and DP collectives."""

from dataclasses import dataclass
import hashlib
import json

import torch


class Dsv4CollectiveError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Dsv4CollectiveError(message)


@dataclass(frozen=True)
class CollectivePhase:
    name: str
    group: str
    operation: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    denominator: int | None = None

    def __post_init__(self):
        require(isinstance(self.name, str) and bool(self.name), "named phase required")
        require(self.group in ("tp", "sp", "dp"), "unknown parallel group")
        require(
            self.operation in ("all_reduce", "all_gather", "reduce_scatter"),
            "unknown collective operation",
        )
        require(
            isinstance(self.shape, tuple)
            and bool(self.shape)
            and all(type(n) is int and n >= 0 for n in self.shape),
            "fixed shape required",
        )
        require(
            self.dtype in (torch.float32, torch.float16, torch.bfloat16),
            "unvalidated collective dtype",
        )
        require(
            self.operation != "reduce_scatter" or self.shape[0] > 0,
            "empty reduce-scatter requires separate signoff",
        )
        require(
            self.denominator is None
            or (
                self.group == "dp"
                and self.operation == "all_reduce"
                and self.dtype == torch.float32
                and type(self.denominator) is int
                and self.denominator > 0
            ),
            "normalization requires FP32 DP all-reduce",
        )


@dataclass(frozen=True)
class Dsv4CollectiveContract:
    global_ranks: tuple[int, ...]
    tp_groups: tuple[tuple[int, ...], ...]
    sp_groups: tuple[tuple[int, ...], ...]
    dp_groups: tuple[tuple[int, ...], ...]
    phases: tuple[CollectivePhase, ...]
    model_version: str
    graph: bool = False
    max_size_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        ranks = self.global_ranks
        require(
            isinstance(ranks, tuple)
            and bool(ranks)
            and all(type(r) is int and r >= 0 for r in ranks)
            and tuple(sorted(set(ranks))) == ranks,
            "global ranks must be sorted and unique",
        )
        require(
            isinstance(self.model_version, str) and bool(self.model_version),
            "explicit model/scale version required",
        )
        require(self.graph is False, "DSV4 collective Graph profile is not validated")
        require(
            type(self.max_size_bytes) is int and self.max_size_bytes > 0,
            "positive collective workspace budget required",
        )
        require(
            isinstance(self.phases, tuple)
            and bool(self.phases)
            and all(isinstance(phase, CollectivePhase) for phase in self.phases)
            and len({phase.name for phase in self.phases}) == len(self.phases),
            "unique ordered collective phases required",
        )
        for dimension in ("tp", "sp", "dp"):
            groups = getattr(self, f"{dimension}_groups")
            require(isinstance(groups, tuple) and bool(groups), f"{dimension} groups required")
            members = []
            for group in groups:
                require(
                    isinstance(group, tuple)
                    and len(group) in (1, 2, 4, 8)
                    and tuple(sorted(set(group))) == group,
                    f"{dimension} group must be sorted, unique and supported",
                )
                members.extend(group)
            require(tuple(sorted(members)) == ranks, f"{dimension} groups must partition ranks")
            require(
                len({len(group) for group in groups}) == 1,
                f"{dimension} groups must have a uniform size",
            )
        require(
            all(any(phase.group == dim for phase in self.phases) for dim in ("tp", "sp", "dp")),
            "all three dimensions need a phase",
        )
        for phase in self.phases:
            size = len(getattr(self, f"{phase.group}_groups")[0])
            require(
                phase.operation != "reduce_scatter" or phase.shape[0] % size == 0,
                f"{phase.name} leading dimension must divide {phase.group} size",
            )
            numel = 1
            for dim in phase.shape:
                numel *= dim
            require(
                numel * torch.empty((), dtype=phase.dtype).element_size() <= self.max_size_bytes,
                f"{phase.name} exceeds collective capacity",
            )

    @property
    def fingerprint(self):
        values = {
            "global_ranks": self.global_ranks,
            "groups": {dim: getattr(self, f"{dim}_groups") for dim in ("tp", "sp", "dp")},
            "phases": [
                (p.name, p.group, p.operation, p.shape, str(p.dtype), p.denominator)
                for p in self.phases
            ],
            "model_version": self.model_version,
            "graph": self.graph,
            "max_size_bytes": self.max_size_bytes,
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
