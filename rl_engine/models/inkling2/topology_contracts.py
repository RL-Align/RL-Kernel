# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini distributed topology and feasibility contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TopologyContractError(ValueError):
    """Raised when a distributed topology cannot satisfy the Inkling contract."""


@dataclass(frozen=True)
class InklingTopology:
    tensor_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: int = 1
    pipeline_parallel: int = 1
    world_size: int = 1

    def validate(self) -> None:
        values = {
            "tensor_parallel": self.tensor_parallel,
            "expert_parallel": self.expert_parallel,
            "context_parallel": self.context_parallel,
            "sequence_parallel": self.sequence_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "world_size": self.world_size,
        }
        if any(value <= 0 for value in values.values()):
            raise TopologyContractError("topology sizes must be positive")
        expected_world = (
            self.tensor_parallel
            * self.expert_parallel
            * self.context_parallel
            * self.sequence_parallel
            * self.pipeline_parallel
        )
        if expected_world != self.world_size:
            raise TopologyContractError(
                f"world_size={self.world_size} does not match topology product={expected_world}"
            )
        if self.tensor_parallel not in {1, 2, 4, 8}:
            raise TopologyContractError("tensor parallel size must be 1, 2, 4, or 8")
        if self.expert_parallel not in {1, 2, 4, 8}:
            raise TopologyContractError("expert parallel size must be 1, 2, 4, or 8")

    def to_dict(self) -> dict[str, int]:
        return {
            "tensor_parallel": self.tensor_parallel,
            "expert_parallel": self.expert_parallel,
            "context_parallel": self.context_parallel,
            "sequence_parallel": self.sequence_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class TopologyEvidence:
    framework: str
    platform: str
    topology: InklingTopology
    attention_backend: str
    router_backend: str
    cache_isolation: bool
    reduction_order: str
    source: str

    def validate(self) -> None:
        self.topology.validate()
        if not self.framework or not self.platform or not self.source:
            raise TopologyContractError("framework, platform, and source must be nonempty")
        if not self.attention_backend or not self.router_backend or not self.reduction_order:
            raise TopologyContractError("runtime backend fields must be nonempty")
        if not self.cache_isolation:
            raise TopologyContractError("topology evidence requires per-visit cache isolation")


@dataclass(frozen=True)
class FeasibilityReport:
    item: str
    status: str
    blockers: tuple[str, ...] = ()
    evidence: TopologyEvidence | None = None

    @property
    def ready(self) -> bool:
        return self.status == "qualified" and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "status": self.status,
            "blockers": list(self.blockers),
            "evidence": None if self.evidence is None else self.evidence.__dict__,
        }


def qualify_tp_ep(
    topology: InklingTopology, *, evidence: TopologyEvidence | None
) -> FeasibilityReport:
    """Qualify the CUDA TP2/TP4 and EP route contract after runtime readback."""
    try:
        topology.validate()
    except TopologyContractError as error:
        return FeasibilityReport("INK-36", "rejected", (str(error),))
    if topology.tensor_parallel not in {2, 4}:
        return FeasibilityReport("INK-36", "gated", ("TP2 or TP4 is required",), evidence)
    if evidence is None:
        return FeasibilityReport("INK-36", "gated", ("SGLang topology evidence is required",))
    try:
        evidence.validate()
    except TopologyContractError as error:
        return FeasibilityReport("INK-36", "rejected", (str(error),), evidence)
    return FeasibilityReport("INK-36", "qualified", evidence=evidence)


def assess_deferred_parallelism(item: str, topology: InklingTopology) -> FeasibilityReport:
    """Keep CP, SP, and PP feasibility explicit until external runtime evidence exists."""
    if item not in {"INK-37", "INK-38", "INK-39"}:
        raise TopologyContractError("item must be INK-37, INK-38, or INK-39")
    try:
        topology.validate()
    except TopologyContractError as error:
        return FeasibilityReport(item, "rejected", (str(error),))
    field = {
        "INK-37": "context_parallel",
        "INK-38": "sequence_parallel",
        "INK-39": "pipeline_parallel",
    }[item]
    size = getattr(topology, field)
    if size == 1:
        return FeasibilityReport(item, "gated", (f"{field} runtime evidence is required",))
    return FeasibilityReport(item, "gated", (f"{field}={size} requires Miles and SGLang evidence",))
