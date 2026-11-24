# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Routed-expert EP placement and the ``expert_tensor_parallel_size == 1`` gate (P5-7).

Under expert parallelism every routed expert is computed whole on exactly one
rank; ranks only move rows (dispatch / return), they never add partial
results. That is what keeps EP=1/2/4/8 byte-equal to the single-card P5-S0
oracle. Expert tensor parallelism (etp > 1) would split each expert's weights
and bring back a cross-rank all-reduce, so it is rejected fail-closed.

``MockExchangePlan`` stands in for the P4 ``ExchangePlan``: a values-only,
single-process dispatch/return over an already-routed ``ExpertBatch``. It
carries no transport and defines no P4 schema; P6 consumes plans only by
fingerprint (``exchange_ref``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor

PLACEMENT_SCHEMA = "p5-ep-placement-v1"
PLACEMENT_POLICY = "linear-contiguous"
MOCK_EXCHANGE_SCHEMA = "p5-ep-mock-exchange-v1"


class ExpertParallelConfigError(ValueError):
    """A parallel config that would split a routed expert across ranks."""


def _field(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def resolve_expert_tensor_parallel_size(config: Any) -> int:
    """The effective etp, following Megatron-core's defaulting rule.

    ``ModelParallelConfig.expert_tensor_parallel_size = None`` resolves to
    ``tensor_model_parallel_size`` in ``__post_init__``, so an unset etp under
    TP=4 is etp=4, not 1. Unresolvable configs fail closed.
    """
    etp = _field(config, "expert_tensor_parallel_size")
    if etp is None:
        etp = _field(config, "tensor_model_parallel_size")
    if etp is None:
        raise ExpertParallelConfigError(
            "cannot resolve expert_tensor_parallel_size: neither it nor "
            "tensor_model_parallel_size is set (fail-closed)"
        )
    if type(etp) is not int or etp < 1:
        raise ExpertParallelConfigError(
            f"expert_tensor_parallel_size must be a positive int, got {etp!r}"
        )
    return etp


def check_expert_parallel_config(config: Any) -> None:
    """Startup gate: reject any config that shards a routed expert's weights."""
    etp = resolve_expert_tensor_parallel_size(config)
    if etp != 1:
        raise ExpertParallelConfigError(
            f"expert_tensor_parallel_size={etp}: routed experts must not be tensor-split; "
            "etp>1 adds a cross-rank all-reduce per expert output and breaks the "
            "no-cross-rank-float-reduction MoE contract (fail-closed)"
        )


@dataclass(frozen=True)
class ExpertPlacement:
    """Logical-expert -> (rank, local index) map; linear contiguous blocks.

    Rank ``r`` owns logical experts ``[r * E_local, (r + 1) * E_local)``, the
    same placement vLLM's linear ``expert_map`` and Megatron EP use.
    """

    n_experts: int
    ep_size: int
    etp_size: int = 1

    def __post_init__(self) -> None:
        check_expert_parallel_config({"expert_tensor_parallel_size": self.etp_size})
        if type(self.n_experts) is not int or self.n_experts < 1:
            raise ValueError(f"n_experts must be a positive int, got {self.n_experts!r}")
        if type(self.ep_size) is not int or self.ep_size < 1:
            raise ValueError(f"ep_size must be a positive int, got {self.ep_size!r}")
        if self.n_experts % self.ep_size != 0:
            raise ValueError(f"n_experts {self.n_experts} not divisible by ep_size {self.ep_size}")

    @property
    def experts_per_rank(self) -> int:
        return self.n_experts // self.ep_size

    def rank_of(self, logical_expert_id: int) -> int:
        self._check_expert(logical_expert_id)
        return logical_expert_id // self.experts_per_rank

    def local_index(self, logical_expert_id: int) -> int:
        self._check_expert(logical_expert_id)
        return logical_expert_id % self.experts_per_rank

    def experts_of(self, rank: int) -> range:
        if not 0 <= rank < self.ep_size:
            raise ValueError(f"rank {rank} out of range for ep_size {self.ep_size}")
        lo = rank * self.experts_per_rank
        return range(lo, lo + self.experts_per_rank)

    def _check_expert(self, logical_expert_id: int) -> None:
        if not 0 <= logical_expert_id < self.n_experts:
            raise ValueError(
                f"logical expert {logical_expert_id} out of range [0, {self.n_experts})"
            )

    @property
    def fingerprint(self) -> str:
        """``placement_map_version`` in the P3 RoutePlan envelope."""
        payload = {
            "schema": PLACEMENT_SCHEMA,
            "policy": PLACEMENT_POLICY,
            "n_experts": self.n_experts,
            "ep_size": self.ep_size,
            "etp_size": self.etp_size,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _slice_experts(w: MXTensor, lo: int, hi: int) -> MXTensor:
    return MXTensor(
        w.codes[lo:hi].contiguous(),
        w.scales[lo:hi].contiguous(),
        w.elem_format,
        (hi - lo, *w.shape[1:]),
        w.packing,
    )


@dataclass(frozen=True)
class MockExchangePlan:
    """Values-only stand-in for the P4 ``ExchangePlan`` (no transport).

    The global ``ExpertBatch`` is the EP=1 view: rows sorted by logical
    expert. Since placement is contiguous, the rows rank ``r`` receives are
    one contiguous range, so dispatch is a slice and return is a scatter by
    ``output_slot``. Base and LoRA use this same plan; LoRA A/B travel to every
    rank unsplit.
    """

    placement: ExpertPlacement
    row_ranges: tuple[tuple[int, int], ...]  # global rows received by each expert host
    rows: int

    @classmethod
    def from_batch(cls, batch: ExpertBatch, placement: ExpertPlacement) -> "MockExchangePlan":
        n_experts = batch.expert_offsets.numel() - 1
        if n_experts != placement.n_experts:
            raise ValueError(
                f"batch has {n_experts} experts, placement expects {placement.n_experts}"
            )
        offsets = [int(v) for v in batch.expert_offsets.cpu()]
        ranges = []
        for rank in range(placement.ep_size):
            experts = placement.experts_of(rank)
            ranges.append((offsets[experts.start], offsets[experts.stop]))
        return cls(placement=placement, row_ranges=tuple(ranges), rows=batch.rows)

    @property
    def recv_counts(self) -> tuple[int, ...]:
        return tuple(hi - lo for lo, hi in self.row_ranges)

    def dispatch(self, batch: ExpertBatch) -> tuple[ExpertBatch, ...]:
        """Each expert host's local ``ExpertBatch`` (rebased offsets, local weights)."""
        if batch.rows != self.rows:
            raise ValueError(f"batch has {batch.rows} rows, plan was built for {self.rows}")
        shards = []
        for rank, (lo, hi) in enumerate(self.row_ranges):
            experts = self.placement.experts_of(rank)
            local = dataclasses.replace(
                batch,
                x=batch.x[lo:hi],
                expert_offsets=(batch.expert_offsets[experts.start : experts.stop + 1] - lo).to(
                    torch.int32
                ),
                p_s=batch.p_s[lo:hi],
                w1=_slice_experts(batch.w1, experts.start, experts.stop),
                w2=_slice_experts(batch.w2, experts.start, experts.stop),
                output_slot=batch.output_slot[lo:hi],
                weight_fingerprint="",
            )
            shards.append(
                dataclasses.replace(local, weight_fingerprint=local.compute_weight_fingerprint())
            )
        return tuple(shards)

    def dispatch_rows(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Send a row-aligned tensor (e.g. ``dy``) along the same plan."""
        if t.shape[0] != self.rows:
            raise ValueError(f"tensor has {t.shape[0]} rows, plan was built for {self.rows}")
        return tuple(t[lo:hi] for lo, hi in self.row_ranges)

    def return_rows(
        self, shards: Sequence[ExpertBatch], outputs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Scatter per-host results back to token-host slots by ``output_slot``.

        Missing or duplicate slots fail closed; this is a byte copy, no arithmetic.
        """
        if len(shards) != self.placement.ep_size or len(outputs) != self.placement.ep_size:
            raise ValueError("need one shard and one output per EP rank")
        result: torch.Tensor | None = None
        filled = torch.zeros(self.rows, dtype=torch.bool)
        for shard, out in zip(shards, outputs, strict=True):
            if out.shape[0] != shard.rows:
                raise ValueError(f"output has {out.shape[0]} rows, shard has {shard.rows}")
            if result is None:
                result = out.new_empty((self.rows, *out.shape[1:]))
            slots = shard.output_slot.long().cpu()
            if bool(filled[slots].any()) or slots.unique().numel() != slots.numel():
                raise ValueError("duplicate output_slot on return (fail-closed)")
            filled[slots] = True
            result[slots.to(out.device)] = out
        if result is None or not bool(filled.all()):
            raise ValueError("missing output_slot on return (fail-closed)")
        return result

    @property
    def fingerprint(self) -> str:
        """What a P6 ``CombinePlan.exchange_ref`` would point at."""
        payload = {
            "schema": MOCK_EXCHANGE_SCHEMA,
            "placement": self.placement.fingerprint,
            "row_ranges": [list(r) for r in self.row_ranges],
            "rows": self.rows,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()
