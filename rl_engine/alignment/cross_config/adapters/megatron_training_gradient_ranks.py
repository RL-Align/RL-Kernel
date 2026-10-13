# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2): cross-rank gradient evidence for Megatron DDP.

Each rank closes its step window with a :class:`GradientStepEvidence` recorded
with ``observe_sync=True`` (and, under the distributed optimizer, the ranges its
optimizer owns), adds the coordinates it observes, and the readbacks are
exchanged over a CPU gloo group. :func:`validate_gradient_rank_set` then checks,
per gradient-reduction (DP) group:

* every expected rank reported once, in one step window;
* the DDP buffer/bucket layout and the per-parameter write decisions agree
  across replicas;
* every bucket was synchronised exactly once on the DP group, with the
  collective the configuration asks for (reduce-scatter with the distributed
  optimizer, all-reduce otherwise), and each rank received its own shard;
* the distributed-optimizer ranges lie inside the rank's reduce-scatter shard
  and together cover every trainable parameter exactly once.

The ``dp`` coordinate is the group DDP reduces gradients on (Megatron's DP×CP
group when CP is used). A passing set is complete, not certified: the producer
gradient set (P5) is still missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    T05ContractError,
    canonical_sha256,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_ranks import (
    ExpectedRankTopology,
)
from rl_engine.integrations.megatron_training_gradients import (
    GRADIENT_EVIDENCE_SCHEMA_VERSION,
    GradientStepEvidence,
)

GRADIENT_RANK_READBACK_SCHEMA_VERSION = "rlkernel.p7.t05.gradient_rank_readback.v1-draft"
GRADIENT_RANK_SET_SCHEMA_VERSION = "rlkernel.p7.t05.gradient_rank_set_report.v1-draft"

_PRODUCER_MISSING = {
    "field": "gradient.producer_emitted",
    "status": "UPSTREAM_EVIDENCE_MISSING",
    "detail": "producer gradient set to cross-check the observed writes",
    "dependency": "P5 (gradient-set publication)",
}


def _fail(status: str, field_name: str, detail: str) -> T05ContractError:
    return T05ContractError(status, field_name, detail)


@dataclass(frozen=True)
class GradientRankReadback:
    """One rank's gradient step evidence, observed coordinates and optimizer ownership."""

    coordinates: Mapping[str, Any]
    evidence: Mapping[str, Any]  # GradientStepEvidence.to_dict()
    ownership: Optional[Mapping[str, Any]] = None
    schema_version: str = GRADIENT_RANK_READBACK_SCHEMA_VERSION

    @classmethod
    def from_evidence(
        cls,
        evidence: GradientStepEvidence,
        coordinates: Mapping[str, Any],
        ownership: Optional[Mapping[str, Any]] = None,
    ) -> "GradientRankReadback":
        return cls(
            coordinates=dict(coordinates),
            evidence=evidence.to_dict(),
            ownership=None if ownership is None else dict(ownership),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GradientRankReadback":
        if not isinstance(payload, Mapping):
            raise _fail("SCHEMA_MISMATCH", "gradient_rank_readback", type(payload).__name__)
        version = payload.get("schema_version")
        if version != GRADIENT_RANK_READBACK_SCHEMA_VERSION:
            raise _fail("SCHEMA_MISMATCH", "gradient_rank_readback.schema_version", repr(version))
        ownership = payload.get("ownership")
        return cls(
            coordinates=dict(payload["coordinates"]),
            evidence=dict(payload["evidence"]),
            ownership=None if ownership is None else dict(ownership),
            schema_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinates": dict(self.coordinates),
            "evidence": dict(self.evidence),
            "ownership": None if self.ownership is None else dict(self.ownership),
        }

    @property
    def global_rank(self) -> int:
        return int(self.coordinates["global_rank"])

    @property
    def dp_rank(self) -> int:
        return int(self.coordinates["dp"]["rank"])

    @property
    def dp_ranks(self) -> tuple[int, ...]:
        return tuple(int(rank) for rank in self.coordinates["dp"]["ranks"])


def gather_gradient_readbacks(
    local: GradientRankReadback, group: Any = None
) -> list[GradientRankReadback]:
    """All-gather readbacks as Python objects over a CPU gloo group (no numeric collective)."""

    import torch.distributed as dist

    backend = str(dist.get_backend(group))
    if backend != "gloo":
        raise _fail(
            "UNSUPPORTED_CAPABILITY",
            "gather_group",
            f"gradient readbacks are exchanged on a CPU gloo group only, got {backend!r}",
        )
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local.to_dict(), group=group)
    return [GradientRankReadback.from_dict(item) for item in gathered]


@dataclass(frozen=True)
class GradientRankSetReport:
    """Cross-rank gradient integrity for one step; complete is not certified."""

    topology: Mapping[str, int]
    collective: str
    groups: tuple[Mapping[str, Any], ...]
    missing_evidence: tuple[Mapping[str, str], ...]
    complete: bool = True
    schema_version: str = GRADIENT_RANK_SET_SCHEMA_VERSION

    @property
    def certified(self) -> bool:
        return self.complete and not self.missing_evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology": dict(self.topology),
            "collective": self.collective,
            "groups": [dict(group) for group in self.groups],
            "complete": self.complete,
            "certified": self.certified,
            "missing_evidence": [dict(item) for item in self.missing_evidence],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def _check_schema(readback: GradientRankReadback) -> None:
    if readback.schema_version != GRADIENT_RANK_READBACK_SCHEMA_VERSION:
        raise _fail(
            "SCHEMA_MISMATCH", "gradient_rank_readback.schema_version", readback.schema_version
        )
    version = readback.evidence.get("schema_version")
    if version != GRADIENT_EVIDENCE_SCHEMA_VERSION:
        raise _fail("SCHEMA_MISMATCH", "evidence.schema_version", repr(version))
    if readback.coordinates.get("source") != "observed":
        raise _fail("MISSING_PROVENANCE", "coordinates.source", "rank coordinates must be observed")
    sync = readback.evidence.get("sync")
    if not isinstance(sync, Mapping) or sync.get("source") != "observed":
        raise _fail(
            "MISSING_PROVENANCE",
            "evidence.sync",
            f"rank {readback.global_rank}: gradient sync was not observed (observe_sync=True)",
        )


def _check_rank_set(
    readbacks: Sequence[GradientRankReadback], topology: ExpectedRankTopology
) -> dict[tuple[int, ...], dict[int, GradientRankReadback]]:
    seen: dict[int, GradientRankReadback] = {}
    for readback in readbacks:
        if readback.global_rank in seen:
            raise _fail(
                "IDENTITY_DRIFT", "coordinates.global_rank", f"rank {readback.global_rank} twice"
            )
        seen[readback.global_rank] = readback
    world = topology.tp * topology.cp * topology.dp * topology.pp
    for readback in readbacks:
        if int(readback.coordinates["world_size"]) != world:
            raise _fail(
                "IDENTITY_DRIFT",
                "coordinates.world_size",
                f"rank {readback.global_rank}: world "
                f"{readback.coordinates['world_size']} != {world}",
            )
    missing = sorted(set(range(world)) - set(seen))
    if missing:
        raise _fail("MISSING_RANK", "coordinates.global_rank", f"no readback from ranks {missing}")
    groups: dict[tuple[int, ...], dict[int, GradientRankReadback]] = {}
    for readback in readbacks:
        ranks = readback.dp_ranks
        if len(ranks) != topology.dp or ranks[readback.dp_rank] != readback.global_rank:
            raise _fail(
                "IDENTITY_DRIFT",
                "coordinates.dp",
                f"rank {readback.global_rank}: DP group {list(ranks)} (size {topology.dp} "
                f"expected) does not place it at {readback.dp_rank}",
            )
        members = groups.setdefault(ranks, {})
        if readback.dp_rank in members:
            raise _fail(
                "IDENTITY_DRIFT", "coordinates.dp.rank", f"DP rank {readback.dp_rank} twice"
            )
        members[readback.dp_rank] = readback
    for ranks, members in groups.items():
        for dp_rank, global_rank in enumerate(ranks):
            if dp_rank not in members or members[dp_rank].global_rank != global_rank:
                raise _fail(
                    "IDENTITY_DRIFT",
                    "coordinates.dp.ranks",
                    f"DP group {list(ranks)}: rank {global_rank} did not report as {dp_rank}",
                )
    return groups


def _decisions(readback: GradientRankReadback) -> list[tuple[Any, ...]]:
    return [
        (item["name"], item["write_path"], item["wrote"], item["buffer_index"], item["bucket_id"])
        for item in readback.evidence["records"]
    ]


def _check_replicas(members: Sequence[GradientRankReadback]) -> None:
    first = members[0]
    for readback in members:
        if readback.evidence.get("unknown"):
            raise _fail(
                "MISSING_PROVENANCE",
                "gradient.unknown",
                f"rank {readback.global_rank}: write source unknown for "
                f"{list(readback.evidence['unknown'])}",
            )
    for readback in members[1:]:
        pairs = (
            ("step_index", readback.evidence["step_index"], first.evidence["step_index"]),
            (
                "sync.buckets",
                readback.evidence["sync"]["buckets"],
                first.evidence["sync"]["buckets"],
            ),
            ("records", _decisions(readback), _decisions(first)),
            (
                "sync.use_distributed_optimizer",
                readback.evidence["sync"]["use_distributed_optimizer"],
                first.evidence["sync"]["use_distributed_optimizer"],
            ),
        )
        for name, value, reference in pairs:
            if value != reference:
                raise _fail(
                    "IDENTITY_DRIFT",
                    f"gradient.{name}",
                    f"DP replicas disagree: rank {readback.global_rank} vs {first.global_rank}",
                )


def _check_sync_calls(
    members: Sequence[GradientRankReadback], dp_ranks: tuple[int, ...], expected_kind: str
) -> None:
    size = len(dp_ranks)
    for readback in members:
        sync = readback.evidence["sync"]
        buckets = {(b["buffer_index"], b["bucket_id"]): b for b in sync["buckets"]}
        if not buckets:
            raise _fail(
                "MISSING_PROVENANCE",
                "sync.buckets",
                f"rank {readback.global_rank}: no DDP bucket observed",
            )
        calls: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for call in sync["calls"]:
            calls.setdefault((call["buffer_index"], call["bucket_id"]), []).append(call)
        for key, bucket in sorted(buckets.items()):
            observed = calls.get(key, [])
            where = f"rank {readback.global_rank} bucket {key}"
            if not observed:
                raise _fail(
                    "MISSING_PROVENANCE", "sync.calls", f"{where}: no gradient sync observed"
                )
            if len(observed) > 1:
                raise _fail(
                    "IDENTITY_DRIFT",
                    "sync.calls",
                    f"{where}: synchronised {len(observed)} times in one step",
                )
            call = observed[0]
            if call["kind"] != expected_kind:
                raise _fail(
                    "SILENT_FALLBACK",
                    "sync.calls.kind",
                    f"{where}: {call['kind']} observed, configuration requires {expected_kind}",
                )
            if tuple(call["group_ranks"]) != dp_ranks:
                raise _fail(
                    "IDENTITY_DRIFT",
                    "sync.calls.group_ranks",
                    f"{where}: synchronised on {call['group_ranks']}, DP group {list(dp_ranks)}",
                )
            if call["input_numel"] != bucket["numel"]:
                raise _fail(
                    "IDENTITY_DRIFT",
                    "sync.calls.input_numel",
                    f"{where}: collective input {call['input_numel']} != bucket {bucket['numel']}",
                )
            if expected_kind == "reduce_scatter":
                shard = bucket["numel"] // size
                if bucket["numel"] % size or call["output_numel"] != shard:
                    raise _fail(
                        "IDENTITY_DRIFT",
                        "sync.calls.output_numel",
                        f"{where}: shard {call['output_numel']} for {bucket['numel']} over {size}",
                    )
                if call["output_offset"] != readback.dp_rank * shard:
                    raise _fail(
                        "IDENTITY_DRIFT",
                        "sync.calls.output_offset",
                        f"{where}: DP rank {readback.dp_rank} received the shard at "
                        f"{call['output_offset']}, owns {readback.dp_rank * shard}",
                    )
            elif (call["output_offset"], call["output_numel"]) != (0, bucket["numel"]):
                raise _fail(
                    "IDENTITY_DRIFT",
                    "sync.calls.output",
                    f"{where}: all-reduce did not cover the whole bucket",
                )


def _check_ownership(members: Sequence[GradientRankReadback]) -> int:
    """Distributed-optimizer ranges: inside each rank's shard, covering every param once."""

    size = len(members)
    layout = members[0].evidence["sync"]["buckets"]
    params: dict[str, tuple[int, int, int, int]] = {}
    for bucket in layout:
        for name, start, end in bucket["params"]:
            params[name] = (bucket["buffer_index"], bucket["bucket_id"], start, end)
    pieces: dict[str, list[tuple[int, int, int]]] = {name: [] for name in params}
    for readback in members:
        if readback.ownership is None or readback.ownership.get("source") != "observed":
            raise _fail(
                "MISSING_PROVENANCE",
                "ownership",
                f"rank {readback.global_rank}: distributed-optimizer ownership not observed",
            )
        shards = {
            (b["buffer_index"], b["bucket_id"]): (
                b["offset"] + readback.dp_rank * (b["numel"] // size),
                b["offset"] + (readback.dp_rank + 1) * (b["numel"] // size),
            )
            for b in layout
        }
        for item in readback.ownership["ranges"]:
            name = item["name"]
            where = f"rank {readback.global_rank} {name}"
            if name not in params:
                raise _fail(
                    "IDENTITY_DRIFT", "ownership.ranges", f"{where}: not a DDP-managed parameter"
                )
            buffer_index, bucket_id, start, end = params[name]
            if (item["buffer_index"], item["bucket_id"]) != (buffer_index, bucket_id):
                raise _fail("IDENTITY_DRIFT", "ownership.ranges", f"{where}: wrong bucket")
            low, high = shards[(buffer_index, bucket_id)]
            if not (low <= item["world_start"] < item["world_end"] <= high):
                raise _fail(
                    "IDENTITY_DRIFT",
                    "ownership.ranges",
                    f"{where}: owns [{item['world_start']}, {item['world_end']}) outside its "
                    f"reduce-scatter shard [{low}, {high})",
                )
            if (
                item["param_start"] != item["world_start"] - start
                or item["param_end"] != item["world_end"] - start
            ):
                raise _fail(
                    "IDENTITY_DRIFT", "ownership.ranges", f"{where}: parameter range misaligned"
                )
            pieces[name].append((item["world_start"], item["world_end"], readback.global_rank))
    for name, (_, _, start, end) in sorted(params.items()):
        cursor = start
        for low, high, rank in sorted(pieces[name]):
            if low < cursor:
                raise _fail(
                    "IDENTITY_DRIFT",
                    "ownership.coverage",
                    f"{name}: [{low}, {high}) from rank {rank} overlaps another owner",
                )
            if low > cursor:
                raise _fail(
                    "INCOMPLETE_ARTIFACT",
                    "ownership.coverage",
                    f"{name}: [{cursor}, {low}) owned by no DP rank",
                )
            cursor = high
        if cursor != end:
            raise _fail(
                "INCOMPLETE_ARTIFACT",
                "ownership.coverage",
                f"{name}: [{cursor}, {end}) owned by no DP rank",
            )
    return sum(len(items) for items in pieces.values())


def validate_gradient_rank_set(
    readbacks: Sequence[GradientRankReadback], topology: ExpectedRankTopology
) -> GradientRankSetReport:
    """Validate one gradient step across ranks; raise on the first integrity violation."""

    if not readbacks:
        raise _fail("MISSING_RANK", "gradient_rank_set", "no gradient readbacks")
    for readback in readbacks:
        _check_schema(readback)
    groups = _check_rank_set(readbacks, topology)
    configured = {
        readback.evidence["sync"]["use_distributed_optimizer"]["value"] for readback in readbacks
    }
    if len(configured) != 1 or None in configured:
        raise _fail(
            "MISSING_PROVENANCE",
            "sync.use_distributed_optimizer",
            f"configuration not uniform or absent: {sorted(map(str, configured))}",
        )
    distributed = configured.pop()
    kind = "reduce_scatter" if distributed else "all_reduce"
    summaries = []
    for dp_ranks, by_rank in sorted(groups.items()):
        members = [by_rank[index] for index in range(len(dp_ranks))]
        _check_replicas(members)
        _check_sync_calls(members, dp_ranks, kind)
        owned = _check_ownership(members) if distributed else None
        evidence = members[0].evidence
        summaries.append(
            {
                "dp_ranks": list(dp_ranks),
                "buckets": len(evidence["sync"]["buckets"]),
                "emitted": list(evidence["emitted"]),
                "owned_ranges": owned,
                "evidence_fingerprints": [
                    canonical_sha256(readback.evidence) for readback in members
                ],
            }
        )
    return GradientRankSetReport(
        topology=topology.to_dict(),
        collective=kind,
        groups=tuple(summaries),
        missing_evidence=(dict(_PRODUCER_MISSING),),
    )


__all__ = [
    "GRADIENT_RANK_READBACK_SCHEMA_VERSION",
    "GRADIENT_RANK_SET_SCHEMA_VERSION",
    "GradientRankReadback",
    "GradientRankSetReport",
    "gather_gradient_readbacks",
    "validate_gradient_rank_set",
]
