# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2): per-rank training readback and rank-set integrity.

Each rank turns its validated :class:`TrainingM10Evidence` plus the process
coordinates it observes from ``torch.distributed`` into a :class:`RankReadback`.
Readbacks are exchanged with ``all_gather_object`` on a CPU gloo group, so no
numeric collective is added to the training step. :func:`validate_rank_set`
then checks the expected rank set (final pipeline stage only), TP group
membership, vocab ownership coverage, target ownership, replicated merge
output, row/global-token mapping and fingerprint consistency, and fails closed
with a contract section 7 status on the first violation.

A passing rank set is *complete*, not *certified*: the union of the per-rank
missing-evidence ledgers is carried forward, and sealing belongs to T10.
"""

from __future__ import annotations

import itertools
import socket
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    T05ContractError,
    canonical_sha256,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_gate import (
    TRAINING_M10_EVIDENCE_SCHEMA_VERSION,
    TrainingM10Evidence,
)

RANK_READBACK_SCHEMA_VERSION = "rlkernel.p7.t05.rank_readback.v1-draft"
RANK_SET_SCHEMA_VERSION = "rlkernel.p7.t05.rank_set_report.v1-draft"

# Fields that must be identical on every rank of one training call.
_RANK_INVARIANT_FIELDS = (
    "path",
    "entrypoint",
    "runtime_platform",
    "real_vocab_size",
    "padded_vocab_size",
    "canonical_configured",
    "build",
)


def _fail(status: str, field_name: str, detail: str) -> T05ContractError:
    return T05ContractError(status, field_name, detail)


def _group_view(group: Any, global_rank: int) -> dict[str, Any]:
    import torch.distributed as dist

    if group is None:
        return {"rank": 0, "world_size": 1, "ranks": [global_rank]}
    return {
        "rank": int(dist.get_rank(group=group)),
        "world_size": int(dist.get_world_size(group=group)),
        "ranks": [int(rank) for rank in dist.get_process_group_ranks(group)],
    }


def _device_identity() -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {"type": "cpu", "index": None, "uuid": None}
    index = torch.cuda.current_device()
    return {
        "type": "cuda",
        "index": int(index),
        "uuid": str(torch.cuda.get_device_properties(index).uuid),
    }


def observe_rank_coordinates(
    *,
    tp_group: Any,
    cp_group: Any = None,
    dp_group: Any = None,
    pp_group: Any = None,
) -> dict[str, Any]:
    """Coordinates this process observes from ``torch.distributed`` (observed, not configured).

    A ``None`` group means that parallel dimension has size 1.
    """

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise _fail("MISSING_PROVENANCE", "coordinates", "torch.distributed is not initialized")
    global_rank = int(dist.get_rank())
    return {
        "source": "observed",
        "global_rank": global_rank,
        "world_size": int(dist.get_world_size()),
        "hostname": socket.gethostname(),
        "device": _device_identity(),
        "tp": _group_view(tp_group, global_rank),
        "cp": _group_view(cp_group, global_rank),
        "dp": _group_view(dp_group, global_rank),
        "pp": _group_view(pp_group, global_rank),
    }


@dataclass(frozen=True)
class RankReadback:
    """One rank's training-call evidence plus the coordinates it observed."""

    coordinates: Mapping[str, Any]
    evidence: Mapping[str, Any]  # TrainingM10Evidence.to_dict()
    schema_version: str = RANK_READBACK_SCHEMA_VERSION

    @classmethod
    def from_evidence(
        cls, evidence: TrainingM10Evidence, coordinates: Mapping[str, Any]
    ) -> "RankReadback":
        return cls(coordinates=dict(coordinates), evidence=evidence.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RankReadback":
        if not isinstance(payload, Mapping):
            raise _fail("SCHEMA_MISMATCH", "rank_readback", f"{type(payload).__name__}")
        version = payload.get("schema_version")
        if version != RANK_READBACK_SCHEMA_VERSION:
            raise _fail("SCHEMA_MISMATCH", "rank_readback.schema_version", repr(version))
        return cls(
            coordinates=dict(payload["coordinates"]),
            evidence=dict(payload["evidence"]),
            schema_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinates": dict(self.coordinates),
            "evidence": dict(self.evidence),
        }

    @property
    def key(self) -> tuple[int, int, int]:
        """(tp_rank, cp_rank, dp_rank) as observed."""

        coords = self.coordinates
        return (int(coords["tp"]["rank"]), int(coords["cp"]["rank"]), int(coords["dp"]["rank"]))


def gather_rank_readbacks(local: RankReadback, group: Any = None) -> list[RankReadback]:
    """All-gather readbacks as Python objects over a CPU gloo group (no numeric collective)."""

    import torch.distributed as dist

    backend = str(dist.get_backend(group))
    if backend != "gloo":
        raise _fail(
            "UNSUPPORTED_CAPABILITY",
            "gather_group",
            f"rank readbacks are exchanged on a CPU gloo group only, got {backend!r}",
        )
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local.to_dict(), group=group)
    return [RankReadback.from_dict(item) for item in gathered]


@dataclass(frozen=True)
class ExpectedRankTopology:
    """Expected parallel sizes, from the profile/manifest; only the last PP stage emits rows."""

    tp: int
    cp: int = 1
    dp: int = 1
    pp: int = 1

    def __post_init__(self) -> None:
        if min(self.tp, self.cp, self.dp, self.pp) < 1:
            raise ValueError("parallel sizes must be >= 1")

    def expected_keys(self) -> frozenset[tuple[int, int, int]]:
        return frozenset(itertools.product(range(self.tp), range(self.cp), range(self.dp)))

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "cp": self.cp, "dp": self.dp, "pp": self.pp}


@dataclass(frozen=True)
class RankSetReport:
    """Integrity result for one training call across ranks; complete is not certified."""

    topology: Mapping[str, int]
    ranks: tuple[Mapping[str, Any], ...]
    vocab_ownership: tuple[tuple[int, int], ...]
    active_tokens: int
    missing_evidence: tuple[Mapping[str, str], ...]
    complete: bool = True
    schema_version: str = RANK_SET_SCHEMA_VERSION

    @property
    def certified(self) -> bool:
        return self.complete and not self.missing_evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology": dict(self.topology),
            "ranks": [dict(rank) for rank in self.ranks],
            "vocab_ownership": [list(item) for item in self.vocab_ownership],
            "active_tokens": self.active_tokens,
            "complete": self.complete,
            "certified": self.certified,
            "missing_evidence": [dict(item) for item in self.missing_evidence],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def _check_schema(readback: RankReadback) -> None:
    if readback.schema_version != RANK_READBACK_SCHEMA_VERSION:
        raise _fail("SCHEMA_MISMATCH", "rank_readback.schema_version", readback.schema_version)
    version = readback.evidence.get("schema_version")
    if version != TRAINING_M10_EVIDENCE_SCHEMA_VERSION:
        raise _fail("SCHEMA_MISMATCH", "evidence.schema_version", repr(version))
    if readback.coordinates.get("source") != "observed":
        raise _fail(
            "MISSING_PROVENANCE",
            "coordinates.source",
            f"rank coordinates must be observed, got {readback.coordinates.get('source')!r}",
        )


def _check_coordinates(readback: RankReadback, topology: ExpectedRankTopology) -> None:
    coords, evidence = readback.coordinates, readback.evidence
    rank = coords["global_rank"]
    for dim, size in (("tp", topology.tp), ("cp", topology.cp), ("dp", topology.dp)):
        view = coords[dim]
        if int(view["world_size"]) != size:
            raise _fail(
                "IDENTITY_DRIFT",
                f"coordinates.{dim}",
                f"rank {rank}: observed {dim} size {view['world_size']} != expected {size}",
            )
        ranks = list(view["ranks"])
        if len(ranks) != size or ranks[int(view["rank"])] != rank:
            raise _fail(
                "IDENTITY_DRIFT",
                f"coordinates.{dim}.ranks",
                f"rank {rank}: {dim} group {ranks} does not place it at {view['rank']}",
            )
    pp = coords["pp"]
    if int(pp["world_size"]) != topology.pp or int(pp["rank"]) != topology.pp - 1:
        raise _fail(
            "IDENTITY_DRIFT",
            "coordinates.pp",
            f"rank {rank}: PP {pp['rank']}/{pp['world_size']} is not the final stage of "
            f"{topology.pp}; only the last stage emits final rows",
        )
    if (evidence["tp_rank"], evidence["tp_world_size"]) != (
        int(coords["tp"]["rank"]),
        int(coords["tp"]["world_size"]),
    ):
        raise _fail(
            "IDENTITY_DRIFT",
            "evidence.tp_rank",
            f"rank {rank}: evidence TP {evidence['tp_rank']}/{evidence['tp_world_size']} differs "
            f"from observed coordinates",
        )


def _check_rank_set(
    readbacks: Sequence[RankReadback], topology: ExpectedRankTopology
) -> dict[tuple[int, int, int], RankReadback]:
    by_key: dict[tuple[int, int, int], RankReadback] = {}
    global_ranks: set[int] = set()
    devices: set[tuple[str, str]] = set()
    for readback in readbacks:
        coords = readback.coordinates
        rank = int(coords["global_rank"])
        if rank in global_ranks:
            raise _fail("IDENTITY_DRIFT", "coordinates.global_rank", f"rank {rank} reported twice")
        global_ranks.add(rank)
        device = coords.get("device") or {}
        if device.get("uuid") is not None:
            slot = (str(coords.get("hostname")), str(device["uuid"]))
            if slot in devices:
                raise _fail("IDENTITY_DRIFT", "coordinates.device", f"device {slot} shared")
            devices.add(slot)
        key = readback.key
        if key in by_key:
            raise _fail(
                "IDENTITY_DRIFT",
                "coordinates",
                f"(tp, cp, dp) {key} reported by ranks {by_key[key].coordinates['global_rank']} "
                f"and {rank}",
            )
        by_key[key] = readback
    expected = topology.expected_keys()
    unexpected = sorted(set(by_key) - expected)
    if unexpected:
        raise _fail("IDENTITY_DRIFT", "coordinates", f"unexpected (tp, cp, dp) {unexpected}")
    missing = sorted(expected - set(by_key))
    if missing:
        raise _fail("MISSING_RANK", "coordinates", f"no readback for (tp, cp, dp) {missing}")
    return by_key


def _check_group_membership(
    by_key: Mapping[tuple[int, int, int], RankReadback], topology: ExpectedRankTopology
) -> None:
    """Each TP/CP/DP group's observed member list matches the ranks that reported in it."""

    for dim, position in (("tp", 0), ("cp", 1), ("dp", 2)):
        groups: dict[tuple[int, ...], dict[int, int]] = {}
        for key, readback in by_key.items():
            outer = key[:position] + key[position + 1 :]
            groups.setdefault(outer, {})[key[position]] = int(readback.coordinates["global_rank"])
        for outer, members in groups.items():
            actual = [members[index] for index in sorted(members)]
            for index in sorted(members):
                readback = by_key[outer[:position] + (index,) + outer[position:]]
                observed = list(readback.coordinates[dim]["ranks"])
                if observed != actual:
                    raise _fail(
                        "IDENTITY_DRIFT",
                        f"coordinates.{dim}.ranks",
                        f"group {outer}: rank {actual[index]} sees {observed}, reporters {actual}",
                    )


def _check_invariants(readbacks: Sequence[RankReadback]) -> None:
    reference = readbacks[0]
    for name in _RANK_INVARIANT_FIELDS:
        expected = canonical_sha256(reference.evidence.get(name))
        for readback in readbacks[1:]:
            if canonical_sha256(readback.evidence.get(name)) != expected:
                raise _fail(
                    "IDENTITY_DRIFT",
                    f"evidence.{name}",
                    f"rank {readback.coordinates['global_rank']} differs from rank "
                    f"{reference.coordinates['global_rank']}",
                )


def _check_tp_group(
    members: Sequence[RankReadback], topology: ExpectedRankTopology
) -> tuple[tuple[int, int], ...]:
    """Vocab coverage without gap/overlap, unique target owner, replicated merge output."""

    ordered = sorted(members, key=lambda item: item.evidence["tp_rank"])
    padded = int(ordered[0].evidence["padded_vocab_size"])
    intervals = tuple(
        (int(item.evidence["vocab_ownership"][0]), int(item.evidence["vocab_ownership"][1]))
        for item in ordered
    )
    cursor = 0
    for rank, (start, end) in enumerate(intervals):
        if start != cursor:
            kind = "gap" if start > cursor else "overlap"
            raise _fail(
                "IDENTITY_DRIFT",
                "evidence.vocab_ownership",
                f"{kind} before TP rank {rank}: [{start}, {end}) after {cursor}",
            )
        cursor = end
    if cursor != padded:
        raise _fail(
            "IDENTITY_DRIFT", "evidence.vocab_ownership", f"coverage ends at {cursor} != {padded}"
        )
    reference = ordered[0].evidence
    for item in ordered[1:]:
        evidence = item.evidence
        rank = item.coordinates["global_rank"]
        if evidence["row_keys"] != reference["row_keys"]:
            raise _fail(
                "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
                "evidence.row_keys",
                f"rank {rank}: TP peers disagree on the rows of this call",
            )
        if evidence["target_owner_ranks"] != reference["target_owner_ranks"]:
            raise _fail(
                "IDENTITY_DRIFT",
                "evidence.target_owner_ranks",
                f"rank {rank}: TP peers disagree on the selected-target owners",
            )
        if evidence["active_bytes_sha256"] != reference["active_bytes_sha256"]:
            raise _fail(
                "IDENTITY_DRIFT",
                "evidence.active_bytes_sha256",
                f"rank {rank}: merged selected logprob is not replicated across the TP group",
            )
    for owner in reference["target_owner_ranks"]:
        start, end = intervals[owner] if 0 <= owner < len(intervals) else (0, 0)
        if not 0 <= owner < topology.tp or start >= end:
            raise _fail(
                "IDENTITY_DRIFT", "evidence.target_owner_ranks", f"owner {owner} owns no vocab"
            )
    return intervals


def _check_weight_replicas(by_key: Mapping[tuple[int, int, int], RankReadback]) -> None:
    shards: dict[int, tuple[str, int]] = {}
    for key, readback in sorted(by_key.items()):
        shard = readback.evidence["weight_shard"]
        tp_rank = key[0]
        digest = canonical_sha256(shard)
        if tp_rank not in shards:
            shards[tp_rank] = (digest, int(readback.coordinates["global_rank"]))
        elif shards[tp_rank][0] != digest:
            raise _fail(
                "PRE_UPDATE_WEIGHT_DRIFT",
                "evidence.weight_shard",
                f"TP rank {tp_rank}: shard on rank {readback.coordinates['global_rank']} differs "
                f"from the replica on rank {shards[tp_rank][1]}",
            )


def _check_global_tokens(
    by_key: Mapping[tuple[int, int, int], RankReadback],
    expected_row_keys: Optional[Iterable[tuple[str, int]]],
) -> None:
    """Rows of different CP/DP slots are disjoint; optionally they cover the expected set."""

    owner: dict[tuple[str, int], tuple[int, int]] = {}
    for (tp_rank, cp_rank, dp_rank), readback in sorted(by_key.items()):
        if tp_rank != 0:
            continue
        for key in readback.evidence["row_keys"]:
            if key is None:
                continue
            row = (str(key[0]), int(key[1]))
            if row in owner:
                raise _fail(
                    "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
                    "evidence.row_keys",
                    f"row {row} held by (cp, dp) {owner[row]} and {(cp_rank, dp_rank)}",
                )
            owner[row] = (cp_rank, dp_rank)
    if expected_row_keys is None:
        return
    expected = {(str(sample), int(position)) for sample, position in expected_row_keys}
    missing = sorted(expected - set(owner))
    extra = sorted(set(owner) - expected)
    if missing:
        raise _fail(
            "INCOMPLETE_ARTIFACT",
            "evidence.row_keys",
            f"{len(missing)} expected rows absent: {missing[:4]}",
        )
    if extra:
        raise _fail(
            "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
            "evidence.row_keys",
            f"{len(extra)} rows outside the failure batch: {extra[:4]}",
        )


def _merged_missing(readbacks: Sequence[RankReadback]) -> tuple[Mapping[str, str], ...]:
    merged: dict[str, Mapping[str, str]] = {}
    for readback in readbacks:
        for item in readback.evidence.get("missing_evidence", ()):
            merged.setdefault(str(item["field"]), dict(item))
    return tuple(merged[name] for name in sorted(merged))


def validate_rank_set(
    readbacks: Sequence[RankReadback],
    topology: ExpectedRankTopology,
    *,
    expected_row_keys: Optional[Iterable[tuple[str, int]]] = None,
) -> RankSetReport:
    """Validate one training call across ranks; raise on the first integrity violation."""

    if not readbacks:
        raise _fail("MISSING_RANK", "rank_set", "no rank readbacks")
    for readback in readbacks:
        _check_schema(readback)
        _check_coordinates(readback, topology)
    by_key = _check_rank_set(readbacks, topology)
    _check_group_membership(by_key, topology)
    ordered = [by_key[key] for key in sorted(by_key)]
    _check_invariants(ordered)
    _check_weight_replicas(by_key)

    intervals: Optional[tuple[tuple[int, int], ...]] = None
    for cp_rank, dp_rank in itertools.product(range(topology.cp), range(topology.dp)):
        members = [by_key[(tp_rank, cp_rank, dp_rank)] for tp_rank in range(topology.tp)]
        group_intervals = _check_tp_group(members, topology)
        if intervals is None:
            intervals = group_intervals
        elif group_intervals != intervals:
            raise _fail(
                "IDENTITY_DRIFT",
                "evidence.vocab_ownership",
                f"(cp, dp) {(cp_rank, dp_rank)} partitions the vocab as {group_intervals}, "
                f"others as {intervals}",
            )
    _check_global_tokens(by_key, expected_row_keys)
    assert intervals is not None

    active = sum(
        int(readback.evidence["active_count"]) for key, readback in by_key.items() if key[0] == 0
    )
    ranks = tuple(
        {
            "global_rank": int(readback.coordinates["global_rank"]),
            "tp_cp_dp": list(key),
            "hostname": readback.coordinates.get("hostname"),
            "device": dict(readback.coordinates.get("device") or {}),
            "entrypoint": readback.evidence["entrypoint"],
            "vocab_ownership": list(readback.evidence["vocab_ownership"]),
            "weight_shard_sha256": readback.evidence["weight_shard"]["sha256"],
            "active_bytes_sha256": readback.evidence["active_bytes_sha256"],
            "snapshot_fingerprint": readback.evidence["snapshot_fingerprint"],
        }
        for key, readback in sorted(by_key.items())
    )
    return RankSetReport(
        topology=topology.to_dict(),
        ranks=ranks,
        vocab_ownership=intervals,
        active_tokens=active,
        missing_evidence=_merged_missing(ordered),
    )


__all__ = [
    "ExpectedRankTopology",
    "RANK_READBACK_SCHEMA_VERSION",
    "RANK_SET_SCHEMA_VERSION",
    "RankReadback",
    "RankSetReport",
    "gather_rank_readbacks",
    "observe_rank_coordinates",
    "validate_rank_set",
]
