# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P2b: correlate recorded P1–P6 ``TraceEnvelope`` evidence with a training call.

Envelopes are forwarded verbatim (deep JSON copy plus digest); T05 never
recomputes, patches or re-judges producer results. Semantic judgement of the
forwarded verdicts belongs to T07/T09. The required field list is a T05 draft
until T01 freezes the public ``TraceEnvelope`` schema and is injectable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    T05ContractError,
    canonical_sha256,
)

UPSTREAM_EVIDENCE_SCHEMA_VERSION = "rlkernel.p7.t05.upstream_evidence.v1-draft"
COMPARISON_MODES = frozenset(
    {"exact_identity", "exact_discrete", "byte_exact", "upstream_contract"}
)
DRAFT_REQUIRED_ENVELOPE_FIELDS = (
    "producer",
    "upstream_contract_id",
    "producer_schema_version",
    "boundary",
    "run_id",
    "case_id",
    "absolute_layer",
    "sublayer_site",
    "pass_name",
    "event_index",
    "global_token_ids",
    "rank",
    "requested_backend",
    "actual_backend",
    "kernel_build",
    "comparison_mode",
    "producer_verdict",
    "fixture_checksum",
)


@dataclass(frozen=True)
class UpstreamCorrelationKey:
    """Identity a forwarded envelope must belong to."""

    case_id: str
    run_id: str
    rank: str
    global_tokens: tuple[tuple[str, int], ...]  # (sample_id, position) of the call's active rows

    def __post_init__(self) -> None:
        tokens = tuple((str(sample), int(position)) for sample, position in self.global_tokens)
        if not tokens or len(set(tokens)) != len(tokens):
            raise T05ContractError(
                "AMBIGUOUS_GLOBAL_TOKEN_MAPPING", "global_tokens", "empty or duplicate tokens"
            )
        object.__setattr__(self, "global_tokens", tokens)


@dataclass(frozen=True)
class UpstreamRequirement:
    """One producer boundary T05 must correlate, with its pinned schema version."""

    producer: str
    upstream_contract_id: str
    boundary: str
    pinned_schema_version: str
    absolute_layer: int
    pass_name: str
    sublayer_site: str


@dataclass(frozen=True)
class ForwardedEnvelope:
    requirement: UpstreamRequirement
    payload: Mapping[str, Any]
    payload_sha256: str

    @property
    def order_key(self) -> tuple[int, str, str, int]:
        """Producer runtime order; M numbers are owner labels, never a sort key."""

        return (
            int(self.payload["absolute_layer"]),
            str(self.payload["sublayer_site"]),
            str(self.payload["pass_name"]),
            int(self.payload["event_index"]),
        )


@dataclass(frozen=True)
class UpstreamEvidence:
    key: UpstreamCorrelationKey
    forwarded: tuple[ForwardedEnvelope, ...]
    required_fields: tuple[str, ...]
    schema_version: str = UPSTREAM_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": {
                "case_id": self.key.case_id,
                "run_id": self.key.run_id,
                "rank": self.key.rank,
                "global_tokens": [list(token) for token in self.key.global_tokens],
            },
            "required_fields": list(self.required_fields),
            "forwarded": [
                {
                    "producer": item.requirement.producer,
                    "upstream_contract_id": item.requirement.upstream_contract_id,
                    "boundary": item.requirement.boundary,
                    "payload_sha256": item.payload_sha256,
                    "payload": copy.deepcopy(dict(item.payload)),
                }
                for item in self.forwarded
            ],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fail(status: str, field_name: str, detail: str) -> T05ContractError:
    return T05ContractError(status, field_name, detail)


def _matches(requirement: UpstreamRequirement, payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("producer") == requirement.producer
        and payload.get("upstream_contract_id") == requirement.upstream_contract_id
        and payload.get("boundary") == requirement.boundary
        and payload.get("absolute_layer") == requirement.absolute_layer
        and payload.get("pass_name") == requirement.pass_name
        and payload.get("sublayer_site") == requirement.sublayer_site
    )


def _check_envelope(
    requirement: UpstreamRequirement,
    payload: Mapping[str, Any],
    key: UpstreamCorrelationKey,
    required_fields: Sequence[str],
) -> None:
    label = f"{requirement.producer}:{requirement.boundary}"
    if payload.get("producer_verdict") in (None, ""):
        raise _fail("UPSTREAM_VERDICT_MISSING", f"{label}.producer_verdict", "no producer verdict")
    for name in required_fields:
        if payload.get(name) in (None, "", []):
            raise _fail("UPSTREAM_EVIDENCE_MISSING", f"{label}.{name}", "required field absent")
    if payload["producer_schema_version"] != requirement.pinned_schema_version:
        raise _fail(
            "UPSTREAM_CONTRACT_MISMATCH",
            f"{label}.producer_schema_version",
            f"{payload['producer_schema_version']!r} != pinned "
            f"{requirement.pinned_schema_version!r}",
        )
    mode = payload["comparison_mode"]
    if mode not in COMPARISON_MODES:
        raise _fail("UPSTREAM_CONTRACT_MISMATCH", f"{label}.comparison_mode", repr(mode))
    if mode == "upstream_contract" and payload.get("tolerance_contract_version") in (None, ""):
        raise _fail(
            "UPSTREAM_EVIDENCE_MISSING",
            f"{label}.tolerance_contract_version",
            "upstream_contract mode needs a tolerance contract version",
        )
    for field_name, expected in (("run_id", key.run_id), ("case_id", key.case_id)):
        if payload[field_name] != expected:
            raise _fail(
                "STALE_RUN_METADATA",
                f"{label}.{field_name}",
                f"{payload[field_name]!r} != {expected!r}",
            )
    if str(payload["rank"]) != key.rank:
        raise _fail("IDENTITY_DRIFT", f"{label}.rank", f"{payload['rank']!r} != {key.rank!r}")
    covered = {(str(sample), int(position)) for sample, position in payload["global_token_ids"]}
    uncovered = [token for token in key.global_tokens if token not in covered]
    if uncovered:
        raise _fail(
            "IDENTITY_DRIFT",
            f"{label}.global_token_ids",
            f"{len(uncovered)} active tokens of this call are not covered, e.g. {uncovered[:3]}",
        )


def correlate_upstream(
    key: UpstreamCorrelationKey,
    requirements: Iterable[UpstreamRequirement],
    envelopes: Iterable[Mapping[str, Any]],
    *,
    required_fields: Optional[Sequence[str]] = None,
) -> UpstreamEvidence:
    """Bind exactly one envelope to each requirement and forward it verbatim."""

    fields = tuple(required_fields or DRAFT_REQUIRED_ENVELOPE_FIELDS)
    payloads = [_plain(envelope) for envelope in envelopes]
    reqs = tuple(requirements)
    if not reqs:
        raise _fail("UPSTREAM_EVIDENCE_MISSING", "requirements", "no upstream requirement declared")
    forwarded: list[ForwardedEnvelope] = []
    for requirement in reqs:
        label = f"{requirement.producer}:{requirement.boundary}"
        matching = [payload for payload in payloads if _matches(requirement, payload)]
        if not matching:
            raise _fail("UPSTREAM_EVIDENCE_MISSING", label, "no envelope for this boundary")
        if len(matching) > 1:
            raise _fail(
                "UPSTREAM_CONTRACT_MISMATCH", label, f"{len(matching)} envelopes claim one boundary"
            )
        payload = matching[0]
        _check_envelope(requirement, payload, key, fields)
        forwarded.append(
            ForwardedEnvelope(
                requirement=requirement,
                payload=copy.deepcopy(payload),
                payload_sha256=canonical_sha256(payload),
            )
        )
    forwarded.sort(key=lambda item: item.order_key + (item.requirement.producer,))
    return UpstreamEvidence(key=key, forwarded=tuple(forwarded), required_fields=fields)


__all__ = [
    "COMPARISON_MODES",
    "DRAFT_REQUIRED_ENVELOPE_FIELDS",
    "UPSTREAM_EVIDENCE_SCHEMA_VERSION",
    "ForwardedEnvelope",
    "UpstreamCorrelationKey",
    "UpstreamEvidence",
    "UpstreamRequirement",
    "correlate_upstream",
]
