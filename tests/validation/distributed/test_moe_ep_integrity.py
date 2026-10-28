# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from dataclasses import replace

import pytest
import torch

from rl_engine.validation.moe.ep_integrity import EPRankResults, audit_ep_results


def _case(world=2, skew=False):
    counts = (2,) + (0,) * (world - 1)
    owners = tuple(range(world))
    routes = torch.tensor(
        [
            [
                0,
                token,
                slot,
                0 if skew else (token + slot) % world,
                0 if skew else (token + slot) % world,
            ]
            for token in range(2)
            for slot in range(2)
        ],
        dtype=torch.int64,
    )
    values = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    batches = []
    for rank in range(world):
        indices = torch.nonzero(routes[:, 4] == rank).flatten().flip(0)
        batches.append(
            EPRankResults(
                rank=rank,
                mapping_run_id="run-1",
                producer_backend="simulation",
                producer_build_fingerprint="fixture-v1",
                routes=routes[indices, :4],
                values=values[indices],
            )
        )
    return dict(
        expected_routes=routes,
        expected_values=values,
        rank_results=batches,
        source_num_tokens=counts,
        top_k=2,
        expert_to_rank=owners,
        mapping_run_id="run-1",
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("skew", [False, True])
def test_ep_integrity_accepts_complete_shuffled_slots_and_empty_ranks(world, skew):
    case = _case(world, skew)
    report = audit_ep_results(**case)
    assert report.passed
    assert report.expected_slots == report.received_slots == 4
    assert report.issue_count == 0
    assert len(report.producers) == world
    case["rank_results"] = tuple(reversed(case["rank_results"]))
    assert audit_ep_results(**case).to_dict() == report.to_dict()


@pytest.mark.parametrize(
    "fault,code",
    [
        ("missing", "missing_token_slot"),
        ("duplicate", "duplicate_token_slot"),
        ("token", "unexpected_token_slot"),
        ("expert", "expert_mismatch"),
        ("payload", "payload_bytes_mismatch"),
        ("run", "mapping_run_id_mismatch"),
        ("schema", "mapping_schema_mismatch"),
        ("rank", "missing_rank"),
        ("owner", "expert_rank_mismatch"),
        ("duplicate_rank", "duplicate_rank"),
    ],
)
def test_ep_integrity_detects_injected_faults(fault, code):
    case = _case()
    batches = list(case["rank_results"])
    batch = batches[0]
    if fault == "missing":
        batches[0] = replace(batch, routes=batch.routes[:1], values=batch.values[:1])
    elif fault == "duplicate":
        batches[0] = replace(batch, routes=batch.routes[[0, 0]], values=batch.values[[0, 0]])
    elif fault in ("token", "expert"):
        routes = batch.routes.clone()
        routes[0, 1 if fault == "token" else 3] = 99
        batches[0] = replace(batch, routes=routes)
    elif fault == "payload":
        values = batch.values.clone()
        values[0, 0] += 1
        batches[0] = replace(batch, values=values)
    elif fault == "run":
        batches[0] = replace(batch, mapping_run_id="old-run")
    elif fault == "schema":
        batches[0] = replace(batch, mapping_schema="old-schema")
    elif fault == "rank":
        batches.pop()
    elif fault == "owner":
        batches = [replace(batches[1], rank=0), replace(batches[0], rank=1)]
    else:
        batches.append(batch)
    case["rank_results"] = batches
    report = audit_ep_results(**case)
    assert not report.passed
    assert code in {issue.code for issue in report.issues}


def test_ep_integrity_checks_original_bytes_and_bounded_error_logs():
    case = _case(1)
    case["expected_values"][0, 0] = 0.0
    batch = case["rank_results"][0]
    values = batch.values.clone()
    # Rows arrive reversed; logical row zero is last.
    values[-1, 0] = -0.0
    case["rank_results"] = [replace(batch, values=values, mapping_run_id="stale")]
    report = audit_ep_results(**case, max_issues=1)
    assert not report.passed
    assert report.issue_count == 2
    assert len(report.issues) == 1
    report = audit_ep_results(**case)
    issue = next(issue for issue in report.issues if issue.code == "payload_bytes_mismatch")
    assert (issue.source_rank, issue.token, issue.slot) == (0, 0, 0)


def test_ep_integrity_rejects_invalid_or_vacuous_oracles():
    case = _case()
    case["source_num_tokens"] = (0, 0)
    with pytest.raises(ValueError, match="non-empty"):
        audit_ep_results(**case)
    case = _case()
    case["expected_routes"][0] = case["expected_routes"][1]
    with pytest.raises(ValueError, match="duplicate"):
        audit_ep_results(**case)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("slot", "unexpected_token_slot"),
        ("dtype", "payload_layout_mismatch"),
        ("routes", "route_layout_mismatch"),
        ("producer", "missing_producer_identity"),
    ],
)
def test_ep_integrity_rejects_bad_received_layout_and_identity(fault, code):
    case = _case()
    batches = list(case["rank_results"])
    batch = batches[0]
    if fault == "slot":
        routes = batch.routes.clone()
        routes[0, 2] = 2
        batches[0] = replace(batch, routes=routes)
    elif fault == "dtype":
        batches[0] = replace(batch, values=batch.values.float())
    elif fault == "routes":
        batches[0] = replace(batch, routes=batch.routes.int())
    else:
        batches[0] = replace(batch, producer_backend="")
    case["rank_results"] = batches
    report = audit_ep_results(**case)
    assert not report.passed
    assert code in {issue.code for issue in report.issues}
