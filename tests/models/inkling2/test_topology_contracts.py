# SPDX-License-Identifier: Apache-2.0
"""INK-36, INK-37, INK-38 and INK-39 topology tests."""

import pytest

from rl_engine.models.inkling2.topology_contracts import (
    InklingTopology,
    TopologyEvidence,
    TopologyContractError,
    assess_deferred_parallelism,
    qualify_tp_ep,
)


def _evidence(topology):
    return TopologyEvidence(
        framework="sglang",
        platform="cuda",
        topology=topology,
        attention_backend="inkling2-full-attention",
        router_backend="inkling2-top8-router",
        cache_isolation=True,
        reduction_order="fixed-tree",
        source="integration-readback",
    )


def test_tp2_and_tp4_require_runtime_evidence():
    tp2 = InklingTopology(tensor_parallel=2, world_size=2)
    assert qualify_tp_ep(tp2, evidence=None).status == "gated"
    report = qualify_tp_ep(tp2, evidence=_evidence(tp2))
    assert report.ready
    assert qualify_tp_ep(
        InklingTopology(tensor_parallel=4, world_size=4),
        evidence=_evidence(InklingTopology(tensor_parallel=4, world_size=4)),
    ).ready


def test_invalid_topology_is_rejected_before_integration():
    invalid = InklingTopology(tensor_parallel=2, expert_parallel=2, world_size=2)
    report = qualify_tp_ep(invalid, evidence=None)
    assert report.status == "rejected"
    with pytest.raises(TopologyContractError, match="product"):
        invalid.validate()


def test_cp_sp_pp_are_explicitly_gated():
    for item, topology in (
        ("INK-37", InklingTopology(context_parallel=2, world_size=2)),
        ("INK-38", InklingTopology(sequence_parallel=2, world_size=2)),
        ("INK-39", InklingTopology(pipeline_parallel=2, world_size=2)),
    ):
        report = assess_deferred_parallelism(item, topology)
        assert report.status == "gated"
        assert report.blockers


def test_evidence_requires_cache_and_backend_readback():
    topology = InklingTopology(tensor_parallel=2, world_size=2)
    evidence = TopologyEvidence(
        "sglang", "cuda", topology, "attention", "router", False, "fixed-tree", "probe"
    )
    report = qualify_tp_ep(topology, evidence=evidence)
    assert report.status == "rejected"
    assert "cache" in report.blockers[0]
