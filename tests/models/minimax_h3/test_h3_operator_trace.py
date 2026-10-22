# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_operator_trace`` (RFC #420): the per-node ledger of what actually ran.

CPU tests drive the ledger with synthetic runs over the full fingerprinted graph and
check that every way a run can fall short of "strict" is reported. A CUDA test runs
the one-block replay of block 0 with the trace attached and checks it reports the
bindings ``h3_block`` declares.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3 import operator_trace as ot
from rl_engine.runtime.semantic_registry import (
    OperatorBackendDescriptor,
    OperatorRequirements,
    SemanticOperatorCatalog,
)

GRAPH = fp.build_graph()
BLOCK0 = [n.path for n in GRAPH if n.path.startswith("transformer_blocks.0.")]


def _run(trace: ot.OperatorTrace, nodes=GRAPH, **overrides) -> ot.OperatorTrace:
    for node in nodes:
        values = dict(status="row", backend=f"cuda.{node.row}", dtype="bfloat16")
        values.update(overrides)
        trace.record_node(node.path, **values)
    return trace


def test_full_graph_trace_with_all_rows_is_strict() -> None:
    trace = _run(ot.OperatorTrace())
    assert len(trace.calls) == len(GRAPH)
    trace.assert_strict()
    assert trace.violations() == []
    assert trace.coverage() == {"wrong_count": [], "unexpected": []}


def test_accumulator_follows_the_row_contract() -> None:
    trace = _run(ot.OperatorTrace())
    by_path = {call.path: call for call in trace.calls}
    assert by_path["transformer_blocks.3.q_proj"].accumulator == "fp32"
    assert by_path["transformer_blocks.3.attention"].accumulator == "fp32"
    assert by_path["transformer_blocks.3.swiglu"].accumulator == "none"
    assert by_path["transformer_blocks.3.rope_q"].accumulator == "none"
    assert ot.expected_accumulator("video_input_projection_fp32") == "fp32"


def test_record_rejects_unknown_paths_wrong_rows_and_statuses() -> None:
    trace = ot.OperatorTrace()
    with pytest.raises(ValueError, match="not a node"):
        trace.record_node("transformer_blocks.50.attention", status="row", backend="x")
    with pytest.raises(ValueError, match="graph assigns"):
        trace.record(
            ot.OperatorCall("transformer_blocks.0.attention", "h3_swiglu", "row", "x")
        )
    with pytest.raises(ValueError, match="status"):
        trace.record_node("transformer_blocks.0.attention", status="maybe", backend="x")


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"status": "interim"}, "bound as 'interim'"),
        ({"status": "reference"}, "bound as 'reference'"),
        ({"fallback": True}, "fallback"),
        ({"requested_backend": "rlkernel.other"}, "requested"),
        ({"split_policy": "split_k"}, "split policy"),
        ({"accumulator": "bf16"}, "accumulator"),
        ({"backend": ""}, "no backend"),
    ],
)
def test_every_strictness_violation_is_reported(overrides, needle) -> None:
    trace = ot.OperatorTrace()
    _run(trace, nodes=GRAPH)
    bad = ot.OperatorTrace()
    for call in trace.calls:
        bad.record(replace(call, **overrides) if call.path == "transformer_blocks.7.q_proj" else call)
    problems = bad.violations()
    assert any("transformer_blocks.7.q_proj" in p and needle in p for p in problems), problems
    with pytest.raises(ValueError, match="not strict"):
        bad.assert_strict()


def test_interim_and_reference_can_be_allowed_explicitly() -> None:
    trace = ot.OperatorTrace()
    for node in GRAPH:
        status = "interim" if node.node in ("q_proj", "k_proj") else "row"
        trace.record_node(node.path, status=status, backend="b")
    assert trace.violations()
    assert trace.violations(allowed_statuses=("row", "interim")) == []


def test_coverage_reports_missing_repeated_and_foreign_nodes() -> None:
    trace = _run(ot.OperatorTrace(), nodes=[n for n in GRAPH if n.path != "norm_out"])
    trace.record_node("proj_in", status="row", backend="b")  # twice now
    cov = trace.coverage()
    assert any(line.startswith("norm_out: ran 0x") for line in cov["wrong_count"])
    assert any(line.startswith("proj_in: ran 2x") for line in cov["wrong_count"])
    assert trace.coverage(paths=["proj_in"])["unexpected"] != []  # everything else is foreign


def test_forwards_expects_each_node_that_many_times() -> None:
    trace = ot.OperatorTrace(forwards=2)
    _run(trace)
    assert trace.violations() != []
    _run(trace)
    assert trace.violations() == []


def test_subset_expectation_for_a_one_block_replay() -> None:
    trace = _run(ot.OperatorTrace(), nodes=[n for n in GRAPH if n.path in BLOCK0])
    assert trace.violations() != []  # the full graph did not run
    assert trace.violations(paths=BLOCK0) == []


def test_digest_is_stable_and_content_sensitive() -> None:
    a, b = _run(ot.OperatorTrace()), _run(ot.OperatorTrace())
    assert a.digest == b.digest
    c = _run(ot.OperatorTrace(), backend="cuda.other")
    assert c.digest != a.digest
    d = ot.OperatorTrace(metadata={"seed": 1})
    assert _run(d).digest != a.digest


def test_roundtrip_and_tamper_detection() -> None:
    trace = _run(ot.OperatorTrace(metadata={"run": "x"}))
    again = ot.OperatorTrace.loads(trace.dumps())
    assert again.digest == trace.digest and again.calls == trace.calls
    raw = trace.to_dict()
    raw["calls"][5]["backend"] = "edited"
    with pytest.raises(ValueError, match="trace_sha256"):
        ot.OperatorTrace.from_dict(raw)
    raw = trace.to_dict()
    raw["graph"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="different H3 graph"):
        ot.OperatorTrace.from_dict(raw)
    raw = trace.to_dict()
    raw["schema_version"] = "v0"
    with pytest.raises(ValueError, match="schema"):
        ot.OperatorTrace.from_dict(raw)


def test_first_difference_names_the_first_diverging_call() -> None:
    a, b = _run(ot.OperatorTrace()), ot.OperatorTrace()
    for call in a.calls:
        b.record(replace(call, backend="other") if call.path.endswith("4.attention") else call)
    diff = ot.first_difference(a, b)
    assert diff["path"] == "transformer_blocks.4.attention"
    assert diff["fields"] == ["backend"]
    assert ot.first_difference(a, a) is None
    shorter = ot.OperatorTrace()
    shorter.calls = a.calls[:-3]
    assert ot.first_difference(a, shorter)["fields"] == ["length"]


def test_status_summary_counts_per_row() -> None:
    trace = ot.OperatorTrace()
    for node in GRAPH:
        trace.record_node(node.path, status="row" if node.node != "o_proj" else "interim", backend="b")
    summary = trace.status_summary()
    assert summary["h3_attention_o_gemm"] == {"interim": 50 + 2}
    assert summary["h3_full_attention"] == {"row": 50 + 2}


def test_merge_requires_the_same_graph() -> None:
    a = ot.OperatorTrace(graph=GRAPH[:20])
    b = ot.OperatorTrace()
    with pytest.raises(ValueError, match="different graphs"):
        a.merge(b)


@pytest.mark.parametrize(
    "name, block, expected",
    [
        ("block.7.attention", None, "transformer_blocks.7.attention"),
        ("refiner.1.q_proj", None, "token_refiner.refiner_blocks.1.q_proj"),
        ("attention", 3, "transformer_blocks.3.attention"),
        ("norm_out", None, "norm_out"),
    ],
)
def test_graph_path_normalisation(name, block, expected) -> None:
    assert ot.graph_path(name, block_index=block) == expected


# --- registry bridge ----------------------------------------------------------------------------


class _FakeNorm:
    def __call__(self, x):
        return x


def _descriptor(**overrides) -> OperatorBackendDescriptor:
    values = dict(
        semantic_op="h3_rmsnorm",
        backend_id="rlkernel.test_h3_rmsnorm",
        supported_targets=frozenset({"rollout", "training"}),
        supported_devices=frozenset({"cpu"}),
        supported_dtypes=frozenset({"bfloat16"}),
        supported_topologies={"*": "*"},
        determinism_or_alignment_properties={"strict_observable": True},
        lifecycle="request",
        implementation_class_or_factory=_FakeNorm,
        fallback_policy="error",
        version_or_build_fingerprint="test-1",
        is_smoke_only=True,
    )
    values.update(overrides)
    return OperatorBackendDescriptor(**values)


def _resolve(descriptor: OperatorBackendDescriptor, **topology):
    from rl_engine.runtime.semantic_registry import OperatorResolutionPolicy

    catalog = SemanticOperatorCatalog([descriptor])
    session = catalog.session(OperatorResolutionPolicy(strict=True, allow_test_backends=True))
    resolution = session.resolve(
        semantic_op=descriptor.semantic_op,
        requested_backend=descriptor.backend_id,
        target="training",
        requirements=OperatorRequirements(device="cpu", dtype="bfloat16", topology=topology),
    )
    instance = session.instantiate(resolution)
    return resolution, session.instance_provenance(resolution, instance)


def test_call_from_a_resolution_carries_registry_provenance() -> None:
    resolution, provenance = _resolve(_descriptor())
    call = ot.trace_call_from_resolution(
        "transformer_blocks.0.norm1", resolution=resolution, provenance=provenance, shapes=[(1, 8)]
    )
    assert call.row == "h3_rmsnorm" and call.backend == "rlkernel.test_h3_rmsnorm"
    assert call.requested_backend == call.backend
    assert call.instance_fingerprint == provenance.instance_fingerprint != ""
    assert call.implementation.endswith("_FakeNorm")
    assert call.fallback is False and call.shapes == ((1, 8),)
    assert call.accumulator == "fp32"
    trace = ot.OperatorTrace()
    trace.record(call)
    assert trace.violations(paths=["transformer_blocks.0.norm1"]) == []


def test_call_from_a_resolution_keeps_topology_and_rejects_unknown_paths() -> None:
    resolution, provenance = _resolve(_descriptor(), tp=4, cp=2)
    call = ot.trace_call_from_resolution(
        "transformer_blocks.0.norm1", resolution=resolution, provenance=provenance
    )
    assert dict(call.topology) == {"tp": 4, "cp": 2}
    with pytest.raises(ValueError, match="not a node"):
        ot.trace_call_from_resolution("nope", resolution=resolution, provenance=provenance)


def test_unresolved_resolution_cannot_become_a_call() -> None:
    resolution, _ = _resolve(_descriptor())
    broken = replace(resolution, trace=replace(resolution.trace, status="unsupported"))
    with pytest.raises(ValueError, match="nothing ran"):
        ot.trace_call_from_resolution("transformer_blocks.0.norm1", resolution=broken)


# --- the real one-block replay (CUDA) ----------------------------------------------------------


def test_block_replay_trace_matches_h3_block_bindings(h3_block_weights_cuda) -> None:
    torch = pytest.importorskip("torch")
    from rl_engine.validation.models import h3_block

    ops = h3_block.CandidateOps()
    ctx = h3_block.make_block_context(h3_block_weights_cuda, layout="tiny", seed=0, batch=1)
    trace = ot.OperatorTrace(mode="candidate", metadata={"layout": "tiny", "block": 0})
    ctx["node_observer"] = ot.block_observer(trace, block_index=0)
    with torch.no_grad():
        h3_block.run_graph("candidate", ctx, ops)

    assert [c.path for c in trace.calls] == BLOCK0
    assert trace.coverage(paths=BLOCK0) == {"wrong_count": [], "unexpected": []}
    bindings = {n.name: n.binding for n in h3_block.NODES}
    for call, node in zip(trace.calls, h3_block.NODES):
        assert (call.status, call.backend, call.row) == (
            bindings[node.name].status,
            bindings[node.name].backend,
            bindings[node.name].rfc_row,
        )
    # Today's stack still has interim and reference nodes: the strict gate must say so.
    assert trace.violations(paths=BLOCK0)
    assert trace.violations(
        paths=BLOCK0, allowed_statuses=("row", "interim", "reference")
    ) == []
    # Replaying twice gives the same ledger: the trace is part of the repeatable surface.
    again = ot.OperatorTrace(mode="candidate", metadata={"layout": "tiny", "block": 0})
    ctx["node_observer"] = ot.block_observer(again, block_index=0)
    with torch.no_grad():
        h3_block.run_graph("candidate", ctx, ops)
    assert again.digest == trace.digest
