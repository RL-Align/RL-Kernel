"""P7 cross_config binding tests (T08).

These drive the *real* `RuntimeTools`, not a stand-in: the point of the
binding is that P7's existing control plane accepts it, and a mock of
`RuntimeTools` would prove nothing about that. The adapter contract
`RuntimeTools` enforces (one application per path, lifecycle and critical
flags matching the descriptor, applied values matching the normalized case)
is therefore exercised on every case here.
"""

import pytest

from rl_engine.alignment.cross_config.planner import V1_KNOBS
from rl_engine.alignment.cross_config.runtime import (
    RuntimeBinding,
    RuntimeMaterializationError,
    RuntimeTools,
)
from rl_engine.alignment.cross_config.schema import (
    ExperimentCase,
    MaterializationStatus,
    SemanticIdentitySpec,
)
from rl_engine.p3.collective_provenance import CollectiveProvenance, GradientBucket
from rl_engine.p3.contract import P3Error, P3Verdict
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.ownership import ParallelConfig, TokenLayout, expected_decisions
from rl_engine.p3.p7_binding import (
    P3_ROUTER_KNOB_DESCRIPTORS,
    P3_ROUTER_KNOBS,
    P3_ROUTER_NORMALIZERS,
    ROUTER_OWNED_PATHS,
    P3RouterMaterializer,
    config_from_binding,
    layout_from_case,
    materialize_rank_rows,
    parallel_config_from_knobs,
    placement_from_knobs,
    rank_token_manifest,
)
from rl_engine.p3.placement import SEMANTICS_RANK_LOCAL

ROW = ((0, 1, 2, 3),)
IDENTITY = SemanticIdentitySpec(
    checkpoint_id="ckpt",
    model_version="m1",
    tokenizer_policy="frozen",
    token_ids=ROW,
    selected_token_ids=ROW,
    active_mask=((True,) * 4,),
    attention_mask=((True,) * 4,),
    position_ids=ROW,
    pre_update_state="locked",
)
LAYOUT = TokenLayout(sequences=((0, 1, 2, 3), (4, 5, 6, 7)), layers=(0, 1, 2, 3))


def _case(training=None, router=None, extra=None):
    requested = {
        "training": {
            "tensor_parallel_size": 2,
            "context_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 2,
            "expert_parallel_size": 2,
            "sequence_parallel": False,
            **(training or {}),
        },
        "router": {
            "placement_version": "ep2-block-v1",
            "placement_strategy": "block",
            "placement_semantics": SEMANTICS_RANK_LOCAL,
            **(router or {}),
        },
        "batch": {"size": 8},
        **(extra or {}),
    }
    return ExperimentCase(
        case_id="c0",
        experiment_id="e0",
        scenario_id="s0",
        identity=IDENTITY,
        requested=requested,
    )


def _collective(dp_size=2, **overrides):
    """A complete gate-gradient collective record for a DP configuration."""
    fields = {
        "dp_size": dp_size,
        "dp_ranks": tuple(range(dp_size)),
        "algorithm": "ring",
        "configured_algorithm": "ring",
        "reduce_op": "avg",
        "accumulate_dtype": "fp32",
        "process_group": f"dp:0-{dp_size - 1}",
        "source": "torch.distributed readback",
        "buckets": (GradientBucket(0, ("gate.weight",), 4096 * 256, "fp32"),),
        **overrides,
    }
    return CollectiveProvenance(**fields)


def _materialize(case=None, **kwargs):
    tools = RuntimeTools(P3_ROUTER_KNOBS)
    kwargs.setdefault("num_experts", 8)
    kwargs.setdefault("collective", _collective())
    return tools.materialize(case or _case(), P3RouterMaterializer(**kwargs))


# --- knob allowlist ----------------------------------------------------------


def test_router_knobs_extend_v1_without_editing_it():
    assert set(V1_KNOBS) < set(P3_ROUTER_KNOBS)
    for path, descriptor in V1_KNOBS.items():
        assert P3_ROUTER_KNOBS[path] is descriptor
    assert set(P3_ROUTER_NORMALIZERS) >= set(P3_ROUTER_KNOBS) - {
        "router.placement_version",
        "router.placement_strategy",
        "router.placement_semantics",
    }


def test_router_knobs_cover_every_parallel_dimension():
    paths = {descriptor.path for descriptor in P3_ROUTER_KNOB_DESCRIPTORS}
    for dimension in ("tensor", "context", "pipeline", "data", "expert"):
        assert f"training.{dimension}_parallel_size" in paths
    assert "training.sequence_parallel" in paths
    assert paths <= ROUTER_OWNED_PATHS


# --- materialization ---------------------------------------------------------


def test_valid_case_is_applied_and_executable():
    materialization = _materialize()
    assert materialization.materialized_case.status is MaterializationStatus.APPLIED
    assert materialization.executable_in_strict_mode
    RuntimeTools.require_executable(materialization, strict=True)


def test_binding_reports_the_composed_topology():
    binding = _materialize().binding
    topology = binding.topology["training"]
    assert topology["world_size"] == 32  # 2*2*2*2*2
    assert [topology[d] for d in ("tp", "cp", "pp", "dp", "ep")] == [2, 2, 2, 2, 2]
    assert topology["sequence_parallel"] is False
    assert binding.runtime_kind == "p3_router"


def test_binding_surfaces_the_ownership_table():
    side = _materialize().binding.side_configs["training"]
    assert dict(side["ownership"]) == {
        "tp": "replica",
        "cp": "partition",
        "dp": "partition",
        "pp": "replica_by_layer",
        "ep": "not_a_token_dimension",
    }
    assert side["parallel"] == "tp2-cp2-dp2-pp2-ep2"
    assert side["placement"]["version"] == "ep2-block-v1"


def test_sequence_parallel_flips_the_tp_ownership_mode():
    side = _materialize(_case(training={"sequence_parallel": True})).binding.side_configs
    assert side["training"]["ownership"]["tp"] == "partition_by_sequence"


def test_config_round_trips_through_the_binding():
    binding = _materialize().binding
    assert config_from_binding(binding).describe() == "tp2-cp2-dp2-pp2-ep2"


def test_config_from_a_foreign_binding_fails_closed():
    binding = RuntimeBinding(
        batch_size=1,
        side_configs={"rollout": {}, "training": {}},
        topology={"rollout": {"world_size": 1}, "training": {"world_size": 1}},
        scorer={},
        operator_backends={"rollout": "x", "training": "x"},
        runtime_kind="something_else",
    )
    with pytest.raises(P3Error) as excinfo:
        config_from_binding(binding)
    assert excinfo.value.verdict == P3Verdict.UPSTREAM_CONTRACT_MISMATCH


def test_a_case_declaring_no_placement_still_materializes():
    case = _case(router={"placement_version": ""})
    materialization = _materialize(case)
    assert materialization.materialized_case.status is MaterializationStatus.APPLIED
    assert materialization.binding.side_configs["training"]["placement"] is None


# --- failure reporting -------------------------------------------------------


def test_ungeneratable_placement_strategy_is_an_error_not_a_fallback():
    materialization = _materialize(_case(router={"placement_strategy": "nonsense"}))
    assert materialization.materialized_case.status is MaterializationStatus.ERROR
    assert not materialization.executable_in_strict_mode
    reasons = [
        a.evidence["reason"]
        for a in materialization.applications
        if a.status is MaterializationStatus.ERROR
    ]
    assert reasons and "nonsense" in reasons[0]
    with pytest.raises(RuntimeMaterializationError):
        RuntimeTools.require_executable(materialization, strict=False)


def test_non_positive_parallel_size_is_an_error():
    materialization = _materialize(_case(training={"expert_parallel_size": 0}))
    assert materialization.materialized_case.status is MaterializationStatus.ERROR


def test_an_engine_knob_is_unobservable_not_silently_applied():
    """The router materializer must not claim to have applied another adapter's knob."""
    case = _case(extra={"rollout": {"dtype": "bfloat16"}})
    materialization = _materialize(case)
    unobservable = {
        a.path: a.evidence["reason"]
        for a in materialization.applications
        if a.status is MaterializationStatus.UNOBSERVABLE
    }
    assert "rollout.dtype" in unobservable
    assert "engine adapter" in unobservable["rollout.dtype"]
    assert not materialization.executable_in_strict_mode


def test_a_path_without_a_descriptor_fails_closed():
    tools = RuntimeTools({path: P3_ROUTER_KNOBS[path] for path in ("batch.size",)})
    with pytest.raises(P3Error) as excinfo:
        tools.materialize(_case(), P3RouterMaterializer(num_experts=8))
    assert excinfo.value.verdict == P3Verdict.SCHEMA_MISMATCH


# --- reuse -------------------------------------------------------------------


def test_identical_cases_are_reusable_and_different_topologies_are_not():
    first = _materialize()
    second = _materialize()
    assert RuntimeTools.can_reuse(first, second)
    other = _materialize(_case(training={"expert_parallel_size": 4}))
    assert not RuntimeTools.can_reuse(first, other)


def test_parallel_sizes_are_process_scoped():
    """Changing EP cannot be done inside a live process, and must say so."""
    assert _materialize().materialized_case.isolation_scope.value == "process"


# --- knob readers ------------------------------------------------------------


def test_absent_dimensions_default_to_one():
    config = parallel_config_from_knobs({"training.expert_parallel_size": 4})
    assert (config.tp, config.cp, config.dp, config.pp, config.ep) == (1, 1, 1, 1, 4)
    assert config.world_size == 4


def test_no_placement_is_declared_rather_than_invented():
    assert placement_from_knobs({"training.expert_parallel_size": 4}) is None


def test_placement_follows_the_declared_semantics():
    flat = {
        "router.placement_version": "v1",
        "router.placement_strategy": "interleave",
        "router.placement_semantics": "global_permutation",
        "training.expert_parallel_size": 4,
    }
    placement = placement_from_knobs(flat, num_experts=8)
    assert placement.semantics == "global_permutation"
    assert sorted(placement.physical_of) == list(range(8))


def test_rank_order_override_is_honoured():
    order = ("pp", "cp", "ep", "dp", "tp")
    config = parallel_config_from_knobs(
        {"training.tensor_parallel_size": 2, "training.pipeline_parallel_size": 2},
        rank_order=order,
    )
    assert config.rank_order == order
    assert config.coordinates(1)["pp"] == 1


# --- consumer hooks ----------------------------------------------------------


def test_rank_token_manifest_covers_the_world_exactly_once_per_carrier():
    config = ParallelConfig(dp=2, pp=2)
    manifest = rank_token_manifest(config, LAYOUT)
    assert set(manifest) == set(range(config.world_size))
    for rank, by_layer in manifest.items():
        flat = {(layer, token) for layer, tokens in by_layer.items() for token in tokens}
        assert flat == expected_decisions(config, LAYOUT, rank)


def test_materialize_rank_rows_slices_a_recorded_case():
    case = next(c for c in catalog() if c.row_active.any())
    tokens = [int(t) for t in case.global_token_id[case.row_active]]
    layout = layout_from_case(case, (case.absolute_layer,))
    # CP splits within a sequence; DP splits across sequences, and
    # layout_from_case builds exactly one sequence.
    config = ParallelConfig(cp=len(tokens))

    seen = []
    for rank in range(config.world_size):
        sliced = materialize_rank_rows(case, config, layout, rank)
        held = [int(t) for t in sliced.global_token_id[sliced.row_active]]
        assert len(held) == 1
        seen.extend(held)
    assert sorted(seen) == sorted(tokens)


def test_over_provisioned_ranks_are_named_as_such():
    """More DP ranks than sequences is degenerate but legal, and not a gap."""
    case = next(c for c in catalog() if c.row_active.any())
    layout = layout_from_case(case, (case.absolute_layer,))  # one sequence
    config = ParallelConfig(dp=3)
    assert materialize_rank_rows(case, config, layout, 0) is not None
    for rank in (1, 2):
        with pytest.raises(P3Error) as excinfo:
            materialize_rank_rows(case, config, layout, rank)
        assert excinfo.value.verdict == P3Verdict.ZERO_ACTIVE_TOKENS


def test_materialize_rank_rows_rejects_a_layer_the_rank_does_not_own():
    case = next(c for c in catalog() if c.row_active.any())
    layout = layout_from_case(case, (case.absolute_layer, case.absolute_layer + 1))
    config = ParallelConfig(pp=2)
    # Stage 1 owns the second layer, so the case's own layer is not its work.
    with pytest.raises(P3Error) as excinfo:
        materialize_rank_rows(case, config, layout, 1)
    assert excinfo.value.verdict == P3Verdict.MISSING_RANK


def test_layout_from_case_uses_active_tokens_only():
    case = next(c for c in catalog() if not case_fully_active(c))
    layout = layout_from_case(case, (case.absolute_layer,))
    active = {int(t) for t in case.global_token_id[case.row_active]}
    assert set(layout.sequences[0]) == active
    assert -1 not in layout.sequences[0]  # padding never enters the decision space


def case_fully_active(case):
    return bool(case.row_active.all())


# --- gate-gradient collective provenance -------------------------------------


def test_dp_without_collective_evidence_is_unobservable():
    """DP runs an all-reduce; its reduction order cannot simply go unrecorded."""
    tools = RuntimeTools(P3_ROUTER_KNOBS)
    materialization = tools.materialize(_case(), P3RouterMaterializer(num_experts=8))
    assert materialization.materialized_case.status is MaterializationStatus.UNOBSERVABLE
    assert not materialization.executable_in_strict_mode
    gaps = {
        a.path: a.evidence["reason"]
        for a in materialization.applications
        if a.status is MaterializationStatus.UNOBSERVABLE
    }
    assert "training.data_parallel_size" in gaps
    assert "all-reduce" in gaps["training.data_parallel_size"]


def test_dp1_needs_no_collective_record():
    """No all-reduce runs at DP=1, so demanding evidence of one is theatre."""
    tools = RuntimeTools(P3_ROUTER_KNOBS)
    case = _case(training={"data_parallel_size": 1})
    materialization = tools.materialize(case, P3RouterMaterializer(num_experts=8))
    assert materialization.materialized_case.status is MaterializationStatus.APPLIED
    side = materialization.binding.side_configs["training"]
    assert side["gate_grad_collective"]["source"] == "no_collective"


def test_binding_carries_the_collective_fingerprint():
    side = _materialize().binding.side_configs["training"]
    assert side["gate_grad_collective"]["algorithm"] == "ring"
    assert side["gate_grad_collective_fingerprint"] == _collective().fingerprint()


def test_a_malformed_collective_record_is_an_error_not_a_gap():
    """Evidence was handed over and does not hold together: that is a defect."""
    broken = _collective(dp_ranks=(0, 0))  # two ranks, one of them twice
    materialization = _materialize(collective=broken)
    assert materialization.materialized_case.status is MaterializationStatus.ERROR
    reasons = [
        a.evidence["reason"]
        for a in materialization.applications
        if a.status is MaterializationStatus.ERROR
    ]
    assert reasons and "distinct ranks" in reasons[0]
