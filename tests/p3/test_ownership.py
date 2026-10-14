"""Token ownership tests (T08): six parallel modes, composed and gated.

The positive layer pins what each dimension does to the carrier set, alone and
combined. Every negative case injects exactly one defect, so a correct verdict
cannot be a coincidence of two interacting faults. Controls pin the inverse:
pipeline stages holding different layers, and TP replicas holding the same
token twice, must not be flagged.
"""

from dataclasses import replace

import pytest

from rl_engine.p3.contract import P3Error, P3Verdict
from rl_engine.p3.mocks import OWNERSHIP
from rl_engine.p3.ownership import (
    DEFAULT_RANK_ORDER,
    ParallelConfig,
    RankEvidence,
    TokenLayout,
    carriers,
    check_cross_config,
    check_rank_ownership,
    check_rank_set,
    check_replica_agreement,
    expected_decisions,
    partition_slices,
    replica_dimensions,
    validate_ownership,
)

LAYOUT = TokenLayout(sequences=((0, 1, 2, 3), (4, 5, 6, 7)), layers=(0, 1, 2, 3))
TOTAL = len(LAYOUT.decisions())  # 4 layers x 8 tokens


def _fp(layer, token):
    """A deterministic stand-in for the per-decision semantic fingerprint."""
    return f"{layer:04d}-{token:04d}"


def _evidence(config, layout=LAYOUT, fp=_fp, group=None):
    """Perfect evidence: every rank reports exactly what it carries."""
    group = group or config.describe()
    out = []
    for rank in range(config.world_size):
        decisions = expected_decisions(config, layout, rank)
        for layer in sorted({layer for layer, _ in decisions}):
            tokens = {token: fp(layer, token) for lay, token in decisions if lay == layer}
            out.append(RankEvidence(rank, group, layer, tokens))
    return out


def _for_rank(evidence, rank):
    """Every evidence record one rank produced, across all layers it owns."""
    return [item for item in evidence if item.rank == rank]


def _verdict(excinfo):
    return excinfo.value.verdict


# --- configuration arithmetic ------------------------------------------------


def test_partition_slices_gives_the_remainder_to_the_first_holders():
    assert [len(s) for s in partition_slices(10, 4)] == [3, 3, 2, 2]
    assert [len(s) for s in partition_slices(8, 4)] == [2, 2, 2, 2]
    assert [len(s) for s in partition_slices(2, 4)] == [1, 1, 0, 0]


def test_rank_coordinates_round_trip():
    config = ParallelConfig(tp=2, cp=2, dp=2, pp=2, ep=2)
    assert config.world_size == 32
    for rank in range(config.world_size):
        assert config.global_rank(**config.coordinates(rank)) == rank


def test_rank_order_is_declared_and_changes_the_decomposition():
    layout = ParallelConfig(tp=2, pp=2)
    reordered = ParallelConfig(tp=2, pp=2, rank_order=("pp", "cp", "ep", "dp", "tp"))
    assert layout.coordinates(1) == {"tp": 1, "cp": 0, "ep": 0, "dp": 0, "pp": 0}
    assert reordered.coordinates(1) == {"pp": 1, "cp": 0, "ep": 0, "dp": 0, "tp": 0}


def test_invalid_configuration_fails_closed():
    for kwargs in ({"tp": 0}, {"dp": -1}, {"rank_order": ("tp", "cp")}):
        with pytest.raises(P3Error) as excinfo:
            ParallelConfig(**kwargs)
        assert _verdict(excinfo) == P3Verdict.SCHEMA_MISMATCH


def test_rank_outside_the_world_fails_closed():
    with pytest.raises(P3Error) as excinfo:
        ParallelConfig(tp=2).coordinates(2)
    assert _verdict(excinfo) == P3Verdict.MISSING_RANK


def test_ownership_modes_come_from_the_shared_table():
    assert ParallelConfig(tp=2).ownership_of("tp") == OWNERSHIP["tp"] == "replica"
    assert ParallelConfig(tp=2, sp=True).ownership_of("tp") == OWNERSHIP["sp"]
    assert ParallelConfig().ownership_of("pp") == "replica_by_layer"
    assert ParallelConfig().ownership_of("ep") == "not_a_token_dimension"


# --- what each dimension does ------------------------------------------------


@pytest.mark.parametrize(
    ("config", "per_rank", "replicated"),
    [
        (ParallelConfig(), TOTAL, False),
        (ParallelConfig(dp=2), TOTAL // 2, False),
        (ParallelConfig(cp=2), TOTAL // 2, False),
        (ParallelConfig(pp=2), TOTAL // 2, False),
        (ParallelConfig(tp=2, sp=True), TOTAL // 2, False),
        (ParallelConfig(tp=2), TOTAL, True),
        (ParallelConfig(ep=2), TOTAL, True),
    ],
)
def test_single_dimension_ownership(config, per_rank, replicated):
    owners = carriers(config, LAYOUT)
    assert len(owners) == TOTAL  # the job's decision space is always covered
    for rank in range(config.world_size):
        assert len(expected_decisions(config, LAYOUT, rank)) == per_rank
    multi = [d for d, ranks in owners.items() if len(ranks) > 1]
    assert bool(multi) is replicated


def test_pipeline_stages_own_disjoint_layers():
    config = ParallelConfig(pp=2)
    stage0 = {layer for layer, _ in expected_decisions(config, LAYOUT, 0)}
    stage1 = {layer for layer, _ in expected_decisions(config, LAYOUT, 1)}
    assert stage0 == {0, 1}
    assert stage1 == {2, 3}
    assert not stage0 & stage1


def test_sequence_parallel_subdivides_inside_the_tp_group():
    plain = ParallelConfig(tp=2)
    sharded = ParallelConfig(tp=2, sp=True)
    assert expected_decisions(plain, LAYOUT, 0) == expected_decisions(plain, LAYOUT, 1)
    assert not expected_decisions(sharded, LAYOUT, 0) & expected_decisions(sharded, LAYOUT, 1)


def test_cp_then_sp_split_the_sequence_in_that_order():
    layout = TokenLayout.flat(range(8), layers=(0,))
    config = ParallelConfig(tp=2, sp=True, cp=2)
    held = {r: sorted(t for _, t in expected_decisions(config, layout, r)) for r in range(4)}
    # CP halves the sequence first; SP halves each CP chunk inside the TP group.
    assert sorted(held.values()) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_combined_configuration_replicates_across_tp_and_ep_only():
    config = ParallelConfig(tp=2, cp=2, dp=2, pp=2, ep=2)
    owners = carriers(config, LAYOUT)
    assert len(owners) == TOTAL
    assert {len(ranks) for ranks in owners.values()} == {4}  # tp x ep
    assert replica_dimensions(config) == ("tp", "ep")
    assert replica_dimensions(ParallelConfig(tp=2, sp=True, ep=2)) == ("ep",)


def test_perfect_evidence_passes_for_every_configuration():
    for config in (
        ParallelConfig(),
        ParallelConfig(dp=2),
        ParallelConfig(cp=2),
        ParallelConfig(pp=2),
        ParallelConfig(tp=2),
        ParallelConfig(tp=2, sp=True),
        ParallelConfig(ep=2),
        ParallelConfig(tp=2, cp=2, dp=2, pp=2, ep=2),
    ):
        validate_ownership(config, LAYOUT, _evidence(config))


# --- layout ------------------------------------------------------------------


def test_layout_rejects_a_repeated_global_token_id():
    with pytest.raises(P3Error) as excinfo:
        TokenLayout(sequences=((0, 1), (1, 2)), layers=(0,))
    assert _verdict(excinfo) == P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_layout_rejects_an_empty_space():
    for kwargs in ({"sequences": ()}, {"layers": ()}):
        with pytest.raises(P3Error) as excinfo:
            TokenLayout(**{"sequences": ((0,),), "layers": (0,), **kwargs})
        assert _verdict(excinfo) == P3Verdict.SCHEMA_MISMATCH


# --- rank-set gate -----------------------------------------------------------


def test_missing_rank():
    config = ParallelConfig(dp=2)
    evidence = [e for e in _evidence(config) if e.rank != 1]
    with pytest.raises(P3Error) as excinfo:
        check_rank_set(config, evidence)
    assert _verdict(excinfo) == P3Verdict.MISSING_RANK


def test_duplicate_rank_layer_evidence():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    with pytest.raises(P3Error) as excinfo:
        check_rank_set(config, evidence + [evidence[0]])
    assert _verdict(excinfo) == P3Verdict.STALE_RUN_METADATA


def test_unexpected_rank():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    with pytest.raises(P3Error) as excinfo:
        check_rank_set(config, evidence + [replace(evidence[0], rank=9)])
    assert _verdict(excinfo) == P3Verdict.STALE_RUN_METADATA


def test_mixed_groups():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    evidence[0] = replace(evidence[0], group="another-run")
    with pytest.raises(P3Error) as excinfo:
        check_rank_set(config, evidence)
    assert _verdict(excinfo) == P3Verdict.STALE_RUN_METADATA


def test_stale_evidence_outranks_a_simultaneous_missing_rank():
    """A contaminated rank set makes the missing-rank finding an artefact."""
    config = ParallelConfig(dp=4)
    evidence = [e for e in _evidence(config) if e.rank != 3]
    evidence[0] = replace(evidence[0], group="another-run")
    with pytest.raises(P3Error) as excinfo:
        check_rank_set(config, evidence)
    assert _verdict(excinfo) == P3Verdict.STALE_RUN_METADATA


# --- per-rank ownership gate -------------------------------------------------


def test_rank_holding_a_decision_it_does_not_carry():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    item = evidence[0]
    stolen = next(t for t in range(8) if t not in item.fingerprints)
    item.fingerprints[stolen] = _fp(item.layer, stolen)
    with pytest.raises(P3Error) as excinfo:
        check_rank_ownership(config, LAYOUT, 0, _for_rank(evidence, 0))
    assert _verdict(excinfo) == P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_rank_missing_a_decision_it_owes():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    evidence[0].fingerprints.pop(sorted(evidence[0].fingerprints)[0])
    with pytest.raises(P3Error) as excinfo:
        check_rank_ownership(config, LAYOUT, 0, _for_rank(evidence, 0))
    assert _verdict(excinfo) == P3Verdict.INCOMPLETE_ARTIFACT


def test_unauthorized_outranks_a_simultaneous_gap():
    """Holding another rank's decision means the partition itself is wrong."""
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    item = evidence[0]
    owned = {token for _, token in expected_decisions(config, LAYOUT, 0)}
    item.fingerprints.pop(sorted(item.fingerprints)[0])
    stolen = next(t for t in range(8) if t not in owned)  # belongs to rank 1
    item.fingerprints[stolen] = _fp(item.layer, stolen)
    with pytest.raises(P3Error) as excinfo:
        check_rank_ownership(config, LAYOUT, 0, _for_rank(evidence, 0))
    assert _verdict(excinfo) == P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_pipeline_stage_is_not_flagged_for_missing_other_layers():
    """Control: a stage owes only its own layers."""
    config = ParallelConfig(pp=2)
    evidence = _evidence(config)
    for rank in range(config.world_size):
        check_rank_ownership(config, LAYOUT, rank, _for_rank(evidence, rank))


def test_tp_replicas_holding_the_same_token_are_not_duplicates():
    """Control: replication is not ambiguous ownership."""
    config = ParallelConfig(tp=2)
    evidence = _evidence(config)
    rank0 = next(e for e in evidence if e.rank == 0 and e.layer == 0)
    rank1 = next(e for e in evidence if e.rank == 1 and e.layer == 0)
    assert rank0.decisions() == rank1.decisions()
    validate_ownership(config, LAYOUT, evidence)


# --- replica agreement -------------------------------------------------------


def test_single_tp_replica_drifting_is_caught():
    config = ParallelConfig(tp=2)
    evidence = _evidence(config)
    token = sorted(evidence[1].fingerprints)[0]
    evidence[1].fingerprints[token] = "drifted"
    with pytest.raises(P3Error) as excinfo:
        check_replica_agreement(config, LAYOUT, evidence)
    assert _verdict(excinfo) == P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert "tp" in str(excinfo.value)


def test_ep_induced_drift_names_the_ep_dimension():
    """An EP-tracking difference means placement leaked into the decision."""
    config = ParallelConfig(ep=2)
    evidence = _evidence(config)
    token = sorted(evidence[1].fingerprints)[0]
    evidence[1].fingerprints[token] = "drifted"
    with pytest.raises(P3Error) as excinfo:
        check_replica_agreement(config, LAYOUT, evidence)
    assert _verdict(excinfo) == P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert "ep" in str(excinfo.value)


def test_partitioned_configurations_have_nothing_to_agree_on():
    config = ParallelConfig(dp=2, cp=2)
    check_replica_agreement(config, LAYOUT, _evidence(config))


def test_rank_set_gate_precedes_ownership():
    config = ParallelConfig(dp=2)
    evidence = _evidence(config)
    evidence[0].fingerprints[99] = "ghost"  # would be AMBIGUOUS on its own
    with pytest.raises(P3Error) as excinfo:
        validate_ownership(config, LAYOUT, [e for e in evidence if e.rank != 1])
    assert _verdict(excinfo) == P3Verdict.MISSING_RANK


# --- cross-config ------------------------------------------------------------


BASE = ParallelConfig(dp=2)
OTHER = ParallelConfig(tp=2, cp=2, pp=2)


def test_cross_config_agreement_passes():
    check_cross_config(LAYOUT, (BASE, _evidence(BASE)), (OTHER, _evidence(OTHER)))


def test_cross_config_is_token_centric_not_rank_against_rank():
    """Rank 0 holds different tokens under the two configurations."""
    base_rank0 = {t for _, t in expected_decisions(BASE, LAYOUT, 0)}
    other_rank0 = {t for _, t in expected_decisions(OTHER, LAYOUT, 0)}
    assert base_rank0 != other_rank0
    check_cross_config(LAYOUT, (BASE, _evidence(BASE)), (OTHER, _evidence(OTHER)))


def test_cross_config_decision_drift():
    other = _evidence(OTHER)
    token = sorted(other[0].fingerprints)[0]
    for item in other:  # change it on every carrier: a real drift, not a replica split
        if token in item.fingerprints and item.layer == other[0].layer:
            item.fingerprints[token] = "drifted"
    with pytest.raises(P3Error) as excinfo:
        check_cross_config(LAYOUT, (BASE, _evidence(BASE)), (OTHER, other))
    assert _verdict(excinfo) == P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH


def test_cross_config_rejects_a_broken_side_before_comparing():
    other = [e for e in _evidence(OTHER) if e.rank != 1]
    with pytest.raises(P3Error) as excinfo:
        check_cross_config(LAYOUT, (BASE, _evidence(BASE)), (OTHER, other))
    assert _verdict(excinfo) == P3Verdict.MISSING_RANK


# --- evidence construction ---------------------------------------------------


def test_rank_evidence_reads_a_route_plan():
    plan = {"per_token_fingerprints": {"0": "aa", "1": "bb"}}
    item = RankEvidence.from_plan(3, "tp1-cp1-dp1-pp1-ep1", 7, plan)
    assert item.fingerprints == {0: "aa", 1: "bb"}
    assert item.decisions() == frozenset({(7, 0), (7, 1)})


def test_describe_is_stable_and_names_sequence_parallelism():
    assert ParallelConfig().describe() == "tp1-cp1-dp1-pp1-ep1"
    assert ParallelConfig(tp=2, sp=True).describe().startswith("tp2sp-")
    assert DEFAULT_RANK_ORDER == ("tp", "cp", "ep", "dp", "pp")
