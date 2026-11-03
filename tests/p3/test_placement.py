"""Placement map tests (T08): both physical-id semantics, gates, and interop.

The unit layer builds plans synthetically so it runs anywhere. One integration
test drives the real ``assemble`` and the start-kit ``checker`` to prove the
global-permutation path stays byte-compatible with what the kit already
produces; it is skipped where the native bitmath library cannot be loaded.

Every negative case injects exactly one defect, so a correct verdict cannot be
a coincidence of two interacting faults.
"""

from dataclasses import replace

import pytest

from rl_engine.p3.contract import E, P3Error, P3Verdict
from rl_engine.p3.placement import (
    PADDING_EXPERT_ID,
    SEMANTICS_GLOBAL,
    SEMANTICS_RANK_LOCAL,
    STRATEGY_BLOCK,
    STRATEGY_EXPLICIT,
    STRATEGY_IDENTITY,
    STRATEGY_INTERLEAVE,
    PlacementMap,
    check_placement_version,
    placement_problems,
    validate_placement_map,
    validate_plan_placement,
)

VERSION = "ep8-block-v1"
SEMANTIC_IDS = [SEMANTICS_GLOBAL, SEMANTICS_RANK_LOCAL]


def _map(ep_size=8, strategy=STRATEGY_BLOCK, semantics=SEMANTICS_RANK_LOCAL, version=VERSION):
    return PlacementMap.from_ep_config(version, ep_size, strategy, semantics)


def _plan(placement, *, tokens=3, padding=1):
    """A minimal RoutePlan carrying only what the placement gates read."""
    core, envelopes, row = [], [], 0
    for token in range(tokens):
        for slot in range(6):
            logical = (token * 6 + slot) % placement.num_experts
            core.append({"logical_expert_id": logical, "global_token_id": token})
            envelopes.append(
                {
                    "source_row": row,
                    "physical_expert_id": placement.physical_expert_id(logical),
                    "placement_map_version": placement.version,
                    "rank": 3,  # a token-dimension rank, unrelated to ep_rank
                }
            )
        row += 1
    for _ in range(padding):
        for _slot in range(6):
            core.append({"logical_expert_id": PADDING_EXPERT_ID, "global_token_id": -1})
            envelopes.append(
                {
                    "source_row": row,
                    "physical_expert_id": PADDING_EXPERT_ID,
                    "placement_map_version": placement.version,
                    "rank": 3,
                }
            )
        row += 1
    return {"core": core, "envelope": envelopes, "placement_map": placement.plan_fragment()}


# --- generation --------------------------------------------------------------


@pytest.mark.parametrize("semantics", SEMANTIC_IDS)
@pytest.mark.parametrize("strategy", [STRATEGY_BLOCK, STRATEGY_INTERLEAVE])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
def test_generated_placements_are_valid(semantics, strategy, ep_size):
    placement = _map(ep_size, strategy, semantics)
    assert placement_problems(placement) == []
    assert sum(len(placement.experts_on_rank(r)) for r in range(ep_size)) == E
    assert len(placement.experts_on_rank(0)) == E // ep_size


@pytest.mark.parametrize("semantics", SEMANTIC_IDS)
def test_uneven_division_spreads_the_remainder(semantics):
    placement = PlacementMap.from_ep_config("v", 4, STRATEGY_BLOCK, semantics, num_experts=10)
    assert placement_problems(placement) == []
    assert [len(placement.experts_on_rank(r)) for r in range(4)] == [3, 3, 2, 2]
    assert placement.experts_on_rank(0) == (0, 1, 2)
    assert placement.experts_on_rank(3) == (8, 9)


def test_rank_local_ids_repeat_across_ranks():
    placement = _map(8, STRATEGY_BLOCK, SEMANTICS_RANK_LOCAL)
    assert placement.locate(0) == (0, 0)
    assert placement.locate(32) == (1, 0)  # same physical id, different rank
    assert max(placement.physical_of) == E // 8 - 1


def test_global_ids_are_unique_across_ranks():
    placement = _map(8, STRATEGY_BLOCK, SEMANTICS_GLOBAL)
    assert sorted(placement.physical_of) == list(range(E))
    # Rank-major concatenation: a contiguous block layout is the identity.
    assert list(placement.physical_of) == list(range(E))


def test_interleave_global_is_a_nontrivial_permutation():
    placement = PlacementMap.from_ep_config("v", 4, STRATEGY_INTERLEAVE, SEMANTICS_GLOBAL, 10)
    assert list(placement.physical_of) == [0, 3, 6, 8, 1, 4, 7, 9, 2, 5]
    assert sorted(placement.physical_of) == list(range(10))


def test_both_semantics_agree_on_position_within_a_rank():
    local = _map(8, STRATEGY_INTERLEAVE, SEMANTICS_RANK_LOCAL)
    glob = _map(8, STRATEGY_INTERLEAVE, SEMANTICS_GLOBAL)
    for rank in range(8):
        assert local.experts_on_rank(rank) == glob.experts_on_rank(rank)


def test_generation_rejects_bad_arguments():
    for kwargs in (
        {"ep_size": 0},
        {"num_experts": 0},
        {"strategy": STRATEGY_EXPLICIT},
        {"semantics": "guesswork"},
    ):
        with pytest.raises(P3Error) as excinfo:
            PlacementMap.from_ep_config("v", **{"ep_size": 8, **kwargs})
        assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP


# --- explicit tables ---------------------------------------------------------


def test_identity_matches_the_assembler_default():
    placement = PlacementMap.identity()
    assert placement_problems(placement) == []
    assert placement.strategy == STRATEGY_IDENTITY
    assert placement.semantics == SEMANTICS_GLOBAL
    assert placement.logical_to_physical() == list(range(E))


def test_explicit_pairs_accept_an_irregular_but_bijective_table():
    pairs = {0: (1, 0), 1: (0, 0), 2: (1, 1), 3: (0, 1)}
    placement = PlacementMap.from_pairs("irregular-v1", pairs, 2, num_experts=4)
    assert placement_problems(placement) == []
    assert placement.experts_on_rank(0) == (1, 3)
    assert placement.experts_on_rank(1) == (0, 2)


def test_lookup_outside_the_table_fails_closed():
    with pytest.raises(P3Error) as excinfo:
        _map().locate(E)
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP


# --- the ambiguity this module makes explicit --------------------------------


def test_rank_local_table_is_rejected_when_declared_global():
    """The open contract question, pinned as a test.

    A rank-local table judged under the global reading fails, and vice versa.
    Neither reading is assumed: the map declares which one it was built under.
    """
    rank_local = _map(8, STRATEGY_BLOCK, SEMANTICS_RANK_LOCAL)
    assert placement_problems(rank_local) == []

    mislabeled = replace(rank_local, semantics=SEMANTICS_GLOBAL)
    problems = placement_problems(mislabeled)
    assert problems and "permutation" in problems[0]

    glob = _map(8, STRATEGY_BLOCK, SEMANTICS_GLOBAL)
    assert placement_problems(replace(glob, semantics=SEMANTICS_RANK_LOCAL))


def test_fingerprint_separates_the_two_readings():
    local = _map(8, STRATEGY_BLOCK, SEMANTICS_RANK_LOCAL)
    glob = _map(8, STRATEGY_BLOCK, SEMANTICS_GLOBAL)
    assert local.fingerprint() != glob.fingerprint()


def test_fingerprint_separates_version_strategy_and_table():
    base = _map()
    assert base.fingerprint() == _map().fingerprint()
    assert base.fingerprint() != _map(version="ep8-block-v2").fingerprint()
    assert base.fingerprint() != _map(strategy=STRATEGY_INTERLEAVE).fingerprint()
    assert base.fingerprint() != _map(ep_size=4).fingerprint()


# --- structural gate ---------------------------------------------------------


def _assert_invalid(placement, needle):
    problems = placement_problems(placement)
    assert problems, "expected a structural defect"
    assert any(needle in p for p in problems), problems
    with pytest.raises(P3Error) as excinfo:
        validate_placement_map(placement)
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP


def test_gap():
    pairs = {i: (0, i) for i in range(4)}
    del pairs[2]  # one hole, nothing else touched
    _assert_invalid(PlacementMap.from_pairs("v", pairs, 1, num_experts=4), "gaps")


def test_overlap_rank_local():
    pairs = {0: (0, 0), 1: (0, 1), 2: (0, 1), 3: (0, 2)}  # 1 and 2 collide
    _assert_invalid(
        PlacementMap.from_pairs("v", pairs, 1, num_experts=4), "more than one logical expert"
    )


def test_duplicate_physical_under_global():
    pairs = {0: (0, 0), 1: (0, 1), 2: (0, 1), 3: (0, 2)}
    placement = PlacementMap.from_pairs("v", pairs, 1, semantics=SEMANTICS_GLOBAL, num_experts=4)
    _assert_invalid(placement, "repeated physical ids")


def test_ep_rank_out_of_range():
    pairs = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (9, 0)}  # ep_size is 1
    _assert_invalid(PlacementMap.from_pairs("v", pairs, 1, num_experts=4), "outside [0, 1)")


def test_noncontiguous_rank_local_slots():
    pairs = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 7)}
    _assert_invalid(PlacementMap.from_pairs("v", pairs, 1, num_experts=4), "not 0..3")


def test_table_length_disagrees_with_num_experts():
    _assert_invalid(replace(_map(), num_experts=E + 1), "num_experts is 257")


def test_rank_and_physical_tables_must_align():
    placement = _map()
    _assert_invalid(replace(placement, ep_rank_of=placement.ep_rank_of[:-1]), "rank entries")


def test_empty_version():
    _assert_invalid(_map(version=""), "version is empty")


def test_unknown_semantics():
    _assert_invalid(replace(_map(), semantics="guesswork"), "unknown semantics")


def test_unknown_strategy():
    _assert_invalid(replace(_map(), strategy="mystery"), "unknown strategy")


# --- version gate ------------------------------------------------------------


def test_version_agreement_passes():
    check_placement_version(VERSION, _map())


def test_version_mismatch_fails_closed():
    with pytest.raises(P3Error) as excinfo:
        check_placement_version("ep8-block-v2", _map())
    assert excinfo.value.verdict == P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH


# --- plan gate ---------------------------------------------------------------


@pytest.mark.parametrize("semantics", SEMANTIC_IDS)
def test_wellformed_plan_passes(semantics):
    placement = _map(8, STRATEGY_BLOCK, semantics)
    validate_plan_placement(_plan(placement), placement)


@pytest.mark.parametrize("semantics", SEMANTIC_IDS)
def test_plan_fragment_round_trips_without_an_external_map(semantics):
    placement = _map(8, STRATEGY_INTERLEAVE, semantics)
    plan = _plan(placement)
    validate_plan_placement(plan)  # map rebuilt from the fragment
    rebuilt = PlacementMap.from_plan_fragment(plan["placement_map"])
    assert rebuilt.fingerprint() == placement.fingerprint()


def test_envelope_rank_may_differ_from_ep_rank():
    """Control: EP is not a token dimension, so this must not be flagged."""
    placement = _map(8, STRATEGY_BLOCK, SEMANTICS_RANK_LOCAL)
    plan = _plan(placement)
    assert all(env["rank"] == 3 for env in plan["envelope"])
    active = [c for c in plan["core"] if c["logical_expert_id"] != PADDING_EXPERT_ID]
    assert {placement.ep_rank(c["logical_expert_id"]) for c in active} != {3}
    validate_plan_placement(plan, placement)


def test_row_physical_disagreeing_with_the_map():
    placement = _map()
    plan = _plan(placement)
    plan["envelope"][7]["physical_expert_id"] += 1
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, placement)
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP
    assert "row 7" in str(excinfo.value)


def test_padding_row_must_carry_minus_one():
    placement = _map()
    plan = _plan(placement)
    plan["envelope"][-1]["physical_expert_id"] = 0
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, placement)
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP
    assert "padding" in str(excinfo.value)


def test_envelope_version_drift_inside_a_plan():
    placement = _map()
    plan = _plan(placement)
    plan["envelope"][2]["placement_map_version"] = "ep8-block-v2"
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, placement)
    assert excinfo.value.verdict == P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH


def test_core_and_envelope_must_align():
    placement = _map()
    plan = _plan(placement)
    plan["envelope"].pop()
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, placement)
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP


def test_version_gate_precedes_the_structural_gate():
    """Both defects present: the verdict must name the version.

    A map from another placement cannot be judged structurally relevant.
    """
    placement = _map()
    plan = _plan(placement)
    broken = replace(placement, version="other-v9", strategy="mystery")
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, broken)
    assert excinfo.value.verdict == P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH


def test_structural_gate_precedes_row_comparison():
    placement = _map()
    plan = _plan(placement)
    plan["envelope"][0]["physical_expert_id"] = 99
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, replace(placement, strategy="mystery"))
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP
    assert "unknown strategy" in str(excinfo.value)


def test_plan_recorded_under_interleave_fails_against_a_block_map():
    interleave = _map(8, STRATEGY_INTERLEAVE, SEMANTICS_RANK_LOCAL)
    plan = _plan(interleave)
    with pytest.raises(P3Error) as excinfo:
        validate_plan_placement(plan, _map(8, STRATEGY_BLOCK, SEMANTICS_RANK_LOCAL))
    assert excinfo.value.verdict == P3Verdict.INVALID_PLACEMENT_MAP


# --- interop with the start kit ----------------------------------------------


def _assembled_plan(case, placement):
    from rl_engine.p3 import oracle
    from rl_engine.p3.assembler import assemble
    from rl_engine.p3.contract import P3OpResult

    score = oracle.router_sqrt_softplus_fwd(case.z, case.round_policy)
    s = score["s"]
    route = (
        oracle.hash_route_fwd(case.input_token_id, s, case.table)
        if case.router_mode == "hash"
        else oracle.learned_route_fwd(s, case.bias)
    )
    bundle = assemble(
        case,
        P3OpResult(P3Verdict.PASS, score),
        P3OpResult(P3Verdict.PASS, route),
        placement=placement.logical_to_physical(),
        placement_version=placement.version,
    )
    return bundle["RoutePlan"]


def _native_math_available():
    from rl_engine.p3 import bitmath

    try:
        import numpy as np

        bitmath.apply(np.zeros(1, dtype=np.float32), "sqrt")
    except OSError:
        return False
    return True


@pytest.mark.skipif(
    not _native_math_available(), reason="native bitmath library unavailable on this platform"
)
@pytest.mark.parametrize("strategy", [STRATEGY_BLOCK, STRATEGY_INTERLEAVE])
def test_global_placement_round_trips_through_the_real_assembler(strategy):
    """A generated global placement must satisfy both gates and the kit's own.

    This is the compatibility claim: under the global reading nothing about
    the existing assembler or checker has to change.
    """
    from rl_engine.p3.checker import validate_plan
    from rl_engine.p3.fixtures import catalog

    placement = PlacementMap.from_ep_config(f"ep8-{strategy}-v1", 8, strategy, SEMANTICS_GLOBAL)
    case = next(c for c in catalog() if c.row_active.any())
    plan = _assembled_plan(case, placement)

    validate_plan_placement(plan, placement)
    validate_plan({"RoutePlan": plan})
