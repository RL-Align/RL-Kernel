# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

import pytest

from rl_engine.p6.contract import CombinePlan, ContractError, check_compatibility, manifest
from rl_engine.p6.fixtures import make_case
from rl_engine.p6.mocks import ep_return
from rl_engine.p6.negative import NEGATIVES, run_negative
from rl_engine.p6.oracle import forward
from rl_engine.p6.recordings import boundary_recordings, catalog, run_reference

DATA = Path(__file__).parent / "data"
RECORDINGS = boundary_recordings()


def test_frozen_contract_and_catalog():
    assert manifest() == json.loads((DATA / "contract.v1.json").read_text())
    assert catalog() == json.loads((DATA / "catalog.v1.json").read_text())


def test_contract_drift_rejected():
    candidate = manifest()
    candidate["policy"]["route_weight_applied_count"] = 2
    with pytest.raises(ContractError, match="SCHEMA_MISMATCH"):
        check_compatibility(candidate)


@pytest.mark.parametrize(
    "record", RECORDINGS, ids=[r["case_id"] + ":" + r["operator"] for r in RECORDINGS]
)
def test_each_owner_can_run_independent_recording(record):
    before = copy.deepcopy(record["inputs"])
    assert run_reference(record["operator"], record["inputs"]) == record["expected"]
    assert record["inputs"] == before


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_executable_negative_fixture(name):
    assert run_negative(name)["actual"] == NEGATIVES[name]


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["full", "mixed", "zero-routes"])
def test_ep_mock_layouts_and_zero_count_peers(ep, mode):
    case = make_case("mock", n=1, h=7, mode=mode)
    plan = CombinePlan.from_dict(case["plan"])
    baseline = forward(plan, case["rows"], case["shared"], case["residual"], plan.context)
    for placement in range(min(ep, 2)):
        for reverse in (False, True):
            result = ep_return(
                plan, case["rows"], plan.context, ep=ep, placement=placement, reverse=reverse
            )
            assert sum(result["counts"]) == len(case["rows"])
            assert result["transport_executed"] is False
            actual = forward(plan, result["rows"], case["shared"], case["residual"], plan.context)
            assert actual["stages"] == baseline["stages"]
            if ep == 8 and mode == "full":
                assert 0 in result["counts"]
