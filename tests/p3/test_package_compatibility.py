"""P3 router exports preserve typed saved objects and independent reference entry points."""

import json
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest

from rl_engine.p3 import reference
from rl_engine.p3.contract import P3OpCtxHost, P3Verdict, SavedRouteSealedV1, SavedScoreSealedV1
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.recordings import record_case


@pytest.mark.parametrize(
    "legacy,canonical,names",
    [
        (
            "router_contract",
            "contract",
            (
                "H",
                "E",
                "K",
                "EPSILON",
                "SCALE",
                "P3Verdict",
                "P3OpCtxHost",
                "SavedRouteSealedV1",
                "manifest",
            ),
        ),
        ("router_oracle", "oracle", ("stable_topk6", "hash_route_fwd", "router_sqrt_softplus_bwd")),
        ("router_torch_reference", "torch_reference", ("forward",)),
    ],
)
def test_router_paths_export_same_objects(legacy, canonical, names):
    old = import_module(f"rl_engine.p3.{legacy}")
    new = import_module(f"rl_engine.p3.{canonical}")
    for name in names:
        assert getattr(old, name) is getattr(new, name)


def test_canonical_cli_manifest_matches_contract():
    from rl_engine.p3.contract import manifest
    from rl_engine.p3.serialization import encode

    output = subprocess.check_output([sys.executable, "-m", "rl_engine.p3", "manifest"], text=True)
    assert json.loads(output) == encode(manifest())


@pytest.mark.parametrize(
    "operator",
    [
        "router_sqrt_softplus_fwd",
        "router_sqrt_softplus_bwd",
        "hash_route_fwd",
        "hash_route_bwd",
        "stable_topk6_fwd",
        "learned_route_fwd",
        "learned_route_bwd",
    ],
)
def test_independent_reference_entry_points(operator):
    case = catalog()[4 if operator.startswith("hash") else 0]
    recorded = record_case(case)
    ctx = P3OpCtxHost(
        "reference-test",
        "recorded",
        0,
        case.row_active.copy(),
        recorded["bundle"]["RoutePlan"]["route_artifact_fingerprint"],
    )
    op = recorded["operators"][operator]
    inputs = list(op["inputs"])
    if operator.endswith("_bwd"):
        key = "score" if operator.startswith("router_sqrt") else "route"
        cls = SavedScoreSealedV1 if key == "score" else SavedRouteSealedV1
        inputs[-1] = cls(**recorded["saved"][key])
    result = getattr(reference, operator)(ctx, *inputs)
    assert result.verdict == P3Verdict.PASS
    for name, expected in op["payload"].items():
        if isinstance(expected, np.ndarray):
            actual = result.payload[name]
            assert actual.dtype == expected.dtype
            # Canonical padding records belong to the assembler; raw padding saved is archive-only.
            assert actual[case.row_active].tobytes() == expected[case.row_active].tobytes()
