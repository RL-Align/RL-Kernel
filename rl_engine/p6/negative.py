# SPDX-License-Identifier: Apache-2.0
"""Executable negative fixtures; a rejection is evidence, not production PASS."""

from dataclasses import replace

from .contract import CombinePlan, ContractError, production_provider, require
from .fixtures import make_case
from .oracle import backward, forward, mock_return

NEGATIVES = {
    "wrong_run": "STALE_RUN_METADATA",
    "wrong_checkpoint": "STALE_RUN_METADATA",
    "wrong_schema": "SCHEMA_MISMATCH",
    "policy_drift": "SCHEMA_MISMATCH",
    "duplicate_slot": "INVALID_DISCRETE_PLAN",
    "missing_slot": "INVALID_DISCRETE_PLAN",
    "bad_slot": "INVALID_DISCRETE_PLAN",
    "bad_token": "INVALID_DISCRETE_PLAN",
    "stale_backward": "STALE_RUN_METADATA",
    "corrupt_saved": "CORRUPT_METADATA",
    "wrong_gradient_boundary": "GRADIENT_BOUNDARY_MISMATCH",
    "duplicate_arrival": "INVALID_DISCRETE_PLAN",
    "active_nonfinite": "NON_FINITE",
    "early_dtype": "DTYPE_MISMATCH",
    "production_unavailable": "UNSUPPORTED_CAPABILITY",
}


def run_negative(name):
    require(name in NEGATIVES, "UNSUPPORTED_CAPABILITY", name)
    case = make_case("negative", n=1, h=3)
    plan = CombinePlan.from_dict(case["plan"])
    context = plan.context
    saved = forward(plan, case["rows"], case["shared"], case["residual"], context)["saved"]

    def invoke():
        nonlocal plan, context, saved
        if name == "wrong_run":
            context = replace(context, run_id="other")
        elif name == "wrong_checkpoint":
            context = replace(context, checkpoint_id="other")
        elif name == "wrong_schema":
            plan = replace(plan, schema="old")
        elif name == "policy_drift":
            plan = replace(plan, policy_hash="0" * 64)
        elif name in ("duplicate_slot", "missing_slot", "bad_slot", "bad_token"):
            inv = list(plan.inverse_map)
            if name == "duplicate_slot":
                inv[1] = inv[0]
            elif name == "missing_slot":
                inv.pop()
            elif name == "bad_slot":
                inv[0] = (inv[0][0], 6, True)
            else:
                inv[0] = (0, inv[0][1], True)
            plan = replace(plan, inverse_map=tuple(inv))
        elif name == "stale_backward":
            return saved.restore(replace(context, forward_id="other"), plan.fingerprint)
        elif name == "corrupt_saved":
            return replace(saved, plan_json="{}").restore(context, plan.fingerprint)
        elif name == "wrong_gradient_boundary":
            return backward(
                saved, case["dx_rows"], case["dx_shared"], context, plan.fingerprint, "raw_residual"
            )
        elif name == "duplicate_arrival":
            return mock_return(plan, case["rows"], [0] * len(case["rows"]), context)
        elif name == "active_nonfinite":
            case["rows"][0][0] = float("inf")
        elif name == "early_dtype":
            case["shared"][0][0] = 1 + 2**-10
        elif name == "production_unavailable":
            return production_provider()
        return forward(plan, case["rows"], case["shared"], case["residual"], context)

    try:
        invoke()
    except ContractError as exc:
        require(exc.code == NEGATIVES[name], "BYTE_MISMATCH", f"{name}: wrong rejection {exc.code}")
        return {
            "case_id": "P6-N-" + name + ".v1",
            "status": "EXPECTED_REJECTION",
            "expected": NEGATIVES[name],
            "actual": exc.code,
        }
    raise ContractError("BYTE_MISMATCH", name + " unexpectedly accepted")


def negative_fixtures():
    return [run_negative(name) for name in NEGATIVES]
