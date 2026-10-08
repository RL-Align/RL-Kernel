"""Local P3 consumer contract; it does not impersonate an approved Foundation ABI."""

from copy import deepcopy

from .contract import CORE_SCHEMA, ENVELOPE_SCHEMA, SEED_SCHEMA, P3Verdict
from .oracle import require

BOUNDARY_VERSION = "p3-local-boundary.v1"


def boundary_manifest():
    return {
        "schema_version": BOUNDARY_VERSION,
        "foundation_compatibility": "UNVERIFIED",
        "foundation_approval_required": True,
        "produces": {
            "RoutePlanCore": CORE_SCHEMA,
            "RoutePlanEnvelope": ENVELOPE_SCHEMA,
            "CombinePlanSeed": SEED_SCHEMA,
        },
        "consumers": {"P4": "RoutePlan", "P6": "CombinePlanSeed", "P7": "RoutePlan"},
        "capacity_policy": "dropless_v1",
        "slot_order": list(range(6)),
        "weight_application_owner": "P5",
        "combine_owner": "P6",
    }


def consume(
    bundle,
    consumer,
    *,
    expected_identity,
    expected_boundary=BOUNDARY_VERSION,
    capacity_policy="dropless_v1",
    require_foundation=False,
):
    """Versioned consumer mock shared by the three downstream handoff examples."""
    from .checker import validate_plan
    from .serialization import canonical

    require(expected_boundary == BOUNDARY_VERSION, P3Verdict.SCHEMA_MISMATCH, "boundary version")
    require(
        capacity_policy == "dropless_v1",
        P3Verdict.UNSUPPORTED_CAPABILITY,
        "finite capacity/overflow requires a named contract delta",
    )
    require(
        not require_foundation,
        P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
        "approved Foundation version and owner binding not supplied",
    )
    require(
        consumer in boundary_manifest()["consumers"],
        P3Verdict.UNSUPPORTED_CAPABILITY,
        "unknown consumer",
    )
    validate_plan(bundle)
    require(
        canonical(bundle["RoutePlan"]["identity"]) == canonical(expected_identity),
        P3Verdict.IDENTITY_DRIFT,
        "consumer case/checkpoint/weight/token identity",
    )
    return deepcopy(bundle[boundary_manifest()["consumers"][consumer]])
