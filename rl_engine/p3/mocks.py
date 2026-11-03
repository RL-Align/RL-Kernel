"""T07/T08 local materialization hooks. No collective or second P7 runner."""

from .contract import E, H, P3Verdict
from .oracle import require


def validate_unsharded_gate(declared, actual, hidden_shape, q_shape):
    require(
        declared == actual, P3Verdict.GATE_SHARDING_MISMATCH, "declaration differs from actual gate"
    )
    require(
        declared.get("gate") == "unsharded",
        P3Verdict.UNSUPPORTED_CAPABILITY,
        "sharded gate unsupported",
    )
    require(hidden_shape[-1] == H, P3Verdict.SCHEMA_MISMATCH, "incomplete hidden features")
    require(
        q_shape[-1] == E, P3Verdict.FORBIDDEN_LOCAL_SHARD_TOPK, "local shard selection forbidden"
    )


def materialize_tokens(case, expected_token_set):
    """P7 invokes this hook with its existing planner's rank ownership manifest."""
    expected = set(expected_token_set)
    actual = set(case.global_token_id[case.row_active].tolist())
    require(expected <= actual, P3Verdict.INCOMPLETE_ARTIFACT, "missing expected token")
    indices = [
        i
        for i, token in enumerate(case.global_token_id)
        if case.row_active[i] and int(token) in expected
    ]
    return case.rows(indices)


def validate_ownership(plan, expected_token_set):
    require(
        isinstance(expected_token_set, (list, tuple, set, frozenset))
        and all(type(t) is int and t >= 0 for t in expected_token_set)
        and len(set(expected_token_set)) == len(expected_token_set),
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        "expected tokens must be unique nonnegative integers",
    )
    actual = [int(k) for k in plan["per_token_fingerprints"]]
    expected = set(expected_token_set)
    require(
        len(set(actual)) == len(actual) and set(actual) <= expected,
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        "duplicate/unauthorized token",
    )
    require(set(actual) == expected, P3Verdict.INCOMPLETE_ARTIFACT, "missing token")


OWNERSHIP = {
    "tp": "replica",
    "sp": "partition_by_sequence",
    "cp": "partition",
    "dp": "partition",
    "pp": "replica_by_layer",
    "ep": "not_a_token_dimension",
}


def validate_ownership_metadata(ownership, expected_token_set):
    """Validate T07/T08 declarations; no partition planner or collective is added."""
    require(
        isinstance(ownership, dict)
        and set(ownership) == set(OWNERSHIP)
        and all(
            ownership[k] == v if k != "tp" else ownership[k] in ("replica", "partition_by_sequence")
            for k, v in OWNERSHIP.items()
        ),
        P3Verdict.SCHEMA_MISMATCH,
        "unsupported token ownership declaration",
    )
    require(
        isinstance(expected_token_set, list)
        and all(type(t) is int and 0 <= t < 2**63 for t in expected_token_set)
        and len(set(expected_token_set)) == len(expected_token_set),
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        "expected rank token set must be explicit and unique",
    )
