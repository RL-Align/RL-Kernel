# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pinned operator case identities and complete execution verdicts."""

import hashlib
import json
from pathlib import Path


def load_cases(path: Path, suite: str) -> list[str]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported operator case manifest schema")
    groups = manifest["suites"]
    all_ids = [case for cases in groups.values() for case in cases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Operator case IDs must be unique across suites")
    names = ("norm", "core") if suite == "all" else (suite,)
    cases = sorted(case for name in names for case in groups[name])
    if not cases or any(not isinstance(case, str) or "::test_" not in case for case in cases):
        raise ValueError("Operator suites must contain nonempty pytest case IDs")
    return cases


def case_digest(cases: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(cases)) + "\n").encode()).hexdigest()


def execution_verdict(expected: list[str], collected: list[str], outcomes: dict) -> dict:
    """Require the complete declared collection and successful setup/call/teardown."""
    missing = sorted(set(expected) - set(collected))
    unexpected = sorted(set(collected) - set(expected))
    duplicates = len(collected) != len(set(collected))
    counts = {"tests": len(expected), "failures": 0, "errors": 0, "skipped": 0}
    incomplete = []
    for case in expected:
        phases = outcomes.get(case, {})
        if any(value == "skipped" for value in phases.values()):
            counts["skipped"] += 1
        elif phases.get("setup") == "failed" or phases.get("teardown") == "failed":
            counts["errors"] += 1
        elif phases.get("call") == "failed":
            counts["failures"] += 1
        elif phases != {"setup": "passed", "call": "passed", "teardown": "passed"}:
            incomplete.append(case)
    passed = not any(
        (
            missing,
            unexpected,
            duplicates,
            incomplete,
            counts["failures"],
            counts["errors"],
            counts["skipped"],
        )
    )
    return {
        "passed": passed,
        "expected_case_ids": expected,
        "expected_case_ids_sha256": case_digest(expected),
        "collected_case_ids": collected,
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_collection": duplicates,
        "incomplete": incomplete,
        "counts": counts,
        "outcomes": outcomes,
    }
