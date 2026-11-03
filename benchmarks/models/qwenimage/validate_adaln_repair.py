#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check completeness and acceptance without reinterpreting tolerance failures."""
import argparse
import json
from pathlib import Path

from benchmarks.models.qwenimage.streams import STREAMS

# Checks per (stream, dtype) case; final modulation adds long-sequence cases.
CHECKS_PER_CASE = {
    "accuracy": {"image": 2, "text": 2, "final": 5},
    "accuracy_large": {"image": 2, "text": 2, "final": 2},
    "bi": {"image": 402, "text": 402, "final": 408},
    "workload": {"image": 15, "text": 15, "final": 15},
    "parameters": {"image": 21, "text": 21, "final": 21},
}

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--allow-fail", action="store_true")
p.add_argument("--allow-dirty", action="store_true", help="Accept reports from an uncommitted tree")
a = p.parse_args()
rows, failures, extensions, commits = [], [], set(), set()
streams = [stream for stream, _ in STREAMS]
for phase, seeds in [
    ("accuracy", [0, 127, 2026]),
    ("accuracy_large", [0, 127, 2026]),
    ("bi", [0]),
    ("workload", [0]),
    ("parameters", [0]),
]:
    expected = 2 * sum(CHECKS_PER_CASE[phase][stream] for stream in streams)
    for seed in seeds:
        path = a.directory / f"{phase}-{seed}.json"
        report = json.loads(path.read_text())
        assert a.allow_dirty or report["source"]["clean_commit"], (path, "dirty tree")
        commits.add(report["source"]["base_sha"])
        extensions.add(report["source"]["extension_sha256"])
        assert [c["stream"] for c in report["cases"][::2]] == streams, path
        checks = sum(len(c["checks"]) for c in report["cases"])
        assert checks == expected, (path, checks, expected)
        bad = []
        for c in report["cases"]:
            for check in c["checks"]:
                for field in ["contract", "cpu_fp32_contract"]:
                    for tensor, result in check.get(field, {}).items():
                        if not result["passed"]:
                            bad.append(
                                dict(
                                    stream=c["stream"],
                                    dtype=c["dtype"],
                                    batch=check.get("batch"),
                                    sequence=check.get("sequence"),
                                    field=field,
                                    tensor=tensor,
                                    result=result,
                                )
                            )
                if any(check.get("mismatched_rows", {}).values()) or not all(
                    check.get("all_equal", {}).values()
                ):
                    bad.append(dict(stream=c["stream"], dtype=c["dtype"], check=check))
        rows.append(dict(file=path.name, checks=checks, failures=len(bad)))
        failures.extend(dict(file=path.name, **b) for b in bad)
screen = json.loads((a.directory / "cancellation-screen.json").read_text())
assert a.allow_dirty or screen["source"]["clean_commit"], "dirty tree"
commits.add(screen["source"]["base_sha"])
extensions.add(screen["source"]["extension_sha256"])
assert len(screen["cases"]) == 8
bad = [c for c in screen["cases"] if not c["fixed_contract"]["passed"]]
rows.append(dict(file="cancellation-screen.json", checks=8, failures=len(bad)))
failures.extend(dict(file="cancellation-screen.json", check=c) for c in bad)
assert len(extensions) == 1, "Do not combine results from different extension builds"
assert len(commits) == 1, "Do not combine results from different commits"
result = dict(
    qualified=not failures,
    commit=next(iter(commits)),
    extension_sha256=next(iter(extensions)),
    reports=rows,
    failures=failures,
)
(a.directory / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if failures and not a.allow_fail:
    raise SystemExit(1)
