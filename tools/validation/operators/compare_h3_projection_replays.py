#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare raw output digests from two independently executed GPU reports."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(first, second):
    if first["status"] != "passed" or second["status"] != "passed":
        raise ValueError("both reports must have passed their own checks")
    for field in (
        "source_revision",
        "source_hashes",
        "checkpoint_revision",
        "seed",
        "contract_sha256",
    ):
        if first["provenance"][field] != second["provenance"][field]:
            raise ValueError("replay provenance mismatch: " + field)

    def cases(report):
        result = {}
        for case in report["cases"]:
            key = (case["semantic_op"], case["input_family"], case["rows"])
            if key in result:
                raise ValueError("ambiguous duplicate replay case")
            result[key] = case["output_digests"]
        return result

    original, replay = cases(first), cases(second)
    if not replay or not replay.keys() <= original.keys():
        raise ValueError("replay cases must be a nonempty subset of the original")
    checks = []
    for key, outputs in replay.items():
        for tensor in ("y", "dx", "dw"):
            checks.append(
                {
                    "case": list(key),
                    "tensor": tensor,
                    "passed": outputs[tensor] == original[key][tensor],
                }
            )
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = (("original", args.original), ("replay", args.replay))
    output = args.output.resolve()
    for _, path in sources:
        try:
            same_file = output.samefile(path)
        except OSError:
            same_file = output == path.resolve()
        if same_file:
            parser.error("--output must differ from both input reports")

    reports, report_sha256, errors = [], [], []
    for label, path in sources:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            report_sha256.append(None)
            errors.append(f"could not read {label} report: {exc.strerror or type(exc).__name__}")
            continue
        report_sha256.append(hashlib.sha256(raw).hexdigest())
        try:
            reports.append(json.loads(raw))
        except ValueError as exc:
            errors.append(f"invalid {label} report JSON: {exc}")

    if errors:
        result = {"status": "error", "error": "\n".join(errors), "checks": []}
    else:
        try:
            result = compare(*reports)
        except (ValueError, KeyError, TypeError) as exc:
            result = {"status": "error", "error": str(exc), "checks": []}
    result["report_sha256"] = report_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return {"passed": 0, "failed": 1, "error": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
