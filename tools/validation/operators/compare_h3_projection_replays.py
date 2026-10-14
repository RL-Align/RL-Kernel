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
    raw = [p.read_bytes() for p in (args.original, args.replay)]
    result = compare(*(json.loads(data) for data in raw))
    result["report_sha256"] = [hashlib.sha256(data).hexdigest() for data in raw]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
