# SPDX-License-Identifier: Apache-2.0
"""INK-33: compare complete real-weight 48-visit captures without tolerance."""

import argparse
import json
import re
from pathlib import Path

from rl_engine.validation.models.inkling2.visit_trace import visits

IDENTITY = (
    "revision",
    "profile",
    "config_sha256",
    "quant_metadata_sha256",
    "weights_sha256",
    "input_sha256",
    "hardware",
    "token_count",
)
DIGEST = re.compile(r"[0-9a-f]{64}")


def _check_report(report: dict, *, require_backward: bool) -> tuple[dict, dict, list]:
    if report.get("schema") != "inkling2-visit-trace-v1":
        raise ValueError("unsupported trace schema")
    meta = report["metadata"]
    if meta.get("capture_mode") != "real-weight" or meta.get("profile") != "bf16-dequant":
        raise ValueError("only real-weight BF16-DEQUANT capture reports are accepted")
    if meta.get("physical_layers") != 24 or meta.get("logical_visits") != 48:
        raise ValueError("report does not identify the full released topology")
    if any(not meta.get(key) for key in IDENTITY + ("engine", "runtime_revision", "trace_scope")):
        raise ValueError("missing capture provenance")
    if any(not DIGEST.fullmatch(str(meta[key])) for key in IDENTITY if key.endswith("sha256")):
        raise ValueError("invalid SHA256 provenance")
    if type(meta["token_count"]) is not int or meta["token_count"] <= 0:
        raise ValueError("invalid token count")
    bits = report.get("selected_logprob_bits")
    if not isinstance(bits, list) or len(bits) != meta["token_count"]:
        raise ValueError("missing or misaligned selected logprob bits")
    for bit in bits:
        if (
            type(bit) is not int
            or not -(2**31) <= bit < 2**31
            or (bit & 0x7F800000) == 0x7F800000
        ):
            raise ValueError("invalid or nonfinite FP32 logprob bit pattern")
    expected = visits()
    records = {}
    directions = {"forward", "backward"} if require_backward else {"forward"}
    for row in report["records"]:
        j = row["logical"]
        if type(j) is not int or not 0 <= j < 48:
            raise ValueError("invalid logical visit")
        if row["physical"] != expected[j].physical or row["pass_index"] != expected[j].pass_index:
            raise ValueError("trace disagrees with physical-layer/pass mapping")
        if row["direction"] not in ("forward", "backward"):
            raise ValueError("invalid trace direction")
        if row["direction"] not in directions:
            continue
        if not all(row.get(key) for key in ("role", "backend", "kernel", "dtype", "sha256")):
            raise ValueError("missing boundary provenance")
        if not DIGEST.fullmatch(row["sha256"]):
            raise ValueError("invalid boundary digest")
        key = (j, row["direction"], row["role"])
        if key in records:
            raise ValueError("duplicate boundary")
        records[key] = row
    for direction in directions:
        if any((j, direction, "partial") not in records for j in range(48)):
            raise ValueError(f"incomplete 48-visit {direction} partial trace")
    return meta, records, bits


def compare_chain(left: dict, right: dict, *, require_backward: bool = False) -> dict:
    """Exact bytes for matched recorded boundaries, not a model certification.

    Forward-only comparison supports trainer vs rollout. Backward is opt-in
    for two differentiable runs. A full operator-level WS1 claim needs actual
    strict bindings and complete operator coverage beyond block boundaries.
    """
    lm, lr, lb = _check_report(left, require_backward=require_backward)
    rm, rr, rb = _check_report(right, require_backward=require_backward)
    if any(lm[key] != rm[key] for key in IDENTITY):
        raise ValueError("incomparable checkpoint/input/hardware provenance")
    if lm["trace_scope"] != rm["trace_scope"] or lr.keys() != rr.keys():
        raise ValueError("trace scopes or boundary sets differ")
    ordered = sorted(
        lr,
        key=lambda k: (
            k[1] != "forward", k[0] if k[1] == "forward" else -k[0], lr[k]["ordinal"]
        ),
    )
    first = None
    count = 0
    for key in ordered:
        a, b = lr[key], rr[key]
        if any(a[k] != b[k] for k in ("ordinal", "dtype", "shape", "sha256")):
            count += 1
            if first is None:
                first = {
                    "logical": key[0],
                    "direction": key[1],
                    "role": key[2],
                    "left": a,
                    "right": b,
                }
    token_mismatches = [i for i, (a, b) in enumerate(zip(lb, rb, strict=True)) if a != b]
    return {
        "trace_equal": count == 0 and not token_mismatches,
        "trace_scope": lm["trace_scope"],
        "compared_boundaries": len(ordered),
        "boundary_mismatches": count,
        "first_boundary_drift": first,
        "selected_token_count": len(lb),
        "selected_token_mismatches": len(token_mismatches),
        "first_selected_token_drift": token_mismatches[0] if token_mismatches else None,
        "require_backward": require_backward,
        "ws1_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--require-backward", action="store_true")
    args = parser.parse_args()
    result = compare_chain(
        json.loads(args.left.read_text()),
        json.loads(args.right.read_text()),
        require_backward=args.require_backward,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["trace_equal"] else 1)


if __name__ == "__main__":
    main()
