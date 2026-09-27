# SPDX-License-Identifier: Apache-2.0
"""Independent inputs and recorded bytes for each P6 operator boundary."""

import copy
from dataclasses import asdict
from pathlib import Path
import struct

from .contract import CombinePlan, OPERATORS, SavedForward, digest, exact_keys, require
from .fixtures import evaluate, load_golden
from . import reference

GOLDEN = Path(__file__).resolve().parents[2] / "tests/p6/data/golden.v1.json"


def decode_rows(raw, n, h):
    values = struct.unpack("<" + "f" * (n * h), bytes.fromhex(raw))
    return [list(values[i * h : (i + 1) * h]) for i in range(n)]


def golden_records():
    return load_golden(GOLDEN)["payload"]["cases"]


def catalog(records=None):
    records = golden_records() if records is None else records
    return [
        {
            "case_id": "P6-F-" + r["input"]["name"] + ".v1",
            "recipe": r["input"]["name"],
            "input_sha256": digest(r["input"]),
            "operators": [op["name"] for op in OPERATORS],
            "evidence_scope": "synthetic_reference",
        }
        for r in records
    ]


def boundary_recordings(records=None):
    records = golden_records() if records is None else records
    result = []
    for entry, source in zip(catalog(records), records, strict=True):
        c, e = source["input"], source["expected"]
        plan = CombinePlan.from_dict(c["plan"])
        n, h = len(plan.token_ids), plan.hidden_size
        flat = decode_rows(e["forward"]["canonical_fp32"], n * 6, h)
        inputs = {"plan": c["plan"], "context": asdict(plan.context)}
        by_operator = {
            "canonical_unpermute_fwd": {**inputs, "rows": c["rows"]},
            "fixed_order_combine_fwd": {
                **inputs,
                "canonical_rows": [flat[i * 6 : (i + 1) * 6] for i in range(n)],
            },
            "shared_residual_merge_fwd": {
                **inputs,
                "routed": decode_rows(e["forward"]["routed_fp32"], n, h),
                "shared": c["shared"],
                "residual": c["residual"],
            },
            "fused_moe_combine_fwd": {
                **inputs,
                "rows": c["rows"],
                "shared": c["shared"],
                "residual": c["residual"],
            },
            "fused_dx_fanin_bwd": {
                "saved": e["saved_forward"],
                "dx_rows": c["dx_rows"],
                "dx_shared": c["dx_shared"],
                "context": asdict(plan.context),
                "expected_fingerprint": plan.fingerprint,
                "shared_boundary": plan.gradient_boundary,
            },
        }
        for op in OPERATORS:
            stages = e["backward"] if op["phase"] == "backward" else e["forward"]
            expected = (
                stages if op["name"].startswith("fused_") else {k: stages[k] for k in op["outputs"]}
            )
            record = {
                "case_id": entry["case_id"],
                "operator": op["name"],
                "boundary": op["boundary"],
                "phase": op["phase"],
                "inputs": copy.deepcopy(by_operator[op["name"]]),
                "expected": copy.deepcopy(expected),
                "plan_fingerprint": plan.fingerprint,
                "order_hash": plan.order_hash,
            }
            record["checksum"] = digest(record)
            result.append(record)
    return result


def run_reference(operator, inputs):
    from .contract import Context

    spec = next((op for op in OPERATORS if op["name"] == operator), None)
    require(spec is not None, "UNSUPPORTED_CAPABILITY", operator)
    exact_keys(inputs, spec["inputs"], "operator inputs")
    args = copy.deepcopy(inputs)
    args["context"] = Context(**args["context"])
    if "plan" in args:
        args["plan"] = CombinePlan.from_dict(args["plan"])
    if "saved" in args:
        args["saved"] = SavedForward(**args["saved"])
    output = getattr(reference, operator)(**args)
    return (
        output["stages"]
        if operator.startswith("fused_")
        else {k: output[k] for k in spec["outputs"]}
    )


def verify_recordings(records):
    require(bool(records), "INCOMPLETE_ARTIFACT", "empty operator recordings")
    seen = set()
    for record in records:
        body = {k: v for k, v in record.items() if k != "checksum"}
        require(digest(body) == record["checksum"], "CORRUPT_ARTIFACT", "operator checksum")
        key = record["case_id"], record["operator"]
        require(key not in seen, "SCHEMA_MISMATCH", "duplicate operator recording")
        seen.add(key)
        actual = run_reference(record["operator"], record["inputs"])
        require(actual == record["expected"], "BYTE_MISMATCH", str(key))
    return len(seen)


def verify_source_cases(records):
    require(bool(records), "INCOMPLETE_ARTIFACT", "empty case collection")
    names = set()
    for record in records:
        name = record["input"]["name"]
        require(name not in names, "SCHEMA_MISMATCH", "duplicate case")
        names.add(name)
        require(evaluate(record["input"]) == record["expected"], "BYTE_MISMATCH", name)
