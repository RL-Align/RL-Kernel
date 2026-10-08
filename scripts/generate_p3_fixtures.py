"""Reproduce frozen T01 fixtures. --check never writes or silently upgrades goldens."""

import argparse
import gzip
import io
import json
from pathlib import Path

from rl_engine.p3 import bitmath
from rl_engine.p3.boundary import boundary_manifest
from rl_engine.p3.contract import GOLDEN_SCHEMA, manifest
from rl_engine.p3.fixtures import catalog, fixture_manifest
from rl_engine.p3.recordings import record_case
from rl_engine.p3.serialization import encode, fingerprint


def publications():
    rows = []
    for case in catalog():
        r = record_case(case)
        rows.append(
            {
                "case_id": case.case_id,
                "source_checksum": case.checksum(),
                "producer_verdict": r["producer_verdict"],
                "operators": {
                    name: {
                        "inputs_checksum": op["input_checksum"],
                        "payload_checksum": fingerprint(op["payload"]),
                        "payload": op["payload"],
                    }
                    for name, op in r["operators"].items()
                },
                "saved": r["saved"],
                "torch_paired_required": True,
                "per_token_fingerprints": (
                    r["bundle"]["RoutePlan"]["per_token_fingerprints"] if r["bundle"] else {}
                ),
            }
        )
    sample = record_case(catalog()[0])["bundle"]
    from rl_engine.p3.negative import run_negative_fixtures

    return {
        "tests/p3/data/contract.v23.json": manifest(),
        "tests/p3/data/boundary.v1.json": boundary_manifest(),
        "tests/p3/data/catalog.v2.json": fixture_manifest(),
        "tests/p3/data/negative.v2.json": run_negative_fixtures(),
        "tests/p3/data/golden.v2.json.gz": {
            "schema": GOLDEN_SCHEMA,
            "math_source": bitmath.source_fingerprint(),
            "cases": rows,
        },
        "tests/p3/data/l3b.pending.v1.json": {
            "schema": "p3-l3b-paired.v1",
            "verdict": "MISSING_PROVENANCE",
            "miles_router_anchor": manifest()["miles_router_anchor"],
            "required_engines": ["Megatron", "Miles"],
            "recorded_artifacts": [],
            "required_fields": [
                "case_id",
                "checkpoint_id",
                "weight_id",
                "fixture_checksum",
                "boundary_trace",
                "actual_provenance",
                "RoutePlan",
                "torch_paired",
            ],
            "note": "Fill from sanitized real engine traces; synthetic pairs never certify L3b.",
        },
        "examples/dsv4_p3_startkit/handoff.v1.json": sample,
        "examples/dsv4_p3_startkit/engine-pair.synthetic.v1.json": {
            "scope": "synthetic_dual_engine_transport_example",
            "certifies_L3b": False,
            "training": record_case(catalog()[0], engine_id="synthetic-training")["bundle"],
            "inference": record_case(catalog()[0], engine_id="synthetic-inference")["bundle"],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    for relative, content in publications().items():
        path = root / relative
        rendered = json.dumps(encode(content), indent=2) + "\n"
        if args.check:
            existing = None
            if path.exists():
                existing = (
                    gzip.decompress(path.read_bytes()).decode("utf-8")
                    if path.suffix == ".gz"
                    else path.read_text()
                )
            if existing != rendered:
                raise SystemExit(f"frozen publication differs: {relative}; contract delta required")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".gz":
                output = io.BytesIO()
                with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as zipped:
                    zipped.write(rendered.encode("utf-8"))
                path.write_bytes(output.getvalue())
            else:
                path.write_text(rendered)
    print("P3 frozen publications match" if args.check else "P3 publications generated")


if __name__ == "__main__":
    main()
