# SPDX-License-Identifier: Apache-2.0
"""Frozen CLI: python -m rl_engine.p2 {manifest,catalog,conformance,verify}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact import provenance, read_artifact, seal
from .contract import (
    MODES,
    REFERENCE_PROFILE,
    SCHEMA_VERSION,
    ContractError,
    manifest,
    runtime_policy,
)
from .fixtures import boundary_recordings, catalog, required_evidence, state_sequence
from .negative import negative_cases


def build_payload(tokens: int = 129) -> dict:
    sequences = {}
    page_size = 4 if tokens > 128 else 16
    for layer in ("C0", "C4", "C128"):
        baseline = state_sequence(layer, tokens, "training", page_size=page_size)
        mode_hashes = {}
        for mode in MODES:
            chunks = tuple([7] * (tokens // 7) + ([tokens % 7] if tokens % 7 else []))
            states = state_sequence(
                layer,
                tokens,
                mode,
                chunks if mode == "prefill" else None,
                page_size=page_size,
                resume_at=min(65, tokens) if mode == "graph_decode" else None,
            )
            mode_hashes[mode] = [s["state_hash"] for s in states]
        sequences[layer] = {
            "tokens": tokens,
            "snapshots": baseline,
            "mode_state_hashes": mode_hashes,
        }
    return {
        "schema": SCHEMA_VERSION,
        "profile": REFERENCE_PROFILE,
        "contract": manifest(),
        "catalog": catalog(),
        "provenance": provenance(),
        "recordings": boundary_recordings(),
        "required_evidence": required_evidence(),
        "negative_cases": negative_cases(),
        "runtime_policy": runtime_policy(),
        "sequences": sequences,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    commands.add_parser("catalog")
    commands.add_parser("negative-fixtures")
    recorded = commands.add_parser("recordings")
    recorded.add_argument("--operator", choices=[op["name"] for op in manifest()["operators"]])
    run = commands.add_parser("conformance")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--tokens", type=int, default=129)
    verify = commands.add_parser("verify")
    verify.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            report = manifest()
        elif args.command == "catalog":
            report = catalog()
        elif args.command == "negative-fixtures":
            report = negative_cases()
        elif args.command == "recordings":
            records = boundary_recordings()
            report = records[args.operator] if args.operator else records
        elif args.command == "conformance":
            report = seal(args.output, build_payload(args.tokens))
        else:
            _, report = read_artifact(args.artifact)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print(json.dumps({"status": exc.status.value, "detail": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
