#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Emit WS1 C2 (#268) workload reference identity (no full-model forward).

Example:
  python tools/validation/models/ws1_reference.py --dtype bf16 --cell-id BN/full
  python tools/validation/models/ws1_reference.py --emit-json -
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_workload_module():
    """Load the pure-Python C2 module without importing torch-heavy package helpers."""
    module_path = Path(__file__).resolve().parents[3] / "rl_engine/config/workload.py"
    spec = importlib.util.spec_from_file_location("_ws1_workload_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the pinned WS1 canonical workload reference payload: "
            "workload_id, seed, dtype, fixture hash, model identity, and matrix cell."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to ws1_manifest.json (default: package manifest).",
    )
    parser.add_argument(
        "--workload-id",
        default=None,
        help="If set, must match the manifest workload_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, must match the manifest seed (does not reseed fixtures).",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Execution dtype label for the emission (bf16/bfloat16 or fp32/float32).",
    )
    parser.add_argument(
        "--cell-id",
        default=None,
        help="Optional primary matrix cell_id (e.g. BN/full).",
    )
    parser.add_argument(
        "--emit-json",
        default=None,
        metavar="PATH",
        help="Write full JSON payload to PATH, or '-' for stdout only JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    workload = _load_workload_module()
    WorkloadError = workload.WorkloadError

    args = build_parser().parse_args(argv)
    try:
        manifest = workload.load_manifest(args.manifest)
        if args.workload_id is not None and args.workload_id != manifest.workload_id:
            raise WorkloadError(
                f"--workload-id {args.workload_id!r} does not match manifest "
                f"{manifest.workload_id!r}"
            )
        if args.seed is not None and int(args.seed) != manifest.seed:
            raise WorkloadError(f"--seed {args.seed} does not match manifest seed {manifest.seed}")
        payload = workload.reference_payload(manifest, cell_id=args.cell_id, dtype=args.dtype)
    except (WorkloadError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.emit_json == "-":
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    # Human-readable summary (always includes the three required identity fields).
    print(f"workload_id: {payload['workload_id']}")
    print(f"seed: {payload['seed']}")
    print(f"dtype: {payload['dtype']}")
    print(f"fixture_hash: {payload['fixture_hash']}")
    print(f"model_id: {payload['model_id']}")
    print(f"revision: {payload['revision']}")
    print(f"clip_interval: {payload['clip_interval']}")
    if payload.get("cell_id"):
        print(f"cell_id: {payload['cell_id']}")
    print(f"active_token_count: {payload['active_token_count']}")
    print(f"chunk_spans: {payload['chunk_plan']['chunk_spans']}")
    missing = payload["profile_missing_required"]
    for profile_id, nodes in missing.items():
        if nodes:
            print(f"profile {profile_id} missing_required: {nodes}")

    if args.emit_json:
        out_path = Path(args.emit_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
