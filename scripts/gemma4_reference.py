#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Emit the Gemma 4 architecture fingerprint reference identity (no model forward).

Example:
  python scripts/gemma4_reference.py
  python scripts/gemma4_reference.py --emit-json -
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_workload_module():
    """Load the pure-Python fingerprint module without importing torch-heavy package helpers."""
    module_path = Path(__file__).resolve().parents[1] / "rl_engine/testing/gemma4_workload.py"
    spec = importlib.util.spec_from_file_location("_gemma4_workload_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the pinned Gemma 4 fingerprint reference payload: workload_id, version, "
            "fixture identity hash, model identity, config fingerprint, and weight snapshot."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to gemma4_manifest.json (default: package manifest).",
    )
    parser.add_argument(
        "--workload-id",
        default=None,
        help="If set, must match the manifest workload_id.",
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
    Gemma4WorkloadError = workload.Gemma4WorkloadError

    args = build_parser().parse_args(argv)
    try:
        manifest = workload.load_manifest(args.manifest)
        if args.workload_id is not None and args.workload_id != manifest.workload_id:
            raise Gemma4WorkloadError(
                f"--workload-id {args.workload_id!r} does not match manifest "
                f"{manifest.workload_id!r}"
            )
        payload = workload.reference_payload(manifest)
    except (Gemma4WorkloadError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.emit_json == "-":
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    text = payload["config_fingerprint"]["text_config"]
    weight = payload["weight_snapshot"]
    print(f"workload_id: {payload['workload_id']}")
    print(f"version: {payload['version']}")
    print(f"fixture_identity_sha256: {payload['fixture_identity_sha256']}")
    print(f"model_id: {payload['model_id']}")
    print(f"revision: {payload['revision']}")
    source = payload["semantics_source"]
    print(f"semantics_source: {source['package']}=={source['version']} @ {source['git_commit']}")
    print(f"num_hidden_layers: {text['num_hidden_layers']}")
    print(f"layer_types: {text['layer_types'].count('sliding_attention')} sliding_attention")
    print(f"             {text['layer_types'].count('full_attention')} full_attention")
    print(f"weight_shards: {len(weight['shards'])}")
    print(f"weight_content_hash: {weight['content_hash']}")

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
