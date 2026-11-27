#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Emit the Muse Glimmer architecture fingerprint and optionally verify a checkpoint.

No model forward is run.

Example:
  python tools/validation/models/museglimmer_reference.py
  python tools/validation/models/museglimmer_reference.py --emit-json -
  python tools/validation/models/museglimmer_reference.py --checkpoint /path/to/Muse-Glimmer-30B
  python tools/validation/models/museglimmer_reference.py --checkpoint DIR --verify-shard-sha256
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.models.muse_glimmer import workload  # noqa: E402
from rl_engine.models.muse_glimmer.loader import verify_checkpoint  # noqa: E402
from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the pinned Muse-Glimmer-30B fingerprint: model identity, config, frozen "
            "semantics, projection grouping, parallel layout and weight snapshot."
        )
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Manifest path (default: package manifest)."
    )
    parser.add_argument(
        "--workload-id", default=None, help="If set, must match the manifest workload_id."
    )
    parser.add_argument(
        "--emit-json",
        default=None,
        metavar="PATH",
        help="Write the full JSON payload to PATH, or '-' for JSON-only stdout.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Local snapshot directory to verify."
    )
    parser.add_argument(
        "--verify-shard-sha256",
        action="store_true",
        help="With --checkpoint, also hash every shard (about 60 GB of reads).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = workload.load_manifest(args.manifest)
        if args.workload_id is not None and args.workload_id != manifest.workload_id:
            raise workload.MuseGlimmerWorkloadError(
                f"--workload-id {args.workload_id!r} does not match manifest "
                f"{manifest.workload_id!r}"
            )
        payload = workload.reference_payload(manifest)
        spec = MuseGlimmerSpec.from_manifest(manifest)
        report = (
            verify_checkpoint(
                args.checkpoint, manifest, verify_shard_sha256=args.verify_shard_sha256
            )
            if args.checkpoint is not None
            else None
        )
    except (workload.MuseGlimmerWorkloadError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.emit_json == "-":
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    text = payload["config_fingerprint"]["text_config"]
    weight = payload["weight_snapshot"]
    source = payload["semantics_source"]
    layout = payload["parallel_layout"]
    print(f"workload_id: {payload['workload_id']}")
    print(f"version: {payload['version']}")
    print(f"fixture_identity_sha256: {payload['fixture_identity_sha256']}")
    print(f"model_id: {payload['model_id']}")
    print(f"revision: {payload['revision']}")
    print(f"semantics_source: {source['package']}=={source['version']} @ {source['git_commit']}")
    print(f"num_hidden_layers: {text['num_hidden_layers']}")
    print(f"layer_types: {text['layer_types'].count(workload.SLIDING_ATTENTION)} sliding_attention")
    print(f"             {text['layer_types'].count(workload.FULL_ATTENTION)} full_attention")
    print(f"projection_grouping: {payload['projection_grouping']['attention']}")
    print(
        f"canonical_tp: {layout['canonical_tp']}  padded_vocab_size: {layout['padded_vocab_size']}"
    )
    print(f"weight_shards: {len(weight['shards'])}")
    print(f"weight_content_hash: {weight['content_hash']}")
    rows = Counter(node.row for node in spec.trace())
    print(f"operator_trace: {sum(rows.values())} nodes")
    for row, count in sorted(rows.items()):
        print(f"  {row}: {count}")
    if report is not None:
        hashed = "all shards hashed" if report.shards_hashed else "shard hashes not checked"
        print(f"checkpoint: OK {report.path} ({report.text_tensor_count} text tensors, {hashed})")

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
