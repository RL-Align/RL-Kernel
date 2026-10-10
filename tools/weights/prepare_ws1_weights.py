#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Materialize and verify the manifest-pinned Qwen3-8B snapshot for WS1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.config.workload import load_manifest  # noqa: E402
from rl_engine.models.qwen3.reference import Qwen3DenseSpec, verify_hf_weight_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned WS1 Qwen3-8B weights."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only verify an existing snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    spec = Qwen3DenseSpec.from_manifest(manifest)
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download WS1 weights") from exc
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_dir=args.output,
            allow_patterns=[
                spec.weight_index_file,
                "model-*.safetensors",
                "config.json",
            ],
        )
    files = verify_hf_weight_snapshot(spec, args.output)
    print(
        f"verified {spec.model_id}@{spec.revision} "
        f"shards={len(files)} content_hash={spec.weight_content_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
