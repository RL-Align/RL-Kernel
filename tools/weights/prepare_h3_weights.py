#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fetch and pin the MiniMax-H3 conditioning tensors used by the RFC #420 tests.

Downloads only the config, the shard index and the shards that hold the
manifest tensors (shard 1 of 14, ~4.8 GB) at the pinned revision, verifies
every sha256 against ``rl_engine/validation/models/h3_manifest.json`` and writes the
tensors to ``<out>/h3_conditioning.safetensors``. Point the tests at it with
``export RL_KERNEL_H3_WEIGHTS=<out>``.

    python tools/weights/prepare_h3_weights.py --out ~/.cache/rl-kernel/h3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.validation.models.h3_weights import (  # noqa: E402
    EXTRACTED_FILE,
    load_h3_manifest,
    sha256_file,
    sha256_tensor,
)


def _check(path: Path, expected: str) -> None:
    """Verify a downloaded file's SHA-256, exiting on mismatch or printing success."""

    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"sha256 mismatch for {path.name}: {actual} != {expected}")
    print(f"ok  {path.name}  {actual}")


def main() -> None:
    """Download pinned model artifacts, verify tensor identities, and write extracted weights."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="where the raw shard goes (default: <out>/hf)",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    manifest = load_h3_manifest()
    identity = manifest["model_identity"]
    out_dir: Path = args.out.expanduser()
    download_dir = (args.download_dir or out_dir / "hf").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(filename: str) -> Path:
        """Download one file from the manifest's model subfolder and pinned revision."""

        return Path(
            hf_hub_download(
                identity["hf_repo"],
                f"{identity['subfolder']}/{filename}",
                revision=identity["revision"],
                local_dir=str(download_dir),
            )
        )

    _check(fetch("config.json"), identity["config_sha256"])
    index_path = fetch(identity["index_file"])
    _check(index_path, identity["index_sha256"])
    weight_map = json.loads(index_path.read_text())["weight_map"]

    tensors = {}
    for shard in sorted({weight_map[name] for name in manifest["tensors"]}):
        if shard not in manifest["weight_shards"]:
            raise SystemExit(f"shard {shard} is not pinned in the manifest")
        shard_path = fetch(shard)
        _check(shard_path, manifest["weight_shards"][shard]["sha256"])
        with safe_open(str(shard_path), framework="pt") as handle:
            for name in manifest["tensors"]:
                if weight_map[name] == shard:
                    tensors[name] = handle.get_tensor(name).contiguous()

    for name, spec in manifest["tensors"].items():
        tensor = tensors[name]
        if str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
            raise SystemExit(f"{name}: dtype {tensor.dtype} != manifest {spec['dtype']}")
        if list(tensor.shape) != spec["shape"]:
            raise SystemExit(f"{name}: shape {list(tensor.shape)} != manifest {spec['shape']}")
        actual = sha256_tensor(tensor)
        if actual != spec["sha256"]:
            raise SystemExit(f"{name}: sha256 {actual} != manifest {spec['sha256']}")

    target = out_dir / EXTRACTED_FILE
    save_file(tensors, str(target), metadata={"revision": identity["revision"]})
    print(f"wrote {len(tensors)} tensors to {target}")
    print(f"export RL_KERNEL_H3_WEIGHTS={out_dir}")


if __name__ == "__main__":
    main()
