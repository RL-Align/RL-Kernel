# SPDX-License-Identifier: Apache-2.0
"""Exact additional attention tensors for original KV/index source mechanisms."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from .fetch_model_weights import copy_shard
from rl_engine.models.deepseek_v41.checkpoint import PinnedCheckpointReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reader = PinnedCheckpointReader()
    selected = {
        k: v
        for k, v in reader.index.items()
        if any(k.startswith(f"layers.{i}.attn.") for i in (8, 14, 20, 24, 28, 32, 36))
    }
    grouped = defaultdict(list)
    for key, shard in selected.items():
        grouped[shard].append(key)
    evidence = []
    for shard, keys in sorted(grouped.items()):
        output = args.output / shard
        marker = args.output / (shard + ".complete.json")
        if marker.exists():
            record = json.loads(marker.read_text())
            if record["keys"] != sorted(keys) or output.stat().st_size != record["bytes"]:
                raise ValueError("incompatible cached attention slice")
        else:
            record = copy_shard(reader, shard, keys, output)
            record["keys"] = sorted(keys)
            marker.write_text(json.dumps(record, indent=2) + "\n")
        evidence.append(record)
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": selected}, indent=2) + "\n"
    )
    (args.output / "manifest.json").write_text(
        json.dumps({"revision": reader.revision, "shards": evidence}, indent=2) + "\n"
    )
    print("COMPLETE_ATTENTION", len(selected), flush=True)


if __name__ == "__main__":
    main()
