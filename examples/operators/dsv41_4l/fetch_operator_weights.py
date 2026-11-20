# SPDX-License-Identifier: Apache-2.0
"""Fetch small official operator slices, never a replacement four-layer checkpoint."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from rl_engine.models.deepseek_v41.checkpoint import PinnedCheckpointReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    reader = PinnedCheckpointReader()
    prefixes = [
        "layers.2.attn.wq_a",
        "layers.2.attn.compressor.wkv",
        "layers.2.attn.compressor.wgate",
        "layers.2.ffn.experts.0.w1",
        "layers.2.ffn.experts.0.w2",
        "layers.2.ffn.experts.0.w3",
    ]
    keys = []
    for prefix in prefixes:
        keys.append(prefix + ".weight")
        if prefix + ".scale" in reader.index:
            keys.append(prefix + ".scale")
    keys += [
        "layers.2.attn.compressor.norm.weight",
        "layers.2.attn.q_norm.weight",
        "layers.1.engram.q_weight",
        "layers.1.engram.k_weight",
    ]
    tensors = {}
    for key in keys:
        tensors[key] = reader.read_tensor(key)
        print(key, tuple(tensors[key].shape), tensors[key].dtype, flush=True)
    save_file(tensors, str(args.output / "operators.safetensors"))
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "operators.safetensors" for k in keys}}, indent=2) + "\n"
    )
    # Exact real rows suffice for the row-lookup component test, not a model execution.
    row_ids = [0, 1, 16000001, 384006167]
    embed = "layers.1.engram.embed"
    rows = {
        f"{embed}.weight": reader.read_rows(embed + ".weight", row_ids),
        f"{embed}.scale": reader.read_rows(embed + ".scale", row_ids),
        "original_row_ids": __import__("torch").tensor(row_ids),
    }
    save_file(rows, str(args.output / "engram-rows.safetensors"))
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model": reader.base,
                "revision": reader.revision,
                "scope": "operator tensors and four real Engram rows; not a model checkpoint",
                "tensor_keys": keys,
                "engram_original_row_ids": row_ids,
                "ranges": reader.evidence,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
