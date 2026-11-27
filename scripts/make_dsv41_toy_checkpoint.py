# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Write a random DSv4.1 toy checkpoint in the release's layout, loadable by vLLM.

``config.json`` is the release's (``text_config`` replaced by :func:`toy_config`: one layer of
every role, an Engram layer), the tokenizer is the release's (so the Engram token map has the
release's 99092 compressed ids), and the weights are :func:`random_checkpoint` in the release
formats. No vision, aligner or DSpark tensors: serve with ``limit_mm_per_prompt={"image": 0}``.

    PYTHONPATH=. python scripts/make_dsv41_toy_checkpoint.py --meta DSV41_META --out DIR
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True, help="release metadata (config, tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-intermediate", type=int, default=256, help="2304 in the release")
    ap.add_argument("--shard-gib", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from safetensors.torch import save_file

    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.loader import random_checkpoint

    meta, out = pathlib.Path(args.meta), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    release = json.loads((meta / "config.json").read_text())
    text = toy_config()
    text["moe_intermediate_size"] = args.moe_intermediate
    quant = text.pop("quantization_config")
    release["text_config"] = {"model_type": release["text_config"]["model_type"], **text}
    release["quantization_config"] = quant
    (out / "config.json").write_text(json.dumps(release, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (meta / name).exists():
            shutil.copy(meta / name, out / name)

    cfg = Dsv41Config.from_pretrained(out)
    tensors = random_checkpoint(cfg, seed=args.seed)
    limit = int(args.shard_gib * 2**30)
    shards: list[dict] = [{}]
    size = 0
    for name in sorted(tensors):
        t = tensors[name].contiguous().cpu()
        nbytes = t.numel() * t.element_size()
        if shards[-1] and size + nbytes > limit:
            shards.append({})
            size = 0
        shards[-1][name] = t
        size += nbytes
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out / fname), metadata={"format": "pt"})
        for name, t in shard.items():
            weight_map[name] = fname
            total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(tensors)} tensors, {total / 2**30:.1f} GiB in {len(shards)} shards -> {out}")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
