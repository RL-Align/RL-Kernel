# SPDX-License-Identifier: Apache-2.0
"""Materialize exact released text tensors by streamed HTTP ranges, not random weights."""

import argparse
import hashlib
import json
import os
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sympy import nextprime
from tokenizers import Tokenizer

from rl_engine.models.deepseek_v41.checkpoint import PinnedCheckpointReader
from rl_engine.models.deepseek_v41.config import DeepSeekV41Config
from rl_engine.models.deepseek_v41.engram import build_compressed_token_map


def copy_shard(reader, shard, keys, output):
    source_offset, header = reader.header(shard)
    ordered = sorted(keys, key=lambda k: header[k]["data_offsets"][0])
    new_header = {}
    cursor = 0
    for key in ordered:
        info = header[key]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        new_header[key] = {**info, "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target_offset = len(encoded) + 8
    groups = []
    for key in ordered:
        start, end = header[key]["data_offsets"]
        if groups and start - groups[-1][1] <= 4096:
            groups[-1][1] = end
            groups[-1][2].append(key)
        else:
            groups.append([start, end, [key]])
    partial = output.with_suffix(".partial")
    evidence = []
    with partial.open("wb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        target.truncate(target_offset + cursor)
        for start, end, names in groups:
            absolute_start, absolute_end = source_offset + start, source_offset + end - 1
            request = urllib.request.Request(
                reader.base + shard, headers={"Range": f"bytes={absolute_start}-{absolute_end}"}
            )
            digest = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(request, timeout=120) as response:
                wanted = f"bytes {absolute_start}-{absolute_end}/"
                if response.status != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(wanted):
                    raise RuntimeError("checkpoint server ignored exact range")
                while received < end - start:
                    block = response.read(min(4 * 1024 * 1024, end - start - received))
                    if not block:
                        raise RuntimeError("truncated checkpoint range")
                    digest.update(block)
                    left = start + received
                    right = left + len(block)
                    for key in names:
                        a, b = header[key]["data_offsets"]
                        lo, hi = max(a, left), min(b, right)
                        if hi > lo:
                            target.seek(target_offset + new_header[key]["data_offsets"][0] + lo - a)
                            target.write(block[lo - left : hi - left])
                    received += len(block)
            evidence.append(
                {
                    "shard": shard,
                    "source_start": absolute_start,
                    "source_end": absolute_end,
                    "sha256": digest.hexdigest(),
                    "tensor_count": len(names),
                }
            )
            print(shard, "range", absolute_start, absolute_end, "copied", received, flush=True)
    os.replace(partial, output)
    return {
        "file": output.name,
        "tensor_count": len(keys),
        "bytes": target_offset + cursor,
        "ranges": evidence,
    }


def hashes_for_tokens(config, mapping, tokens, layer_id):
    t = config.text
    seen = set()
    layer_primes = None
    for layer in t["engram_layer_ids"]:
        current_layer = []
        for _ in range(t["engram_max_ngram_size"] - 1):
            current = t["engram_vocab_size"] - 1
            for _ in range(t["engram_n_heads"]):
                current = int(nextprime(current))
                while current in seen:
                    current = int(nextprime(current))
                seen.add(current)
                current_layer.append(current)
        if layer == layer_id:
            layer_primes = current_layer
    bound = (np.iinfo(np.int64).max // t["engram_compressed_vocab_size"]) // 2
    mult = (
        np.random.default_rng(10007 * layer_id).integers(
            0, bound, size=t["engram_max_ngram_size"], dtype=np.int64
        )
        * 2
        + 1
    )
    offsets = np.cumsum([0, *layer_primes[:-1]]).tolist()
    pad = int(mapping[t["engram_pad_token_id"]])
    rows = set()
    compressed = [int(mapping[i]) for i in tokens]
    for pos in range(len(tokens)):
        rolling = compressed[pos] * int(mult[0])
        for shift in range(1, t["engram_max_ngram_size"]):
            token = compressed[pos - shift] if pos >= shift else pad
            rolling ^= token * int(mult[shift])
            for h in range(t["engram_n_heads"]):
                col = (shift - 1) * t["engram_n_heads"] + h
                rows.add(rolling % layer_primes[col] + offsets[col])
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = DeepSeekV41Config.from_json(args.metadata / "config.json")
    reader = PinnedCheckpointReader()
    tokenizer_path = args.output / "tokenizer.json"
    if not tokenizer_path.exists():
        with urllib.request.urlopen(reader.base + "tokenizer.json", timeout=60) as response:
            tokenizer_path.write_bytes(response.read())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    mapping, size = build_compressed_token_map(tokenizer)
    if size != config.text["engram_compressed_vocab_size"]:
        raise ValueError("released tokenizer vocabulary mismatch")
    save_file({"token_map": mapping}, str(args.output / "token-map.safetensors"))
    prompts = [
        "DeepSeek V4.1 ROCm model test.",
        "Deterministic forward and backward on AMD GPUs.",
        "你好，世界。模型训练与推理一致性测试。",
    ]
    token_sets = [[0] + tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
    cases = [
        {"name": f"prompt-{i}", "text": p, "input_ids": ids[:16]}
        for i, (p, ids) in enumerate(zip(prompts, token_sets, strict=True))
    ]
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    selected = {}
    for key, shard in reader.index.items():
        text_layer = any(key.startswith(f"layers.{i}.") for i in range(4))
        globals_ = key in ("embed.weight", "norm.weight", "head.weight")
        extra = key.startswith("layers.14.engram.") or any(
            key.startswith(f"layers.{i}.attn.compressor.") for i in (8, 14, 20)
        )
        if (text_layer or globals_ or extra) and ".engram.embed." not in key:
            selected[key] = shard
    by_shard = defaultdict(list)
    for key, shard in selected.items():
        by_shard[shard].append(key)
    manifest = {
        "revision": reader.revision,
        "scope": (
            "exact original layers 0,1,2,3 plus explicitly selected targeted components; "
            "frozen packed weights"
        ),
        "shards": [],
        "cases": cases,
    }
    for shard, keys in sorted(by_shard.items()):
        out = args.output / shard
        marker = args.output / (shard + ".complete.json")
        if marker.exists() and out.exists():
            info = json.loads(marker.read_text())
            if info.get("tensor_keys") != sorted(keys) or out.stat().st_size != info["bytes"]:
                raise ValueError("existing checkpoint slice differs from selection")
            print("reuse", shard, info["bytes"], flush=True)
        else:
            info = copy_shard(reader, shard, keys, out)
            info["tensor_keys"] = sorted(keys)
            marker.write_text(json.dumps(info, indent=2) + "\n")
        manifest["shards"].append(info)
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": selected}, indent=2) + "\n"
    )
    for layer in (1, 14):
        row_ids = sorted(
            {
                r
                for case in cases
                for r in hashes_for_tokens(config, mapping, case["input_ids"], layer)
            }
        )
        path = args.output / f"engram-{layer}-rows.safetensors"
        if not path.exists():
            prefix = f"layers.{layer}.engram.embed"
            rows = {
                prefix + ".weight": reader.read_rows(prefix + ".weight", row_ids),
                prefix + ".scale": reader.read_rows(prefix + ".scale", row_ids),
                "original_row_ids": torch.tensor(row_ids),
            }
            save_file(rows, str(path))
        manifest[f"engram_{layer}_row_ids"] = row_ids
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print("COMPLETE", len(selected), "real tensors", flush=True)


if __name__ == "__main__":
    main()
