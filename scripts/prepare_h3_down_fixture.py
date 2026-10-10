#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fetch one pinned H3 down weight by HTTP ranges, with synthetic activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rl_engine.kernels.ops.h3_ffn_down import H3_CHECKPOINT_REVISION  # noqa: E402

BASE = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{H3_CHECKPOINT_REVISION}/"
WEIGHT_KEY = "transformer_blocks.0.ff.net.2.weight"
WEIGHT_SHAPE = [5376, 14336]


def read_range(url, start, end, *, opener=urllib.request.urlopen):
    """Reject servers that ignore Range instead of downloading a whole shard."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with opener(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"expected HTTP 206, got {response.status}; refusing full shard")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or (int(match[1]), int(match[2])) != (start, end):
            raise RuntimeError("Content-Range does not match requested tensor bytes")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("range response length mismatch")
        return data


def tensor_extent(entry, header_size):
    if entry["dtype"] != "BF16" or entry["shape"] != WEIGHT_SHAPE:
        raise ValueError("checkpoint down weight must be BF16 [5376,14336]")
    start, end = entry["data_offsets"]
    expected = 5376 * 14336 * 2
    if start < 0 or end - start != expected:
        raise ValueError("checkpoint tensor extent does not match its declared shape")
    return 8 + header_size + start, 8 + header_size + end


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument(
        "--variant", choices=("transformer", "transformer_ref"), default="transformer"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.rows <= 32768:
        parser.error("rows must be in [1,32768]")
    index_url = BASE + args.variant + "/diffusion_pytorch_model.safetensors.index.json"
    with urllib.request.urlopen(index_url, timeout=60) as response:
        index_raw = response.read(8 << 20)
    index = json.loads(index_raw)
    shard = index["weight_map"][WEIGHT_KEY]
    if Path(shard).name != shard or not shard.endswith(".safetensors"):
        raise ValueError("unexpected checkpoint shard path")
    url = BASE + args.variant + "/" + shard
    header_size = struct.unpack("<Q", read_range(url, 0, 7))[0]
    if not 0 < header_size <= 16 << 20:
        raise ValueError("invalid safetensors header length")
    header_raw = read_range(url, 8, 7 + header_size)
    header = json.loads(header_raw)
    start, end = tensor_extent(header[WEIGHT_KEY], header_size)
    data = bytearray()
    for lo in range(start, end, 8 << 20):
        hi = min(lo + (8 << 20), end) - 1
        data.extend(read_range(url, lo, hi))
        print(f"weight bytes: {len(data)}/{end - start}", flush=True)
    weight = torch.frombuffer(data, dtype=torch.bfloat16).reshape(WEIGHT_SHAPE).clone()
    generator = torch.Generator().manual_seed(args.seed)
    gate = torch.randn(args.rows, 14336, generator=generator)
    up = torch.randn(args.rows, 14336, generator=generator)
    x = (torch.nn.functional.silu(gate) * up).bfloat16()
    grad = (torch.randn(args.rows, 5376, generator=generator) * 0.1).bfloat16()
    metadata = {
        "checkpoint_revision": H3_CHECKPOINT_REVISION,
        "model_variant": args.variant,
        "weight_key": WEIGHT_KEY,
        "weight_source": url,
        "weight_sha256": hashlib.sha256(data).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "activation_source": "seeded synthetic FP32 SiLU(gate) * up, then BF16",
        "activation_kind": "synthetic",
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "weight": weight, "grad_output": grad, "metadata": metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved real checkpoint weight with synthetic activations: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
