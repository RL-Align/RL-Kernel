# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Check P5's scale-residual range on DeepSeek-V4-Flash's real MXFP4 expert scales.

P5's SM90 routed-expert kernel folds each 32-element E8M0 block scale into the E4M3
weight bytes relative to a per-row reference ``ref = max_j s[j] - 6``. That is exact only
when every residual ``s[j] - ref`` lies in [-8, 6], i.e. when the spread
``max_j s[j] - min_j s[j]`` of a weight row (over K) is at most 14.
``sm90_moe_prepare_weight_ref`` fails closed otherwise. This script fetches the expert
scale tensors (``w1`` / ``w3`` ``[F, H/32]`` and ``w2`` ``[H, F/32]``, E8M0) of the
chosen layers and experts with HTTP range requests (no weights are downloaded) and
reports the spread distribution.

The nibble order needs no download: ``scripts/check_dsv4_model_ref.py`` runs DeepSeek's
own ``fp4_gemm`` on bytes written low nibble = even column, and its MoE output agrees
with P5's to 2.5% (cosine 0.9997); swapped nibbles would permute the input features.

    python scripts/check_dsv4_fp4_scales.py --layers 0 10 21 32 42 --experts 0 64 128 255
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import struct
import urllib.request

import numpy as np

REPO = "deepseek-ai/DeepSeek-V4-Flash"


def _url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{name}"


def _get(name: str, lo: int, hi: int) -> bytes:
    req = urllib.request.Request(_url(name), headers={"Range": f"bytes={lo}-{hi - 1}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == hi - lo:
                return data
        except OSError:
            if attempt == 3:
                raise
    raise RuntimeError(f"short read from {name} [{lo}, {hi})")


def _header(shard: str) -> tuple[int, dict]:
    n = struct.unpack("<Q", _get(shard, 0, 8))[0]
    return 8 + n, json.loads(_get(shard, 8, 8 + n))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", type=int, nargs="+", default=[0, 10, 21, 32, 42])
    p.add_argument("--experts", type=int, nargs="+", default=[0, 37, 64, 128, 200, 255])
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    with urllib.request.urlopen(_url("model.safetensors.index.json"), timeout=60) as r:
        weight_map = json.loads(r.read())["weight_map"]
    names = [
        f"layers.{layer}.ffn.experts.{e}.{w}.scale"
        for layer in args.layers
        for e in args.experts
        for w in ("w1", "w3", "w2")
    ]
    missing = [n for n in names if n not in weight_map]
    if missing:
        raise SystemExit(f"not in the checkpoint: {missing[:4]}")
    shards = sorted({weight_map[n] for n in names})
    with cf.ThreadPoolExecutor(args.workers) as pool:
        headers = dict(zip(shards, pool.map(_header, shards)))

    def fetch(name: str):
        base, header = headers[weight_map[name]]
        info = header[name]
        if info["dtype"] != "F8_E8M0":
            raise SystemExit(f"{name}: dtype {info['dtype']}")
        lo, hi = info["data_offsets"]
        codes = np.frombuffer(_get(weight_map[name], base + lo, base + hi), dtype=np.uint8)
        return name, codes.reshape(info["shape"]).astype(np.int32)

    spreads: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    worst = []
    rows = bad = 0
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for name, codes in pool.map(fetch, names):
            spread = codes.max(axis=1) - codes.min(axis=1)
            kind = name.rsplit(".", 2)[-2]
            spreads[kind].update(spread.tolist())
            rows += spread.size
            over = int((spread > 14).sum())
            bad += over
            worst.append((int(spread.max()), name, int(codes.min()), int(codes.max())))

    print(f"layers {args.layers}, experts {args.experts}: {len(names)} scale tensors, {rows} rows")
    for kind in ("w1", "w3", "w2"):
        c = spreads[kind]
        hist = ", ".join(f"{k}:{c[k]}" for k in sorted(c))
        print(f"  {kind}: max spread {max(c)}; spread histogram {{{hist}}}")
    worst.sort(reverse=True)
    for spread, name, lo, hi in worst[:5]:
        print(f"  widest: {name} spread {spread} (codes {lo}..{hi})")
    print(f"rows with spread > 14 (P5 would refuse): {bad} of {rows}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
