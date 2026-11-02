# SPDX-License-Identifier: Apache-2.0
"""Download one pinned 0731 gate tensor using HTTP ranges, never a full shard."""

import argparse
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import torch

from rl_engine.validation.operators.router_gate_fixture import (
    MODEL,
    REVISION,
    SHAPE,
    SHARD,
    TENSOR,
    validate_tensor_metadata,
    validate_weight_bytes,
)

BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
MAX_RANGE_BYTES = 10_000_000


def read_range(url, start, size):
    if (
        type(start) is not int
        or start < 0
        or type(size) is not int
        or not (0 < size <= MAX_RANGE_BYTES)
    ):
        raise ValueError("HTTP range requires a nonnegative start and a bounded positive size")
    request = Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urlopen(request, timeout=60) as response:
        # Never stream through an entire shard when a server ignores Range.
        if response.status != 206:
            raise RuntimeError("checkpoint server must support HTTP byte ranges")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{start + size - 1}/"):
            raise RuntimeError(f"unexpected checkpoint range: {content_range}")
        data = response.read(size + 1)
    if len(data) != size:
        raise RuntimeError("truncated checkpoint range")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(BASE + "model.safetensors.index.json", timeout=60) as response:
        index = json.load(response)
    shard = index["weight_map"][TENSOR]
    if shard != SHARD:
        raise ValueError("gate shard does not match the frozen 0731 checkpoint")
    # A unique query prevents intermediary caches from serving a different range.
    url = BASE + shard
    length = struct.unpack("<Q", read_range(url + "?range=header-size", 0, 8))[0]
    if not 0 < length <= MAX_RANGE_BYTES:
        raise RuntimeError("unexpected safetensors header size")
    header = json.loads(read_range(url + "?range=header", 8, length))
    tensor = header[TENSOR]
    start, end = validate_tensor_metadata(tensor)
    data = read_range(url + "?range=gate-weight", 8 + length + start, end - start)
    digest = validate_weight_bytes(data)
    weight = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(SHAPE)
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "tensor": TENSOR,
        "shard": shard,
        "tensor_metadata": tensor,
        "weight_sha256": digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "metadata": metadata}, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
