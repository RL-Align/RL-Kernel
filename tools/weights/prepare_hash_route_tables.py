# SPDX-License-Identifier: Apache-2.0
"""Fetch only the three pinned 0731 Hash tables, with bounded ranges and fixed SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import torch

from rl_engine.contracts.operators.hash_route import (
    CHECKPOINT_TABLE_SHA256,
    HASH_LAYERS,
    MODEL,
    REVISION,
    SLOTS,
    TABLE_BYTES,
    VOCABULARY,
)

BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"


def read_range(url, start, size, label):
    if type(start) is not int or start < 0 or type(size) is not int or not 0 < size <= 10_000_000:
        raise ValueError(
            "Hash checkpoint range must have bounded positive size and nonnegative start"
        )
    request = Request(
        url + "?hash-route=" + label, headers={"Range": f"bytes={start}-{start + size - 1}"}
    )
    with urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-{start + size - 1}/"
        ):
            raise ValueError("Hash checkpoint server returned an unexpected HTTP byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Hash checkpoint range has an unexpected byte count")
    return data


def validate_header(tensor):
    if (
        not isinstance(tensor, dict)
        or tensor.get("dtype") != "I64"
        or (tensor.get("shape") != [VOCABULARY, SLOTS])
    ):
        raise ValueError("0731 tid2eid must be I64 [129280,6]")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int or value < 0 for value in offsets)
    ):
        raise ValueError("Hash table offsets must be two nonnegative integers")
    start, end = offsets
    if end - start != TABLE_BYTES:
        raise ValueError("Hash table byte span must equal the frozen I64 tensor size")
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(BASE + "model.safetensors.index.json", timeout=60) as response:
        index = json.load(response)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in HASH_LAYERS:
        tensor_name = f"layers.{layer}.ffn.gate.tid2eid"
        shard = f"model-{layer + 2:05d}-of-00048.safetensors"
        if index["weight_map"][tensor_name] != shard:
            raise ValueError("Hash table shard identity differs from the frozen checkpoint")
        url = BASE + shard
        length = struct.unpack("<Q", read_range(url, 0, 8, "size"))[0]
        if not 0 < length <= 10_000_000:
            raise ValueError("unexpected safetensors header size")
        header = json.loads(read_range(url, 8, length, "header"))
        tensor = header[tensor_name]
        start, end = validate_header(tensor)
        data = read_range(url, 8 + length + start, end - start, "table")
        digest = hashlib.sha256(data).hexdigest()
        if digest != CHECKPOINT_TABLE_SHA256[layer]:
            raise ValueError("Hash table bytes do not match the frozen layer SHA256")
        table = (
            torch.frombuffer(bytearray(data), dtype=torch.int64).clone().reshape(VOCABULARY, SLOTS)
        )
        metadata = {
            "model": MODEL,
            "revision": REVISION,
            "absolute_layer": layer,
            "tensor": tensor_name,
            "shard": shard,
            "tensor_metadata": tensor,
            "sha256": digest,
        }
        torch.save(
            {"table": table, "metadata": metadata},
            args.output_dir / f"0731-layer{layer}-tid2eid.pt",
        )
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
