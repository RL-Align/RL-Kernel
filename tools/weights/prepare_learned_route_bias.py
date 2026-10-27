# SPDX-License-Identifier: Apache-2.0
"""Fetch the fixed 1-KiB layer-3 Flash-0731 correction bias, never its shard."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import torch

from rl_engine.contracts.operators.learned_route import (
    CHECKPOINT_BIAS_SHA256,
    CHECKPOINT_MODEL,
    CHECKPOINT_REVISION,
)

BASE = f"https://huggingface.co/{CHECKPOINT_MODEL}/resolve/{CHECKPOINT_REVISION}/"
SHARD = "model-00005-of-00048.safetensors"
TENSOR = "layers.3.ffn.gate.bias"


def read_range(url, start, size, label):
    if type(start) is not int or start < 0 or type(size) is not int or not 0 < size <= 10_000_000:
        raise ValueError("bias range requires a nonnegative start and bounded positive size")
    request = Request(
        url + "?learned-bias=" + label, headers={"Range": f"bytes={start}-{start + size - 1}"}
    )
    with urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-{start + size - 1}/"
        ):
            raise ValueError("unexpected bias HTTP byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("unexpected bias range length")
    return data


def validate_header(tensor):
    if not isinstance(tensor, dict) or tensor.get("dtype") != "F32" or tensor.get("shape") != [256]:
        raise ValueError("checkpoint bias must be F32 [256]")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(v) is not int or v < 0 for v in offsets)
    ):
        raise ValueError("bias offsets must be nonnegative integers")
    if offsets[1] - offsets[0] != 1024:
        raise ValueError("bias byte span must be exactly 1024")
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = BASE + SHARD
    length = struct.unpack("<Q", read_range(url, 0, 8, "size"))[0]
    if not 0 < length <= 10_000_000:
        raise ValueError("unexpected safetensors header length")
    header = json.loads(read_range(url, 8, length, "header"))
    tensor = header[TENSOR]
    start, end = validate_header(tensor)
    data = read_range(url, 8 + length + start, end - start, "bias")
    if hashlib.sha256(data).hexdigest() != CHECKPOINT_BIAS_SHA256:
        raise ValueError("bias payload differs from frozen checkpoint SHA256")
    bias = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    metadata = {
        "model": CHECKPOINT_MODEL,
        "revision": CHECKPOINT_REVISION,
        "absolute_layer": 3,
        "tensor": TENSOR,
        "shard": SHARD,
        "tensor_metadata": tensor,
        "sha256": CHECKPOINT_BIAS_SHA256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bias": bias, "metadata": metadata}, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
