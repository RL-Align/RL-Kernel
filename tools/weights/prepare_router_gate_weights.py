# SPDX-License-Identifier: Apache-2.0
"""Download one pinned 0731 gate tensor using HTTP ranges, never a full shard."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import torch

REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
BASE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REVISION}/"


def read_range(url, start, size):
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
    name = "layers.0.ffn.gate.weight"
    shard = index["weight_map"][name]
    # A unique query prevents intermediary caches from serving a different range.
    url = BASE + shard
    length = struct.unpack("<Q", read_range(url + "?range=header-size", 0, 8))[0]
    if length > 10_000_000:
        raise RuntimeError("unexpected safetensors header size")
    header = json.loads(read_range(url + "?range=header", 8, length))
    tensor = header[name]
    if tensor["dtype"] != "BF16" or tensor["shape"] != [256, 4096]:
        raise RuntimeError(f"0731 gate contract changed: {tensor}")
    start, end = tensor["data_offsets"]
    data = read_range(url + "?range=gate-weight", 8 + length + start, end - start)
    weight = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(256, 4096)
    metadata = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REVISION,
        "tensor": name,
        "shard": shard,
        "tensor_metadata": tensor,
        "weight_sha256": hashlib.sha256(data).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "metadata": metadata}, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
