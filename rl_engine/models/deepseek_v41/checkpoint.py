# SPDX-License-Identifier: Apache-2.0
"""Original-name layer-wise loading and fail-closed pinned HTTP byte ranges.

Only selected tensors are materialized. Packed expert I8 bytes are preserved;
no key renumbering, scale conversion, missing-weight initialization or download
of the 101GB Engram shard is implicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import urllib.request
from pathlib import Path

import torch

from .config import MODEL_ID, OFFICIAL_REVISION

_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


class PinnedCheckpointReader:
    def __init__(self, *, model_id=MODEL_ID, revision=OFFICIAL_REVISION):
        if model_id != MODEL_ID or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("reader requires the official model and an immutable SHA revision")
        self.revision = revision
        self.base = f"https://huggingface.co/{model_id}/resolve/{revision}/"
        self._headers = {}
        self.evidence = []
        with urllib.request.urlopen(
            self.base + "model.safetensors.index.json", timeout=60
        ) as response:
            self.index = json.load(response)["weight_map"]

    def _range(self, filename, start, end):
        if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", filename) is None:
            raise ValueError("invalid checkpoint shard name")
        request = urllib.request.Request(
            self.base + filename, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            if (
                response.status != 206
                or re.fullmatch(rf"bytes {start}-{end}/\d+", content_range) is None
            ):
                raise RuntimeError(
                    f"server did not honor exact range: {response.status} {content_range}"
                )
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("truncated or oversized checkpoint byte range")
        self.evidence.append(
            {
                "shard": filename,
                "start": start,
                "end": end,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        return data

    def header(self, filename):
        if filename not in self._headers:
            size = struct.unpack("<Q", self._range(filename, 0, 7))[0]
            if not 2 <= size <= 64 * 1024 * 1024:
                raise ValueError("invalid safetensors header size")
            self._headers[filename] = (size + 8, json.loads(self._range(filename, 8, size + 7)))
        return self._headers[filename]

    def read_tensor(self, key, *, max_bytes=256 * 1024 * 1024):
        filename = self.index[key]
        offset, header = self.header(filename)
        info = header[key]
        start, end = info["data_offsets"]
        if end - start > max_bytes:
            raise ValueError(
                f"{key} is {end - start} bytes; use explicit row loading or raise max_bytes"
            )
        dtype = _DTYPES[info["dtype"]]
        expected_bytes = math.prod(info["shape"]) * torch.empty((), dtype=dtype).element_size()
        if end - start != expected_bytes:
            raise ValueError("tensor byte length differs from dtype/shape")
        data = self._range(filename, offset + start, offset + end - 1)
        return (
            torch.frombuffer(bytearray(data), dtype=torch.uint8)
            .view(dtype)
            .reshape(info["shape"])
            .clone()
        )

    def read_rows(self, key, row_ids):
        """Read exactly the requested original rows, with no synthetic missing rows."""
        filename = self.index[key]
        offset, header = self.header(filename)
        info = header[key]
        if len(info["shape"]) != 2:
            raise ValueError("row slicing requires a matrix")
        ids = list(row_ids)
        if (
            not ids
            or ids != sorted(set(ids))
            or any(type(i) is not int or not 0 <= i < info["shape"][0] for i in ids)
        ):
            raise ValueError("provide unique sorted in-range original row IDs")
        dtype = _DTYPES[info["dtype"]]
        row_bytes = info["shape"][1] * torch.empty((), dtype=dtype).element_size()
        start, end = info["data_offsets"]
        if end - start != info["shape"][0] * row_bytes:
            raise ValueError("invalid row storage span")
        chunks = [
            self._range(
                filename, offset + start + i * row_bytes, offset + start + (i + 1) * row_bytes - 1
            )
            for i in ids
        ]
        return (
            torch.frombuffer(bytearray(b"".join(chunks)), dtype=torch.uint8)
            .view(dtype)
            .reshape(len(ids), -1)
            .clone()
        )


class LayerCheckpoint:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        with (self.directory / "model.safetensors.index.json").open() as stream:
            self.index = json.load(stream)["weight_map"]

    def keys_for_layers(self, layer_ids):
        prefixes = tuple(f"layers.{i}." for i in layer_ids)
        return tuple(k for k in self.index if k.startswith(prefixes))

    def tensor(self, key, *, device="cpu"):
        from safetensors import safe_open

        path = (self.directory / self.index[key]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("checkpoint shard escapes its directory")
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        with safe_open(path, framework="pt", device=str(target)) as shard:
            return shard.get_tensor(key)

    def projection(self, prefix, *, device):
        w = self.tensor(prefix + ".weight", device=device)
        if w.dtype == torch.float8_e4m3fn:
            s = self.tensor(prefix + ".scale", device=device)
            if s.dtype != torch.float8_e8m0fnu or s.shape != (
                (w.shape[0] + 31) // 32,
                w.shape[1] // 32,
            ):
                raise ValueError(f"invalid block32 scale metadata for {prefix}")
            return w, s
        if w.dtype == torch.int8:
            s = self.tensor(prefix + ".scale", device=device)
            if s.dtype != torch.float8_e8m0fnu or s.shape != (w.shape[0], w.shape[1] // 16):
                raise ValueError(f"invalid packed FP4 scale metadata for {prefix}")
            return w.view(torch.uint8), s.view(torch.uint8)
        if w.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"unsupported projection dtype for {prefix}: {w.dtype}")
        return w, None
