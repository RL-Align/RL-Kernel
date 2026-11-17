# SPDX-License-Identifier: Apache-2.0
"""Lossless content-addressed tensor captures; original dtype/stride metadata retained."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import torch
import zstandard


def save_trace(value, path):
    path = Path(path)
    if os.environ.get("DSV4_TRACE_FORMAT") != "cas-v1":
        torch.save(value, path)
        return path.name
    root = Path("/outputs/trace-tensors")
    root.mkdir(exist_ok=True)

    def capture(x):
        if isinstance(x, torch.Tensor):
            cpu = x.detach().cpu().contiguous()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            sha = hashlib.sha256(raw).hexdigest()
            # The object stores original tensor bytes; metadata describes the source view.
            key = hashlib.sha256((str(cpu.dtype) + str(list(cpu.shape)) + sha).encode()).hexdigest()
            target = root / (key + ".pt.zst")
            if not target.exists():
                temp = target.with_name(target.name + f".{os.getpid()}.tmp")
                with temp.open("wb") as output:
                    with zstandard.ZstdCompressor(level=1).stream_writer(output) as stream:
                        torch.save(cpu, stream)
                try:
                    os.link(temp, target)
                except FileExistsError:
                    pass
                finally:
                    temp.unlink()
            return dict(
                tensor_object=str(target.relative_to(Path("/outputs"))),
                sha256=sha,
                dtype=str(x.dtype),
                shape=list(x.shape),
                stride=list(x.stride()),
                storage_offset=x.storage_offset(),
                device=str(x.device),
                raw_bytes=len(raw),
                first_bytes=raw[:64].hex(),
            )
        if isinstance(x, dict):
            return {str(k): capture(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [capture(v) for v in x]
        return x if x is None or isinstance(x, (str, bool, int, float)) else repr(type(x))

    target = path.with_suffix(".json")
    target.write_text(json.dumps(capture(value)) + "\n")
    return target.name


def load_tensor(root, record):
    path = Path(root) / record["tensor_object"]
    if path.suffix == ".zst":
        # Torch ZIP loading needs a seekable buffer; the original tensor bytes stay unchanged.
        import io

        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as stream:
                tensor = torch.load(
                    io.BytesIO(stream.read()), map_location="cpu", weights_only=True
                )
    else:
        with gzip.open(path, "rb") as stream:
            tensor = torch.load(stream, map_location="cpu", weights_only=True)
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("captured original tensor bytes were corrupted")
    if str(tensor.dtype) != record["dtype"] or list(tensor.shape) != record["shape"]:
        raise ValueError("captured original tensor dtype/shape changed")
    return tensor


def load_capture(path):
    """Read legacy .pt or lossless CAS captures from a completed run on any host."""
    path = Path(path)
    if path.suffix != ".json":
        return torch.load(path, map_location="cpu", weights_only=True)
    root = next((p for p in path.parents if (p / "trace-tensors").is_dir()), None)
    if root is None:
        raise ValueError("capture is missing its original trace-tensors store")

    def restore(value):
        if isinstance(value, dict):
            if "tensor_object" in value:
                return load_tensor(root, value)
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(json.loads(path.read_text()))
