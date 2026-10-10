# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pinned MiniMax-H3 conditioning weights for the RFC #420 tests.

Only the tensors listed in ``h3_manifest.json`` are used. They all live in
shard 1 of 14, so ``tools/weights/prepare_h3_weights.py`` downloads that shard,
checks it against the manifest and extracts the tensors into
``$RL_KERNEL_H3_WEIGHTS/h3_conditioning.safetensors``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

MANIFEST_PATH = Path(__file__).with_name("h3_manifest.json")
WEIGHTS_ENV = "RL_KERNEL_H3_WEIGHTS"
EXTRACTED_FILE = "h3_conditioning.safetensors"


def load_h3_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def h3_weights_dir() -> Path | None:
    value = os.environ.get(WEIGHTS_ENV, "").strip()
    return Path(value).expanduser() if value else None


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_h3_conditioning_weights(
    device: torch.device | str = "cpu", names: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load the extracted pinned tensors; raise if they are missing or off-manifest."""

    from safetensors.torch import load_file

    root = h3_weights_dir()
    if root is None:
        raise FileNotFoundError(f"set {WEIGHTS_ENV} to the prepare_h3_weights.py output dir")
    path = root / EXTRACTED_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run tools/weights/prepare_h3_weights.py")
    manifest = load_h3_manifest()
    tensors = load_file(str(path), device=str(device))
    wanted = names if names is not None else list(manifest["tensors"])
    out: dict[str, torch.Tensor] = {}
    for name in wanted:
        spec = manifest["tensors"][name]
        tensor = tensors[name]
        if str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
            raise ValueError(f"{name}: dtype {tensor.dtype} != manifest {spec['dtype']}")
        if list(tensor.shape) != spec["shape"]:
            raise ValueError(f"{name}: shape {list(tensor.shape)} != manifest {spec['shape']}")
        out[name] = tensor
    return out
