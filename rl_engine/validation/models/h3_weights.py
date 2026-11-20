# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pinned MiniMax-H3 conditioning weights for the RFC #420 tests.

Only the tensors listed in ``h3_manifest.json`` are used. They all live in
shard 1 of 14, so ``tools/weights/prepare_h3_weights.py`` downloads that shard,
checks it against the manifest and extracts the tensors into
``$RL_KERNEL_H3_WEIGHTS/h3_conditioning.safetensors``. With ``--block`` it also
extracts block 0's attention and FFN weights (manifest ``block_tensors``,
about 1.1 GB) into ``h3_block0.safetensors`` for ``ws1_one_h3_block``.
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
BLOCK_FILE = "h3_block0.safetensors"


def load_h3_manifest() -> dict[str, Any]:
    """Read the bundled checkpoint identity and pinned tensor contracts."""

    return json.loads(MANIFEST_PATH.read_text())


def h3_weights_dir() -> Path | None:
    """Expand the configured extraction directory, or return None when unset or blank."""

    value = os.environ.get(WEIGHTS_ENV, "").strip()
    return Path(value).expanduser() if value else None


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """Return a file SHA256 while reading bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash a CPU tensor's contiguous bytes without safetensors metadata."""

    tensor_bytes = tensor.contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(tensor_bytes)).hexdigest()


def _load_pinned(
    filename: str, section: str, device: torch.device | str, names: list[str] | None
) -> dict[str, torch.Tensor]:
    """Load one extracted file and check every tensor of a manifest section against it."""

    from safetensors.torch import load_file

    root = h3_weights_dir()
    if root is None:
        raise FileNotFoundError(f"set {WEIGHTS_ENV} to the prepare_h3_weights.py output dir")
    path = root / filename
    if not path.is_file():
        flag = " --block" if section == "block_tensors" else ""
        raise FileNotFoundError(f"{path} missing; run tools/weights/prepare_h3_weights.py{flag}")
    specs = load_h3_manifest()[section]
    tensors = load_file(str(path), device="cpu")
    wanted = names if names is not None else list(specs)
    for name, spec in specs.items():
        tensor = tensors[name]
        if str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
            raise ValueError(f"{name}: dtype {tensor.dtype} != manifest {spec['dtype']}")
        if list(tensor.shape) != spec["shape"]:
            raise ValueError(f"{name}: shape {list(tensor.shape)} != manifest {spec['shape']}")
        actual = sha256_tensor(tensor)
        if actual != spec["sha256"]:
            raise ValueError(f"{name}: sha256 {actual} != manifest {spec['sha256']}")
    out: dict[str, torch.Tensor] = {}
    for name in wanted:
        if name not in specs:
            raise KeyError(name)
        out[name] = tensors[name].to(device=device)
    return out


def load_h3_conditioning_weights(
    device: torch.device | str = "cpu", names: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load the extracted pinned tensors; raise if they are missing or off-manifest."""

    return _load_pinned(EXTRACTED_FILE, "tensors", device, names)


def load_h3_block_weights(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Conditioning tensors plus block 0's attention/FFN weights, every one sha-checked."""

    return {
        **load_h3_conditioning_weights(device),
        **_load_pinned(BLOCK_FILE, "block_tensors", device, None),
    }


# --- Tensors straight from the released shards --------------------------------------------------

SHARD_ROOTS = ("hf_full/transformer", "hf/transformer")  # under $RL_KERNEL_H3_WEIGHTS


def h3_shard_root() -> Path | None:
    """The first shard directory under the weights dir that has an index, or None."""

    base = h3_weights_dir()
    if base is None:
        return None
    for sub in SHARD_ROOTS:
        root = base / sub
        if (root / "diffusion_pytorch_model.safetensors.index.json").is_file():
            return root
    return None


def load_h3_tensors(
    names: list[str], device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load named tensors from whichever released shards are present.

    Resolves each name through the shard index and reads only that tensor
    (safetensors memory-maps). Raises ``FileNotFoundError`` when the weights
    directory, the index or a needed shard is missing, so tests can skip.
    """

    from safetensors import safe_open

    root = h3_shard_root()
    if root is None:
        raise FileNotFoundError(f"no shard index under ${WEIGHTS_ENV}/{{{','.join(SHARD_ROOTS)}}}")
    weight_map = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"{name} is not in the H3 shard index")
        by_shard.setdefault(weight_map[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for shard, wanted in by_shard.items():
        path = root / shard
        if not path.is_file():
            raise FileNotFoundError(f"shard {shard} is not downloaded under {root}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in wanted:
                out[name] = handle.get_tensor(name).to(device)
    return out
