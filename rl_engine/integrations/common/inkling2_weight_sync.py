# SPDX-License-Identifier: Apache-2.0
"""INK-41: versioned readback of canonical text weights and loop aliases."""

import hashlib
import json
from collections.abc import Mapping

import torch


def weight_digest(weights: Mapping[str, torch.Tensor]) -> str:
    """Hash canonical HF-keyed, already-mapped tensors; preserve exact dtypes."""
    if not weights:
        raise ValueError("empty canonical weight map")
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        if value.dtype not in (torch.bfloat16, torch.float32) or value.layout != torch.strided:
            raise ValueError(f"unsupported canonical tensor: {name}")
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(header.encode() + b"\0")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def validate_loop_aliases(aliases: Mapping[int, int]) -> None:
    physical = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    if dict(aliases) != dict(enumerate(physical)):
        raise ValueError("expected 48 logical visits sharing 24 physical weight owners")


class WeightSync:
    """Fail-closed eager coordinator; engine readback and cache reset are callbacks.

    Call under a quiesced rollout worker. An unsuccessful reload invalidates
    the published version; there is no claim of automatic rollback.
    """

    def __init__(self, *, platform: str):
        if platform not in ("cuda", "rocm"):
            raise ValueError("expected cuda or rocm")
        self.platform = platform
        self.version = None
        self.digest = None
        self._last_version = -1

    def publish(self, version, expected, *, reload, readback, aliases, reset_cache):
        if type(version) is not int or version <= self._last_version:
            raise ValueError("weight version must increase")
        self.version = self.digest = None
        self._last_version = version
        reload()
        weights = readback()
        if not weights or any(not t.is_cuda for t in weights.values()):
            raise RuntimeError("weight readback must come from GPU tensors")
        if (torch.version.hip is not None) != (self.platform == "rocm"):
            raise RuntimeError("weight readback platform mismatch")
        validate_loop_aliases(aliases())
        actual = weight_digest(weights)
        if actual != expected:
            raise RuntimeError("weight readback digest mismatch")
        if reset_cache() is not True:
            raise RuntimeError("cache reset not acknowledged")
        self.version, self.digest = version, actual

    def require_version(self, version):
        if self.version is None or self.version != version:
            raise RuntimeError("rollout weight version is not published")
