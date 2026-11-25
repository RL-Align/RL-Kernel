# SPDX-License-Identifier: Apache-2.0
"""F2 shared table ownership. Hashing is initialization-only, never kernel dispatch.

Trusted manifests must come from the canonical producer, not from an untrusted
payload. This is an accidental-misuse contract, not a Python security boundary.
"""

from array import array
from dataclasses import dataclass
import hashlib
import json
import sys
import weakref

import torch

ALGORITHM = "f2-torch-fp32-pow-recip-mul-cos-sin-bf16-v1"
_REGISTRY = {}


@dataclass(frozen=True)
class RoPETableIdentity:
    length: int
    sha256: str
    algorithm: str = ALGORITHM
    theta: float = 500000.0
    dimension: int = 128
    position_start: int = 0
    layout: str = "cos-then-sin/bf16-le/row-major/Px64"


def _metadata(cos, sin):
    if cos.ndim != 2 or cos.shape[1] != 64 or sin.shape != cos.shape:
        raise ValueError("tables must have matching shape [P, 64]")
    if not 0 <= cos.shape[0] <= 131072:
        raise ValueError("table coverage exceeds context")
    if cos.device != sin.device:
        raise ValueError("table devices must match")
    for t in (cos, sin):
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError("tables must be contiguous BF16 tensors")
        if t.is_inference():
            raise ValueError("tables need version counters; create outside inference_mode")


def _digest(cos, sin):
    _metadata(cos, sin)
    if sys.byteorder != "little":
        raise RuntimeError("F2 table serialization requires a little-endian host")
    header = dict(
        algorithm=ALGORITHM,
        theta=500000.0,
        dimension=128,
        position_start=0,
        length=cos.shape[0],
        layout="cos-then-sin/bf16-le/row-major/Px64",
    )
    h = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    for t in (cos, sin):
        # Chunked explicit int16 bits avoid slow per-byte Storage iteration and
        # exclude adjacent storage when the input is a contiguous slice.
        host = t.detach().to("cpu").view(torch.int16).reshape(-1)
        for chunk in host.split(65536):
            h.update(array("h", chunk.tolist()).tobytes())
    return h.hexdigest()


def _register(cos, sin, identity):
    key = (id(cos), id(sin))

    def cleanup(_):
        _REGISTRY.pop(key, None)

    _REGISTRY[key] = (
        weakref.ref(cos, cleanup),
        weakref.ref(sin, cleanup),
        cos._version,
        sin._version,
        identity,
    )
    return cos, sin


def table_identity(cos, sin):
    """Validate registered pair and ordinary PyTorch mutation without GPU sync."""
    entry = _REGISTRY.get((id(cos), id(sin)))
    if entry is None or entry[0]() is not cos or entry[1]() is not sin:
        raise ValueError(
            "unverified RoPE table pair; use build_shared_rope_table or load_shared_rope_table"
        )
    _metadata(cos, sin)
    if (cos._version, sin._version) != entry[2:4]:
        raise ValueError("verified RoPE table was modified; reload the canonical artifact")
    return entry[4]


@torch.no_grad()
def build_shared_rope_table(length, *, device="cuda"):
    """Canonical CUDA producer; distribute its tensors and identity to other GPUs.

    Eager FP32 arange(0,128,2)/128, pow(500000,e), reciprocal, position
    multiplication, cos/sin, then BF16. Independent builds on other software/GPU
    versions are NOT guaranteed identical: compare identities or load one artifact.
    """
    if type(length) is not int or not 0 <= length <= 131072:
        raise ValueError("length must be an integer in [0, 131072]")
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        raise ValueError("canonical table construction requires NVIDIA CUDA")
    with torch.inference_mode(False), torch.autocast("cuda", enabled=False):
        e = torch.arange(0, 128, 2, dtype=torch.float32, device=device) / 128
        inv = 1.0 / (500000.0**e)
        angle = torch.arange(length, dtype=torch.float32, device=device)[:, None] * inv
        cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
        identity = RoPETableIdentity(length, _digest(cos, sin))
        return _register(cos, sin, identity)


@torch.no_grad()
def load_shared_rope_table(cos, sin, *, expected: RoPETableIdentity, device):
    """Verify against a trusted producer identity, then own isolated target copies.

    CPU loading supports artifact validation/tests, not CPU kernel execution.
    Never derive ``expected`` from the untrusted input being validated.
    """
    _metadata(cos, sin)
    if not isinstance(expected, RoPETableIdentity) or expected != RoPETableIdentity(
        cos.shape[0], expected.sha256
    ):
        raise ValueError("unsupported RoPE table identity metadata")
    with torch.inference_mode(False):
        c = cos.detach().to(device).clone()
        s = sin.detach().to(device).clone()
        if _digest(c, s) != expected.sha256:
            raise ValueError("RoPE table content does not match trusted identity")
        return _register(c, s, expected)
