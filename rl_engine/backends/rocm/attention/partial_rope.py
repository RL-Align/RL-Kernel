# SPDX-License-Identifier: Apache-2.0
"""Reuse aiter cached RoPE (#627/#631) and #624's Hadamard/MXFP4; no new GPU kernel."""

import hashlib
import importlib
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_q_preprocess import INDEXER_Q_BACKEND
from rl_engine.contracts.operators.partial_rope import (
    PARTIAL_ROPE_BACKEND,
    PartialRoPEError,
    require,
)

# rotate_style=1 is GPT-J interleaved; cos/sin hold rope_dim/2 entries; NoPE leads.
ROPE_LAUNCH = {"rotate_style": 1, "reuse_freqs_front_part": True, "nope_first": True}
ROPE_SOURCES = (
    "aiter.ops.rope",
    "aiter.jit.module_rope_1c_cached_fwd",
    "aiter.jit.module_rope_1c_cached_bwd",
)


class AiterPartialRoPEBackend:
    backend_id = PARTIAL_ROPE_BACKEND
    device_type = "cuda"

    def __init__(self):
        require(torch.version.hip is not None and torch.cuda.is_available(), "requires a ROCm GPU")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", f"unvalidated architecture {arch}")
        try:
            modules = [importlib.import_module(name) for name in ROPE_SOURCES]
            self._forward, self._backward = modules[0].rope_cached_fwd, modules[0].rope_cached_bwd
        except (ImportError, AttributeError) as exc:
            raise PartialRoPEError("required aiter cached RoPE ABI unavailable") from exc
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "launch": ROPE_LAUNCH,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sources": {
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in zip(ROPE_SOURCES, modules, strict=True)
            },
        }

    @staticmethod
    def plan(contract):
        return dict(ROPE_LAUNCH)

    def rotate(self, x, cos, sin):
        if not x.shape[0]:
            return torch.empty_like(x)
        return self._forward(x[:, None], cos, sin, *ROPE_LAUNCH.values()).squeeze(1)

    def rotate_vjp(self, grad, cos, sin):
        if not grad.shape[0]:
            return torch.empty_like(grad)
        return self._backward(grad[:, None], cos, sin, *ROPE_LAUNCH.values()).squeeze(1)


class AiterIndexerQBackend(AiterPartialRoPEBackend):
    """Item 11 RoPE, then #624's tail-safe Hadamard and even-rule MXFP4."""

    backend_id = INDEXER_Q_BACKEND

    def __init__(self, *, codec=None):
        super().__init__()
        if codec is None:
            from rl_engine.backends.rocm.attention.indexer_compressor_c4 import (
                AiterIndexerC4Backend,
            )

            with torch.cuda.device(self.device):
                codec = AiterIndexerC4Backend()
        require(codec.device == self.device, "codec/backend device mismatch")
        self.codec = codec
        self.provenance["backend"] = self.backend_id
        self.provenance["codec"] = codec.provenance

    def plan(self, contract):
        return {"rope": dict(ROPE_LAUNCH), "quantize": "per_1x32_f4_quant_triton;shuffle=False"}

    def forward(self, q, cos, sin, contract):
        rows, heads, dim = q.shape
        values = self.codec._rotate(self.rotate(q, cos, sin).reshape(-1, dim))
        data, scales = self.codec.quantize(values, contract)
        return (
            values.reshape(rows, heads, dim),
            data.reshape(rows, heads, dim // 2),
            scales.reshape(rows, heads, dim // 32),
        )

    def vjp(self, grad, cos, sin):
        unrotated = self.codec._rotate(grad.reshape(-1, grad.shape[-1])).reshape_as(grad)
        return self.rotate_vjp(unrotated, cos, sin)
