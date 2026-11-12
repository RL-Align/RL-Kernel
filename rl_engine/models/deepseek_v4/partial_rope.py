# SPDX-License-Identifier: Apache-2.0
"""Item 11: GPT-J interleaved RoPE on the trailing 64 dims, by global position."""

import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.partial_rope import PartialRoPEContract, require


def implementation_fingerprint(contract, backend, source):
    identity = {
        "contract": contract.fingerprint,
        "backend": getattr(backend, "provenance", {"test_backend": type(backend).__name__}),
        "model_source": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "plan": backend.plan(contract),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def check_backend(contract, backend, expected_implementation, source):
    require(backend.backend_id == contract.backend_id, "backend identity mismatch")
    fingerprint = implementation_fingerprint(contract, backend, source)
    require(
        expected_implementation in (None, fingerprint),
        "train/rollout implementation fingerprint mismatch",
    )
    return fingerprint


def select_angles(x, positions, cos_cache, sin_cache, *, contract, backend, heads=None):
    """Validate [T,H,D] BF16 rows; return zeroed padding and FP32 [T,1,1,rope/2] angles."""
    require(
        isinstance(x, torch.Tensor)
        and x.ndim == 3
        and x.shape[2] == contract.head_dim
        and heads in (None, x.shape[1])
        and x.dtype == torch.bfloat16
        and x.is_contiguous(),
        "rows must be contiguous BF16 [tokens, heads, head_dim]",
    )
    require(x.device.type == backend.device_type, "tensor/backend device mismatch")
    if hasattr(backend, "device"):
        require(x.device == backend.device, "bind a backend on this device")
    if x.device.type == "cuda":
        with torch.cuda.device(x.device):
            require(not torch.cuda.is_current_stream_capturing(), "Graph capture is not validated")
    require(
        isinstance(positions, torch.Tensor)
        and positions.shape == (x.shape[0],)
        and positions.dtype == torch.int64
        and positions.device == x.device,
        "int64 global positions [tokens] required",
    )
    half = contract.rope_dim // 2
    require(
        cos_cache.ndim == 2
        and cos_cache.shape == sin_cache.shape
        and cos_cache.shape[0] > 0
        and cos_cache.shape[1] == half,
        "cos/sin tables must be [positions, rope_dim/2]",
    )
    require(
        all(
            t.device == x.device
            and t.dtype in (torch.bfloat16, torch.float32)
            and not t.requires_grad
            for t in (cos_cache, sin_cache)
        ),
        "read-only BF16/FP32 RoPE tables on the row device required",
    )
    require(
        bool(((positions >= -1) & (positions < cos_cache.shape[0])).all()),
        "position outside the RoPE table; padding is -1",
    )
    with torch.inference_mode(False), torch.no_grad():
        selected = positions.clamp_min(0)
        cos = cos_cache.index_select(0, selected).float().reshape(-1, 1, 1, half)
        sin = sin_cache.index_select(0, selected).float().reshape(-1, 1, 1, half)
    require(
        bool(
            (torch.isfinite(cos) & torch.isfinite(sin) & (cos.abs() <= 1) & (sin.abs() <= 1)).all()
        ),
        "invalid RoPE angles",
    )
    return torch.where((positions >= 0)[:, None, None], x, 0), cos, sin


class _PartialRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, backend):
        ctx.save_for_backward(cos, sin)
        ctx.backend = backend
        return backend.rotate(x, cos, sin)

    @staticmethod
    def backward(ctx, grad):
        require(not torch.is_grad_enabled(), "higher-order RoPE gradients are not supported")
        cos, sin = ctx.saved_tensors
        grad = grad.to(torch.bfloat16).contiguous()
        return ctx.backend.rotate_vjp(grad, cos, sin), None, None, None


class PartialRoPE:
    """``out[..., -64:]`` rotated pairwise (even, odd); ``out[..., :-64]`` untouched."""

    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(isinstance(contract, PartialRoPEContract), "typed partial RoPE contract required")
        if backend is None:
            from rl_engine.backends.rocm.attention.partial_rope import AiterPartialRoPEBackend

            backend = AiterPartialRoPEBackend()
        self.contract, self.backend = contract, backend
        self.implementation_fingerprint = check_backend(
            contract, backend, expected_implementation, __file__
        )

    def __call__(self, x, positions, cos_cache, sin_cache):
        masked, cos, sin = select_angles(
            x, positions, cos_cache, sin_cache, contract=self.contract, backend=self.backend
        )
        return _PartialRoPE.apply(masked, cos, sin, self.backend)
