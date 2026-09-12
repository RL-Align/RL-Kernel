# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Precompiled deterministic ROCm RoPE matching the shared reference layout."""

from __future__ import annotations

from threading import Lock

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.rotary_embedding.rope import (
    _build_cos_sin,
    _restore_rope,
    _rope_table,
)


@torch.library.custom_op("rl_kernel::deterministic_rope_apply_rocm", mutates_args=())
def _deterministic_rope_apply_rocm(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    direction: float,
) -> Tensor:
    return _C.deterministic_rope_apply_rocm(x, cos, sin, direction)


@_deterministic_rope_apply_rocm.register_fake
def _deterministic_rope_apply_rocm_fake(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    direction: float,
) -> Tensor:
    del cos, sin, direction
    return torch.empty_like(x)


@torch.library.custom_op("rl_kernel::deterministic_rope_apply_token_major_rocm", mutates_args=())
def _deterministic_rope_apply_token_major_rocm(
    x: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    head_dim: int,
    direction: float,
) -> Tensor:
    return _C.deterministic_rope_apply_token_major_rocm(
        x,
        positions,
        cos,
        sin,
        head_dim,
        direction,
    )


@_deterministic_rope_apply_token_major_rocm.register_fake
def _deterministic_rope_apply_token_major_rocm_fake(
    x: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    head_dim: int,
    direction: float,
) -> Tensor:
    del positions, cos, sin, head_dim, direction
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _forward_rope_pair(
    query: Tensor,
    key: Tensor,
    positions: Tensor,
    theta: float,
    *,
    inv_freq: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the paired forward arithmetic and return its shared table."""

    if positions.dim() != 1:
        raise ValueError("paired ROCm RoPE requires a flat position tensor")
    if inv_freq is None:
        query_2d, cos, sin = _rope_table(query, positions, theta)
    else:
        head_dim = query.shape[-1]
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        table_len = int(positions.shape[0])
        query_2d = query.contiguous().reshape(-1, head_dim)
        if query_2d.shape[0] % table_len != 0:
            raise ValueError(
                f"row count {query_2d.shape[0]} not divisible by seq length {table_len}; "
                "expected a [..., S, D] contiguous layout."
            )
        if (
            inv_freq.shape != (head_dim // 2,)
            or inv_freq.dtype != torch.float32
            or inv_freq.device != query.device
        ):
            raise RuntimeError("cached ROCm RoPE inv_freq does not match the input")
        position_rows = positions.to(device=query.device, dtype=torch.float32).reshape(-1, 1)
        frequencies = position_rows * inv_freq
        cos = frequencies.cos().contiguous()
        sin = frequencies.sin().contiguous()
    key_2d = key.contiguous().reshape(-1, key.shape[-1])
    if key_2d.size(0) % cos.size(0):
        raise ValueError(
            f"key row count {key_2d.size(0)} is not divisible by " f"position count {cos.size(0)}"
        )
    query_out = _C.deterministic_rope_apply_rocm(query_2d, cos, sin, 1.0)
    key_out = _C.deterministic_rope_apply_rocm(key_2d, cos, sin, 1.0)
    return query_out.reshape(query.shape), key_out.reshape(key.shape), cos, sin


class _RocmRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin = _rope_table(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.x_shape = tuple(x.shape)
        ctx.pos_dim = positions.dim()
        out_2d = _deterministic_rope_apply_rocm(x_2d, cos, sin, 1.0)
        return _restore_rope(out_2d, x, positions)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.pos_dim == 2 and len(ctx.x_shape) == 4:
                g_2d = grad_out.permute(1, 0, 2, 3).contiguous().reshape(-1, ctx.x_shape[-1])
                out_2d = _deterministic_rope_apply_rocm(g_2d, cos, sin, -1.0)
                heads, batch, seq, dim = (
                    ctx.x_shape[1],
                    ctx.x_shape[0],
                    ctx.x_shape[2],
                    ctx.x_shape[3],
                )
                grad_x = out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()
            else:
                g_2d = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
                grad_x = _deterministic_rope_apply_rocm(g_2d, cos, sin, -1.0).reshape(
                    grad_out.shape
                )
        return grad_x, None, None


class _RocmRoPEPairFunction(torch.autograd.Function):
    """Rotate Q/K with one shared position table and independent HIP launches."""

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        theta: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        query_out, key_out, cos, sin = _forward_rope_pair(query, key, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.query_shape = tuple(query.shape)
        ctx.key_shape = tuple(key.shape)
        return query_out, key_out

    @staticmethod
    def backward(ctx, grad_query: Tensor, grad_key: Tensor):
        cos, sin = ctx.saved_tensors

        def rotate_gradient(grad: Tensor | None, shape: tuple[int, ...]) -> Tensor | None:
            if grad is None:
                return None
            grad_2d = grad.contiguous().reshape(-1, shape[-1])
            return _C.deterministic_rope_apply_rocm(grad_2d, cos, sin, -1.0).reshape(shape)

        query_grad = (
            rotate_gradient(grad_query, ctx.query_shape) if ctx.needs_input_grad[0] else None
        )
        key_grad = rotate_gradient(grad_key, ctx.key_shape) if ctx.needs_input_grad[1] else None
        return query_grad, key_grad, None, None


class RocmDeterministicRoPEOp:
    """Precompiled HIP RoPE path shared by ROCm training and rollout."""

    backend_id = "rlkernel.rocm.deterministic_rope"
    op_class = "elementwise"
    fallback = False

    def __init__(self) -> None:
        if torch.version.hip is None:
            raise RuntimeError("RocmDeterministicRoPEOp requires a ROCm PyTorch build")
        if not _EXT_AVAILABLE or not all(
            hasattr(_C, symbol)
            for symbol in (
                "deterministic_rope_apply_rocm",
                "deterministic_rope_apply_token_major_rocm",
            )
        ):
            raise RuntimeError(
                "ROCm deterministic RoPE is unavailable; rebuild rl_engine._C for ROCm"
            )
        self._inference_inv_freq_cache: dict[tuple[torch.device, object, int, str], Tensor] = {}
        self._inference_inv_freq_lock = Lock()

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        self._validate_input(x)
        return _RocmRoPEFunction.apply(x, positions, theta)

    def forward_pair(
        self,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        *,
        theta: float = 1_000_000.0,
    ) -> tuple[Tensor, Tensor]:
        """Rotate Q/K with one FP32 cos/sin table and unchanged HIP arithmetic."""

        self._validate_input(query)
        self._validate_input(key)
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("paired ROCm RoPE Q/K must share one device and dtype")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("paired ROCm RoPE Q/K must share one head dimension")
        if positions.dim() != 1:
            raise ValueError("paired ROCm RoPE requires a flat position tensor")
        if not torch.is_grad_enabled():
            # vLLM executes rollout under inference/no-grad mode. Calling the
            # custom autograd Function there cannot contribute a backward but
            # still pays its dispatcher/context cost in every decoder layer.
            # Keep the exact same table construction and HIP launches while
            # bypassing only that unused autograd wrapper.
            query_out, key_out, _cos, _sin = _forward_rope_pair(
                query,
                key,
                positions,
                theta,
                inv_freq=self._cached_inference_inv_freq(query, theta),
            )
            return query_out, key_out
        query_out, key_out = _RocmRoPEPairFunction.apply(query, key, positions, theta)
        # A multi-output autograd Function marks both outputs differentiable if
        # either input requires grad. Preserve the two independent-call API.
        if not query.requires_grad:
            query_out = query_out.detach()
        if not key.requires_grad:
            key_out = key_out.detach()
        return query_out, key_out

    def _cached_inference_inv_freq(self, query: Tensor, theta: float) -> Tensor:
        """Reuse only the position-independent FP32 part of the RoPE table."""

        head_dim = query.shape[-1]
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        resolved_theta = float(theta)
        # ``float.hex`` keeps +0.0 and -0.0 distinct, unlike float equality.
        stream = torch.cuda.current_stream(query.device)
        key = (query.device, stream, head_dim, resolved_theta.hex())
        cached = self._inference_inv_freq_cache.get(key)
        if cached is not None:
            return cached
        half = head_dim // 2
        inv_freq = 1.0 / (
            resolved_theta
            ** (
                torch.arange(
                    0,
                    half,
                    dtype=torch.float32,
                    device=query.device,
                )
                / half
            )
        )
        # Build outside the lock, then publish once. The dict never evicts:
        # keys retain their streams and values remain alive for any HIP graph
        # that captured them. A same-key race returns the first published
        # tensor, whose producer and consumers all use that key's stream.
        with self._inference_inv_freq_lock:
            return self._inference_inv_freq_cache.setdefault(key, inv_freq)

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not x.is_cuda:
            raise RuntimeError("ROCm deterministic RoPE requires a GPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm deterministic RoPE requires FP16 or BF16")

    @staticmethod
    def build_position_table(
        max_positions: int,
        head_dim: int,
        *,
        device: torch.device,
        theta: float,
    ) -> tuple[Tensor, Tensor]:
        """Build the training-identical FP32 table once before graph capture."""

        if max_positions <= 0 or head_dim <= 0 or head_dim % 2:
            raise ValueError("ROCm RoPE table dimensions must be positive and even")
        positions = torch.arange(max_positions, dtype=torch.int64, device=device)
        return _build_cos_sin(positions, head_dim // 2, float(theta), device)

    @staticmethod
    def forward_token_major(
        x: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        head_dim: int,
    ) -> Tensor:
        """Rotate vLLM's strided token-major Q/K without layout copies."""

        if x.ndim != 2 or x.stride(1) != 1:
            raise ValueError("token-major ROCm RoPE requires unit inner stride")
        if positions.ndim != 1 or positions.numel() != x.size(0):
            raise ValueError("token-major ROCm RoPE positions must match token rows")
        return _deterministic_rope_apply_token_major_rocm(
            x,
            positions,
            cos,
            sin,
            head_dim,
            1.0,
        )


__all__ = ["RocmDeterministicRoPEOp"]
