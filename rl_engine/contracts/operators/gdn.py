# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared Gated DeltaNet geometry and numerical execution contract."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNContract:
    """V-major state and q|k|v packing; see docs/contracts/gdn.md for arithmetic."""

    num_k_heads: int
    num_v_heads: int
    key_dim: int
    value_dim: int
    scale: float
    use_qk_l2norm: bool = True
    profile: str = "vllm_decode_v1"

    def __post_init__(self) -> None:
        """Reject invalid geometry and unsupported profiles before tensor validation."""
        for name in ("num_k_heads", "num_v_heads", "key_dim", "value_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be divisible by num_k_heads")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
        ):
            raise ValueError("scale must be a finite real number")
        if not isinstance(self.use_qk_l2norm, bool):
            raise ValueError("use_qk_l2norm must be bool")
        if self.profile != "vllm_decode_v1":
            raise ValueError("profile must be 'vllm_decode_v1'; other providers are unqualified")

    def validate(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[int, ...]:
        """Validate shapes/dtypes/packing and return host sequence boundaries.

        The contract is device-neutral. The CPU reference additionally rejects
        non-CPU execution. Indices name distinct slots, including empty sequences.
        No slot is implicitly a null block, and no input is modified.
        """
        width = 2 * self.num_k_heads * self.key_dim + self.num_v_heads * self.value_dim
        if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != width:
            raise ValueError(f"mixed_qkv must have shape [T, {width}] (q|k|v)")
        if mixed_qkv.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("mixed_qkv must be FP32, BF16 or FP16")
        tokens = mixed_qkv.shape[0]
        for name, tensor, shape, dtype in (
            ("a", a, (tokens, self.num_v_heads), mixed_qkv.dtype),
            ("b", b, (tokens, self.num_v_heads), mixed_qkv.dtype),
            ("A_log", A_log, (self.num_v_heads,), torch.float32),
            ("dt_bias", dt_bias, (self.num_v_heads,), torch.float32),
        ):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != mixed_qkv.device:
                raise ValueError(f"{name} has an incompatible shape, dtype or device")
        if (
            state.ndim != 4
            or state.shape[1:] != (self.num_v_heads, self.value_dim, self.key_dim)
            or state.dtype not in (torch.float32, torch.bfloat16)
            or state.device != mixed_qkv.device
        ):
            raise ValueError("state must be FP32/BF16 [blocks, HV, V, K] on the input device")
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device.type != "cpu"
        ):
            raise ValueError("cu_seqlens must be a nonempty CPU int32/int64 vector")
        boundaries = tuple(cu_seqlens.tolist())
        if (
            boundaries[0] != 0
            or boundaries[-1] != tokens
            or any(end < start for start, end in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise ValueError("cu_seqlens must be nondecreasing and span [0, T]")
        if (
            indices.shape != (len(boundaries) - 1,)
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != mixed_qkv.device
            or bool((indices < 0).any())
            or bool((indices >= state.shape[0]).any())
            or indices.unique().numel() != indices.numel()
        ):
            raise ValueError(
                "indices must name unique nonnegative in-range state slots, one per sequence"
            )
        return boundaries
