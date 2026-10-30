# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Out-of-place partial GPT-J rotation using caller-provided tables."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from rl_engine.kernels.dsv4.attention.contract import (
    HEAD_DIM,
    MAIN_ROTARY_END,
    MAIN_ROTARY_START,
    ROPE_DIM,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status

_T02_APPLY: Callable[..., Tensor] | None = None
_T02_LOADED = False


def _load_t02() -> Callable[..., Tensor] | None:
    global _T02_APPLY, _T02_LOADED
    if not _T02_LOADED:
        fn = None
        for path in (
            "rl_engine.kernels.ops.pytorch.rotary_embedding.rope_gptj",
            "rl_engine.kernels.dsv4.attention.rope_gptj",
        ):
            try:
                module = __import__(path, fromlist=["rope_gptj_interleaved_partial"])
                fn = getattr(module, "rope_gptj_interleaved_partial", None)
                if fn is not None:
                    break
            except ImportError:
                continue
        _T02_APPLY = fn
        _T02_LOADED = True
    return _T02_APPLY


def apply_gptj_interleaved_partial(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    inverse: bool,
    rotary_start: int = MAIN_ROTARY_START,
    rotary_end: int = MAIN_ROTARY_END,
    variant: str = "gptj_interleaved_partial",
    inplace: bool = False,
) -> Tensor:
    """Rotate adjacent pairs in the rotary slice and copy the remaining dimensions."""

    if variant != "gptj_interleaved_partial":
        raise DSv4FailClosedError(
            DSv4Status.INVALID_ROPE_VARIANT,
            f"T06 o-proj requires gptj_interleaved_partial, got {variant!r}",
        )
    if inplace:
        raise DSv4FailClosedError(
            DSv4Status.INVALID_ROPE_VARIANT,
            "inverse RoPE must be out-of-place; refusing inplace=True",
        )
    if x.shape[-1] != HEAD_DIM:
        raise DSv4FailClosedError(
            DSv4Status.SCHEMA_MISMATCH,
            f"Main O head dim must be {HEAD_DIM}, got {x.shape[-1]}",
        )
    if rotary_end - rotary_start != ROPE_DIM:
        raise DSv4FailClosedError(
            DSv4Status.INVALID_ROPE_VARIANT,
            f"rotary range must have rope_dim={ROPE_DIM}, got {rotary_start}:{rotary_end}",
        )

    n_pairs = ROPE_DIM // 2
    if cos.shape[-1] != n_pairs or sin.shape[-1] != n_pairs:
        raise DSv4FailClosedError(
            DSv4Status.SCHEMA_MISMATCH,
            f"cos/sin last dim must be {n_pairs}, got cos={tuple(cos.shape)}",
        )

    t02 = _load_t02()
    if t02 is not None:
        y = t02(
            x,
            cos,
            sin,
            inverse=inverse,
            rotary_start=rotary_start,
            rotary_end=rotary_end,
            variant=variant,
            inplace=False,
        )
        if y.data_ptr() == x.data_ptr():
            raise DSv4FailClosedError(
                DSv4Status.IDENTITY_DRIFT,
                "T02 RoPE mutated the input storage; inverse RoPE must be out-of-place",
            )
        return y

    x_f = x.float()
    cos_f = cos.float()
    sin_f = sin.float()
    while cos_f.dim() < x_f.dim():
        cos_f = cos_f.unsqueeze(-2)
        sin_f = sin_f.unsqueeze(-2)
    left = x_f[..., :rotary_start]
    mid = x_f[..., rotary_start:rotary_end]
    right = x_f[..., rotary_end:]
    even = mid[..., 0::2]
    odd = mid[..., 1::2]
    if inverse:
        new_even = even * cos_f + odd * sin_f
        new_odd = -even * sin_f + odd * cos_f
    else:
        new_even = even * cos_f - odd * sin_f
        new_odd = odd * cos_f + even * sin_f
    rotated = torch.stack((new_even, new_odd), dim=-1).reshape_as(mid)
    out = torch.cat((left, rotated, right), dim=-1)
    if out.data_ptr() == x.data_ptr():
        raise DSv4FailClosedError(DSv4Status.IDENTITY_DRIFT, "RoPE apply reused input storage")
    return out.to(dtype=x.dtype) if out.dtype != x.dtype else out


def fixture_cos_sin(
    positions: Tensor, *, n_pairs: int = ROPE_DIM // 2, theta: float = 1e6
) -> tuple[Tensor, Tensor]:
    """Generate synthetic FP32 tables for tests."""

    pos = positions.to(dtype=torch.float32).reshape(-1, 1)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, n_pairs, dtype=torch.float32, device=positions.device) / n_pairs)
    )
    freqs = pos * inv_freq
    return freqs.cos().contiguous(), freqs.sin().contiguous()
