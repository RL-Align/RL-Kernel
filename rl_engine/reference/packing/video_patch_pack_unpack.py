# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reuse vLLM-Omni's H3 video permutation with an independent PyTorch gold.

The public interface keeps the batch dimension. Diffusers flattens the first
two output dimensions after applying the same permutation.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch

_CHANNELS = 24
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2
_WIDTH = _CHANNELS * _PATCH_T * _PATCH_H * _PATCH_W
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

UPSTREAM_REVISION = "b1fc5fcfa3e1a4f76c6686eaf09242f0fc6464bc"
UPSTREAM_SOURCE = "vllm_omni/diffusion/models/minimax_h3/packed_tokens.py"
UPSTREAM_SOURCE_SHA256 = "20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f"


# Vendored video functions and helpers from vLLM-Omni, Apache-2.0.
# Source: vllm_omni/diffusion/models/minimax_h3/packed_tokens.py
# Revision: b1fc5fcfa3e1a4f76c6686eaf09242f0fc6464bc
# The upstream function bodies are unchanged; the public adapter follows below.
def _int_tuple(value: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {list(value)!r}")
    out = tuple(int(item) for item in value)
    if any(item <= 0 for item in out):
        raise ValueError(f"{name} values must be positive, got {list(value)!r}")
    return out


def _rank(tensor: torch.Tensor, name: str, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape={list(tensor.shape)}")


def minimax_h3_patchify_video_latent(
    latent: torch.Tensor,
    *,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Pack video latent [B,C,T,H,W] into DiT token rows."""

    _rank(latent, "video latent", 5)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    batch, channel, full_t, full_h, full_w = (int(dim) for dim in latent.shape)
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            "video latent spatial/time dims must be divisible by patch_size: "
            f"shape={list(latent.shape)}, patch_size={[pt, ph, pw]}"
        )
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(batch, channel, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, channel * pt * ph * pw).contiguous()


def minimax_h3_unpatchify_video_tokens(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Unpack DiT video token rows into latent [B,C,T,H,W]."""

    _rank(rows, "video token rows", 2)
    t, h, w, channel = _int_tuple(latent_shape, "latent_shape", 4)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    expected_dim = pt * ph * pw * channel
    if int(rows.shape[-1]) != expected_dim:
        raise ValueError(
            f"video token dim {int(rows.shape[-1])} != patch volume * channel "
            f"{expected_dim} for latent_shape={list(latent_shape)}, "
            f"patch_size={[pt, ph, pw]}"
        )
    rows_per_sample = t * h * w
    if int(rows.shape[0]) % rows_per_sample:
        raise ValueError(
            f"video rows {int(rows.shape[0])} must be divisible by t*h*w "
            f"{rows_per_sample} for latent_shape={list(latent_shape)}"
        )
    packed = rows.reshape(-1, t, h, w, channel, pt, ph, pw)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, t * pt, h * ph, w * pw).contiguous()


def _validate_pack(x: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError("video latents must be a torch.Tensor")
    if x.ndim != 5:
        raise ValueError(f"video latents must have shape (B,24,T,H,W), got {tuple(x.shape)}")
    b, c, t, h, w = x.shape
    if b < 1 or t < 1 or h < 1 or w < 1:
        raise ValueError("B, T, H and W must be positive")
    if c != _CHANNELS:
        raise ValueError(f"H3 video latents require {_CHANNELS} channels, got {c}")
    if h % _PATCH_H or w % _PATCH_W:
        raise ValueError("H and W must be divisible by the (2,2) spatial patch")
    if x.dtype not in _DTYPES:
        raise TypeError(f"unsupported video latent dtype: {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("video latents must be contiguous")
    return b, t, h, w


def _validate_unpack(
    tokens: torch.Tensor, shape: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int, int]:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("video tokens must be a torch.Tensor")
    if not isinstance(shape, (tuple, torch.Size)) or len(shape) != 5:
        raise ValueError("shape must be (B,24,T,H,W)")
    try:
        b, c, t, h, w = (operator.index(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError("video latent shape dimensions must be integers") from exc
    if min(b, t, h, w) < 1 or c != _CHANNELS or h % 2 or w % 2:
        raise ValueError("invalid H3 video latent shape")
    expected = (b, t * (h // 2) * (w // 2), _WIDTH)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"video tokens must have shape {expected}, got {tuple(tokens.shape)}")
    if tokens.dtype not in _DTYPES:
        raise TypeError(f"unsupported video token dtype: {tokens.dtype}")
    if not tokens.is_contiguous():
        raise ValueError("video tokens must be contiguous")
    return b, c, t, h, w


def pack_video_patch_reference(x: torch.Tensor) -> torch.Tensor:
    """Return (B,S,96), with S ordered frame-major, then row-major."""
    b, t, h, w = _validate_pack(x)
    return (
        x.reshape(b, _CHANNELS, t, h // 2, 2, w // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(b, t * (h // 2) * (w // 2), _WIDTH)
        .contiguous()
    )


def unpack_video_patch_reference(
    tokens: torch.Tensor, shape: tuple[int, int, int, int, int]
) -> torch.Tensor:
    """Invert the H3 patch permutation without arithmetic or dtype conversion."""
    b, _, t, h, w = _validate_unpack(tokens, shape)
    return (
        tokens.reshape(b, t, h // 2, w // 2, _CHANNELS, 1, 2, 2)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(shape)
        .contiguous()
    )


class NativeVideoPatchPackUnpackOp:
    backend_id = "pytorch-video-patch-pack-unpack-v1"

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = _validate_pack(x)
        rows = minimax_h3_patchify_video_latent(x, patch_size=(_PATCH_T, _PATCH_H, _PATCH_W))
        return rows.view(b, t * (h // _PATCH_H) * (w // _PATCH_W), _WIDTH)

    def unpack(self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        _, c, t, h, w = _validate_unpack(tokens, shape)
        return minimax_h3_unpatchify_video_tokens(
            tokens.view(-1, _WIDTH),
            latent_shape=(t, h // _PATCH_H, w // _PATCH_W, c),
            patch_size=(_PATCH_T, _PATCH_H, _PATCH_W),
        )
