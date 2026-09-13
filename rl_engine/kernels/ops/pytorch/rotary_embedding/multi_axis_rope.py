# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
from torch import Tensor


QWEN_IMAGE_AXES_DIM: tuple[int, ...] = (16, 56, 56)
"""Qwen-Image MMDiT rotary axes: (temporal, height, width), head_dim = 128."""


def qwen_image_positions(
    text_length: int,
    image_height: int,
    image_width: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build the [S, 3] axis coordinates of one Qwen-Image sequence.

    Text tokens sit on the grid diagonal (issue #386: coordinate (i, i, i)),
    followed by the image tokens on the (h, w) grid at temporal index 0, in
    row-major order. The result feeds ``NativeMultiAxisRopeOp`` /
    ``MultiAxisRopeAscendOp`` directly.
    """
    if text_length < 0 or image_height <= 0 or image_width <= 0:
        raise ValueError(
            "text_length must be non-negative and image dimensions positive, got "
            f"text_length={text_length}, image_height={image_height}, "
            f"image_width={image_width}"
        )
    text_ids = (
        torch.arange(text_length, device=device, dtype=torch.float32)
        .unsqueeze(-1)
        .repeat(1, 3)
    )
    h = torch.arange(image_height, device=device, dtype=torch.float32)
    w = torch.arange(image_width, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).reshape(-1, 2)
    image_ids = torch.cat(
        [torch.zeros(image_height * image_width, 1, device=device), grid], dim=-1
    )
    return torch.cat([text_ids, image_ids], dim=0)


def build_multi_axis_cos_sin(
    positions: Tensor,
    axes_dim: tuple[int, ...],
    *,
    theta: float,
) -> tuple[Tensor, Tensor]:
    """Build fp32 rotate-half tables [S, D/2] from per-axis coordinates.

    ``positions`` is [S, A] with A == len(axes_dim); each axis contributes
    inv_freq = theta^(-arange(0, d_i, 2) / d_i) and an outer product with its
    coordinate column. The per-axis tables are concatenated to
    [S, sum(d_i)/2]. The rotate-half duplication (cat(freqs, freqs)) is left
    to the consumer so the tables can be handed to the Ascend C kernel, which
    indexes cos[i] for both halves implicitly.
    """
    if positions.dim() != 2:
        raise ValueError(
            f"positions must be 2-D [S, num_axes], got shape {tuple(positions.shape)}"
        )
    if positions.shape[-1] != len(axes_dim):
        raise ValueError(
            f"positions last dimension {positions.shape[-1]} must match the "
            f"number of axes ({len(axes_dim)})"
        )
    if any(d <= 0 or d % 2 != 0 for d in axes_dim):
        raise ValueError(f"every axis dim must be a positive even number, got {axes_dim}")

    pos = positions.to(device=positions.device, dtype=torch.float32)
    freqs: list[Tensor] = []
    for axis, dim in enumerate(axes_dim):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim)
        )
        freqs.append(torch.outer(pos[:, axis], inv_freq))
    table = torch.cat(freqs, dim=-1)
    return table.cos().contiguous(), table.sin().contiguous()


class NativeMultiAxisRopeOp:
    """Pure PyTorch reference RoPE for Qwen-Image (issue #386 `multi_axis_rope`).

    Applies the rotate-half rotation over the concatenated per-axis frequency
    tables (HF/diffusers convention, dimension pairing (i, i + D/2), NOT
    adjacent). cos/sin are computed internally in fp32 from the [S, num_axes]
    coordinates and the per-axis dims — no external cache is accepted.

    Qwen-Image defaults: axes_dim = (16, 56, 56), theta = 1e4, text tokens on
    the grid diagonal (see ``qwen_image_positions``).
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        axes_dim: tuple[int, ...] = QWEN_IMAGE_AXES_DIM,
        theta: float = 10_000.0,
    ) -> Tensor:
        return self.forward(x, positions, axes_dim=axes_dim, theta=theta)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        axes_dim: tuple[int, ...] = QWEN_IMAGE_AXES_DIM,
        theta: float = 10_000.0,
    ) -> Tensor:
        """Apply multi-axis RoPE in input dtype; cos/sin always computed in fp32."""
        cos, sin = self._compute_cos_sin(x, positions, axes_dim=axes_dim, theta=theta)
        xf = x.float()
        half = xf.shape[-1] // 2
        x1, x2 = xf[..., :half], xf[..., half:]
        out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return out.to(dtype=x.dtype)

    def forward_fp32(
        self,
        x: Tensor,
        positions: Tensor,
        *,
        axes_dim: tuple[int, ...] = QWEN_IMAGE_AXES_DIM,
        theta: float = 10_000.0,
    ) -> Tensor:
        """fp32 gold standard: internal computation and output are fp32."""
        cos, sin = self._compute_cos_sin(x, positions, axes_dim=axes_dim, theta=theta)
        xf = x.float()
        half = xf.shape[-1] // 2
        x1, x2 = xf[..., :half], xf[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    @staticmethod
    def _compute_cos_sin(
        x: Tensor,
        positions: Tensor,
        *,
        axes_dim: tuple[int, ...],
        theta: float,
    ) -> tuple[Tensor, Tensor]:
        """Compute fp32 half-dim cos/sin [S, D/2] broadcastable to x halves.

        The rotate-half duplication (cat(freqs, freqs)) is implicit: each
        half of the rotation reuses the same table entry, exactly like the
        Ascend C kernel's indexing.
        """
        dim = x.shape[-1]
        if dim != sum(axes_dim):
            raise ValueError(
                f"x head_dim {dim} must equal sum(axes_dim)={sum(axes_dim)} "
                f"for axes {axes_dim}"
            )
        if positions.dim() == 3 and positions.shape[0] == 1:
            positions = positions[0]
        return build_multi_axis_cos_sin(positions, axes_dim, theta=theta)
