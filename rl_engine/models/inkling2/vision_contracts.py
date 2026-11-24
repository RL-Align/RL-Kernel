# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini vision preprocessing, merge, and token injection contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class VisionContractError(ValueError):
    """Raised when image or token metadata violates the vision contract."""


@dataclass(frozen=True)
class VisionPatchConfig:
    image_size: int = 560
    patch_size: int = 20
    channels: int = 3
    merge_size: int = 2

    def validate(self) -> None:
        if min(self.image_size, self.patch_size, self.channels, self.merge_size) <= 0:
            raise VisionContractError("vision dimensions must be positive")
        if self.image_size % self.patch_size:
            raise VisionContractError("image_size must be divisible by patch_size")
        grid = self.image_size // self.patch_size
        if grid % self.merge_size:
            raise VisionContractError("patch grid must be divisible by merge_size")

    @property
    def grid_size(self) -> int:
        self.validate()
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size**2

    @property
    def merged_count(self) -> int:
        return (self.grid_size // self.merge_size) ** 2


def preprocess_images(
    images: torch.Tensor,
    config: VisionPatchConfig,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    config.validate()
    if images.ndim != 4 or images.shape[1] != config.channels:
        raise VisionContractError("images must have shape [batch, channels, height, width]")
    if not images.is_floating_point() and images.dtype != torch.uint8:
        raise TypeError("images must be floating point or uint8")
    value = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    value = F.interpolate(
        value, (config.image_size, config.image_size), mode="bilinear", align_corners=False
    )
    channel_mean = (
        torch.zeros(config.channels, device=value.device)
        if mean is None
        else mean.to(value.device).float()
    )
    channel_std = (
        torch.ones(config.channels, device=value.device)
        if std is None
        else std.to(value.device).float()
    )
    if channel_mean.shape != (config.channels,) or channel_std.shape != (config.channels,):
        raise VisionContractError("mean and std must have one value per channel")
    if torch.any(channel_std <= 0):
        raise VisionContractError("std must be positive")
    return ((value - channel_mean.view(1, -1, 1, 1)) / channel_std.view(1, -1, 1, 1)).to(
        output_dtype
    )


def patchify_images(images: torch.Tensor, config: VisionPatchConfig) -> torch.Tensor:
    config.validate()
    if images.ndim != 4 or images.shape[1] != config.channels:
        raise VisionContractError("images must have shape [batch, channels, height, width]")
    if images.shape[-2:] != (config.image_size, config.image_size):
        raise VisionContractError("images must be preprocessed to the configured size")
    patches = images.unfold(2, config.patch_size, config.patch_size).unfold(
        3, config.patch_size, config.patch_size
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(images.shape[0], config.patch_count, -1)


def merge_2x2_patches(patch_features: torch.Tensor, config: VisionPatchConfig) -> torch.Tensor:
    config.validate()
    if patch_features.ndim != 3 or patch_features.shape[1] != config.patch_count:
        raise VisionContractError("patch features must have shape [batch, patch_count, hidden]")
    hidden = patch_features.shape[-1]
    grid = patch_features.view(-1, config.grid_size, config.grid_size, hidden)
    m = config.merge_size
    merged = grid.view(-1, config.grid_size // m, m, config.grid_size // m, m, hidden)
    merged = merged.permute(0, 1, 3, 2, 4, 5).contiguous()
    return merged.view(-1, config.merged_count, m * m * hidden)


def project_merged_patches(
    merged: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if merged.ndim != 3 or weight.ndim != 2 or merged.shape[-1] != weight.shape[1]:
        raise VisionContractError("projector shapes are incompatible")
    if weight.device != merged.device or weight.dtype not in (torch.float32, torch.bfloat16):
        raise VisionContractError("projector weight must share device and use BF16 or FP32")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != merged.device):
        raise VisionContractError("projector bias must match the output dimension")
    return F.linear(merged.float(), weight.float(), None if bias is None else bias.float()).to(
        merged.dtype
    )


@dataclass(frozen=True)
class TokenInjectionResult:
    embeddings: torch.Tensor
    image_mask: torch.Tensor


def inject_image_tokens(
    text_embeddings: torch.Tensor,
    image_tokens: torch.Tensor,
    positions: torch.Tensor,
) -> TokenInjectionResult:
    if text_embeddings.ndim != 3 or image_tokens.ndim != 3:
        raise VisionContractError("text embeddings and image tokens must be rank-3")
    if (
        text_embeddings.shape[0] != image_tokens.shape[0]
        or text_embeddings.shape[2] != image_tokens.shape[2]
    ):
        raise VisionContractError(
            "text and image embeddings must share batch and hidden dimensions"
        )
    if positions.shape != image_tokens.shape[:2] or positions.device != text_embeddings.device:
        raise VisionContractError("image positions must match image tokens and device")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("image positions must be integer")
    if torch.any(positions < 0) or torch.any(positions >= text_embeddings.shape[1]):
        raise VisionContractError("image positions are outside the text sequence")
    if positions.shape[1] and any(
        torch.unique(row).numel() != row.numel() for row in positions.unbind(0)
    ):
        raise VisionContractError("image positions must be unique within each row")
    output = text_embeddings.clone()
    mask = torch.zeros(text_embeddings.shape[:2], device=text_embeddings.device, dtype=torch.bool)
    batch = torch.arange(text_embeddings.shape[0], device=text_embeddings.device)[:, None]
    output[batch, positions] = image_tokens.to(dtype=text_embeddings.dtype)
    mask[batch, positions] = True
    return TokenInjectionResult(output, mask)
