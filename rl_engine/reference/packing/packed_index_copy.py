# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch reference for H3 packed text/video/audio row construction."""

from __future__ import annotations

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_inputs(
    text: Tensor,
    video: Tensor,
    audio: Tensor,
    text_indices: Tensor,
    video_indices: Tensor,
    audio_indices: Tensor,
    sequence_length: int,
) -> None:
    """Check metadata without reading device values or synchronizing CUDA."""
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be a positive integer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    for name, source, indices in (
        ("text", text, text_indices),
        ("video", video, video_indices),
        ("audio", audio, audio_indices),
    ):
        if source.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have an fp16, bf16, or fp32 dtype")
        if source.ndim != 3:
            raise ValueError(f"{name} must have shape [B, S, D]")
        if source.shape[0] <= 0 or source.shape[2] <= 0:
            raise ValueError("sources must have non-empty batch and hidden dimensions")
        if source.shape[0] != text.shape[0] or source.shape[2] != text.shape[2]:
            raise ValueError("sources must share batch and hidden dimensions")
        if source.device != text.device or indices.device != text.device:
            raise ValueError("sources and indices must share a device")
        if not source.is_contiguous() or not indices.is_contiguous():
            raise ValueError("sources and indices must be contiguous")
        if indices.dtype != torch.int64:
            raise TypeError("indices must have int64 dtype")
        if indices.ndim != 1 or indices.numel() != source.shape[1]:
            raise ValueError(
                f"{name}_indices must be one-dimensional with one index per source row"
            )
    if text.shape[1] + video.shape[1] + audio.shape[1] == 0:
        raise ValueError("at least one modality must contain a source row")


def _validate_indices(indices: tuple[Tensor, Tensor, Tensor], sequence_length: int) -> None:
    """Reject invalid layouts before index_copy; synchronizes for CUDA tensors."""
    ordered = torch.cat(indices).sort().values
    valid = (ordered[0] >= 0) & (ordered[-1] < sequence_length)
    valid = valid & (ordered[1:] > ordered[:-1]).all()
    if not bool(valid):
        raise ValueError("indices must be in range, unique and disjoint")


class NativePackedIndexCopyOp:
    """Build a zero-filled packed sequence with one writer per addressed row."""

    op_class = "elementwise"

    def __call__(
        self,
        text: Tensor,
        video: Tensor,
        audio: Tensor,
        text_indices: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        sequence_length: int,
    ) -> Tensor:
        return self.forward(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
        )

    def forward(
        self,
        text: Tensor,
        video: Tensor,
        audio: Tensor,
        text_indices: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        sequence_length: int,
    ) -> Tensor:
        return self._copy(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
            dtype=text.dtype,
        )

    def forward_fp32(
        self,
        text: Tensor,
        video: Tensor,
        audio: Tensor,
        text_indices: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        sequence_length: int,
    ) -> Tensor:
        return self._copy(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
            dtype=torch.float32,
        )

    @staticmethod
    def _copy(
        text: Tensor,
        video: Tensor,
        audio: Tensor,
        text_indices: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        sequence_length: int,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        _validate_inputs(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
        )
        _validate_indices((text_indices, video_indices, audio_indices), sequence_length)
        text, video, audio = (source.to(dtype=dtype) for source in (text, video, audio))
        packed = text.new_zeros((text.shape[0], sequence_length, text.shape[2]))
        # Match the pinned Diffusers packed-input path and its cast boundary.
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video)
        return packed.index_copy(1, audio_indices, audio)
