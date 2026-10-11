# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA construction of packed H3 text/video/audio rows."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.packing.packed_index_copy import _validate_inputs


class _PackedIndexCopyCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        text,
        video,
        audio,
        text_indices,
        video_indices,
        audio_indices,
        sequence_length,
    ):
        ctx.save_for_backward(text_indices, video_indices, audio_indices)
        return _C.packed_index_copy_forward(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        gradients = _C.packed_index_copy_backward(
            grad_out.resolve_neg().contiguous(), *ctx.saved_tensors
        )
        return (
            *(grad if ctx.needs_input_grad[i] else None for i, grad in enumerate(gradients)),
            None,
            None,
            None,
            None,
        )


class PackedIndexCopyCudaOp:
    """CUDA row scatter and backward gather, with text-controlled output dtype."""

    op_class = "elementwise"

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not all(
            hasattr(_C, name)
            for name in ("packed_index_copy_forward", "packed_index_copy_backward")
        ):
            raise RuntimeError("packed_index_copy requires the compiled CUDA extension")

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
        if text.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("packed_index_copy requires NVIDIA CUDA tensors")
        _validate_inputs(
            text,
            video,
            audio,
            text_indices,
            video_indices,
            audio_indices,
            sequence_length,
        )
        sources = tuple(source.to(dtype=dtype).resolve_neg() for source in (text, video, audio))
        indices = tuple(
            index.resolve_neg() for index in (text_indices, video_indices, audio_indices)
        )
        return _PackedIndexCopyCudaFunction.apply(*sources, *indices, sequence_length)
