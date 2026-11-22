# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen-Image coordinates for operator and logical-layout validation."""

import math

import torch

from rl_engine.ops.rope.multi_axis_rope import qwen_image_positions


def qwen_image_grid_prefix(sequence_length: int) -> torch.Tensor:
    """Use a centered square-grid prefix for a complete logical sequence.

    CLI sequence lengths need not be square. Grid size depends on the full
    sample length, never the physical batch, padding or current chunk length.
    """
    if sequence_length < 0:
        raise ValueError("sequence_length must be nonnegative")
    side = math.isqrt(max(0, sequence_length - 1)) + 1
    image, _ = qwen_image_positions(side, side, 0)
    return image[:sequence_length]
