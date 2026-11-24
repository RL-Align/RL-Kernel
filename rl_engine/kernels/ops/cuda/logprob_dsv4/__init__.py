# SPDX-License-Identifier: Apache-2.0
"""CUDA DSv4 selected-token log-probability (profile ``dsv4-logprob-v1``)."""

from .dsv4_logprob import (
    DEFAULT_CHUNK_ROWS,
    NUM_TILES,
    PROFILE,
    head_logits,
    logprob_backward_cuda,
    logprob_from_logits_cuda,
    merge_cuda,
    selected_logprob,
    tile_stats_cuda,
)

__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "NUM_TILES",
    "PROFILE",
    "head_logits",
    "logprob_backward_cuda",
    "logprob_from_logits_cuda",
    "merge_cuda",
    "selected_logprob",
    "tile_stats_cuda",
]
