# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic MiniMax-H3 (RFC #420) test inputs shared by tests and scripts."""

from __future__ import annotations

import torch


def h3_timesteps(num: int, *, seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """Distinct-looking timesteps in [0, 1] with both endpoints present."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.rand(num, generator=generator)
    t[0] = 0.0
    if num > 1:
        t[-1] = 1.0
    return t.to(device)


def h3_packed_layout(
    seq_len: int, num_timesteps: int, *, seed: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """(timestep_indices, token_tags) for a packed sequence with every modality.

    Rows are video, text and audio blocks in random order, each block mapped
    to a random timestep, like H3's conditioning/target packing.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_tags = torch.randint(0, 3, (seq_len,), generator=generator)
    timestep_indices = torch.randint(0, num_timesteps, (seq_len,), generator=generator)
    token_tags[: min(3, seq_len)] = torch.arange(min(3, seq_len))
    return timestep_indices.to(device), token_tags.to(device)


def h3_block_layout(
    seq_len: int, num_timesteps: int, *, seed: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """(timestep_indices, token_tags) for block-structured packing, as H3 packs requests.

    Each timestep owns one contiguous run of text, video and audio blocks (about
    5% / 80% / 15% of its tokens), so table rows form contiguous position runs.
    ``h3_packed_layout`` is the interleaved stress case.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    cuts = torch.sort(torch.randperm(seq_len - 1, generator=generator)[: num_timesteps - 1] + 1)
    edges = [0, *cuts.values.tolist(), seq_len]
    timestep_indices = torch.empty(seq_len, dtype=torch.long)
    token_tags = torch.empty(seq_len, dtype=torch.long)
    for t, (lo, hi) in enumerate(zip(edges, edges[1:])):
        text, audio = (hi - lo) // 20, (hi - lo) * 3 // 20
        timestep_indices[lo:hi] = t
        token_tags[lo:hi] = 0  # video
        token_tags[lo : lo + text] = 1
        token_tags[hi - audio : hi] = 2
    return timestep_indices.to(device), token_tags.to(device)
