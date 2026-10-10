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
