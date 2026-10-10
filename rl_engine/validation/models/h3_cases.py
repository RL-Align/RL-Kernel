# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic MiniMax-H3 (RFC #420) test inputs shared by tests and scripts."""

from __future__ import annotations

import torch


def h3_timesteps(num: int, *, seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """Return seeded FP32 timesteps of shape ``(num,)`` on ``device``.

    Requires positive ``num``; the first value is 0 and the last is 1 when
    ``num > 1``, with all remaining values sampled uniformly in [0, 1].
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.rand(num, generator=generator)
    t[0] = 0.0
    if num > 1:
        t[-1] = 1.0
    return t.to(device)


def h3_packed_layout(
    seq_len: int, num_timesteps: int, *, seed: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return seeded int64 timestep indices and modality tags, each shape ``(seq_len,)``.

    Each row independently samples a timestep in ``[0, num_timesteps)`` and
    a video/text/audio tag in ``[0, 3)``. The first three rows cover all tags
    when available; both tensors are returned on ``device``.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_tags = torch.randint(0, 3, (seq_len,), generator=generator)
    timestep_indices = torch.randint(0, num_timesteps, (seq_len,), generator=generator)
    token_tags[: min(3, seq_len)] = torch.arange(min(3, seq_len))
    return timestep_indices.to(device), token_tags.to(device)
