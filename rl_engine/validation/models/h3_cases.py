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


# diffusers@f53d552 modular_pipelines/minimax_h3/before_denoise.py constants.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalised spatial axis, as numpy's ``linspace(endpoint=False)`` builds it."""

    import numpy as np

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def h3_fl2va_layout(
    *,
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    audio_channels: int = 2,
    patch: tuple[int, int] = (2, 2),
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Packed text/audio/video rows of a text-only FL2VA request, laid out as diffusers packs them.

    Replays ``position_ids``/``token_tags`` of the pinned pipeline's FL2VA
    layout step (no keyframe anchors): text rows on the time axis at their row
    index, channel-major audio rows pinned to the width extremes, then the video
    frames on the non-uniform ``5/3 * (1, 4, 4, 4, 4)`` clock. ``position_ids``
    is ``(S, 3)`` float64 as the pipeline makes it; the rotary module casts it.
    ``timestep_indices`` puts text on table row 0 and the generated media on
    row 1, a two-timestep stand-in for the conditioning rows.
    """

    import numpy as np

    patch_h, patch_w = patch
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    seq_len = num_text_tokens + num_audio_rows + num_video_rows
    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    frame_time = float(num_text_tokens) + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )
    video = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video[:, :, 0] = frame_time[:, None]
    video[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video.reshape(-1, 3)

    token_tags = torch.full((seq_len,), VIDEO_TAG, dtype=torch.long)
    token_tags[:num_text_tokens] = TEXT_TAG
    token_tags[audio_start:video_start] = AUDIO_TAG
    timestep_indices = torch.ones(seq_len, dtype=torch.long)
    timestep_indices[:num_text_tokens] = 0
    return {
        "position_ids": position_ids.to(device),
        "token_tags": token_tags.to(device),
        "timestep_indices": timestep_indices.to(device),
    }


# Named FL2VA layouts for the one-block replay: (text, latent frames, latent H, latent W, audio).
H3_BLOCK_LAYOUTS = {
    # 32 + 2*20 + 2*(8*12) = 264 rows: the CI-sized case.
    "tiny": dict(
        num_text_tokens=32,
        num_latent_frames=2,
        latent_height=16,
        latent_width=24,
        num_audio_latents=20,
    ),
    # 77 + 2*40 + 3*(16*16) = 925 rows: a non-power-of-two, three-frame case.
    "small": dict(
        num_text_tokens=77,
        num_latent_frames=3,
        latent_height=32,
        latent_width=32,
        num_audio_latents=40,
    ),
    # 120 + 2*80 + 5*(18*32) = 3160 rows: a 16:9 five-frame case.
    "medium": dict(
        num_text_tokens=120,
        num_latent_frames=5,
        latent_height=36,
        latent_width=64,
        num_audio_latents=80,
    ),
}
