# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared core of RFC #420 Phase B ``fl2va_pipeline_contract`` and ``ref2va_pipeline_contract``.

MiniMax-H3 denoises one packed document: text rows, conditioning rows, target audio rows and
target video rows, scattered by explicit index tensors and placed on one fp64 ``(t, h, w)``
rotary clock. The two released pipelines build that document differently:

* FL2VA (``transformer/``): ``[text | keyframe conditions | target audio | target video]``,
  zero, first, last, or first-and-last keyframes, anchored on the target's own clock;
* Ref2VA (``transformer_ref/``): ``[text | reference blocks | target audio | target video]``,
  where every reference advances the shared clock before the target starts.

This module holds what both share, ported op for op from the pinned Diffusers modular
pipeline (``huggingface/diffusers@f53d552``, ``modular_pipelines/minimax_h3``):

* the canvas and frame-count arithmetic (``resolve_canvas_size``, ``align_num_frames`` ...);
* the fp64 rotary grids, including the two summation orders of the temporal span that the
  reference keeps on purpose: numpy pairwise for the FL2VA ``last`` anchor, Python ``sum``
  for the Ref2VA clock advance. Both are written out here; Python's float ``sum`` changed
  algorithm in CPython 3.12, so the summation of the latter is an explicit, recorded choice
  (``SPAN_SUMMATIONS``) rather than whatever the interpreter does;
* :class:`PackedLayout`, its structural validation and its row-to-timestep plan;
* the conditioning-row packing (patchify, ``t * x0 + (1 - t) * noise`` at ``t = 0.999``, the
  generator draw order);
* the sealed packed-layout artifact required by H-C7: one sha256 per layout component, the
  pinned checkpoint component hashes of the variant, a preprocessing-recipe hash, and a seal
  over all of it, so repeated preprocessing can be compared byte for byte;
* the pipeline identity required by H16: FL2VA and Ref2VA are separate identities even
  though their transformer configs and tensor tables are identical. A layout carries its
  workflow, the structure of the layout must agree with it, and the transformer directory it
  is paired with must hash to the same variant.

Everything here is CPU arithmetic on integers and fp64 / fp32 tensors; it has no reduction
whose order depends on threads, devices or batch size. ``PackedLayout.to`` moves the result to
a device without changing a byte (fp64 is kept: the transformer's rope casts it to fp32).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3.trajectory import tensor_digest

SCHEMA_VERSION = "rlkernel.h3.packed_layout.v1"
PROVIDER = "huggingface/diffusers"
PROVIDER_REVISION = "f53d552036a0d1bd5570782a39cd40cfabf112bc"

# Modality tags index the AdaLN table, so they are a checkpoint contract.
VIDEO_TAG = fp.MODALITY_TAGS["video"]
TEXT_TAG = fp.MODALITY_TAGS["text"]
AUDIO_TAG = fp.MODALITY_TAGS["audio"]

# Geometry of the released checkpoint and its VAEs.
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2  # stereo, packed channel-major
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_SAMPLE_RATE = 32000
VAE_SPATIAL_COMPRESSION = 16
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
VAE_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
CANVAS_MULTIPLE = VAE_SPATIAL_COMPRESSION * PATCH_SIZE[2]  # 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 1 / 4, 4.0
MIN_DURATION, MAX_DURATION = 5.0, 15.0

# Conditioning: visual anchors are held at t = 0.999 for every step; reference soundtracks at 1.0.
CONDITION_NOISE_LEVEL = 0.999
REFERENCE_AUDIO_TIMESTEP = 1.0
CONDITION_ENCODE_SEED = 42  # seed of the VAE posterior sample (visual_vae_contract owns the encode)

# Rotary clock: one latent frame spans 5/3 * (1, 4, 4, 4, 4) units; spatial axes span [0, 32).
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32

# Workflow -> pipeline identity (variant). t2va is FL2VA with zero keyframes.
WORKFLOWS: Mapping[str, str] = {"t2va": "FL2VA", "fl2va": "FL2VA", "ref2va": "Ref2VA"}

LAYOUT_TENSORS = ("position_ids", "token_tags", "text_indices", "video_indices", "audio_indices")
# Pinned component directories shared by both pipelines (taken from the FL2VA / Ref2VA wrapper
# directories, whose copies hash identically) and the schedulers.
SHARED_COMPONENTS = ("text_encoder", "processor", "tokenizer", "video_vae", "audio_vae")


class VariantMismatch(ValueError):
    """An FL2VA artifact met a Ref2VA one (or the reverse): RFC #420 H16."""


# --- canvas and frame count ----------------------------------------------------------------


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    canvas_multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> tuple[int, int]:
    """``(height, width)`` an aspect ratio resolves to (short edge, area cap, rounding)."""

    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"aspect ratio {ratio:g} outside [{min_aspect_ratio:g}, {max_aspect_ratio:g}]"
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    m = canvas_multiple
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def align_num_frames(
    num_frames: int,
    frames_per_chunk: int = VAE_FRAMES_PER_CHUNK,
    latents_per_chunk: int = VAE_LATENTS_PER_CHUNK,
) -> int:
    """Snap up to the next ``17 * n + 5`` the video VAE can encode."""

    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(
    num_frames: int,
    frames_per_chunk: int = VAE_FRAMES_PER_CHUNK,
    latents_per_chunk: int = VAE_LATENTS_PER_CHUNK,
) -> int:
    """``17 * n + 5`` frames -> ``5 * n + 2`` latent frames."""

    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"num_frames must be {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}"
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int, fps: float = FPS, latents_per_second: int = AUDIO_LATENTS_PER_SECOND
) -> int:
    """Audio latents per channel covering ``num_frames`` video frames (rounded half to even)."""

    return int(round(num_frames / fps * latents_per_second))


def resolve_num_frames(
    num_frames: int,
    *,
    fps: float = FPS,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> int:
    """Align ``num_frames`` and check the *aligned* duration lies in ``[5, 15]`` seconds."""

    aligned = align_num_frames(num_frames)
    duration = aligned / fps
    if not min_duration <= duration <= max_duration:
        raise ValueError(
            f"num_frames {num_frames} (aligned {aligned}) is {duration:g} s; MiniMax-H3 "
            f"generates {min_duration:g}-{max_duration:g} s"
        )
    return aligned


def check_canvas(height: int, width: int, multiple: int = CANVAS_MULTIPLE) -> None:
    if height % multiple or width % multiple:
        raise ValueError(f"height and width must be multiples of {multiple}, got {height}x{width}")


# --- rotary grids -------------------------------------------------------------------------


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalised spatial axis: numpy ``linspace(endpoint=False)`` in fp64, times 32."""

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame from ``origin``: fp64 cumulative sum of the spans."""

    spans = torch.tensor(
        [
            ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def frame_position_grid(
    latent_height: int, latent_width: int, patch: tuple[int, int, int] = PATCH_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, w)`` coordinates of one latent frame (row-major) and the width axis they came from."""

    _, patch_h, patch_w = patch
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid


def _span_terms(num_latent_frames: int) -> list[float]:
    return [
        ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
        for i in range(num_latent_frames)
    ]


def numpy_pairwise_sum(values: Sequence[float]) -> float:
    """numpy's float64 ``add.reduce`` (``pairwise_sum``: 8 lanes, 128-element blocks), in Python.

    Written out so the FL2VA ``last`` anchor does not depend on how a numpy build reduces.
    """

    n = len(values)
    if n < 8:
        total = 0.0
        for v in values:
            total += v
        return total
    if n <= 128:
        lanes = list(values[:8])
        i = 8
        while i < n - n % 8:
            for j in range(8):
                lanes[j] += values[i + j]
            i += 8
        total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + (
            (lanes[4] + lanes[5]) + (lanes[6] + lanes[7])
        )
        for k in range(i, n):
            total += values[k]
        return total
    half = n // 2
    half -= half % 8
    return numpy_pairwise_sum(values[:half]) + numpy_pairwise_sum(values[half:])


def sequential_sum(values: Iterable[float]) -> float:
    """Plain left-to-right fp64 accumulation (CPython <= 3.11 ``sum`` of floats)."""

    total = 0.0
    for v in values:
        total += v
    return total


def neumaier_sum(values: Iterable[float]) -> float:
    """CPython >= 3.12 ``sum`` of floats: Neumaier-compensated, compensation added at the end."""

    total = 0.0
    compensation = 0.0
    for v in values:
        t = total + v
        if abs(total) >= abs(v):
            compensation += (total - t) + v
        else:
            compensation += (v - t) + total
        total = t
    if compensation and math.isfinite(compensation):
        total += compensation
    return total


# How the Ref2VA clock advance of a video reference is summed. The pinned Diffusers port (and
# SGLang) call Python's ``sum`` here, whose float algorithm changed in CPython 3.12, so the
# reference itself depends on the interpreter: "neumaier" is what it computes on >= 3.12 (the
# RL-Kernel environment), "sequential" what it computes on <= 3.11 and what its comment
# describes. The two differ in the last fp64 ulp for 6, 10, 11, 15, 16, ... latent frames.
SPAN_SUMMATIONS = {"neumaier": neumaier_sum, "sequential": sequential_sum}
DEFAULT_SPAN_SUMMATION = "neumaier"


def last_anchor_span_pairwise(num_latent_frames: int) -> float:
    """Rotary span of ``num_latent_frames`` frames, numpy pairwise sum (FL2VA ``last`` anchor)."""

    return numpy_pairwise_sum(_span_terms(num_latent_frames))


def video_span(num_latent_frames: int, summation: str = DEFAULT_SPAN_SUMMATION) -> float:
    """Same series, summed as the Ref2VA clock advance (see ``SPAN_SUMMATIONS``)."""

    if summation not in SPAN_SUMMATIONS:
        raise ValueError(
            f"span summation must be one of {sorted(SPAN_SUMMATIONS)}, got {summation!r}"
        )
    return SPAN_SUMMATIONS[summation](_span_terms(num_latent_frames))


def fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int = AUDIO_CHANNELS,
) -> None:
    """One channel-major audio block: no height, pinned to its own block's width extremes."""

    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                ((audio_channels - 1) * num_audio_latents,),
                float(width_grid[-1]),
                dtype=torch.float64,
            ),
        ]
    )


# --- the layout ---------------------------------------------------------------------------


def _as_long(values: torch.Tensor | Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long).reshape(-1).contiguous()


@dataclass(frozen=True)
class PackedLayout:
    """The structural description of one packed H3 document (one request, every batch row).

    ``geometry`` holds the plain numbers the layout was built from (canvas, frame counts,
    keyframe anchors or reference blocks); it is part of the artifact and of the seal.
    """

    workflow: str
    position_ids: torch.Tensor  # (S, 3) float64
    token_tags: torch.Tensor  # (S,) int64
    text_indices: torch.Tensor
    video_indices: torch.Tensor  # conditioning rows first
    audio_indices: torch.Tensor  # reference rows first
    num_condition_video_rows: int
    num_condition_audio_rows: int
    geometry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return WORKFLOWS[self.workflow]

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def num_text_tokens(self) -> int:
        return int(self.text_indices.numel())

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in LAYOUT_TENSORS}

    def denoiser_inputs(self, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """The five structural transformer kwargs, moved without any dtype change."""

        return {
            name: t.to(device) if device is not None else t for name, t in self.tensors().items()
        }

    def validate(self) -> None:
        """Structural invariants; raises ``ValueError`` naming the first broken one."""

        if self.workflow not in WORKFLOWS:
            raise ValueError(f"unknown workflow {self.workflow!r}")
        s = self.sequence_length
        if self.position_ids.dtype != torch.float64 or tuple(self.position_ids.shape) != (s, 3):
            raise ValueError("position_ids must be (S, 3) float64")
        if not bool(torch.isfinite(self.position_ids).all()):
            raise ValueError("position_ids must be finite")
        for name in LAYOUT_TENSORS[1:]:
            tensor = getattr(self, name)
            if tensor.dtype != torch.long or tensor.ndim != 1 or tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a 1-D int64 CPU tensor")
        if self.token_tags.numel() != s:
            raise ValueError("token_tags must have one tag per row")
        rows = torch.cat([self.text_indices, self.video_indices, self.audio_indices])
        if rows.numel() != s or not torch.equal(torch.sort(rows).values, torch.arange(s)):
            raise ValueError("text, video and audio indices must partition the sequence")
        if not torch.equal(self.text_indices, torch.arange(self.num_text_tokens)):
            raise ValueError("text rows must be the leading rows of the sequence")
        if not bool((self.token_tags[self.video_indices] == VIDEO_TAG).all()):
            raise ValueError("every video row must carry the video tag")
        if not bool((self.token_tags[self.audio_indices] == AUDIO_TAG).all()):
            raise ValueError("every audio row must carry the audio tag")
        text_tags = self.token_tags[self.text_indices]
        if not bool(((text_tags == TEXT_TAG) | (text_tags == VIDEO_TAG)).all()):
            raise ValueError("text rows carry the text tag, or the video tag inside a vision block")
        if not 0 <= self.num_condition_video_rows <= self.video_indices.numel():
            raise ValueError("num_condition_video_rows out of range")
        if not 0 <= self.num_condition_audio_rows <= self.audio_indices.numel():
            raise ValueError("num_condition_audio_rows out of range")

    def component_hashes(self) -> dict[str, str]:
        """sha256 over dtype, shape and bytes of every layout tensor (H-C7 components)."""

        return {name: tensor_digest(t) for name, t in self.tensors().items()}


def build_layout(
    *,
    workflow: str,
    text_token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    geometry: Mapping[str, Any],
) -> PackedLayout:
    """Assemble and validate a layout from per-row positions and the media index tensors."""

    num_text = int(text_token_tags.shape[0])
    sequence_length = int(position_ids.shape[0])
    text_indices = torch.arange(num_text)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    layout = PackedLayout(
        workflow=workflow,
        position_ids=position_ids.contiguous(),
        token_tags=token_tags,
        text_indices=text_indices,
        video_indices=_as_long(video_indices),
        audio_indices=_as_long(audio_indices),
        num_condition_video_rows=int(num_condition_video_rows),
        num_condition_audio_rows=int(num_condition_audio_rows),
        geometry=dict(geometry),
    )
    layout.validate()
    return layout


def check_text_token_tags(text_token_tags: torch.Tensor) -> torch.Tensor:
    tags = torch.as_tensor(text_token_tags)
    if tags.ndim != 1 or tags.dtype not in (torch.int32, torch.int64):
        raise ValueError("text_token_tags must be a 1-D integer tensor")
    if not bool(((tags == TEXT_TAG) | (tags == VIDEO_TAG)).all()):
        raise ValueError("text_token_tags may only hold the text tag or the video (vision) tag")
    return tags.to(torch.long).cpu()


# --- timestep plan ------------------------------------------------------------------------


def row_timesteps(
    layout: PackedLayout,
    *,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float = REFERENCE_AUDIO_TIMESTEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distinct fp32 timesteps, sorted; index of every row)``: one forward, several levels.

    Text rows inherit the video timestep; conditioning video rows are pinned at their noise
    level and reference audio rows at ``condition_audio_timestep``.
    """

    video = layout.video_indices
    audio = layout.audio_indices
    rows = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
    rows[video[: layout.num_condition_video_rows]] = condition_video_timestep
    rows[audio[layout.num_condition_audio_rows :]] = audio_timestep
    rows[audio[: layout.num_condition_audio_rows]] = condition_audio_timestep
    timesteps, indices = torch.unique(rows, sorted=True, return_inverse=True)
    return timesteps, indices


def row_timestep_plan(
    layout: PackedLayout,
    video_timesteps: Iterable[float],
    audio_timesteps: Iterable[float],
    *,
    noise_level: float = CONDITION_NOISE_LEVEL,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One ``row_timesteps`` pair per step; visual conditions sit at ``max(t, 0.999)``."""

    return [
        row_timesteps(
            layout,
            video_timestep=float(t),
            audio_timestep=float(a),
            condition_video_timestep=max(float(t), noise_level),
            condition_audio_timestep=REFERENCE_AUDIO_TIMESTEP,
        )
        for t, a in zip(video_timesteps, audio_timesteps, strict=True)
    ]


# --- conditioning and generated rows ------------------------------------------------------


def patchify_video_latents(
    latents: torch.Tensor, patch_size: tuple[int, int, int] = PATCH_SIZE
) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B * patches, C * prod(patch))`` rows, frame- then row-major."""

    patch_t, patch_h, patch_w = patch_size
    b, c, t, h, w = latents.shape
    if t % patch_t or h % patch_h or w % patch_w:
        raise ValueError(f"latents {tuple(latents.shape)} are not divisible by patch {patch_size}")
    x = latents.reshape(b, c, t // patch_t, patch_t, h // patch_h, patch_h, w // patch_w, patch_w)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * patch_t * patch_h * patch_w).contiguous()


def randn(
    shape: Sequence[int], generator: torch.Generator | None, device: torch.device | str
) -> torch.Tensor:
    """FP32 normal draw as ``diffusers.utils.torch_utils.randn_tensor`` makes it.

    A CPU generator draws on the CPU and the result moves to ``device``; otherwise the draw
    happens on ``device``.
    """

    device = torch.device(device)
    draw_device = generator.device if generator is not None else device
    if draw_device.type == "cpu" and device.type != "cpu":
        out = torch.randn(tuple(shape), generator=generator, device="cpu", dtype=torch.float32)
        return out.to(device)
    return torch.randn(tuple(shape), generator=generator, device=device, dtype=torch.float32)


def noise_condition(
    condition: torch.Tensor, noise: torch.Tensor, level: float = CONDITION_NOISE_LEVEL
) -> torch.Tensor:
    """``t * x0 + (1 - t) * noise`` with ``t`` an fp32 tensor, as the pinned pipeline mixes it.

    The pinned scheduler's ``scale_noise`` makes the level a tensor of the sample dtype (fp32),
    so ``1 - t`` is evaluated in fp32. A Python-float ``t`` would keep ``1 - t`` in fp64 and give
    different bytes; that is a probe arm (H15), not this contract. The tensor is built in the
    sample dtype directly: a pass through a half-precision default dtype would round the level.
    """

    t = torch.tensor(level, device=condition.device, dtype=condition.dtype)
    while t.ndim < condition.ndim:
        t = t.unsqueeze(-1)
    return t * condition + (1.0 - t) * noise


def pack_condition_rows(
    condition_latents: Sequence[torch.Tensor],
    *,
    generator: torch.Generator | None,
    device: torch.device | str = "cpu",
    level: float = CONDITION_NOISE_LEVEL,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> torch.Tensor:
    """Noise and pack the visual conditions: one draw per condition, in packed order."""

    packed = []
    for condition in condition_latents:
        noise = randn(condition.shape, generator, device)
        noised = noise_condition(condition.to(device), noise, level)
        packed.append(patchify_video_latents(noised, patch_size))
    if not packed:
        return torch.empty(0, 0, dtype=torch.float32, device=device)
    return torch.cat(packed)


def draw_generated_rows(
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    generator: torch.Generator | None,
    device: torch.device | str = "cpu",
    latent_channels: int = VAE_LATENT_CHANNELS,
    audio_latent_channels: int = AUDIO_LATENT_CHANNELS,
    audio_channels: int = AUDIO_CHANNELS,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target video noise (latent tensor, then patchified) then audio noise (row layout).

    Conditions are drawn *before* this; passing ``latents`` / ``audio_latents`` skips a draw.
    """

    if latents is None:
        latents = randn(
            (1, latent_channels, num_latent_frames, latent_height, latent_width), generator, device
        )
    video_rows = patchify_video_latents(latents.to(device, torch.float32), patch_size)
    if audio_latents is None:
        audio_rows = randn(
            (num_audio_latents * audio_channels, audio_latent_channels), generator, device
        )
    else:
        audio_rows = (
            audio_latents.to(device, torch.float32)
            .permute(0, 2, 1)
            .reshape(-1, audio_latent_channels)
        )
    return video_rows, audio_rows


# --- pipeline identity (H16) --------------------------------------------------------------


@dataclass(frozen=True)
class PipelineIdentity:
    variant: str
    transformer_dir: str
    wrapper_dir: str
    workflows: tuple[str, ...]
    recipe: Mapping[str, Any]

    def recipe_sha256(self) -> str:
        return _sha(self.recipe)


_COMMON_RECIPE: Mapping[str, Any] = {
    "provider": PROVIDER,
    "provider_revision": PROVIDER_REVISION,
    "patch_size": list(PATCH_SIZE),
    "audio_channels": AUDIO_CHANNELS,
    "fps": FPS,
    "audio_latents_per_second": AUDIO_LATENTS_PER_SECOND,
    "audio_sample_rate": AUDIO_SAMPLE_RATE,
    "vae_chunk": [VAE_FRAMES_PER_CHUNK, VAE_LATENTS_PER_CHUNK],
    "canvas": [CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS],
    "aspect": [MIN_ASPECT_RATIO, MAX_ASPECT_RATIO],
    "duration": [MIN_DURATION, MAX_DURATION],
    "tags": {"video": VIDEO_TAG, "text": TEXT_TAG, "audio": AUDIO_TAG},
    "rope": [ROPE_FRAME_RESCALE.hex(), list(ROPE_FRAMES_PER_LATENT), ROPE_SPATIAL_SCALE],
    "condition_noise_level": {"value": CONDITION_NOISE_LEVEL, "one_minus_t": "fp32 tensor"},
    "condition_encode_seed": CONDITION_ENCODE_SEED,
    "pixel_mean_std": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "text_encoder_layer": 50,
}

PIPELINES: dict[str, PipelineIdentity] = {
    "FL2VA": PipelineIdentity(
        variant="FL2VA",
        transformer_dir=fp.VARIANTS["FL2VA"],
        wrapper_dir="FL2VA",
        workflows=("t2va", "fl2va"),
        recipe={
            **_COMMON_RECIPE,
            "layout": "[text | keyframes | target audio | target video]",
            "anchors": ["first", "last"],
            "last_anchor_span": "numpy pairwise sum",
            "keyframe_resize": "first stretched LANCZOS; follower cover-cropped, (src-w)//2",
            "presentation": "<Picture i>: + vision block per keyframe, then prompt; no template",
        },
    ),
    "Ref2VA": PipelineIdentity(
        variant="Ref2VA",
        transformer_dir=fp.VARIANTS["Ref2VA"],
        wrapper_dir="Ref2VA",
        workflows=("ref2va",),
        recipe={
            **_COMMON_RECIPE,
            "layout": "[text | reference blocks | target audio | target video]",
            "limits": {"image": 9, "video": 3, "audio": 3, "total": 12},
            "clock": "image +1; audio +latents; video max(audio latents, video span)",
            "video_span_summation_default": DEFAULT_SPAN_SUMMATION,
            "reference_image_short_edge": 2048,
            "video_reference": "fps resample, truncate, own-aspect canvas LANCZOS; VAE snap down",
            "audio_reference": "truncate at source rate, upmix, one torchaudio resample; mode",
            "conditioner_video_sample_fps": 2.0,
            "presentation": "<Audio j>/<Picture i>/<Video k> + <x.x seconds> blocks, then prompt",
        },
    ),
}


def pipeline_for_workflow(workflow: str) -> PipelineIdentity:
    if workflow not in WORKFLOWS:
        raise ValueError(f"unknown workflow {workflow!r}; expected one of {sorted(WORKFLOWS)}")
    return PIPELINES[WORKFLOWS[workflow]]


def _strip_prefix(files: Iterable[Mapping[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out = []
    for f in files:
        if f["path"].startswith(prefix):
            out.append({**f, "path": f["path"][len(prefix) :]})
    return out


def checkpoint_component_hashes(variant: str) -> dict[str, str]:
    """Pinned component hashes a variant's pipeline runs on (transformer, shared, schedulers)."""

    identity = PIPELINES[variant]
    dirs = fp.pinned_checkpoint()["dirs"]
    out = {
        "transformer": fp.component_hash(
            _strip_prefix(dirs[identity.transformer_dir], identity.transformer_dir + "/")
        ),
        "scheduler": fp.component_hash(dirs["scheduler"]),
        "audio_scheduler": fp.component_hash(dirs["audio_scheduler"]),
    }
    for component in SHARED_COMPONENTS:
        out[component] = fp.component_hash(
            _strip_prefix(dirs[identity.wrapper_dir], f"{identity.wrapper_dir}/{component}/")
        )
    return out


def transformer_shard_hashes(variant: str) -> dict[str, str]:
    """Pinned ``{file name: sha256}`` of a variant's transformer safetensors shards."""

    directory = PIPELINES[variant].transformer_dir
    return {
        f["path"][len(directory) + 1 :]: f["sha256"]
        for f in fp.pinned_checkpoint()["dirs"][directory]
        if f["sha256"]
    }


def _hub_metadata_sha256(path: Path) -> str | None:
    """sha256 recorded by ``hf_hub_download(local_dir=...)``, if this is such a download."""

    for root in path.parents:
        meta = root / ".cache" / "huggingface" / "download"
        candidate = meta / path.relative_to(root)
        candidate = candidate.with_name(candidate.name + ".metadata")
        if candidate.is_file():
            lines = candidate.read_text().splitlines()
            if len(lines) >= 2 and lines[0].strip() == fp.REVISION:
                return lines[1].strip()
            return None
    return None


def _blob_sha256(path: Path) -> str | None:
    """sha256 an HF cache snapshot symlink points at (``blobs/<sha256>``), if it is one."""

    if path.is_symlink():
        name = Path(path.resolve()).name
        if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
            return name
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_transformer_dir(directory: str | Path, *, full_hash: bool = False) -> str:
    """Which released variant a local transformer directory holds (``"FL2VA"`` / ``"Ref2VA"``).

    The two partitions have the same config, index and tensor table, so only the shard
    contents tell them apart. Without ``full_hash``, shard sha256 values come from the Hub
    download metadata or the HF cache blob name (no read of the 66 GB); a shard with neither
    raises :class:`ValueError`. With ``full_hash``, every shard file is read and hashed, and a
    recorded hash that disagrees with the contents raises :class:`VariantMismatch`. A
    directory whose shards match neither variant, or both in part, raises
    :class:`VariantMismatch`.
    """

    directory = Path(directory)
    expected = {v: transformer_shard_hashes(v) for v in PIPELINES}
    names = sorted(next(iter(expected.values())))
    observed: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"missing transformer shard {path}")
        recorded = _hub_metadata_sha256(path) or _blob_sha256(path)
        if not full_hash:
            if recorded is None:
                raise ValueError(
                    f"{path}: no Hub metadata or cache blob name; pass full_hash=True to hash it"
                )
            observed[name] = recorded
            continue
        sha = _file_sha256(path)
        if recorded is not None and recorded != sha:
            raise VariantMismatch(
                f"{path}: contents hash to {sha}, but the download metadata records {recorded}"
            )
        observed[name] = sha
    matches = {
        variant: sum(observed[n] == table[n] for n in names) for variant, table in expected.items()
    }
    for variant, count in matches.items():
        if count == len(names):
            return variant
    raise VariantMismatch(
        f"{directory}: shards match neither variant completely (matching shards per variant: "
        f"{matches} of {len(names)})"
    )


def structural_workflow(layout: PackedLayout) -> str:
    """The workflow a layout's *structure* belongs to, independent of what it declares.

    t2va has no conditioning rows; FL2VA keyframes sit on the target's own clock, so the first
    target video frame starts at ``num_text_tokens``; every Ref2VA reference advances the clock
    by at least one unit, so its target starts later.
    """

    if layout.num_condition_video_rows >= layout.video_indices.numel():
        raise ValueError("layout has no target video rows: not a MiniMax-H3 layout")
    if layout.num_condition_video_rows == 0 and layout.num_condition_audio_rows == 0:
        return "t2va"
    first_target = layout.video_indices[layout.num_condition_video_rows]
    origin = float(layout.position_ids[first_target, 0])
    if origin == float(layout.num_text_tokens) and layout.num_condition_audio_rows == 0:
        return "fl2va"
    return "ref2va"


def check_pipeline_identity(
    layout: PackedLayout,
    *,
    transformer_variant: str | None = None,
    artifact: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed when FL2VA and Ref2VA pieces are mixed (H16).

    Checks that the layout's structure matches its declared workflow, that the transformer it
    is paired with is the same variant, and, if given, that a sealed artifact describes this
    very layout (same workflow, component hashes, condition-row counts and geometry).
    """

    declared = layout.workflow
    structural = structural_workflow(layout)
    if WORKFLOWS[structural] != WORKFLOWS[declared] or (
        structural != declared and structural != "t2va"
    ):
        raise VariantMismatch(
            f"layout declares {declared!r} but its structure is a {structural!r} layout"
        )
    if transformer_variant is not None and transformer_variant not in PIPELINES:
        raise VariantMismatch(
            f"unknown transformer variant {transformer_variant!r}; one of {sorted(PIPELINES)}"
        )
    if transformer_variant is not None and transformer_variant != layout.variant:
        raise VariantMismatch(
            f"{layout.variant} layout ({declared}) paired with the {transformer_variant} "
            f"transformer ({PIPELINES[transformer_variant].transformer_dir}/)"
        )
    if artifact is not None:
        verify_artifact(artifact)
        identity = artifact["pipeline_identity"]
        if identity["variant"] != layout.variant or identity["workflow"] != declared:
            raise VariantMismatch(
                f"artifact is {identity['variant']}/{identity['workflow']}, layout is "
                f"{layout.variant}/{declared}"
            )
        if artifact["layout_component_sha256"] != layout.component_hashes():
            raise VariantMismatch("artifact component hashes do not describe this layout")
        # The tensors do not carry these: the counts set the row-timestep plan, the geometry
        # the rows the initial-row builders expect.
        for key, value in (
            ("num_condition_video_rows", layout.num_condition_video_rows),
            ("num_condition_audio_rows", layout.num_condition_audio_rows),
            ("geometry", _jsonable(layout.geometry)),
        ):
            if artifact[key] != value:
                raise VariantMismatch(f"artifact {key} does not describe this layout")


# --- sealed artifact (H-C7) ---------------------------------------------------------------


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("geometry values must be finite")
        return value
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    raise TypeError(f"geometry value of type {type(value).__name__} is not serialisable")


def layout_artifact(
    layout: PackedLayout, *, inputs: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Sealed, machine-readable record of a packed layout and its pipeline identity.

    ``inputs`` are optional digests of what produced the layout (conditioning rows, text tags,
    normalised reference media ...); they are sealed with it so a repeated run can show that
    every preprocessed component, not only the index tensors, is byte-identical.
    """

    layout.validate()
    identity = PIPELINES[layout.variant]
    body = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"repo": fp.REPO, "revision": fp.REVISION},
        "pipeline_identity": {
            "variant": identity.variant,
            "workflow": layout.workflow,
            "structural_workflow": structural_workflow(layout),
            "transformer_dir": identity.transformer_dir,
            "preprocessing_recipe_sha256": identity.recipe_sha256(),
            "checkpoint_component_sha256": checkpoint_component_hashes(identity.variant),
        },
        "geometry": _jsonable(layout.geometry),
        "sequence_length": layout.sequence_length,
        "num_text_tokens": layout.num_text_tokens,
        "num_condition_video_rows": layout.num_condition_video_rows,
        "num_condition_audio_rows": layout.num_condition_audio_rows,
        "layout_component_sha256": layout.component_hashes(),
        "input_sha256": dict(sorted((inputs or {}).items())),
    }
    return {**body, "artifact_sha256": _sha(body)}


def verify_artifact(artifact: Mapping[str, Any]) -> None:
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unknown packed-layout schema {artifact.get('schema_version')!r}")
    if _sha(body) != artifact.get("artifact_sha256"):
        raise ValueError("packed-layout artifact seal does not match its contents")


def first_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    """Dotted path of the first differing entry of two artifacts (sorted keys), else ``None``."""

    def walk(x: Any, y: Any, path: str) -> str | None:
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            for key in sorted(set(x) | set(y)):
                if key == "artifact_sha256":
                    continue
                if key not in x or key not in y:
                    return f"{path}{key}"
                found = walk(x[key], y[key], f"{path}{key}.")
                if found:
                    return found
            return None
        return None if x == y else path.rstrip(".")

    return walk(a, b, "")


def save_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    verify_artifact(artifact)
    Path(path).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text())
    verify_artifact(artifact)
    return artifact
