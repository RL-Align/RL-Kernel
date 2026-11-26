# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``fl2va_pipeline_contract``: FL2VA preprocessing and packed-layout construction.

FL2VA runs on the ``transformer/`` partition and conditions on zero, first, last, or
first-and-last keyframes (zero keyframes is the ``t2va`` workflow on the same partition).
This module is the strict, pinned version of what the Diffusers modular pipeline
(``diffusers@f53d552``: ``MiniMaxH3ResizeStep``, ``MiniMaxH3FL2VATextEncoderStep``'s
presentation, ``MiniMaxH3PrepareLayoutStep``, ``MiniMaxH3PrepareConditionLatentsStep``,
``MiniMaxH3PrepareLatentsStep``, ``MiniMaxH3FL2VAPrepareLatentsStep``) does before the
transformer runs:

1. :func:`prepare_keyframes` resolves the canvas (from the first keyframe's aspect ratio
   unless ``height`` / ``width`` are given) and puts the keyframes on it: the first one is
   stretched with PIL LANCZOS, a second one is cover-cropped with the reference's own
   rounding and centring (not ``VaeImageProcessor``'s ``crop`` mode, which is off by a pixel
   on about half of all aspect ratios);
2. :func:`fl2va_presentation` tokenizes ``"<Picture i>: "`` + one vision block per keyframe +
   the prompt, and tags the vision block's rows as video;
3. :func:`build_fl2va_layout` builds ``[text | keyframes | target audio | target video]``
   and its fp64 rotary grid; a ``first`` keyframe sits at the target's first frame time, a
   ``last`` one at the last frame time computed with numpy's pairwise sum;
4. :func:`fl2va_initial_rows` noises the keyframe latents to ``t = 0.999`` and draws the
   target noise from the request generator, in the reference's draw order.

The VAE encode of the keyframe pixels is the ``visual_vae_contract`` row and the Qwen3-VL
hidden state is ``qwen3vl_layer50_contract``; this row takes their outputs as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

from rl_engine.models.minimax_h3 import packed_layout as pl

if TYPE_CHECKING:  # Pillow is not a base dependency; only the keyframe path imports it
    from PIL import Image

WORKFLOW = "fl2va"
ANCHORS = ("first", "last")
# Zero, first, last, first-and-last: the four keyframe signatures the released model supports.
ANCHOR_SETS: tuple[tuple[str, ...], ...] = ((), ("first",), ("last",), ("first", "last"))


@dataclass(frozen=True)
class Keyframes:
    """Keyframes on the target canvas, in packed order, with their anchors."""

    height: int
    width: int
    anchors: tuple[str, ...]
    images: tuple[Image.Image, ...]


def keyframe_anchors(image: Any | None, last_image: Any | None) -> tuple[str, ...]:
    return tuple(a for a, k in (("first", image), ("last", last_image)) if k is not None)


def cover_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    """The follower keyframe: LANCZOS to cover the canvas (round), crop at ``(src - w) // 2``."""

    from PIL import Image

    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (
        max(width, round(image.size[0] * scale)),
        max(height, round(image.size[1] * scale)),
    )
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def prepare_keyframes(
    image: Image.Image | None = None,
    last_image: Image.Image | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    short_edge: int = pl.CANVAS_SHORT_EDGE,
    max_pixels: int = pl.CANVAS_MAX_PIXELS,
) -> Keyframes:
    """Resolve the canvas and put the keyframes on it (``MiniMaxH3ResizeStep``).

    Without keyframes and without ``height`` / ``width`` the canvas is MiniMax-H3's own 16:9.
    Keyframes need Pillow, which RL-Kernel does not install by default.
    """

    if (height is None) != (width is None):
        raise ValueError("height and width have to be passed together, or neither")
    images = [k for k in (image, last_image) if k is not None]
    if images:
        from PIL import Image
    anchors = keyframe_anchors(image, last_image)
    if height is None or width is None:
        aspect = images[0].size if images else (16, 9)
        height, width = pl.resolve_canvas_size(
            *aspect, short_edge=short_edge, max_pixels=max_pixels
        )
    pl.check_canvas(height, width)
    prepared = []
    for index, keyframe in enumerate(images):
        if keyframe.size == (width, height):
            prepared.append(keyframe)
        elif index == 0:
            # The geometry anchor is stretched: VaeImageProcessor.resize(PIL, "default") is
            # exactly PIL's resize((width, height), LANCZOS).
            prepared.append(keyframe.resize((width, height), Image.Resampling.LANCZOS))
        else:
            prepared.append(cover_crop(keyframe, height, width))
    return Keyframes(height=height, width=width, anchors=anchors, images=tuple(prepared))


def keyframe_pixels(image: Image.Image) -> torch.Tensor:
    """``(1, 3, 1, H, W)`` uint8: the VAE input of one keyframe (encoded by visual_vae_contract).

    The pinned pipeline packs the PIL array as it is, without a mode conversion, so only a
    3-channel image is accepted; anything else is refused here instead of being converted.
    """

    pixels = np.array(image)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(
            f"a keyframe must be a 3-channel (RGB) image, got mode {image.mode!r} "
            f"(array shape {pixels.shape}); convert it before preprocessing"
        )
    return torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].contiguous()


def fl2va_presentation(
    tokenize: Callable[[str], Sequence[int]],
    token_id: Callable[[str], int],
    prompt: str,
    image_token_counts: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Token ids and per-row modality tags of the conditioner input.

    ``"<Picture i>: "`` (text) + ``<|vision_start|> <|image_pad|> * n <|vision_end|>`` (tagged
    video) per keyframe, then the prompt verbatim; no chat template, no special tokens.
    ``tokenize`` is ``lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]`` and
    ``image_token_counts`` is ``image_grid_thw.prod() // merge_size ** 2`` per keyframe.
    """

    if not isinstance(prompt, str):
        raise ValueError("MiniMax-H3 packs one request into one sequence; prompt must be a str")
    token_ids: list[int] = []
    token_tags: list[int] = []
    for index, count in enumerate(image_token_counts):
        label = list(tokenize(f"<Picture {index + 1}>: "))
        vision = (
            [token_id("<|vision_start|>")]
            + [token_id("<|image_pad|>")] * int(count)
            + [token_id("<|vision_end|>")]
        )
        token_ids += label + vision
        token_tags += [pl.TEXT_TAG] * len(label) + [pl.VIDEO_TAG] * len(vision)
    prompt_ids = list(tokenize(prompt))
    token_ids += prompt_ids
    token_tags += [pl.TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags


def build_fl2va_layout(
    text_token_tags: torch.Tensor,
    *,
    height: int,
    width: int,
    num_frames: int,
    anchors: Sequence[str] = (),
    workflow: str | None = None,
) -> pl.PackedLayout:
    """``[text | keyframes | target audio | target video]`` (``MiniMaxH3PrepareLayoutStep``).

    ``num_frames`` is aligned to ``17 * n + 5`` and the aligned duration must be 5-15 s.
    ``workflow`` defaults to ``"t2va"`` without anchors and ``"fl2va"`` with them.
    """

    anchors = tuple(anchors)
    if anchors not in ANCHOR_SETS:
        raise ValueError(f"keyframe anchors must be one of {ANCHOR_SETS}, got {anchors}")
    workflow = workflow or ("fl2va" if anchors else "t2va")
    if pl.WORKFLOWS.get(workflow) != "FL2VA":
        raise pl.VariantMismatch(f"workflow {workflow!r} is not an FL2VA workflow")
    text_token_tags = pl.check_text_token_tags(text_token_tags)
    pl.check_canvas(height, width)
    num_frames = pl.resolve_num_frames(num_frames)
    num_latent_frames = pl.video_latent_num_frames(num_frames)
    latent_height = height // pl.VAE_SPATIAL_COMPRESSION
    latent_width = width // pl.VAE_SPATIAL_COMPRESSION
    num_audio_latents = pl.audio_latent_num_frames(num_frames)

    _, patch_h, patch_w = pl.PATCH_SIZE
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text = int(text_token_tags.shape[0])
    num_condition_rows = len(anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * pl.AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text + num_condition_rows + num_audio_rows + num_video_rows
    condition_start = num_text
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    frame_grid, width_grid = pl.frame_position_grid(latent_height, latent_width)
    for index, anchor in enumerate(anchors):
        if anchor == "first":
            anchor_time = float(num_text)
        else:
            anchor_time = (
                float(num_text)
                + pl.last_anchor_span_pairwise(num_latent_frames)
                - pl.ROPE_FRAME_RESCALE
            )
        rows = slice(
            condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid
    pl.fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        float(num_text),
        width_grid,
    )
    video = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video[:, :, 0] = pl.temporal_position_grid(num_latent_frames, float(num_text))[:, None]
    video[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video.reshape(-1, 3)

    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    return pl.build_layout(
        workflow=workflow,
        text_token_tags=text_token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
        geometry={
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "num_audio_latents": num_audio_latents,
            "keyframe_anchors": list(anchors),
        },
    )


def fl2va_initial_rows(
    layout: pl.PackedLayout,
    condition_latents: Sequence[torch.Tensor],
    *,
    generator: torch.Generator | None,
    device: torch.device | str = "cpu",
    latent_channels: int = pl.VAE_LATENT_CHANNELS,
    audio_latent_channels: int = pl.AUDIO_LATENT_CHANNELS,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial ``(video rows, audio rows)`` in ``video_indices`` / ``audio_indices`` order.

    Draw order on ``generator``: one noise per keyframe condition, then the target video
    latent, then the target audio rows. The keyframe rows lead the video rows and are never
    written by the denoising loop.
    """

    if layout.variant != "FL2VA":
        raise pl.VariantMismatch(f"{layout.variant} layout handed to the FL2VA row builder")
    if len(condition_latents) != len(layout.geometry["keyframe_anchors"]):
        raise ValueError(
            f"{len(condition_latents)} keyframe latents for anchors "
            f"{layout.geometry['keyframe_anchors']}"
        )
    geometry = layout.geometry
    video_rows: list[torch.Tensor] = []
    if condition_latents:
        condition_rows = pl.pack_condition_rows(
            condition_latents, generator=generator, device=device
        )
        if condition_rows.shape[0] != layout.num_condition_video_rows:
            raise ValueError(
                f"layout reserved {layout.num_condition_video_rows} keyframe rows, the latents "
                f"pack into {condition_rows.shape[0]}: canvas and keyframes disagree"
            )
        video_rows.append(condition_rows)
    target_video, audio_rows = pl.draw_generated_rows(
        num_latent_frames=geometry["num_latent_frames"],
        latent_height=geometry["latent_height"],
        latent_width=geometry["latent_width"],
        num_audio_latents=geometry["num_audio_latents"],
        generator=generator,
        device=device,
        latent_channels=latent_channels,
        audio_latent_channels=audio_latent_channels,
        latents=latents,
        audio_latents=audio_latents,
    )
    video_rows.append(target_video)
    return torch.cat(video_rows), audio_rows
