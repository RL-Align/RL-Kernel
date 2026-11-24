# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``ref2va_pipeline_contract``: reference preprocessing, ``transformer_ref`` mapping.

Ref2VA runs on the ``transformer_ref/`` partition and conditions on up to 12 references
(at most 9 images, 3 videos, 3 audios; audio alone is refused), **in request order**: the
order numbers the references in the conditioner presentation and lays them out on the shared
rotary clock, so a different order is a different request. This module is the strict,
pinned version of the Diffusers modular pipeline (``diffusers@f53d552``:
``MiniMaxH3Ref2VASetupStep``, ``MiniMaxH3Ref2VATextEncoderStep``'s frame sampling and
presentation, the reference geometry of ``MiniMaxH3Ref2VAReferenceEncoderStep``,
``MiniMaxH3Ref2VAPrepareLayoutStep`` and ``MiniMaxH3Ref2VAPrepareLatentsStep``):

1. :func:`validate_references` and :func:`resolve_ref2va_plan`: limits, the 16:9 canvas
   (references never bind the generated geometry), the ``17 * n + 5`` frame count;
2. :func:`normalize_references`: an image to its own 2048-pixel short edge (no area cap,
   LANCZOS); a video to 24 fps (whole-frame drop/duplicate), truncated, LANCZOS onto the
   canvas of its own aspect ratio; a soundtrack truncated at its source rate, upmixed to
   stereo, resampled once to 32 kHz;
3. :func:`sample_video_condition_frames` and :func:`ref2va_presentation`: what the Qwen3-VL
   conditioner reads (2 fps, merged pairs labelled ``"<x.x seconds>"`` with round-half-even);
4. :func:`vae_video_frame_count`: the frame bound a video reference is sliced to before the VAE;
5. :func:`build_ref2va_layout`: ``[text | reference blocks | target audio | target video]``;
   an image advances the clock by one slot, an audio by its latent count, a video (with its
   soundtrack packed just before it) by ``max(audio latents, sequential video span)``;
6. :func:`ref2va_initial_rows`: noised visual references then target noise on the video
   side, clean reference soundtracks then target noise on the audio side;
7. :func:`map_transformer_ref`: the ``transformer_ref/`` state dict through the same
   ``h3_checkpoint_mapper`` tables, tagged with the Ref2VA identity so it cannot be paired
   with an FL2VA layout.

The VAE encodes (``visual_vae_contract`` / ``audio_vae_contract``) and the Qwen3-VL hidden
state (``qwen3vl_layer50_contract``) are other rows; this row consumes their outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import torch

from rl_engine.models.minimax_h3 import checkpoint_mapper
from rl_engine.models.minimax_h3 import packed_layout as pl

if TYPE_CHECKING:  # Pillow is not a base dependency; only the media normalisation imports it
    from PIL import Image

WORKFLOW = "ref2va"
KINDS = ("image", "video", "audio")
LIMITS: Mapping[str, int] = {"image": 9, "video": 3, "audio": 3, "total": 12}
REFERENCE_IMAGE_SHORT_EDGE = 2048
CONDITIONER_VIDEO_SAMPLE_FPS = 2.0
QWEN3VL_TEMPORAL_PATCH = 2


# --- references ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageReference:
    image: Any  # PIL image, (H, W, 3) array or (3, H, W) tensor; uint8 or float in [0, 1]
    kind = "image"
    has_audio = False


@dataclass(frozen=True)
class VideoReference:
    frames: Any  # list of PIL, (T, H, W, 3) array or (T, 3, H, W) tensor
    fps: float
    audio: torch.Tensor | None = None  # (channels, samples)
    sample_rate: int | None = None
    kind = "video"

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


@dataclass(frozen=True)
class AudioReference:
    audio: torch.Tensor  # (channels, samples), mono or stereo
    sample_rate: int | None = None
    kind = "audio"
    has_audio = True


def validate_references(references: Sequence[Any]) -> list[str]:
    """Kinds in request order, after the released limits (Diffusers objects are accepted too)."""

    if not references:
        raise ValueError("ref2va needs at least one reference; use t2va for text only")
    kinds = []
    for index, ref in enumerate(references):
        kind = getattr(ref, "kind", None)
        if kind not in KINDS:
            raise ValueError(f"references[{index}] must be an image, video or audio reference")
        kinds.append(kind)
    for kind in KINDS:
        if kinds.count(kind) > LIMITS[kind]:
            raise ValueError(f"at most {LIMITS[kind]} {kind} references, got {kinds.count(kind)}")
    if len(kinds) > LIMITS["total"]:
        raise ValueError(f"at most {LIMITS['total']} references in total, got {len(kinds)}")
    if set(kinds) == {"audio"}:
        raise ValueError("an audio reference needs at least one image or video reference")
    return kinds


def resolve_ref2va_plan(
    *,
    num_frames: int,
    height: int | None = None,
    width: int | None = None,
    short_edge: int = pl.CANVAS_SHORT_EDGE,
    max_pixels: int = pl.CANVAS_MAX_PIXELS,
) -> tuple[int, int, int]:
    """``(height, width, aligned num_frames)``: 16:9 unless asked, aligned duration in 5-15 s."""

    if (height is None) != (width is None):
        raise ValueError("height and width have to be passed together, or neither")
    if height is None or width is None:
        height, width = pl.resolve_canvas_size(16, 9, short_edge=short_edge, max_pixels=max_pixels)
    pl.check_canvas(height, width)
    return height, width, pl.resolve_num_frames(num_frames)


# --- media normalisation ------------------------------------------------------------------


def _to_pil(image: Any) -> Image.Image:
    """Diffusers' route onto PIL: uint8 -> [0, 1] float -> ``(x * 255).round().astype(uint8)``."""

    from PIL import Image

    if isinstance(image, torch.Tensor):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        image = image.cpu().permute(1, 2, 0).float().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"a reference image must be (H, W, 3) RGB, got {image.shape}")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = Image.fromarray((image * 255).round().astype("uint8"))
    if not isinstance(image, Image.Image):
        raise ValueError(f"unsupported reference image type {type(image).__name__}")
    return image


def normalize_image_reference(
    image: Any,
    *,
    short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    multiple: int = pl.CANVAS_MULTIPLE,
) -> Image.Image:
    """Own short edge (2048), rounded to the canvas multiple, no area cap, PIL LANCZOS."""

    from PIL import Image

    image = _to_pil(image)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"a reference image must have a positive size, got {image.size}")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"a reference image must be within 1:4 and 4:1, got {width}x{height}")
    scale = short_edge / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image


def _frames_uint8_thwc(frames: Any) -> np.ndarray:
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"a reference video must be (T, H, W, 3) RGB, got {tuple(frames.shape)}")
    return frames


def normalize_video_reference(
    frames: Any,
    fps: float,
    num_frames: int,
    *,
    target_fps: float = pl.FPS,
    short_edge: int = pl.CANVAS_SHORT_EDGE,
    max_pixels: int = pl.CANVAS_MAX_PIXELS,
) -> np.ndarray:
    """uint8 ``(T, H, W, 3)`` at 24 fps, truncated to ``num_frames``, on its own-aspect canvas.

    The rate change holds every frame until the next one's slot ``floor(i * 24 / fps + 0.5)``
    (ffmpeg's ``fps`` filter); the LANCZOS rescale comes second.
    """

    frames = _frames_uint8_thwc(frames)
    if fps <= 0:
        raise ValueError(f"a reference video must have a positive frame rate, got {fps}")
    if fps != target_fps:
        scale = target_fps / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        end = math.floor(frames.shape[0] * scale + 0.5)
        frames = np.repeat(frames, np.diff(slots, append=end), axis=0)
    frames = frames[:num_frames]
    height, width = pl.resolve_canvas_size(
        frames.shape[2], frames.shape[1], short_edge=short_edge, max_pixels=max_pixels
    )
    if frames.shape[1:3] == (height, width):
        return frames
    from PIL import Image

    return np.stack(
        [
            np.asarray(Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS))
            for f in frames
        ]
    )


def normalize_audio_reference(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    max_duration: float,
    target_sample_rate: int = pl.AUDIO_SAMPLE_RATE,
) -> torch.Tensor:
    """Stereo fp32 ``(2, samples)`` at 32 kHz: truncate at the source rate, upmix, resample once."""

    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            f"a soundtrack must be (channels, samples) mono/stereo, got {waveform.shape}"
        )
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"resampling a soundtrack from {sample_rate} Hz to {target_sample_rate} Hz needs "
            "torchaudio; pass one already at the target rate"
        ) from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def normalize_references(
    references: Sequence[Any],
    *,
    num_frames: int,
    audio_sample_rate: int = pl.AUDIO_SAMPLE_RATE,
    image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    canvas_short_edge: int = pl.CANVAS_SHORT_EDGE,
    canvas_max_pixels: int = pl.CANVAS_MAX_PIXELS,
) -> list[ImageReference | VideoReference | AudioReference]:
    """Every reference on MiniMax-H3's own rates and resolutions, in packed order."""

    validate_references(references)
    out: list[ImageReference | VideoReference | AudioReference] = []
    for ref in references:
        waveform = None
        if ref.has_audio:
            rate = ref.sample_rate if ref.sample_rate is not None else audio_sample_rate
            waveform = normalize_audio_reference(
                ref.audio,
                rate,
                max_duration=num_frames / pl.FPS,
                target_sample_rate=audio_sample_rate,
            )
        if ref.kind == "image":
            out.append(
                ImageReference(normalize_image_reference(ref.image, short_edge=image_short_edge))
            )
        elif ref.kind == "video":
            frames = normalize_video_reference(
                ref.frames,
                float(ref.fps),
                num_frames,
                short_edge=canvas_short_edge,
                max_pixels=canvas_max_pixels,
            )
            out.append(
                VideoReference(
                    frames=frames,
                    fps=float(pl.FPS),
                    audio=waveform,
                    sample_rate=None if waveform is None else audio_sample_rate,
                )
            )
        else:
            assert waveform is not None
            out.append(AudioReference(audio=waveform, sample_rate=audio_sample_rate))
    return out


# --- conditioner presentation -------------------------------------------------------------


def sample_video_condition_frames(
    frames: np.ndarray,
    *,
    fps: float = pl.FPS,
    sample_fps: float = CONDITIONER_VIDEO_SAMPLE_FPS,
    temporal_patch: int = QWEN3VL_TEMPORAL_PATCH,
) -> tuple[list[np.ndarray], list[float]]:
    """Frames the conditioner reads (every ``fps / sample_fps``-th, deduplicated) and the
    timestamp of every merged vision block (mean of its pair; the tail repeats the last one)."""

    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"a reference video must run at least {minimum} frames at {fps:g} fps, "
            f"got {frames.shape[0]}"
        )
    timestamps = [i / sample_fps for i in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    blocks = [
        (timestamps[i] + timestamps[i + temporal_patch - 1]) / 2
        for i in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[i] for i in indices], blocks


def ref2va_presentation(
    tokenize: Callable[[str], Sequence[int]],
    token_id: Callable[[str], int],
    prompt: str,
    kinds_with_audio: Sequence[tuple[str, bool]],
    image_token_counts: Sequence[int],
    video_block_token_counts: Sequence[int],
    video_block_timestamps: Sequence[Sequence[float]],
) -> tuple[list[int], list[int]]:
    """Token ids and modality tags: labels numbered per modality, in request order.

    ``"<Audio j>: "`` comes first for anything carrying sound (alone for an audio reference),
    then ``"<Picture i>: "`` + one vision block, or ``"<Video k>: "`` + ``"<x.x seconds>"`` +
    a vision block per merged frame pair; then the prompt. Vision rows are tagged video.
    """

    if not isinstance(prompt, str):
        raise ValueError("MiniMax-H3 packs one request into one sequence; prompt must be a str")
    token_ids: list[int] = []
    token_tags: list[int] = []

    def text(value: str) -> None:
        ids = list(tokenize(value))
        token_ids.extend(ids)
        token_tags.extend([pl.TEXT_TAG] * len(ids))

    def vision(pad: str, count: int) -> None:
        ids = [token_id("<|vision_start|>")] + [token_id(pad)] * int(count)
        ids.append(token_id("<|vision_end|>"))
        token_ids.extend(ids)
        token_tags.extend([pl.VIDEO_TAG] * len(ids))

    counts = {"image": 0, "video": 0, "audio": 0}
    for kind, has_audio in kinds_with_audio:
        if has_audio:
            counts["audio"] += 1
            text(f"<Audio {counts['audio']}>: ")
        if kind == "image":
            counts["image"] += 1
            text(f"<Picture {counts['image']}>: ")
            vision("<|image_pad|>", image_token_counts[counts["image"] - 1])
        elif kind == "video":
            counts["video"] += 1
            text(f"<Video {counts['video']}>: ")
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                text(f"<{timestamp:.1f} seconds>")
                vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1])
    text(prompt)
    return token_ids, token_tags


# --- reference geometry and layout --------------------------------------------------------


def vae_video_frame_count(
    num_frames: int,
    frames_per_chunk: int = pl.VAE_FRAMES_PER_CHUNK,
    latents_per_chunk: int = pl.VAE_LATENTS_PER_CHUNK,
) -> int:
    """The pinned pipeline's frame bound for a video reference, ``max(1, (n-5) // 17) * 17 + 5``.

    From 22 frames on, the reference snaps *down* to ``17 * k + 5`` frames and the VAE encodes
    it without padding. A shorter reference (the conditioner needs at least 13 frames at
    24 fps) gets the bound 22, so it keeps all its frames, exactly as in the release; the VAE
    then repeats its last frame up to a multiple of 17, and 13-17 frames encode to 2 latent
    frames, 18-21 to 7. The layout reads latent frame counts from the encoder output
    (:func:`reference_blocks`), not from this bound.
    """

    return (
        max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk
        + latents_per_chunk
    )


@dataclass(frozen=True)
class ReferenceBlock:
    """Latent geometry of one reference, the only thing the layout reads from it."""

    kind: str
    latent_frames: int = 0  # image: 1
    latent_height: int = 0
    latent_width: int = 0
    audio_rows: int = 0  # channel-major rows of the soundtrack (audio, or video with sound)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "latent_frames": self.latent_frames,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "audio_rows": self.audio_rows,
        }


def reference_blocks(
    kinds_with_audio: Sequence[tuple[str, bool]],
    condition_latent_shapes: Sequence[Sequence[int]],
    audio_row_counts: Sequence[int],
) -> list[ReferenceBlock]:
    """Pair the encoder outputs (which skip references they do not apply to) with the kinds."""

    visual = iter(tuple(int(x) for x in shape[2:5]) for shape in condition_latent_shapes)
    audio = iter(int(n) for n in audio_row_counts)
    blocks = []
    for kind, has_audio in kinds_with_audio:
        rows = next(audio, None) if has_audio else 0
        if rows is None:
            raise ValueError("fewer soundtracks than references that carry sound")
        if kind in ("image", "video"):
            shape = next(visual, None)
            if shape is None:
                raise ValueError("fewer visual latents than image and video references")
            frames, height, width = shape
            if kind == "image" and frames != 1:
                raise ValueError(f"an image reference encodes to 1 latent frame, got {frames}")
            blocks.append(ReferenceBlock(kind, frames, height, width, rows))
        else:
            blocks.append(ReferenceBlock(kind, audio_rows=rows))
    for leftover, name in ((visual, "visual latents"), (audio, "soundtracks")):
        if next(leftover, None) is not None:
            raise ValueError(f"more {name} than references that carry them")
    return blocks


def build_ref2va_layout(
    text_token_tags: torch.Tensor,
    blocks: Sequence[ReferenceBlock],
    *,
    height: int,
    width: int,
    num_frames: int,
    span_summation: str = pl.DEFAULT_SPAN_SUMMATION,
) -> pl.PackedLayout:
    """``[text | references | target audio | target video]`` (``MiniMaxH3Ref2VAPrepareLayoutStep``).

    ``span_summation`` is how a video reference's clock advance is summed; the default is what
    the pinned Diffusers port computes on CPython >= 3.12 (see ``packed_layout.SPAN_SUMMATIONS``).
    """

    if not blocks:
        raise ValueError("ref2va needs at least one reference block")
    text_token_tags = pl.check_text_token_tags(text_token_tags)
    pl.check_canvas(height, width)
    num_frames = pl.resolve_num_frames(num_frames)
    num_latent_frames = pl.video_latent_num_frames(num_frames)
    latent_height = height // pl.VAE_SPATIAL_COMPRESSION
    latent_width = width // pl.VAE_SPATIAL_COMPRESSION
    num_audio_latents = pl.audio_latent_num_frames(num_frames)
    _, patch_h, patch_w = pl.PATCH_SIZE
    channels = pl.AUDIO_CHANNELS

    def video_rows(b: ReferenceBlock) -> int:
        return b.latent_frames * (b.latent_height // patch_h) * (b.latent_width // patch_w)

    for b in blocks:
        if b.audio_rows % channels:
            raise ValueError(f"soundtrack rows {b.audio_rows} are not {channels}-channel major")
    num_text = int(text_token_tags.shape[0])
    num_target_video = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio = num_audio_latents * channels
    num_ref_video = sum(video_rows(b) for b in blocks)
    num_ref_audio = sum(b.audio_rows for b in blocks)
    sequence_length = num_text + num_ref_video + num_ref_audio + num_target_audio + num_target_video

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    target_frame_grid, target_width_grid = pl.frame_position_grid(latent_height, latent_width)

    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    cursor = num_text
    rotary_time = float(num_text)
    for b in blocks:
        if b.kind == "image":
            rows = slice(cursor, cursor + video_rows(b))
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = pl.frame_position_grid(b.latent_height, b.latent_width)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0  # one integer slot, not a latent frame's 5/3 units
        elif b.kind == "audio":
            latents = b.audio_rows // channels
            rows = slice(cursor, cursor + b.audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            pl.fill_audio_positions(position_ids, rows, latents, rotary_time, target_width_grid)
            rotary_time += float(latents)
        elif b.kind == "video":
            latents = b.audio_rows // channels
            audio_rows = slice(cursor, cursor + b.audio_rows)
            vrows = slice(audio_rows.stop, audio_rows.stop + video_rows(b))
            cursor = vrows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(vrows.start, vrows.stop))
            frame_grid, width_grid = pl.frame_position_grid(b.latent_height, b.latent_width)
            pl.fill_audio_positions(position_ids, audio_rows, latents, rotary_time, width_grid)
            frame_time = pl.temporal_position_grid(b.latent_frames, rotary_time)
            position_ids[vrows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[vrows, 1:] = frame_grid.repeat(b.latent_frames, 1)
            rotary_time += max(float(latents), pl.video_span(b.latent_frames, span_summation))
        else:
            raise ValueError(f"unknown reference kind {b.kind!r}")

    audio_start = cursor
    video_start = audio_start + num_target_audio
    pl.fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    frame_time = pl.temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    return pl.build_layout(
        workflow=WORKFLOW,
        text_token_tags=text_token_tags,
        position_ids=position_ids,
        video_indices=torch.cat(video_indices + [torch.arange(video_start, sequence_length)]),
        audio_indices=torch.cat(audio_indices + [torch.arange(audio_start, video_start)]),
        num_condition_video_rows=num_ref_video,
        num_condition_audio_rows=num_ref_audio,
        geometry={
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "num_audio_latents": num_audio_latents,
            "reference_blocks": [b.to_dict() for b in blocks],
            "span_summation": span_summation,
        },
    )


def ref2va_initial_rows(
    layout: pl.PackedLayout,
    condition_latents: Sequence[torch.Tensor],
    audio_condition_latents: Sequence[torch.Tensor],
    *,
    generator: torch.Generator | None,
    device: torch.device | str = "cpu",
    latent_channels: int = pl.VAE_LATENT_CHANNELS,
    audio_latent_channels: int = pl.AUDIO_LATENT_CHANNELS,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial ``(video rows, audio rows)``: noised visual references then target noise; clean
    reference soundtracks (``t = 1``) then target audio noise. Draw order: one noise per
    visual reference, then the target video latent, then the target audio rows."""

    if layout.variant != "Ref2VA":
        raise pl.VariantMismatch(f"{layout.variant} layout handed to the Ref2VA row builder")
    geometry = layout.geometry
    condition_rows = pl.pack_condition_rows(condition_latents, generator=generator, device=device)
    if condition_rows.shape[0] != layout.num_condition_video_rows:
        raise ValueError(
            f"layout reserved {layout.num_condition_video_rows} reference video rows, the latents "
            f"pack into {condition_rows.shape[0]}"
        )
    target_video, target_audio = pl.draw_generated_rows(
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
    reference_audio = sum(int(rows.shape[0]) for rows in audio_condition_latents)
    if reference_audio != layout.num_condition_audio_rows:
        raise ValueError(
            f"layout reserved {layout.num_condition_audio_rows} reference audio rows, the "
            f"soundtracks pack into {reference_audio}"
        )
    video = torch.cat([condition_rows, target_video]) if condition_rows.numel() else target_video
    audio = torch.cat([rows.to(device) for rows in audio_condition_latents] + [target_audio])
    return video, audio


# --- transformer_ref mapping --------------------------------------------------------------


@dataclass(frozen=True)
class VariantState:
    """A mapped native state dict that remembers which partition it came from."""

    variant: str
    transformer_dir: str
    state: dict[str, torch.Tensor]


def map_transformer_ref(
    state: Mapping[str, torch.Tensor],
    *,
    source_variant: str = "Ref2VA",
    qkv_layout: str = "module",
) -> VariantState:
    """``transformer_ref/`` Diffusers state -> native names (the ``h3_checkpoint_mapper`` tables).

    The tables are the FL2VA ones: both partitions have the same config and tensor table, so
    the mapping is the same pure data movement. What differs is the identity the result
    carries; ``source_variant`` must be ``"Ref2VA"`` (pass the variant
    :func:`packed_layout.identify_transformer_dir` returned for the directory you loaded).
    """

    if source_variant != "Ref2VA":
        known = pl.PIPELINES.get(source_variant)
        where = f"{known.transformer_dir}/" if known is not None else "unknown variant"
        raise pl.VariantMismatch(
            f"{source_variant} weights ({where}) are not the transformer_ref partition"
        )
    native = checkpoint_mapper.diffusers_to_native(state, qkv_layout=qkv_layout)
    return VariantState("Ref2VA", pl.PIPELINES["Ref2VA"].transformer_dir, native)


def check_layout_weights(layout: pl.PackedLayout, weights: VariantState) -> None:
    """A Ref2VA layout runs only on ``transformer_ref`` weights, an FL2VA one on ``transformer``."""

    pl.check_pipeline_identity(layout, transformer_variant=weights.variant)
