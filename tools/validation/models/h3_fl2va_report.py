# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``fl2va_pipeline_contract`` cases for ``h3_pipeline_contract_report.py --row fl2va``."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from rl_engine.models.minimax_h3 import fl2va
from rl_engine.models.minimax_h3 import packed_layout as pl
from rl_engine.models.minimax_h3.trajectory import tensor_digest

PROMPT = "A skateboarding cat crosses a sunlit plaza; wheels clatter, a crowd cheers."
SIGNATURES = {"zero": (), "first": ("first",), "last": ("last",), "first_last": ("first", "last")}
# Released canvas (16:9 from a 1280x720 keyframe, or given without keyframes) and a portrait
# one, 124 and 345 frames.
CASES = [
    f"{name}/{aspect}/{frames}"
    for name in SIGNATURES
    for aspect in ("landscape", "portrait")
    for frames in (124, 345)
]


def _image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _token_ids(text: str) -> list[int]:
    return [ord(c) for c in text]  # stand-in tokenizer: the tags, not the ids, are the contract


_SPECIAL = {"<|vision_start|>": 151652, "<|image_pad|>": 151655, "<|vision_end|>": 151653}


def preprocess(case: str) -> dict[str, Any]:
    """Keyframes -> presentation tags -> layout -> initial rows -> sealed artifact."""

    name, aspect, frames = case.split("/")
    anchors = SIGNATURES[name]
    size = (1280, 720) if aspect == "landscape" else (720, 1280)
    first = _image(*size, seed=1) if "first" in anchors else None
    # A lone last keyframe sets the canvas, so it has the case's aspect; after a first one it
    # is the square follower that the cover crop handles.
    last = _image(*((1000, 1000) if first else size), seed=2) if "last" in anchors else None
    if anchors:
        keyframes = fl2va.prepare_keyframes(first, last)
    else:
        h, w = (768, 1344) if aspect == "landscape" else (1344, 768)
        keyframes = fl2va.prepare_keyframes(height=h, width=w)
    counts = [(keyframes.height // 32) * (keyframes.width // 32)] * len(keyframes.images)
    ids, tags = fl2va.fl2va_presentation(_token_ids, _SPECIAL.__getitem__, PROMPT, counts)
    layout = fl2va.build_fl2va_layout(
        torch.tensor(tags),
        height=keyframes.height,
        width=keyframes.width,
        num_frames=int(frames),
        anchors=keyframes.anchors,
    )
    g = torch.Generator().manual_seed(123)
    latents = [
        torch.randn(
            1, 24, 1, layout.geometry["latent_height"], layout.geometry["latent_width"], generator=g
        )
        for _ in keyframes.anchors
    ]
    video, audio = fl2va.fl2va_initial_rows(
        layout, latents, generator=torch.Generator().manual_seed(0)
    )
    inputs = {
        f"keyframe_{i}_pixels": tensor_digest(fl2va.keyframe_pixels(k))
        for i, k in enumerate(keyframes.images)
    }
    inputs.update(
        token_ids=tensor_digest(torch.tensor(ids)),
        initial_video_rows=tensor_digest(video),
        initial_audio_rows=tensor_digest(audio),
    )
    return pl.layout_artifact(layout, inputs=inputs)


def _layout_args(anchors, height, width, frames, text):
    tags = torch.ones(text, dtype=torch.long)
    tags[: text // 3] = 0
    layout = fl2va.build_fl2va_layout(
        tags, height=height, width=width, num_frames=frames, anchors=anchors
    )
    return tags, layout


def reference() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
    except Exception as exc:
        out["diffusers_static"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    else:
        build = before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence
        cases = equal = 0
        for anchors in fl2va.ANCHOR_SETS:
            for frames in (120, 124, 171, 220, 300, 345):
                for aspect in ((16, 9), (9, 16), (1, 1), (4, 1), (1, 4), (21, 9), (1280, 721)):
                    height, width = pl.resolve_canvas_size(*aspect)
                    for text in (1, 77, 300):
                        tags, ours = _layout_args(anchors, height, width, frames, text)
                        g = ours.geometry
                        theirs = build(
                            tags,
                            g["num_latent_frames"],
                            g["latent_height"],
                            g["latent_width"],
                            g["num_audio_latents"],
                            (1, 2, 2),
                            2,
                            2,
                            0,
                            anchors,
                        )
                        names = ("position_ids", "token_tags", "video_indices", "audio_indices")
                        same = all(  # dtype, shape and bytes, so -0.0 is not 0.0
                            tensor_digest(getattr(ours, n)) == tensor_digest(t)
                            for n, t in zip(names + ("text_indices",), theirs[:5], strict=True)
                        ) and (
                            ours.num_condition_video_rows,
                            ours.num_condition_audio_rows,
                        ) == tuple(
                            theirs[5:]
                        )
                        cases += 1
                        equal += same
        out["diffusers_static"] = {
            "cases": cases,
            "bitwise_equal": equal,
            "what": "MiniMaxH3PrepareLayoutStep.build_packed_sequence: 4 anchor sets x 6 frame "
            "counts x 7 aspect ratios x 3 text lengths",
        }
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (  # noqa: E501
            packed_sequence,
        )
    except Exception as exc:
        out["sglang_prefix"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    else:
        indices = {(): None, ("first",): [0], ("last",): [-1], ("first", "last"): [0, -1]}
        cases = equal = 0
        for anchors, keyframe_indices in indices.items():
            for frames in (124, 220, 345):
                for aspect in ((16, 9), (9, 16), (1, 1)):
                    height, width = pl.resolve_canvas_size(*aspect)
                    tags = torch.ones(41, dtype=torch.long)
                    ours = fl2va.build_fl2va_layout(
                        tags, height=height, width=width, num_frames=frames, anchors=anchors
                    )
                    g = ours.geometry
                    packed = packed_sequence.minimax_h3_packed_sequence(
                        text_len=41,
                        latent_t=g["num_latent_frames"],
                        latent_h=g["latent_height"],
                        latent_w=g["latent_width"],
                        audio_t=g["num_audio_latents"],
                        include_keyframe_cond=bool(anchors),
                        keyframe_frame_indices=keyframe_indices,
                        frame_count=g["num_frames"] if anchors else None,
                    )
                    used = ours.sequence_length
                    cases += 1
                    pairs = (  # digests: torch.equal would take int32 indices for int64
                        (packed["img_position_ids"][:used], ours.position_ids),
                        (packed["token_tags"][:used], ours.token_tags),
                        (packed["img_pos"], ours.video_indices),
                        (packed["audio_pos"], ours.audio_indices),
                    )
                    equal += all(tensor_digest(a) == tensor_digest(b) for a, b in pairs)
        out["sglang_prefix"] = {
            "cases": cases,
            "bitwise_equal": equal,
            "what": "minimax_h3_packed_sequence, first `used` rows (SGLang pads to 64 rows "
            "tagged -1); FL2VA text-only tags",
        }
    return out


def timing(time_call: Callable[..., dict[str, float]]) -> dict[str, Any]:
    tags = torch.ones(300, dtype=torch.long)
    out = {
        "rl_kernel": {
            **time_call(
                lambda: fl2va.build_fl2va_layout(
                    tags, height=768, width=1344, num_frames=124, anchors=("first", "last")
                )
            ),
            "label": "FL2VA first+last layout, 768x1344, 124 frames",
        }
    }
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise

        build = before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence
        out["diffusers"] = time_call(
            lambda: build(tags, 37, 48, 84, 207, (1, 2, 2), 2, 2, 0, ("first", "last"))
        )
    except Exception:
        pass
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (  # noqa: E501
            packed_sequence,
        )

        out["sglang"] = time_call(
            lambda: packed_sequence.minimax_h3_packed_sequence(
                text_len=300,
                latent_t=37,
                latent_h=48,
                latent_w=84,
                audio_t=207,
                include_keyframe_cond=True,
                keyframe_frame_indices=[0, -1],
                frame_count=124,
            )
        )
    except Exception:
        pass
    return out


def findings() -> dict[str, Any]:
    layout = fl2va.build_fl2va_layout(
        torch.ones(16, dtype=torch.long),
        height=768,
        width=1344,
        num_frames=124,
        anchors=("first", "last"),
    )
    rows_per_frame = 24 * 42
    times = layout.position_ids[:, 0]
    target = layout.video_indices[layout.num_condition_video_rows :]
    return {
        "last_anchor": {
            "time": float(times[16 + rows_per_frame]),
            "target_last_latent_frame_time": float(times[target[-1]]),
            "note": "the last keyframe sits on the last *pixel* frame: the last latent frame "
            "time + 3 * 5/3 (its 4 pixel frames), summed with numpy's pairwise order",
        },
        "pairwise_vs_sequential_first_difference": next(
            n
            for n in range(1, 400)
            if pl.last_anchor_span_pairwise(n) != pl.video_span(n, "sequential")
        ),
    }
