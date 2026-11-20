# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``ref2va_pipeline_contract`` cases for ``h3_pipeline_contract_report.py --row ref2va``."""

from __future__ import annotations

import random
import sys
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from rl_engine.models.minimax_h3 import packed_layout as pl
from rl_engine.models.minimax_h3 import ref2va
from rl_engine.models.minimax_h3.trajectory import tensor_digest

PROMPT = "<Picture 1> walks through <Video 1>; the voice of <Audio 2> narrates."
CASES = ["image", "image_video_audio", "two_videos"]
_SPECIAL = {
    "<|vision_start|>": 151652,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<|vision_end|>": 151653,
}
INTERPRETER_SUMMATION = "neumaier" if sys.version_info >= (3, 12) else "sequential"


def _image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _frames(count: int, height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, height, width, 3), dtype=np.uint8)


def _wave(channels: int, samples: int, seed: int) -> torch.Tensor:
    return torch.randn(channels, samples, generator=torch.Generator().manual_seed(seed)) * 0.1


def _references(case: str) -> list[Any]:
    image = ref2va.ImageReference(_image(320, 240, 1))
    video = ref2va.VideoReference(
        _frames(75, 90, 160, 2), fps=30.0, audio=_wave(1, 44100 * 3, 3), sample_rate=44100
    )
    audio = ref2va.AudioReference(_wave(2, 32000 * 6, 4), sample_rate=32000)
    silent = ref2va.VideoReference(_frames(60, 160, 90, 5), fps=24.0)
    return {
        "image": [image],
        "image_video_audio": [image, video, audio],
        "two_videos": [video, silent],
    }[case]


def _kinds(refs):
    return [(r.kind, r.has_audio) for r in refs]


def preprocess(case: str) -> dict[str, Any]:
    """References -> normalised media -> presentation -> layout -> initial rows -> artifact."""

    height, width, num_frames = ref2va.resolve_ref2va_plan(num_frames=124)
    normalized = ref2va.normalize_references(_references(case), num_frames=num_frames)
    sampled = [
        ref2va.sample_video_condition_frames(r.frames) for r in normalized if r.kind == "video"
    ]
    image_counts = [
        (r.image.size[0] // 32) * (r.image.size[1] // 32) for r in normalized if r.kind == "image"
    ]
    video_counts = [
        (r.frames.shape[1] // 32) * (r.frames.shape[2] // 32)
        for r in normalized
        if r.kind == "video"
    ]
    ids, tags = ref2va.ref2va_presentation(
        lambda s: [ord(c) for c in s],
        _SPECIAL.__getitem__,
        PROMPT,
        _kinds(normalized),
        image_counts,
        video_counts,
        [stamps for _, stamps in sampled],
    )
    g = torch.Generator().manual_seed(11)
    condition_latents, audio_rows = [], []
    for r in normalized:
        if r.kind == "image":
            w, h = r.image.size
            condition_latents.append(torch.randn(1, 24, 1, h // 16, w // 16, generator=g))
        elif r.kind == "video":
            frames = pl.video_latent_num_frames(ref2va.vae_video_frame_count(r.frames.shape[0]))
            h, w = r.frames.shape[1] // 16, r.frames.shape[2] // 16
            condition_latents.append(torch.randn(1, 24, frames, h, w, generator=g))
        if r.has_audio:
            latents = pl.audio_latent_num_frames(
                r.audio.shape[1], fps=pl.AUDIO_SAMPLE_RATE, latents_per_second=40
            )
            audio_rows.append(torch.randn(2 * latents, 32, generator=g))
    blocks = ref2va.reference_blocks(
        _kinds(normalized), [c.shape for c in condition_latents], [a.shape[0] for a in audio_rows]
    )
    layout = ref2va.build_ref2va_layout(
        torch.tensor(tags), blocks, height=height, width=width, num_frames=num_frames
    )
    video, audio = ref2va.ref2va_initial_rows(
        layout, condition_latents, audio_rows, generator=torch.Generator().manual_seed(0)
    )
    inputs = {"token_ids": tensor_digest(torch.tensor(ids))}
    for i, r in enumerate(normalized):
        if r.kind == "image":
            inputs[f"ref{i}_image"] = tensor_digest(torch.from_numpy(np.array(r.image)))
        if r.kind == "video":
            inputs[f"ref{i}_frames"] = tensor_digest(torch.from_numpy(r.frames))
        if r.has_audio:
            inputs[f"ref{i}_audio"] = tensor_digest(r.audio)
    inputs.update(initial_video_rows=tensor_digest(video), initial_audio_rows=tensor_digest(audio))
    return pl.layout_artifact(layout, inputs=inputs)


def _random_request(rng: random.Random):
    kinds, shapes, audio = [], [], []
    for _ in range(rng.randint(1, 6)):
        kind = rng.choice(["image", "video", "video", "audio"])
        if kind == "image":
            h, w = rng.choice([(128, 192), (96, 96), (128, 256)])
            shapes.append((1, 24, 1, h, w))
            kinds.append((kind, False))
        elif kind == "video":
            frames = rng.choice([7, 12, 17, 22, 37, 47])
            h, w = rng.choice([(48, 84), (84, 48), (48, 48)])
            shapes.append((1, 24, frames, h, w))
            sound = rng.random() < 0.6
            if sound:
                audio.append(2 * rng.choice([40, 93, 207, 400]))
            kinds.append((kind, sound))
        else:
            audio.append(2 * rng.choice([1, 40, 207, 600]))
            kinds.append((kind, True))
    if {k for k, _ in kinds} == {"audio"}:
        kinds.insert(0, ("image", False))
        shapes.insert(0, (1, 24, 1, 96, 96))
    return kinds, shapes, audio


def reference() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise, before_encoder, encoders
        from diffusers.modular_pipelines.minimax_h3 import references as dref
    except Exception as exc:
        out["diffusers_static"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    else:
        build = before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
        rng = random.Random(0)
        cases = equal = 0
        for _ in range(200):
            kinds, shapes, audio = _random_request(rng)
            refs: list[Any] = []
            for kind, sound in kinds:
                if kind == "image":
                    refs.append(dref.MiniMaxH3ImageReference(image=None))
                elif kind == "video":
                    refs.append(
                        dref.MiniMaxH3VideoReference(
                            frames=None, fps=24.0, audio=torch.zeros(2, 1) if sound else None
                        )
                    )
                else:
                    refs.append(dref.MiniMaxH3AudioReference(audio=torch.zeros(2, 1)))
            frames = rng.choice([121, 124, 200, 345])
            height, width = rng.choice([(768, 1344), (1344, 768), (768, 768)])
            tags = torch.ones(rng.randint(1, 400), dtype=torch.long)
            blocks = ref2va.reference_blocks(kinds, shapes, audio)
            ours = ref2va.build_ref2va_layout(
                tags,
                blocks,
                height=height,
                width=width,
                num_frames=frames,
                span_summation=INTERPRETER_SUMMATION,
            )
            g = ours.geometry
            theirs = build(
                tags,
                refs,
                [torch.empty(s, device="meta") for s in shapes],
                [torch.empty(n, 32, device="meta") for n in audio],
                g["num_latent_frames"],
                g["latent_height"],
                g["latent_width"],
                g["num_audio_latents"],
                (1, 2, 2),
                2,
                2,
                0,
            )
            names = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
            cases += 1
            equal += all(  # dtype, shape and bytes, so -0.0 is not 0.0
                tensor_digest(getattr(ours, n)) == tensor_digest(t)
                for n, t in zip(names, theirs[:5], strict=True)
            ) and (ours.num_condition_video_rows, ours.num_condition_audio_rows) == tuple(
                theirs[5:]
            )
        out["diffusers_static"] = {
            "cases": cases,
            "bitwise_equal": equal,
            "span_summation": INTERPRETER_SUMMATION,
            "what": "MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence on 200 random "
            "requests (1-6 references of every kind, with and without sound)",
        }
        setup = before_encoder.MiniMaxH3Ref2VASetupStep
        frames = _frames(50, 90, 160, 0)
        cases = equal = 0
        for fps, n in ((30.0, 124), (24.0, 30), (12.5, 124), (60.0, 40), (29.97, 124)):
            cases += 1
            equal += bool(
                np.array_equal(
                    ref2va.normalize_video_reference(frames, fps, n),
                    setup._normalize_video_condition(frames, fps, n, 32, 768, 768 * 1344, 24.0),
                )
            )
        for wave, rate in (
            (_wave(1, 44100 * 3, 0), 44100),
            (_wave(2, 48000 * 2, 1), 48000),
            (_wave(2, 32000, 2), 32000),
        ):
            cases += 1
            equal += tensor_digest(
                ref2va.normalize_audio_reference(wave, rate, max_duration=1.5)
            ) == tensor_digest(setup._normalize_audio_condition(wave, rate, 32000, 1.5))
        from diffusers.image_processor import VaeImageProcessor

        processor = VaeImageProcessor(vae_scale_factor=16)
        for size in ((320, 240), (240, 320), (2048, 2048), (100, 400)):
            image = _image(*size, seed=3)
            ours_image = ref2va.normalize_image_reference(image)
            scale = 2048 / min(size)
            th = max(32, round(size[1] * scale / 32) * 32)
            tw = max(32, round(size[0] * scale / 32) * 32)
            theirs_image = (
                image if image.size == (tw, th) else processor.resize(image, height=th, width=tw)
            )
            cases += 1
            equal += bool(np.array_equal(np.array(ours_image), np.array(theirs_image)))
        text_step = encoders.MiniMaxH3Ref2VATextEncoderStep
        for n in (13, 24, 49, 50):
            cases += 1
            a = ref2va.sample_video_condition_frames(frames[:n])
            b = text_step._sample_video_condition_frames(frames[:n], 24.0, 2.0, 2)
            equal += a[1] == b[1] and all(
                np.array_equal(x, y) for x, y in zip(a[0], b[0], strict=True)
            )
        out["diffusers_media"] = {
            "cases": cases,
            "bitwise_equal": equal,
            "what": "Ref2VASetupStep video (5 rates) / audio (3 rates, torchaudio) normalisation, "
            "image short-edge resize (4 sizes), conditioner frame sampling (4 lengths)",
        }
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (  # noqa: E501
            packed_sequence,
        )
    except Exception as exc:
        out["sglang_prefix"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    else:
        rng = random.Random(1)
        cases = equal = 0
        for _ in range(100):
            kinds, shapes, audio = _random_request(rng)
            blocks = ref2va.reference_blocks(kinds, shapes, audio)
            ours = ref2va.build_ref2va_layout(
                torch.ones(33, dtype=torch.long),
                blocks,
                height=768,
                width=1344,
                num_frames=124,
                span_summation=INTERPRETER_SUMMATION,
            )
            g = ours.geometry
            wire: list[dict[str, Any]] = []
            for b in blocks:
                if b.kind == "image":
                    wire.append(
                        {"kind": "image", "latent_h": b.latent_height, "latent_w": b.latent_width}
                    )
                elif b.kind == "audio":
                    wire.append({"kind": "audio", "ref_audio_t": b.audio_rows // 2})
                else:
                    wire.append(
                        {
                            "kind": "video",
                            "ref_audio_t": b.audio_rows // 2,
                            "latent_t": b.latent_frames,
                            "latent_h": b.latent_height,
                            "latent_w": b.latent_width,
                        }
                    )
            packed = packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(
                text_len=33,
                latent_t=g["num_latent_frames"],
                latent_h=g["latent_height"],
                latent_w=g["latent_width"],
                audio_t=g["num_audio_latents"],
                ref_blocks=wire,
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
            "span_summation": INTERPRETER_SUMMATION,
            "what": "minimax_h3_packed_sequence_ref2va_blocks on 100 random requests, first `used` "
            "rows (SGLang pads to 64 rows tagged -1)",
        }
    return out


def _timing_request():
    kinds = [("image", False), ("video", True), ("audio", True)]
    shapes = [(1, 24, 1, 128, 192), (1, 24, 37, 48, 84)]
    audio = [2 * 207, 2 * 207]
    return kinds, shapes, audio


def timing(time_call: Callable[..., dict[str, float]]) -> dict[str, Any]:
    kinds, shapes, audio = _timing_request()
    tags = torch.ones(300, dtype=torch.long)
    blocks = ref2va.reference_blocks(kinds, shapes, audio)
    out = {
        "rl_kernel": {
            **time_call(
                lambda: ref2va.build_ref2va_layout(
                    tags, blocks, height=768, width=1344, num_frames=124
                )
            ),
            "label": "Ref2VA image+video(sound)+audio layout, 768x1344, 124 frames",
        }
    }
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
        from diffusers.modular_pipelines.minimax_h3 import references as dref

        build = before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
        refs = [
            dref.MiniMaxH3ImageReference(image=None),
            dref.MiniMaxH3VideoReference(frames=None, fps=24.0, audio=torch.zeros(2, 1)),
            dref.MiniMaxH3AudioReference(audio=torch.zeros(2, 1)),
        ]
        latents = [torch.empty(s, device="meta") for s in shapes]
        rows = [torch.empty(n, 32, device="meta") for n in audio]
        out["diffusers"] = time_call(
            lambda: build(tags, refs, latents, rows, 37, 48, 84, 207, (1, 2, 2), 2, 2, 0)
        )
    except Exception:
        pass
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (  # noqa: E501
            packed_sequence,
        )

        wire = [
            {"kind": "image", "latent_h": 128, "latent_w": 192},
            {"kind": "video", "ref_audio_t": 207, "latent_t": 37, "latent_h": 48, "latent_w": 84},
            {"kind": "audio", "ref_audio_t": 207},
        ]
        out["sglang"] = time_call(
            lambda: packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(
                text_len=300, latent_t=37, latent_h=48, latent_w=84, audio_t=207, ref_blocks=wire
            )
        )
    except Exception:
        pass
    return out


def findings() -> dict[str, Any]:
    differ = [
        n for n in range(1, 200) if pl.video_span(n, "neumaier") != pl.video_span(n, "sequential")
    ]
    reference_latent_frames = [5 * n + 2 for n in range(1, 40)]  # 17n+5 frames -> 5n+2 latents
    rng = random.Random(2)
    layouts = position_rows = fp64_rows = fp32_rows = 0
    for _ in range(300):
        kinds, shapes, audio = _random_request(rng)
        blocks = ref2va.reference_blocks(kinds, shapes, audio)
        tags = torch.ones(rng.randint(1, 400), dtype=torch.long)
        a, b = (
            ref2va.build_ref2va_layout(
                tags, blocks, height=768, width=1344, num_frames=124, span_summation=s
            )
            for s in ("neumaier", "sequential")
        )
        layouts += 1
        position_rows += a.sequence_length
        fp64_rows += int((a.position_ids != b.position_ids).any(dim=1).sum())
        fp32_rows += int(
            (a.position_ids.to(torch.float32) != b.position_ids.to(torch.float32)).any(dim=1).sum()
        )
    return {
        "video_span_summation": {
            "interpreter": f"CPython {sys.version.split()[0]}",
            "interpreter_sum_is": INTERPRETER_SUMMATION,
            "default": pl.DEFAULT_SPAN_SUMMATION,
            "latent_frames_where_neumaier_differs_from_sequential": differ[:40],
            "count_below_200": len(differ),
            "released_reference_latent_frame_counts_affected": [
                n for n in reference_latent_frames if n in set(differ)
            ][:20],
            "random_requests": {
                "layouts": layouts,
                "rows": position_rows,
                "rows_with_fp64_difference": fp64_rows,
                "rows_with_fp32_difference_after_cast": fp32_rows,
            },
            "note": "Diffusers and SGLang sum a video reference's clock advance with Python's "
            "sum(), which is Neumaier-compensated from CPython 3.12 and plain left-to-right "
            "before; the released reference therefore depends on the interpreter. RL-Kernel "
            "writes both out and records the choice in the layout geometry.",
        }
    }
