# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``ref2va_pipeline_contract``: reference preprocessing, ``transformer_ref`` mapping."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3 import packed_layout as pl
from rl_engine.models.minimax_h3 import ref2va
from rl_engine.models.minimax_h3.trajectory import tensor_digest

REPO = Path(__file__).resolve().parents[3]
PROMPT = "<Picture 1> rides <Video 1> while <Audio 1> plays"


class FakeTokenizer:
    """One id per character, fixed ids for the vision specials; also callable like HF's."""

    SPECIAL = {
        "<|vision_start|>": 900001,
        "<|image_pad|>": 900002,
        "<|video_pad|>": 900004,
        "<|vision_end|>": 900003,
    }

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def token_id(self, token: str) -> int:
        return self.SPECIAL[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.tokenize(text)}

    convert_tokens_to_ids = token_id


def _image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _frames(count: int, height: int, width: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (count, height, width, 3), dtype=np.uint8)


def _wave(channels: int, samples: int, seed: int) -> torch.Tensor:
    return torch.randn(channels, samples, generator=torch.Generator().manual_seed(seed)) * 0.1


def _kinds(refs):
    return [(r.kind, r.has_audio) for r in refs]


def ref2va_preprocess(
    *, device: str = "cpu", span_summation: str = pl.DEFAULT_SPAN_SUMMATION
) -> dict:
    """Whole Ref2VA chain on synthetic media at a small canvas; returns the sealed artifact."""

    height, width, num_frames = ref2va.resolve_ref2va_plan(num_frames=121, height=128, width=224)
    refs = [
        ref2va.ImageReference(_image(300, 200, 1)),
        ref2va.VideoReference(
            _frames(80, 90, 160, 2), fps=30.0, audio=_wave(1, 44100 * 3, 3), sample_rate=44100
        ),
        ref2va.AudioReference(_wave(2, 32000 * 6, 4), sample_rate=32000),
        ref2va.ImageReference(torch.rand(3, 64, 100, generator=torch.Generator().manual_seed(5))),
    ]
    normalized = ref2va.normalize_references(
        refs,
        num_frames=num_frames,
        image_short_edge=256,
        canvas_short_edge=128,
        canvas_max_pixels=128 * 224,
    )
    tok = FakeTokenizer()
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
        tok.tokenize,
        tok.token_id,
        PROMPT,
        _kinds(normalized),
        image_counts,
        video_counts,
        [timestamps for _, timestamps in sampled],
    )
    # Stand-ins for the VAE encoders: latents shaped as the encoders would emit them.
    g = torch.Generator().manual_seed(11)
    condition_latents, audio_rows = [], []
    for r in normalized:
        if r.kind == "image":
            w, h = r.image.size
            condition_latents.append(torch.randn(1, 24, 1, h // 16, w // 16, generator=g))
        elif r.kind == "video":
            frames = ref2va.vae_video_frame_count(r.frames.shape[0])
            condition_latents.append(
                torch.randn(
                    1,
                    24,
                    pl.video_latent_num_frames(frames),
                    r.frames.shape[1] // 16,
                    r.frames.shape[2] // 16,
                    generator=g,
                )
            )
        if r.has_audio:
            latents = pl.audio_latent_num_frames(
                r.audio.shape[1], fps=pl.AUDIO_SAMPLE_RATE, latents_per_second=40
            )
            audio_rows.append(torch.randn(2 * latents, 32, generator=g))
    blocks = ref2va.reference_blocks(
        _kinds(normalized), [c.shape for c in condition_latents], [a.shape[0] for a in audio_rows]
    )
    layout = ref2va.build_ref2va_layout(
        torch.tensor(tags),
        blocks,
        height=height,
        width=width,
        num_frames=num_frames,
        span_summation=span_summation,
    )
    video, audio = ref2va.ref2va_initial_rows(
        layout,
        condition_latents,
        audio_rows,
        generator=torch.Generator().manual_seed(0),
        device=device,
    )
    # The digests below do not see the device; the rows must be where they were asked for.
    assert video.device.type == audio.device.type == torch.device(device).type
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


# --- request validation and normalisation -------------------------------------------------


def test_reference_limits_and_audio_alone_are_refused():
    img = ref2va.ImageReference(_image(8, 8, 0))
    aud = ref2va.AudioReference(_wave(1, 10, 0), 32000)
    vid = ref2va.VideoReference(_frames(5, 8, 8, 0), fps=24.0)
    assert ref2va.validate_references([img, aud]) == ["image", "audio"]
    for refs, match in (
        ([], "at least one"),
        ([aud], "image or video"),
        ([img] * 10, "at most 9 image"),
        ([vid] * 4, "at most 3 video"),
        ([img, aud, aud, aud, aud], "at most 3 audio"),
        ([img] * 9 + [vid] * 3 + [aud], "at most 12"),
        ([object()], "image, video or audio"),
    ):
        with pytest.raises(ValueError, match=match):
            ref2va.validate_references(refs)


def test_plan_defaults_to_16_9_and_aligns_frames():
    assert ref2va.resolve_ref2va_plan(num_frames=121) == (768, 1344, 124)
    assert ref2va.resolve_ref2va_plan(num_frames=200, height=1344, width=768) == (1344, 768, 209)
    with pytest.raises(ValueError, match="together"):
        ref2va.resolve_ref2va_plan(num_frames=121, height=768)


def test_image_reference_goes_to_its_own_2048_short_edge():
    pil = _image(300, 200, 1)
    out = ref2va.normalize_image_reference(pil)
    assert out.size == (3072, 2048)
    assert np.array_equal(
        np.array(out), np.array(pil.resize((3072, 2048), Image.Resampling.LANCZOS))
    )
    arr = np.array(pil)
    for other in (arr, torch.from_numpy(arr).permute(2, 0, 1), arr.astype(np.float32) / 255.0):
        assert np.array_equal(np.array(ref2va.normalize_image_reference(other)), np.array(out))
    with pytest.raises(ValueError, match="1:4"):
        ref2va.normalize_image_reference(_image(500, 100, 1))


def test_video_reference_resamples_whole_frames_then_rescales():
    frames = _frames(10, 64, 112, 0)
    out = ref2va.normalize_video_reference(frames, 30.0, 124, short_edge=64, max_pixels=64 * 128)
    # 10 frames at 30 fps -> slots floor(i * 0.8 + 0.5) = 0,1,2,2,3,4,5,6,6,7, end 8: a frame
    # is held until the next slot, so frames 2 and 7 (whose slot the next frame reuses) drop.
    assert out.shape == (8, 64, 128, 3)
    resized = [
        np.asarray(Image.fromarray(f).resize((128, 64), Image.Resampling.LANCZOS)) for f in frames
    ]
    kept = [0, 1, 3, 4, 5, 6, 8, 9]
    assert np.array_equal(out, np.stack([resized[i] for i in kept]))
    same = ref2va.normalize_video_reference(
        _frames(30, 64, 128, 1), 24.0, 12, short_edge=64, max_pixels=64 * 128
    )
    assert same.shape == (12, 64, 128, 3)


def test_audio_reference_truncates_upmixes_and_resamples_once():
    pytest.importorskip("torchaudio")  # 48 kHz is resampled
    import torchaudio

    mono = _wave(1, 48000 * 3, 0)
    out = ref2va.normalize_audio_reference(mono, 48000, max_duration=2.0)
    assert out.shape[0] == 2 and torch.equal(out[0], out[1])

    expected = torchaudio.transforms.Resample(48000, 32000)(
        mono[:, :96000].expand(2, -1).contiguous()
    )
    assert torch.equal(out, expected)
    same = ref2va.normalize_audio_reference(_wave(2, 32000 * 3, 1), 32000, max_duration=2.0)
    assert same.shape == (2, 64000)


def test_conditioner_frame_sampling_and_presentation():
    frames, stamps = ref2va.sample_video_condition_frames(_frames(60, 4, 4, 0))
    assert len(frames) == 5 and stamps == [0.25, 1.25, 2.0]
    assert [f"<{s:.1f} seconds>" for s in stamps] == [
        "<0.2 seconds>",
        "<1.2 seconds>",
        "<2.0 seconds>",
    ]
    with pytest.raises(ValueError, match="at least 13 frames"):
        ref2va.sample_video_condition_frames(_frames(12, 4, 4, 0))
    tok = FakeTokenizer()
    kinds = [("video", True), ("image", False), ("audio", True), ("video", False)]
    ids, tags = ref2va.ref2va_presentation(
        tok.tokenize, tok.token_id, "x", kinds, [4], [2, 3], [[0.25], [0.25, 1.25]]
    )
    text = "".join(chr(i) if i < 900000 else "|" for i in ids)
    assert text.startswith(
        "<Audio 1>: <Video 1>: <0.2 seconds>||||<Picture 1>: ||||||<Audio 2>: <Video 2>: "
    )
    assert text.endswith("<1.2 seconds>|||||x")
    assert all(t == 0 for i, t in zip(ids, tags, strict=True) if i >= 900000)
    assert all(t == 1 for i, t in zip(ids, tags, strict=True) if i < 900000)


def test_vae_frame_count_snaps_down():
    # The pinned pipeline's own bound, max(1, ...): below 22 frames it is 22, so a short
    # reference is not cut (13 frames is the shortest one the conditioner accepts).
    assert [ref2va.vae_video_frame_count(n) for n in (5, 13, 21, 22, 23, 38, 39, 124)] == [
        22,
        22,
        22,
        22,
        22,
        22,
        39,
        124,
    ]


# --- geometry and layout ------------------------------------------------------------------


def test_reference_blocks_pair_encoder_outputs_with_kinds():
    kinds = [("image", False), ("video", True), ("audio", True), ("video", False)]
    blocks = ref2va.reference_blocks(
        kinds, [(1, 24, 1, 8, 12), (1, 24, 7, 8, 14), (1, 24, 2, 4, 4)], [20, 10]
    )
    assert [b.to_dict() for b in blocks] == [
        {
            "kind": "image",
            "latent_frames": 1,
            "latent_height": 8,
            "latent_width": 12,
            "audio_rows": 0,
        },
        {
            "kind": "video",
            "latent_frames": 7,
            "latent_height": 8,
            "latent_width": 14,
            "audio_rows": 20,
        },
        {
            "kind": "audio",
            "latent_frames": 0,
            "latent_height": 0,
            "latent_width": 0,
            "audio_rows": 10,
        },
        {
            "kind": "video",
            "latent_frames": 2,
            "latent_height": 4,
            "latent_width": 4,
            "audio_rows": 0,
        },
    ]
    with pytest.raises(ValueError, match="more soundtracks"):
        ref2va.reference_blocks(
            kinds, [(1, 24, 1, 8, 12), (1, 24, 7, 8, 14), (1, 24, 2, 4, 4)], [20, 10, 4]
        )
    with pytest.raises(ValueError, match="1 latent frame"):
        ref2va.reference_blocks([("image", False)], [(1, 24, 2, 8, 8)], [])
    # Fewer encoder outputs than references that need them.
    with pytest.raises(ValueError, match="fewer visual latents"):
        ref2va.reference_blocks(kinds, [(1, 24, 1, 8, 12), (1, 24, 7, 8, 14)], [20, 10])
    with pytest.raises(ValueError, match="fewer soundtracks"):
        ref2va.reference_blocks(
            kinds, [(1, 24, 1, 8, 12), (1, 24, 7, 8, 14), (1, 24, 2, 4, 4)], [20]
        )


def test_clock_advance_per_reference_kind():
    blocks = [
        ref2va.ReferenceBlock("image", 1, 8, 12),
        ref2va.ReferenceBlock("audio", audio_rows=2 * 30),
        ref2va.ReferenceBlock("video", 17, 8, 14, audio_rows=2 * 50),
    ]
    layout = ref2va.build_ref2va_layout(
        torch.ones(5, dtype=torch.long), blocks, height=64, width=96, num_frames=124
    )
    times = layout.position_ids[:, 0]
    image_rows = 4 * 6
    assert bool((times[5 : 5 + image_rows] == 5.0).all())  # image at the text end, +1
    audio = slice(5 + image_rows, 5 + image_rows + 60)
    assert times[audio].tolist() == [6.0 + i for i in range(30)] * 2  # audio from 6, +30
    video_audio = slice(audio.stop, audio.stop + 100)
    assert float(times[video_audio][0]) == 36.0
    video_span = pl.video_span(17)
    assert video_span > 50  # the video span, not the soundtrack, advances the clock
    target = layout.video_indices[layout.num_condition_video_rows]
    assert float(times[target]) == 36.0 + video_span
    assert (
        layout.num_condition_audio_rows == 160
        and layout.num_condition_video_rows == 24 + 17 * 4 * 7
    )
    assert pl.structural_workflow(layout) == "ref2va"
    # The two span summations put the target 1 ulp apart for a 17-latent-frame reference.
    other = ref2va.build_ref2va_layout(
        torch.ones(5, dtype=torch.long),
        blocks,
        height=64,
        width=96,
        num_frames=124,
        span_summation="sequential",
    )
    delta = (other.position_ids - layout.position_ids)[:, 0]
    assert bool((delta[: video_audio.stop] == 0).all())
    assert 0 < float(delta.abs().max()) < 1e-13
    assert other.component_hashes()["position_ids"] != layout.component_hashes()["position_ids"]
    assert other.geometry["span_summation"] == "sequential"


def test_initial_rows_prepend_noised_visuals_and_clean_soundtracks():
    blocks = [ref2va.ReferenceBlock("image", 1, 4, 6), ref2va.ReferenceBlock("audio", audio_rows=8)]
    layout = ref2va.build_ref2va_layout(
        torch.ones(3, dtype=torch.long), blocks, height=64, width=96, num_frames=124
    )
    image = torch.full((1, 4, 1, 4, 6), 2.0)
    soundtrack = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    video, audio = ref2va.ref2va_initial_rows(
        layout,
        [image],
        [soundtrack],
        generator=torch.Generator().manual_seed(3),
        latent_channels=4,
        audio_latent_channels=8,
    )
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(image.shape, generator=g)
    target = torch.randn(1, 4, 37, 4, 6, generator=g)
    target_audio = torch.randn(2 * 207, 8, generator=g)
    t = torch.tensor(0.999)
    assert torch.equal(
        video,
        torch.cat(
            [
                pl.patchify_video_latents(t * image + (1 - t) * noise),
                pl.patchify_video_latents(target),
            ]
        ),
    )
    assert torch.equal(audio, torch.cat([soundtrack, target_audio]))
    with pytest.raises(ValueError, match="reference audio rows"):
        ref2va.ref2va_initial_rows(
            layout, [image], [], generator=None, latent_channels=4, audio_latent_channels=8
        )


# --- H-C7: repeated preprocessing ---------------------------------------------------------


def test_repeated_preprocessing_is_byte_identical():
    pytest.importorskip("torchaudio")  # ref2va_preprocess resamples a 44.1 kHz soundtrack
    first = ref2va_preprocess()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        second = ref2va_preprocess()
    finally:
        torch.set_num_threads(threads)
    assert pl.first_difference(first, second) is None and first == second


def test_repeated_preprocessing_in_a_fresh_process_is_byte_identical():
    pytest.importorskip("torchaudio")
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from tests.models.minimax_h3.test_h3_ref2va_pipeline import ref2va_preprocess;"
        "print(json.dumps(ref2va_preprocess()))"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "3", "PYTHONHASHSEED": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ref2va_preprocess()


def test_layout_and_rows_run_without_pillow():
    # Pillow is not a base dependency: the module, the layout and the initial rows must not
    # need it; normalising an image reference does.
    no_pillow = (
        "import sys; sys.path.insert(0, sys.argv[1]); sys.modules['PIL'] = None;"
        "import torch; from rl_engine.models.minimax_h3 import ref2va;"
    )
    rows = (
        "blocks = ref2va.reference_blocks([('image', False)], [(1, 24, 1, 4, 6)], []);"
        "layout = ref2va.build_ref2va_layout(torch.ones(5, dtype=torch.long), blocks,"
        " height=64, width=96, num_frames=124);"
        "ref2va.ref2va_initial_rows(layout, [torch.zeros(1, 24, 1, 4, 6)], [],"
        " generator=torch.Generator().manual_seed(0));"
        "print(layout.workflow)"
    )
    out = subprocess.run(
        [sys.executable, "-c", no_pillow + rows, str(REPO)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "ref2va"
    image = "ref2va.normalize_image_reference(torch.zeros(3, 64, 64))"
    failed = subprocess.run(
        [sys.executable, "-c", no_pillow + image, str(REPO)], capture_output=True, text=True
    )
    assert failed.returncode != 0 and "PIL" in failed.stderr.splitlines()[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_initial_rows_on_cuda_equal_cpu_bytes():
    pytest.importorskip("torchaudio")
    assert ref2va_preprocess(device="cuda") == ref2va_preprocess()


# --- H16: variant identity and the transformer_ref mapping --------------------------------


def _meta_state(variant: str) -> dict[str, torch.Tensor]:
    table = fp.expected_tensor_table(fp.transformer_config(variant))
    dtypes = {"F32": torch.float32, "BF16": torch.bfloat16}
    return {
        name: torch.empty(shape, dtype=dtypes[dtype], device="meta")
        for name, (dtype, shape) in table.items()
    }


def test_transformer_ref_maps_through_the_shared_tables():
    state = _meta_state("Ref2VA")
    assert len(state) == 638
    mapped = ref2va.map_transformer_ref(state)
    assert (mapped.variant, mapped.transformer_dir) == ("Ref2VA", "transformer_ref")
    assert len(mapped.state) == 638 - 2 * 52  # q/k/v fused per attention (50 blocks + 2 refiners)
    with pytest.raises(pl.VariantMismatch, match="not the transformer_ref"):
        ref2va.map_transformer_ref(state, source_variant="FL2VA")
    with pytest.raises(pl.VariantMismatch, match="not the transformer_ref"):
        ref2va.map_transformer_ref(state, source_variant="ref2va")  # a workflow name


def test_ref2va_layout_refuses_fl2va_pieces():
    blocks = [ref2va.ReferenceBlock("image", 1, 4, 6)]
    layout = ref2va.build_ref2va_layout(
        torch.ones(3, dtype=torch.long), blocks, height=64, width=96, num_frames=124
    )
    weights = ref2va.VariantState("FL2VA", "transformer", {})
    with pytest.raises(pl.VariantMismatch, match="FL2VA transformer"):
        ref2va.check_layout_weights(layout, weights)
    ref2va.check_layout_weights(layout, ref2va.VariantState("Ref2VA", "transformer_ref", {}))
    relabelled = pl.build_layout(
        workflow="fl2va",
        text_token_tags=torch.ones(3, dtype=torch.long),
        position_ids=layout.position_ids,
        video_indices=layout.video_indices,
        audio_indices=layout.audio_indices,
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows,
        geometry=layout.geometry,
    )
    with pytest.raises(pl.VariantMismatch, match="structure is a 'ref2va' layout"):
        pl.check_pipeline_identity(relabelled)
    assert pl.layout_artifact(layout)["pipeline_identity"]["transformer_dir"] == "transformer_ref"


# --- the pinned Diffusers pipeline --------------------------------------------------------


def _diffusers():
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
    from diffusers.modular_pipelines.minimax_h3 import (
        before_denoise,
        before_encoder,
        encoders,
        references,
    )

    return before_denoise, before_encoder, encoders, references


def _random_request(rng: random.Random):
    """Random reference list (kinds, sound, geometry) the released limits allow."""

    kinds = []
    for _ in range(rng.randint(1, 6)):
        kinds.append(rng.choice(["image", "video", "video", "audio"]))
    if set(kinds) == {"audio"}:
        kinds[0] = "image"
    blocks, shapes, audio = [], [], []
    for kind in kinds:
        if kind == "image":
            h, w = rng.choice([(128, 192), (96, 96), (128, 256)])
            shapes.append((1, 24, 1, h, w))
            blocks.append((kind, False))
        elif kind == "video":
            frames = rng.choice([7, 12, 17, 22, 37, 47])
            h, w = rng.choice([(48, 84), (84, 48), (48, 48)])
            shapes.append((1, 24, frames, h, w))
            sound = rng.random() < 0.6
            if sound:
                audio.append(2 * rng.choice([40, 93, 207, 400]))
            blocks.append((kind, sound))
        else:
            audio.append(2 * rng.choice([1, 40, 207, 600]))
            blocks.append((kind, True))
    return blocks, shapes, audio


def test_layout_matches_diffusers_static_builder_on_random_requests():
    before_denoise, _, _, references = _diffusers()
    build = before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
    summation = "neumaier" if sys.version_info >= (3, 12) else "sequential"
    rng = random.Random(0)
    differs = 0
    for case in range(120):
        kinds, shapes, audio = _random_request(rng)
        refs = []
        for kind, sound in kinds:
            if kind == "image":
                refs.append(references.MiniMaxH3ImageReference(image=None))
            elif kind == "video":
                refs.append(
                    references.MiniMaxH3VideoReference(
                        frames=None, fps=24.0, audio=torch.zeros(2, 1) if sound else None
                    )
                )
            else:
                refs.append(references.MiniMaxH3AudioReference(audio=torch.zeros(2, 1)))
        num_frames = rng.choice([121, 124, 200, 345])
        height, width = rng.choice([(768, 1344), (1344, 768), (768, 768)])
        tags = torch.ones(rng.randint(1, 400), dtype=torch.long)
        blocks = ref2va.reference_blocks(kinds, shapes, audio)
        ours = ref2va.build_ref2va_layout(
            tags,
            blocks,
            height=height,
            width=width,
            num_frames=num_frames,
            span_summation=summation,
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
        for name, ref in zip(
            pl.LAYOUT_TENSORS[:2] + ("video_indices", "audio_indices", "text_indices"),
            theirs[:5],
            strict=True,
        ):
            assert (
                torch.equal(getattr(ours, name), ref) and getattr(ours, name).dtype == ref.dtype
            ), (case, name)
        assert (ours.num_condition_video_rows, ours.num_condition_audio_rows) == theirs[5:]
        other = "sequential" if summation == "neumaier" else "neumaier"
        alt = ref2va.build_ref2va_layout(
            tags, blocks, height=height, width=width, num_frames=num_frames, span_summation=other
        )
        differs += not torch.equal(alt.position_ids, ours.position_ids)
    assert differs > 0  # the interpreter-dependent ulp shows up on realistic requests


def test_media_normalisation_matches_diffusers():
    _, before_encoder, encoders, references = _diffusers()
    setup = before_encoder.MiniMaxH3Ref2VASetupStep
    frames = _frames(50, 90, 160, 0)
    for fps, n in ((30.0, 124), (24.0, 30), (12.5, 124), (60.0, 40)):
        ours = ref2va.normalize_video_reference(
            frames, fps, n, short_edge=128, max_pixels=128 * 224
        )
        theirs = setup._normalize_video_condition(frames, fps, n, 32, 128, 128 * 224, 24.0)
        assert np.array_equal(ours, theirs), fps
    pytest.importorskip("torchaudio")
    for wave, rate in (
        (_wave(1, 44100 * 3, 0), 44100),
        (_wave(2, 48000 * 2, 1), 48000),
        (_wave(2, 32000, 2), 32000),
    ):
        ours = ref2va.normalize_audio_reference(wave, rate, max_duration=1.5)
        theirs = setup._normalize_audio_condition(wave, rate, 32000, 1.5)
        assert torch.equal(ours, theirs)
    text_step = encoders.MiniMaxH3Ref2VATextEncoderStep
    for n in (13, 24, 49, 120, 124):
        ours = ref2va.sample_video_condition_frames(
            frames[:n] if n <= 50 else np.repeat(frames, 3, axis=0)[:n]
        )
        theirs = text_step._sample_video_condition_frames(
            frames[:n] if n <= 50 else np.repeat(frames, 3, axis=0)[:n], 24.0, 2.0, 2
        )
        assert ours[1] == theirs[1] and all(
            np.array_equal(a, b) for a, b in zip(ours[0], theirs[0], strict=True)
        )
    tok = FakeTokenizer()
    refs = [
        references.MiniMaxH3VideoReference(frames=None, fps=24.0, audio=torch.zeros(2, 1)),
        references.MiniMaxH3ImageReference(image=None),
        references.MiniMaxH3AudioReference(audio=torch.zeros(2, 1)),
        references.MiniMaxH3VideoReference(frames=None, fps=24.0),
    ]
    stamps = [[0.25, 1.25], [0.25]]
    ours = ref2va.ref2va_presentation(
        tok.tokenize, tok.token_id, PROMPT, _kinds(refs), [7], [3, 5], stamps
    )
    theirs = text_step._build_presentation(tok, PROMPT, refs, [7], [3, 5], stamps)
    assert ours == theirs


def test_preprocessing_matches_the_tiny_diffusers_pipeline():
    from tests.models.minimax_h3._tiny_pipeline import capture_steps, load_tiny_pipeline

    pytest.importorskip("torchaudio")
    pipe = load_tiny_pipeline(reference_image_short_edge=96)
    _, _, _, references = _diffusers()
    rate = pipe.audio_sampling_rate
    refs = [
        references.MiniMaxH3ImageReference(image=_image(90, 60, 1)),
        references.MiniMaxH3VideoReference(
            frames=_frames(70, 45, 80, 2), fps=30.0, audio=_wave(1, 4410 * 3, 3), sample_rate=4410
        ),
        references.MiniMaxH3AudioReference(audio=_wave(2, rate * 7, 4), sample_rate=rate),
    ]
    with capture_steps() as captured:
        pipe(
            prompt=PROMPT,
            references=refs,
            num_frames=121,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(0),
            output="latents",
        )
    setup = captured["MiniMaxH3Ref2VASetupStep"]
    height, width, num_frames = ref2va.resolve_ref2va_plan(
        num_frames=121, height=None, width=None, short_edge=64, max_pixels=64 * 128
    )
    assert (height, width, num_frames) == (setup["height"], setup["width"], setup["num_frames"])
    normalized = ref2va.normalize_references(
        refs,
        num_frames=num_frames,
        audio_sample_rate=rate,
        image_short_edge=96,
        canvas_short_edge=64,
        canvas_max_pixels=64 * 128,
    )
    for mine, theirs in zip(normalized, setup["normalized_references"], strict=True):
        assert mine.kind == theirs.kind and mine.has_audio == theirs.has_audio
        if mine.kind == "image":
            assert np.array_equal(np.array(mine.image), np.array(theirs.image))
        if mine.kind == "video":
            assert np.array_equal(mine.frames, theirs.frames)
        if mine.has_audio:
            assert torch.equal(mine.audio, theirs.audio)

    processor, tokenizer = pipe.processor, pipe.tokenizer
    merge = processor.image_processor.merge_size**2
    images = [r.image for r in normalized if r.kind == "image"]
    image_counts = [
        int(g.prod()) // merge
        for g in processor.image_processor(images=images, return_tensors="pt")["image_grid_thw"]
    ]
    sampled = [
        ref2va.sample_video_condition_frames(
            r.frames, temporal_patch=processor.video_processor.temporal_patch_size
        )
        for r in normalized
        if r.kind == "video"
    ]
    grids = processor.video_processor(
        videos=[np.stack(f) for f, _ in sampled], do_sample_frames=False, return_tensors="pt"
    )["video_grid_thw"]
    video_counts = [int(g[1]) * int(g[2]) // merge for g in grids]
    _, tags = ref2va.ref2va_presentation(
        lambda s: tokenizer(s, add_special_tokens=False)["input_ids"],
        tokenizer.convert_tokens_to_ids,
        PROMPT,
        _kinds(normalized),
        image_counts,
        video_counts,
        [t for _, t in sampled],
    )
    assert torch.equal(
        torch.tensor(tags), captured["MiniMaxH3Ref2VATextEncoderStep"]["text_token_tags"]
    )

    encoded = captured["MiniMaxH3Ref2VAReferenceEncoderStep"]
    video_ref = next(r for r in normalized if r.kind == "video")
    assert encoded["condition_latents"][1].shape[2] == pl.video_latent_num_frames(
        ref2va.vae_video_frame_count(video_ref.frames.shape[0])
    )
    blocks = ref2va.reference_blocks(
        _kinds(normalized),
        [c.shape for c in encoded["condition_latents"]],
        [a.shape[0] for a in encoded["audio_condition_latents"]],
    )
    summation = "neumaier" if sys.version_info >= (3, 12) else "sequential"
    layout = ref2va.build_ref2va_layout(
        torch.tensor(tags),
        blocks,
        height=height,
        width=width,
        num_frames=num_frames,
        span_summation=summation,
    )
    ref = captured["MiniMaxH3Ref2VAPrepareLayoutStep"]
    for name in pl.LAYOUT_TENSORS:
        assert (
            torch.equal(getattr(layout, name), ref[name].cpu())
            and getattr(layout, name).dtype == ref[name].dtype
        ), name
    assert (layout.num_condition_video_rows, layout.num_condition_audio_rows) == (
        ref["num_condition_video_rows"],
        ref["num_condition_audio_rows"],
    )
    plan = captured["MiniMaxH3SetTimestepsStep"]
    for (a, b), (c, d) in zip(
        pl.row_timestep_plan(layout, plan["timesteps"].tolist(), plan["audio_timesteps"].tolist()),
        plan["row_timestep_plan"],
        strict=True,
    ):
        assert torch.equal(a, c.cpu()) and torch.equal(b, d.cpu())
    video, audio = ref2va.ref2va_initial_rows(
        layout,
        encoded["condition_latents"],
        encoded["audio_condition_latents"],
        generator=torch.Generator().manual_seed(0),
        latent_channels=pipe.vae_latent_channels,
        audio_latent_channels=pipe.audio_latent_channels,
    )
    final = captured["MiniMaxH3Ref2VAPrepareLatentsStep"]
    assert torch.equal(video, final["latents"]) and torch.equal(audio, final["audio_latents"])


# --- SGLang (prior art) -------------------------------------------------------------------


def test_layout_matches_sglang_ref2va_blocks_prefix():
    module = pytest.importorskip(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.packed_sequence"
    )
    summation = "neumaier" if sys.version_info >= (3, 12) else "sequential"
    rng = random.Random(1)
    for _ in range(60):
        kinds, shapes, audio = _random_request(rng)
        blocks = ref2va.reference_blocks(kinds, shapes, audio)
        layout = ref2va.build_ref2va_layout(
            torch.ones(33, dtype=torch.long),
            blocks,
            height=768,
            width=1344,
            num_frames=124,
            span_summation=summation,
        )
        g = layout.geometry
        wire = []
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
        packed = module.minimax_h3_packed_sequence_ref2va_blocks(
            text_len=33,
            latent_t=g["num_latent_frames"],
            latent_h=g["latent_height"],
            latent_w=g["latent_width"],
            audio_t=g["num_audio_latents"],
            ref_blocks=wire,
        )
        used = layout.sequence_length
        assert torch.equal(packed["img_position_ids"][:used], layout.position_ids)
        assert torch.equal(packed["token_tags"][:used], layout.token_tags)
        assert torch.equal(packed["img_pos"], layout.video_indices)
        assert torch.equal(packed["audio_pos"], layout.audio_indices)


def test_report_counts_a_dtype_change_as_a_difference(monkeypatch):
    """The evidence tool's SGLang count must see dtypes: a stand-in that returns our own
    layout with int32 indices is equal by value and not bitwise."""

    import importlib.util
    import types

    sys.path.insert(0, str(REPO / "tools/validation/models"))
    try:
        spec = importlib.util.spec_from_file_location(
            "h3_ref2va_report", REPO / "tools/validation/models/h3_ref2va_report.py"
        )
        assert spec is not None and spec.loader is not None
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
    finally:
        sys.path.remove(str(REPO / "tools/validation/models"))
    built: list = []
    build = ref2va.build_ref2va_layout
    monkeypatch.setattr(
        ref2va, "build_ref2va_layout", lambda *a, **k: built.append(build(*a, **k)) or built[-1]
    )

    def stand_in(index_dtype):
        def packed_sequence(**kw):
            ours = built[-1]  # the layout the report built just before calling SGLang
            return {
                "img_position_ids": ours.position_ids,
                "token_tags": ours.token_tags,
                "img_pos": ours.video_indices.to(index_dtype),
                "audio_pos": ours.audio_indices.to(index_dtype),
            }

        return types.SimpleNamespace(minimax_h3_packed_sequence_ref2va_blocks=packed_sequence)

    sglang = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3"
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", None)
    for index_dtype, equal in ((torch.long, 100), (torch.int32, 0)):
        fake = types.ModuleType(sglang)
        fake.packed_sequence = stand_in(index_dtype)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, sglang, fake)
        out = report.reference()["sglang_prefix"]
        assert (out["cases"], out["bitwise_equal"]) == (100, equal), index_dtype
