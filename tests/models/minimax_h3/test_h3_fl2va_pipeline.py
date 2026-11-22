# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``fl2va_pipeline_contract``: zero/first/last/first-last preprocessing and layout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

try:  # the keyframe tests need Pillow; the T2VA tests and the no-Pillow check do not
    from PIL import Image
except ImportError:
    Image = None
from rl_engine.models.minimax_h3 import fl2va
from rl_engine.models.minimax_h3 import packed_layout as pl
from rl_engine.models.minimax_h3.trajectory import tensor_digest

REPO = Path(__file__).resolve().parents[3]


class FakeTokenizer:
    """One id per character, a fixed id per special token: enough to check the tags."""

    SPECIAL = {"<|vision_start|>": 900001, "<|image_pad|>": 900002, "<|vision_end|>": 900003}

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def token_id(self, token: str) -> int:
        return self.SPECIAL[token]


def _pillow():
    """``PIL.Image``, or skip the calling test (a keyframe test) when Pillow is missing."""

    if Image is None:
        pytest.skip("keyframe tests need Pillow")
    return Image


def _image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return _pillow().fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def fl2va_preprocess(anchors: tuple[str, ...], *, device: str = "cpu") -> dict:
    """The whole FL2VA preprocessing chain on synthetic media; returns the sealed artifact."""

    first = _image(1280, 720, 1) if "first" in anchors else None
    last = _image(1000, 1000, 2) if "last" in anchors else None
    keyframes = fl2va.prepare_keyframes(first, last)
    tok = FakeTokenizer()
    # 768x1344 at Qwen3-VL's 32x32 merged patches: 24 * 42 tokens per keyframe.
    counts = [(keyframes.height // 32) * (keyframes.width // 32)] * len(keyframes.images)
    ids, tags = fl2va.fl2va_presentation(
        tok.tokenize, tok.token_id, "a cat on a skateboard", counts
    )
    layout = fl2va.build_fl2va_layout(
        torch.tensor(tags),
        height=keyframes.height,
        width=keyframes.width,
        num_frames=124,
        anchors=keyframes.anchors,
    )
    # Stand-in for visual_vae_contract: deterministic latents shaped like a keyframe's.
    g = torch.Generator().manual_seed(123)
    geometry = layout.geometry
    latents = [
        torch.randn(1, 24, 1, geometry["latent_height"], geometry["latent_width"], generator=g)
        for _ in keyframes.anchors
    ]
    video_rows, audio_rows = fl2va.fl2va_initial_rows(
        layout, latents, generator=torch.Generator().manual_seed(0), device=device
    )
    # The digests below do not see the device; the rows must be where they were asked for.
    assert video_rows.device.type == audio_rows.device.type == torch.device(device).type
    inputs = {
        f"keyframe_{i}_pixels": tensor_digest(fl2va.keyframe_pixels(k))
        for i, k in enumerate(keyframes.images)
    }
    inputs.update(
        token_ids=tensor_digest(torch.tensor(ids)),
        initial_video_rows=tensor_digest(video_rows),
        initial_audio_rows=tensor_digest(audio_rows),
    )
    return pl.layout_artifact(layout, inputs=inputs)


# --- the four keyframe signatures ---------------------------------------------------------


@pytest.mark.parametrize("anchors", fl2va.ANCHOR_SETS)
def test_layout_structure_per_anchor_set(anchors):
    tags = torch.tensor([1] * 7 + [0] * 5 + [1] * 4)
    layout = fl2va.build_fl2va_layout(tags, height=768, width=1344, num_frames=124, anchors=anchors)
    g = layout.geometry
    rows_per_frame = (g["latent_height"] // 2) * (g["latent_width"] // 2)
    assert (g["num_latent_frames"], g["latent_height"], g["latent_width"]) == (37, 48, 84)
    assert g["num_audio_latents"] == 207
    assert layout.num_condition_video_rows == len(anchors) * rows_per_frame
    assert layout.num_condition_audio_rows == 0
    assert layout.sequence_length == 16 + (len(anchors) + 37) * rows_per_frame + 2 * 207
    assert layout.workflow == ("fl2va" if anchors else "t2va")
    assert pl.structural_workflow(layout) == layout.workflow
    # keyframe rows sit right after the text, before the audio
    assert layout.video_indices[: layout.num_condition_video_rows].tolist() == list(
        range(16, 16 + layout.num_condition_video_rows)
    )
    times = layout.position_ids[:, 0]
    for i, anchor in enumerate(anchors):
        block = times[16 + i * rows_per_frame : 16 + (i + 1) * rows_per_frame]
        expected = 16.0 if anchor == "first" else 16.0 + pl.last_anchor_span_pairwise(37) - 5 / 3
        assert bool((block == expected).all())
    target = layout.video_indices[layout.num_condition_video_rows :]
    # The last keyframe sits on the last *pixel* frame: the last latent frame (4 pixel frames
    # since 37 % 5 != 1) starts 3 * 5/3 rotary units earlier.
    if "last" in anchors:
        last_latent = float(times[target[-1]])
        anchor = 16.0 + pl.last_anchor_span_pairwise(37) - 5 / 3
        assert anchor == pytest.approx(last_latent + 3 * 5 / 3, abs=1e-9)
    pl.check_pipeline_identity(layout, transformer_variant="FL2VA")


def test_anchor_signatures_outside_the_released_four_are_refused():
    tags = torch.tensor([1, 1])
    for anchors in (("last", "first"), ("first", "first"), ("middle",)):
        with pytest.raises(ValueError, match="anchors"):
            fl2va.build_fl2va_layout(tags, height=768, width=1344, num_frames=124, anchors=anchors)


def test_keyframes_resolve_canvas_and_resize_like_the_reference():
    first, last = _image(1280, 720, 1), _image(1000, 1000, 2)
    kf = fl2va.prepare_keyframes(first, last)
    assert (kf.height, kf.width, kf.anchors) == (768, 1344, ("first", "last"))
    stretched = first.resize((1344, 768), _pillow().Resampling.LANCZOS)
    assert np.array_equal(np.array(kf.images[0]), np.array(stretched))
    # follower: cover 1344x1344, crop rows (1344 - 768) // 2 = 288
    cover = last.resize((1344, 1344), _pillow().Resampling.LANCZOS).crop((0, 288, 1344, 288 + 768))
    assert np.array_equal(np.array(kf.images[1]), np.array(cover))
    only_last = fl2va.prepare_keyframes(None, last)
    assert (only_last.height, only_last.width, only_last.anchors) == (768, 768, ("last",))
    t2va = fl2va.prepare_keyframes()
    assert (t2va.height, t2va.width, t2va.anchors, t2va.images) == (768, 1344, (), ())
    assert fl2va.keyframe_pixels(kf.images[0]).shape == (1, 3, 1, 768, 1344)


def test_keyframe_pixels_refuse_images_that_are_not_three_channel():
    # The pinned pipeline packs np.array(image) unconverted: a gray or palette image has no
    # channel axis, an RGBA / LA one has 4 / 2 channels. All are refused, not converted.
    for mode in ("L", "P", "RGBA", "LA"):
        with pytest.raises(ValueError, match="3-channel"):
            fl2va.keyframe_pixels(_pillow().new(mode, (8, 4)))
    pixels = fl2va.keyframe_pixels(_pillow().new("RGB", (8, 4), (1, 2, 3)))
    assert pixels.shape == (1, 3, 1, 4, 8) and pixels.dtype == torch.uint8
    assert pixels[0, :, 0, 0, 0].tolist() == [1, 2, 3]


def test_presentation_tags_vision_blocks_as_video():
    tok = FakeTokenizer()
    ids, tags = fl2va.fl2va_presentation(tok.tokenize, tok.token_id, "go", [3, 2])
    label = len("<Picture 1>: ")
    assert tags[:label] == [1] * label
    assert ids[label] == 900001 and tags[label : label + 5] == [0] * 5
    assert ids.count(900002) == 5 and tags[-2:] == [1, 1]
    assert len(ids) == len(tags) == 2 * label + 5 + 4 + 2


def test_initial_rows_draw_conditions_then_video_then_audio():
    layout = fl2va.build_fl2va_layout(
        torch.tensor([1, 1]), height=64, width=96, num_frames=124, anchors=("first", "last")
    )
    latents = [torch.full((1, 4, 1, 4, 6), float(i)) for i in range(2)]
    video, audio = fl2va.fl2va_initial_rows(
        layout,
        latents,
        generator=torch.Generator().manual_seed(5),
        latent_channels=4,
        audio_latent_channels=8,
    )
    g = torch.Generator().manual_seed(5)
    noises = [torch.randn(latents[0].shape, generator=g) for _ in range(2)]
    target = torch.randn(1, 4, 37, 4, 6, generator=g)
    audio_noise = torch.randn(2 * 207, 8, generator=g)
    t = torch.tensor(0.999)
    expected = torch.cat(
        [
            pl.patchify_video_latents(t * x + (1 - t) * n)
            for x, n in zip(latents, noises, strict=True)
        ]
        + [pl.patchify_video_latents(target)]
    )
    assert torch.equal(video, expected) and torch.equal(audio, audio_noise)
    assert video.shape[0] == layout.video_indices.numel()
    with pytest.raises(ValueError, match="keyframe latents"):
        fl2va.fl2va_initial_rows(layout, latents[:1], generator=None)


# --- H-C7: repeated preprocessing ---------------------------------------------------------


@pytest.mark.parametrize("anchors", fl2va.ANCHOR_SETS)
def test_repeated_preprocessing_is_byte_identical(anchors):
    first = fl2va_preprocess(anchors)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        second = fl2va_preprocess(anchors)
    finally:
        torch.set_num_threads(threads)
    assert pl.first_difference(first, second) is None
    assert first == second


def test_repeated_preprocessing_in_a_fresh_process_is_byte_identical():
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from tests.models.minimax_h3.test_h3_fl2va_pipeline import fl2va, fl2va_preprocess;"
        "print(json.dumps([fl2va_preprocess(a) for a in fl2va.ANCHOR_SETS]))"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "3", "PYTHONHASHSEED": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, env=env, check=True
    )
    remote = json.loads(out.stdout.strip().splitlines()[-1])
    local = [fl2va_preprocess(a) for a in fl2va.ANCHOR_SETS]
    assert remote == local


def test_t2va_runs_without_pillow():
    # Pillow is not a base dependency: importing the module and the zero-keyframe path must
    # not need it; a keyframe does.
    no_pillow = (
        "import sys; sys.path.insert(0, sys.argv[1]); sys.modules['PIL'] = None;"
        "import torch; from rl_engine.models.minimax_h3 import fl2va;"
    )
    t2va = (
        "k = fl2va.prepare_keyframes(height=768, width=1344);"
        "layout = fl2va.build_fl2va_layout(torch.ones(5, dtype=torch.long), height=k.height,"
        " width=k.width, num_frames=124);"
        "fl2va.fl2va_initial_rows(layout, [], generator=torch.Generator().manual_seed(0));"
        "print(layout.workflow)"
    )
    out = subprocess.run(
        [sys.executable, "-c", no_pillow + t2va, str(REPO)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "t2va"
    keyframe = "fl2va.prepare_keyframes(object())"
    failed = subprocess.run(
        [sys.executable, "-c", no_pillow + keyframe, str(REPO)], capture_output=True, text=True
    )
    assert failed.returncode != 0 and "PIL" in failed.stderr.splitlines()[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_initial_rows_on_cuda_equal_cpu_bytes():
    """``device="cuda"`` with a CPU generator.

    ``randn_tensor`` draws on the generator's device, so every noise tensor here is a CPU draw
    moved to the GPU and only the condition blend runs there. This checks that the blend and
    the transfers keep the CPU bytes; it does not compare CUDA RNG with CPU RNG (a CUDA
    generator draws other numbers by design), nor VAE or transformer kernels (other rows).
    """

    cpu = fl2va_preprocess(("first", "last"))
    cuda = fl2va_preprocess(("first", "last"), device="cuda")
    assert cpu == cuda


# --- H16: variant identity ----------------------------------------------------------------


def test_fl2va_layout_refuses_ref2va_pieces():
    tags = torch.tensor([1, 1, 1])
    layout = fl2va.build_fl2va_layout(tags, height=64, width=96, num_frames=124, anchors=("first",))
    with pytest.raises(pl.VariantMismatch, match="Ref2VA transformer"):
        pl.check_pipeline_identity(layout, transformer_variant="Ref2VA")
    with pytest.raises(pl.VariantMismatch, match="not an FL2VA workflow"):
        fl2va.build_fl2va_layout(tags, height=64, width=96, num_frames=124, workflow="ref2va")
    mislabelled = fl2va.build_fl2va_layout(
        tags, height=64, width=96, num_frames=124, anchors=("first",), workflow="t2va"
    )
    with pytest.raises(pl.VariantMismatch, match="structure"):
        pl.check_pipeline_identity(mislabelled)
    artifact = pl.layout_artifact(layout)
    assert artifact["pipeline_identity"]["transformer_dir"] == "transformer"
    other = fl2va.build_fl2va_layout(tags, height=64, width=96, num_frames=124, anchors=("last",))
    with pytest.raises(pl.VariantMismatch, match="do not describe"):
        pl.check_pipeline_identity(other, artifact=artifact)


# --- the pinned Diffusers pipeline --------------------------------------------------------


def _diffusers_layout_steps():
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
    from diffusers.modular_pipelines.minimax_h3 import before_denoise, modular_pipeline

    return before_denoise, modular_pipeline


def test_layout_matches_diffusers_static_builder_sweep():
    before_denoise, mp = _diffusers_layout_steps()
    build = before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence
    cases = 0
    for anchors in fl2va.ANCHOR_SETS:
        for num_frames in (120, 124, 171, 220, 300, 345):
            for aspect in ((16, 9), (9, 16), (1, 1), (4, 1), (1, 4), (21, 9), (1280, 721)):
                height, width = mp.resolve_canvas_size(*aspect, 32, 768, 768 * 1344)
                assert pl.resolve_canvas_size(*aspect) == (height, width)
                for text in (1, 77, 300):
                    tags = torch.ones(text, dtype=torch.long)
                    tags[: text // 3] = 0
                    ours = fl2va.build_fl2va_layout(
                        tags, height=height, width=width, num_frames=num_frames, anchors=anchors
                    )
                    g = ours.geometry
                    assert g["num_frames"] == mp.align_num_frames(num_frames, 17, 5)
                    assert g["num_latent_frames"] == mp.video_latent_num_frames(
                        g["num_frames"], 17, 5
                    )
                    assert g["num_audio_latents"] == mp.audio_latent_num_frames(g["num_frames"])
                    ref = build(
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
                    names = (
                        "position_ids",
                        "token_tags",
                        "video_indices",
                        "audio_indices",
                        "text_indices",
                    )
                    for name, theirs in zip(names, ref[:5], strict=True):
                        mine = getattr(ours, name)
                        assert mine.dtype == theirs.dtype and torch.equal(mine, theirs), name
                    assert (ours.num_condition_video_rows, ours.num_condition_audio_rows) == ref[5:]
                    cases += 1
    assert cases == 4 * 6 * 7 * 3


def test_row_timestep_plan_matches_diffusers():
    before_denoise, _ = _diffusers_layout_steps()
    from diffusers import MiniMaxH3Scheduler

    layout = fl2va.build_fl2va_layout(
        torch.ones(9, dtype=torch.long),
        height=768,
        width=1344,
        num_frames=124,
        anchors=("first", "last"),
    )
    video, audio = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(50)
    audio.set_timesteps(50)
    ours = pl.row_timestep_plan(layout, video.timesteps.tolist(), audio.timesteps.tolist())
    for (t, a), (ts, idx) in zip(
        zip(video.timesteps, audio.timesteps, strict=True), ours, strict=True
    ):
        ref = before_denoise.MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout.video_indices,
            layout.audio_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
            layout.num_text_tokens,
            float(t),
            float(a),
            max(float(t), 0.999),
            1.0,
        )
        assert torch.equal(ts, ref[0]) and torch.equal(idx, ref[1])


@pytest.mark.parametrize("anchors", fl2va.ANCHOR_SETS)
def test_preprocessing_matches_the_tiny_diffusers_pipeline(anchors):
    from tests.models.minimax_h3._tiny_pipeline import capture_steps, load_tiny_pipeline

    pipe = load_tiny_pipeline()
    first = _image(80, 48, 1) if "first" in anchors else None
    last = _image(70, 70, 2) if "last" in anchors else None
    kwargs = {"image": first, "last_image": last} if anchors else {"height": 64, "width": 128}
    with capture_steps() as captured:
        pipe(
            prompt="a cat on a skateboard",
            num_frames=121,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(0),
            output="latents",
            **kwargs,
        )

    if anchors:
        mine = fl2va.prepare_keyframes(first, last, short_edge=64, max_pixels=64 * 128)
        theirs = captured["MiniMaxH3ResizeStep"]
        assert (mine.height, mine.width, mine.anchors) == (
            theirs["height"],
            theirs["width"],
            theirs["keyframe_anchors"],
        )
        for a, b in zip(mine.images, theirs["keyframes"], strict=True):
            assert np.array_equal(np.array(a), np.array(b))
        height, width = mine.height, mine.width
        tokenizer, processor = pipe.tokenizer, pipe.processor
        grids = processor.image_processor(images=list(mine.images), return_tensors="pt")[
            "image_grid_thw"
        ]
        counts = [int(g.prod()) // processor.image_processor.merge_size**2 for g in grids]
        ids, tags = fl2va.fl2va_presentation(
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"],
            tokenizer.convert_tokens_to_ids,
            "a cat on a skateboard",
            counts,
        )
        text_tags = torch.tensor(tags)
        assert torch.equal(text_tags, captured["MiniMaxH3FL2VATextEncoderStep"]["text_token_tags"])
    else:
        height, width = 64, 128
        ids = pipe.tokenizer("a cat on a skateboard", add_special_tokens=False)["input_ids"]
        text_tags = torch.ones(len(ids), dtype=torch.long)
        assert torch.equal(text_tags, captured["MiniMaxH3TextEncoderStep"]["text_token_tags"])

    layout = fl2va.build_fl2va_layout(
        text_tags, height=height, width=width, num_frames=121, anchors=anchors
    )
    ref = captured["MiniMaxH3PrepareLayoutStep"]
    for name in pl.LAYOUT_TENSORS:
        assert torch.equal(getattr(layout, name), ref[name].cpu()), name
        assert getattr(layout, name).dtype == ref[name].dtype, name
    assert layout.num_condition_video_rows == ref["num_condition_video_rows"]
    assert layout.num_condition_audio_rows == ref["num_condition_audio_rows"]

    plan = captured["MiniMaxH3SetTimestepsStep"]
    ours = pl.row_timestep_plan(
        layout, plan["timesteps"].tolist(), plan["audio_timesteps"].tolist()
    )
    for (a, b), (c, d) in zip(ours, plan["row_timestep_plan"], strict=True):
        assert torch.equal(a, c.cpu()) and torch.equal(b, d.cpu())

    if anchors:
        latents = captured["MiniMaxH3KeyframeVaeEncoderStep"]["condition_latents"]
        video, _ = fl2va.fl2va_initial_rows(
            layout,
            latents,
            generator=torch.Generator().manual_seed(0),
            latent_channels=pipe.vae_latent_channels,
            audio_latent_channels=pipe.audio_latent_channels,
        )
        assert torch.equal(
            video[: layout.num_condition_video_rows],
            captured["MiniMaxH3PrepareConditionLatentsStep"]["condition_rows"],
        )
        assert torch.equal(video, captured["MiniMaxH3FL2VAPrepareLatentsStep"]["latents"])


# --- SGLang (prior art) -------------------------------------------------------------------


def test_layout_matches_sglang_packed_sequence_prefix():
    module = pytest.importorskip(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.packed_sequence"
    )
    signature = {(): None, ("first",): [0], ("last",): [-1], ("first", "last"): [0, -1]}
    for anchors, indices in signature.items():
        for num_frames, (height, width) in (
            (124, (768, 1344)),
            (220, (1344, 768)),
            (345, (768, 768)),
        ):
            layout = fl2va.build_fl2va_layout(
                torch.ones(41, dtype=torch.long),
                height=height,
                width=width,
                num_frames=num_frames,
                anchors=anchors,
            )
            g = layout.geometry
            packed = module.minimax_h3_packed_sequence(
                text_len=41,
                latent_t=g["num_latent_frames"],
                latent_h=g["latent_height"],
                latent_w=g["latent_width"],
                audio_t=g["num_audio_latents"],
                include_keyframe_cond=bool(anchors),
                keyframe_frame_indices=indices,
                frame_count=g["num_frames"] if anchors else None,
            )
            used = layout.sequence_length
            # SGLang pads to the next multiple of 64 rows: exactly -used % 64 rows tagged -1.
            assert packed["seq_len"] == used + (-used % 64)
            assert torch.equal(packed["img_position_ids"][:used], layout.position_ids)
            assert bool((packed["token_tags"][used:] == -1).all())  # SGLang pads to 64 rows
            assert torch.equal(packed["token_tags"][:used], layout.token_tags)
            assert torch.equal(packed["img_pos"], layout.video_indices)
            assert torch.equal(packed["audio_pos"], layout.audio_indices)


def test_report_counts_a_dtype_change_as_a_difference(monkeypatch):
    """The evidence tool's SGLang count must see dtypes: a stand-in that returns our own
    layout with int32 indices is equal by value and not bitwise."""

    _pillow()  # the evidence tool draws keyframes with Pillow
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "h3_fl2va_report", REPO / "tools/validation/models/h3_fl2va_report.py"
    )
    assert spec is not None and spec.loader is not None
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    anchors_of = {None: (), (0,): ("first",), (-1,): ("last",), (0, -1): ("first", "last")}

    def stand_in(index_dtype):
        def packed_sequence(**kw):
            n = (kw["latent_t"] - 2) // 5
            ours = fl2va.build_fl2va_layout(
                torch.ones(kw["text_len"], dtype=torch.long),
                height=kw["latent_h"] * 16,
                width=kw["latent_w"] * 16,
                num_frames=17 * n + 5,
                anchors=anchors_of[tuple(kw["keyframe_frame_indices"] or ()) or None],
            )
            return {
                "img_position_ids": ours.position_ids,
                "token_tags": ours.token_tags,
                "img_pos": ours.video_indices.to(index_dtype),
                "audio_pos": ours.audio_indices.to(index_dtype),
            }

        return types.SimpleNamespace(minimax_h3_packed_sequence=packed_sequence)

    sglang = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3"
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", None)
    for index_dtype, equal in ((torch.long, 36), (torch.int32, 0)):
        fake = types.ModuleType(sglang)
        fake.packed_sequence = stand_in(index_dtype)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, sglang, fake)
        out = report.reference()["sglang_prefix"]
        assert (out["cases"], out["bitwise_equal"]) == (36, equal), index_dtype


def test_report_cases_have_the_canvas_their_name_says():
    _pillow()  # the evidence tool draws keyframes with Pillow
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "h3_fl2va_report", REPO / "tools/validation/models/h3_fl2va_report.py"
    )
    assert spec is not None and spec.loader is not None
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    canvas = {"landscape": (768, 1344), "portrait": (1344, 768)}
    for case in report.CASES:
        if case.endswith("/124"):
            g = report.preprocess(case)["geometry"]
            assert (g["height"], g["width"]) == canvas[case.split("/")[1]], case
