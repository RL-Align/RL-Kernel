# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``phase_b_image_full_chain`` (RFC #738): the H3-Image chain against the pinned pipeline.

Tiny random models (transformer, VAE) and fixed text features stand in for the released
weights; the released-weight run is ``tools/validation/models/h3_image_full_chain.py``. Pins:

* the reference resize equals ``MiniMaxH3ImageEditResizeStep`` (sizes, bytes, EXIF, errors);
* the sigma grid and per-row timestep table equal ``MiniMaxH3ImageSetTimestepsStep``;
* the chain on RL-Kernel contracts equals the pinned modular pipeline run block by block at
  every stage: T2I on CPU and CUDA, an upscaled canvas with the resize back, and edits with
  one and two references on CUDA (the image VAE contract pins CUDA arithmetic; on CPU the
  first drift is reported at ``reference_encode``, which the test checks);
* a one-ULP drift injected at step ``k`` is reported at ``denoise`` step ``k``; a written
  reference row is refused at the step it happens; artifacts verify and tampering fails.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3_image import full_chain as fc
from rl_engine.models.minimax_h3_image import reference_conditioning as rc

pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
sys.path.insert(0, str(Path(__file__).parent))

import test_h3_image_packed_layout as layout_tests  # noqa: E402
import test_h3_image_vae as vae_tests  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def _image(height: int, width: int, seed: int):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


_MODELS: dict[str, tuple] = {}


def _models(device: str):
    if device not in _MODELS:
        from diffusers import MiniMaxH3ImageTransformer2DModel

        config = dict(layout_tests.TINY)
        config["in_channels"] = 24
        torch.manual_seed(0)
        transformer = MiniMaxH3ImageTransformer2DModel(**config).eval().to(device)
        _MODELS[device] = (transformer, vae_tests._tiny_vae(device))
    return _MODELS[device]


def _scheduler():
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    return MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")


def _text(request: fc.ImageRequest, device: str, vision_rows: int = 3) -> fc.TextConditions:
    rows = 7 + (vision_rows * len(request.references))
    tags = torch.ones(rows, dtype=torch.long)
    start = 1
    for _ in request.references:
        tags[start : start + vision_rows] = 0
        start += vision_rows + 1
    generator = torch.Generator().manual_seed(3)
    embeds = torch.randn(1, rows, 8, generator=generator).to(device)
    return fc.TextConditions("fixed", request.prompt, list(range(100, 100 + rows)), embeds, tags)


def _run_both(request: fc.ImageRequest, device: str, **kwargs):
    transformer, vae = _models(device)
    from rl_engine.models.minimax_h3_image import image_vae as iv

    text = _text(request, device)
    ours = fc.run_chain(
        request,
        text,
        vae_contract=iv.ImageVAEContract(vae, require_release=False),
        denoiser=fc.DiffusersTransformerDenoiser(transformer),
        device=device,
        **kwargs,
    )
    pipe = fc.diffusers_pipeline(vae=vae, transformer=transformer, scheduler=_scheduler())
    theirs = fc.run_diffusers(request, text, pipe)
    return ours, theirs


T2I = fc.ImageRequest("a cat", 64, 96, seed=5, min_generation_area=1024)
UPSCALED = fc.ImageRequest("a cat", 40, 72, seed=5, min_generation_area=64 * 96)
EDIT1 = fc.ImageRequest(
    "edit <Picture 1>",
    64,
    64,
    seed=9,
    min_generation_area=1024,
    references=(_image(1100, 1030, 0),),
)
EDIT2 = fc.ImageRequest(
    "<Picture 1> and <Picture 2>",
    64,
    96,
    seed=9,
    min_generation_area=1024,
    references=(_image(1030, 1100, 1), _image(300, 1500, 2)),
)


# --- preprocessing and schedule ---------------------------------------------------------------


@pytest.mark.parametrize(
    "hw",
    [(1024, 1024), (1100, 1030), (300, 1500), (2500, 2100), (4000, 900), (32, 77), (1023, 2049)],
)
def test_reference_resize_matches_the_pinned_step(hw) -> None:
    pipe = fc.diffusers_pipeline()
    image = _image(*hw, seed=hw[0])
    request = fc.ImageRequest("x", references=(image,))
    (vae_image,), (vision_image,) = fc.diffusers_resize(pipe, request)
    ours = fc.prepare_reference(image)
    assert ours.vae_image.size == vae_image.size and ours.vision_image.size == vision_image.size
    assert fc.image_digest(ours.vae_image) == fc.image_digest(vae_image)
    assert fc.image_digest(ours.vision_image) == fc.image_digest(vision_image)
    if ours.vae_image.size != image.size:
        assert all(side % 32 == 0 for side in ours.vae_image.size)


def test_reference_resize_applies_exif_and_rgb() -> None:
    from PIL import Image

    image = _image(1200, 1040, seed=4).convert("RGBA")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    image.info["exif"] = exif.tobytes()
    image = Image.open(_roundtrip_png(image))
    ours = fc.prepare_reference(image)
    (vae_image,), _ = fc.diffusers_resize(
        fc.diffusers_pipeline(), fc.ImageRequest("x", references=(image,))
    )
    assert ours.vae_image.mode == "RGB"
    assert fc.image_digest(ours.vae_image) == fc.image_digest(vae_image)


def _roundtrip_png(image):
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", exif=image.info["exif"])
    buffer.seek(0)
    return buffer


def test_too_small_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        fc.prepare_reference(_image(31, 600, 0))


def test_schedule_and_row_timesteps_match_the_pinned_step() -> None:
    from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    from rl_engine.models.minimax_h3_image import packed_layout as pl

    pipe = fc.diffusers_pipeline(scheduler=_scheduler())
    tags = torch.ones(9, dtype=torch.long)
    tags[1:4] = 0
    layout = pl.build_layout(tags, [(4, 6), (2, 12)], (6, 8))
    state = PipelineState()
    for key in ("image_indices", "text_indices", "num_reference_rows"):
        state.set(key, getattr(layout, key))
    _, state = bd.MiniMaxH3ImageSetTimestepsStep()(pipe, state)
    from rl_engine.models.minimax_h3_image import pdd_schedule as ps

    ours = ps.released_schedule()
    assert torch.equal(ours.sigmas, pipe.scheduler.sigmas)
    assert torch.equal(ours.timesteps, state.get("timesteps"))
    for (u, ix), (du, dix) in zip(
        ps.row_timestep_plan(layout, ours), state.get("row_timestep_plan")
    ):
        assert torch.equal(u, du.cpu()) and torch.equal(ix, dix.cpu())
        assert float(u[-1]) == float(torch.tensor(rc.REFERENCE_NOISE_LEVEL, dtype=torch.float32))


# --- chain vs pinned pipeline -----------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("request_", [T2I, UPSCALED], ids=["t2i", "t2i-upscaled"])
def test_t2i_chain_equals_the_pinned_pipeline(request_, device) -> None:
    ours, theirs = _run_both(request_, device)
    for run in (ours, theirs):
        fc.verify(run.artifact)
    report = fc.compare(ours.artifact, theirs.artifact)
    assert report["gate"] and report["secondary_equal"], report["first_drift"]
    assert report["trajectory_first_drift"] is None
    assert torch.equal(ours.output, theirs.output)
    assert tuple(ours.output.shape[-2:]) == (request_.height, request_.width)


def test_upscaled_canvas_is_resized_back() -> None:
    ours, _ = _run_both(UPSCALED, "cpu")
    stages = {s["stage"]: s for s in ours.artifact["stages"]}
    assert stages["media_preprocess"]["tensors"]["canvas"] != [UPSCALED.height, UPSCALED.width]
    assert stages["resize"]["meta"]["resize"] != "none"


@CUDA
@pytest.mark.parametrize("request_", [EDIT1, EDIT2], ids=["edit-1ref", "edit-2refs"])
def test_edit_chain_equals_the_pinned_pipeline(request_) -> None:
    ours, theirs = _run_both(request_, "cuda")
    report = fc.compare(ours.artifact, theirs.artifact)
    assert report["gate"] and report["secondary_equal"], report["first_drift"]
    assert report["trajectory_first_drift"] is None
    inputs = ours.artifact["trajectory"]["inputs"]
    assert len(inputs["references"]) == len(request_.references)
    for step in ours.artifact["trajectory"]["step_records"]:
        assert (
            step["reference_rows_sha256"] == inputs["layout"]["conditioned_reference_rows_sha256"]
        )


def test_edit_on_cpu_drifts_first_at_reference_encode() -> None:
    # image_vae_contract pins the CUDA pixel arithmetic; Diffusers on CPU uses CPU arithmetic.
    ours, theirs = _run_both(EDIT1, "cpu")
    report = fc.compare(ours.artifact, theirs.artifact)
    assert not report["gate"]
    assert report["first_drift"] == {"stage": "reference_encode", "tensor": "latents"}
    assert report["trajectory_first_drift"]["stage"] == "inputs"


# --- drift localization and integrity --------------------------------------------------------


@pytest.mark.parametrize("step", [0, 3, 5])
def test_injected_drift_is_reported_at_its_step(step) -> None:
    def flip(i, rows):
        if i == step:
            rows = rows.clone()
            rows.view(torch.int32)[-1, 7] ^= 1
        return rows

    clean, _ = _run_both(T2I, "cpu")
    transformer, vae = _models("cpu")
    from rl_engine.models.minimax_h3_image import image_vae as iv

    drifted = fc.run_chain(
        T2I,
        _text(T2I, "cpu"),
        vae_contract=iv.ImageVAEContract(vae, require_release=False),
        denoiser=fc.DiffusersTransformerDenoiser(transformer),
        device="cpu",
        on_step=flip,
    )
    report = fc.compare(clean.artifact, drifted.artifact)
    assert report["first_drift"]["stage"] == "denoise"
    drift = report["trajectory_first_drift"]
    assert drift["step"] == step and drift["kind"] == "latent"


def test_written_reference_row_is_refused() -> None:
    def write(i, rows):
        if i == 2:
            rows = rows.clone()
            rows[0, 0] += 1.0
        return rows

    with pytest.raises(rc.ReferenceRowsChanged, match="step 2"):
        _run_both(EDIT1, "cpu", on_step=write)


def test_run_to_run_is_bitwise() -> None:
    a, _ = _run_both(T2I, "cpu")
    b, _ = _run_both(T2I, "cpu")
    assert a.artifact == b.artifact


def test_tampered_artifacts_fail_to_verify() -> None:
    ours, _ = _run_both(T2I, "cpu")
    for mutate in (
        lambda a: a["stages"][7]["tensors"].update(final_image_rows="0" * 64),
        lambda a: a["stages"].reverse(),
        lambda a: a["trajectory"]["step_records"][2].update(latent_sha256="0" * 64),
    ):
        bad = copy.deepcopy(ours.artifact)
        mutate(bad)
        with pytest.raises(ValueError):
            fc.verify(bad)


def test_stage_log_refuses_out_of_order_stages() -> None:
    log = fc.StageLog()
    log.add("request", {})
    with pytest.raises(ValueError, match="out of order"):
        log.add("denoise", {})


def test_text_for_another_prompt_is_refused() -> None:
    transformer, vae = _models("cpu")
    from rl_engine.models.minimax_h3_image import image_vae as iv

    with pytest.raises(ValueError, match="another prompt"):
        fc.run_chain(
            T2I,
            _text(fc.ImageRequest("a dog"), "cpu"),
            vae_contract=iv.ImageVAEContract(vae, require_release=False),
            denoiser=fc.DiffusersTransformerDenoiser(transformer),
            device="cpu",
        )


@CUDA
@pytest.mark.parametrize("request_", [T2I, EDIT2], ids=["t2i", "edit-2refs"])
def test_chain_with_the_triton_velocity_kernel_equals_the_pinned_pipeline(request_) -> None:
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp
    from rl_engine.models.minimax_h3_image import velocity_step as vs

    sampler = vs.ImageVelocitySampler(op=H3ImageVelocityStepTritonOp(), device="cuda")
    ours, theirs = _run_both(request_, "cuda", sampler=sampler)
    report = fc.compare(ours.artifact, theirs.artifact)
    assert report["gate"] and report["secondary_equal"], report["first_drift"]
    denoise = {s["stage"]: s for s in ours.artifact["stages"]}["denoise"]
    assert denoise["provenance"]["sampler"]["backend_id"] == "H3ImageVelocityStepTritonOp"
