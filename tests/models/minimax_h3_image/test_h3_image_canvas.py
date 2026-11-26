# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_canvas_contract`` (RFC #738): canvas resolution and post-decode resize.

CPU only. Pins:

* the released canvas rule (minimum-area upscale, half-to-even rounding to 32, latent = /16)
  on a committed table, and against the pinned Diffusers ``get_image_canvas_size`` when it
  is installed;
* the same request always gives the same plan and plan digest (RFC acceptance focus);
* the post-decode resize equals the released decode step's ``VaeImageProcessor`` branch byte
  for byte (Diffusers-gated), and is the identity when the canvas equals the request;
* the I5 / I6 probe profiles change the canvas and are rejected as strict;
* invalid sizes, areas and images fail closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3_image import canvas as cv

# (height, width, min_generation_area) -> canvas (height, width)
TABLE = {
    (2048, 2048, cv.DEFAULT_MIN_GENERATION_AREA): (2048, 2048),
    (1024, 1024, cv.DEFAULT_MIN_GENERATION_AREA): (2048, 2048),  # upscaled to the minimum area
    (1080, 1920, cv.DEFAULT_MIN_GENERATION_AREA): (1536, 2720),
    (4096, 1024, cv.DEFAULT_MIN_GENERATION_AREA): (4096, 1024),  # already large enough
    (1000, 700, 1024): (992, 704),
    (80, 80, 1024): (64, 64),  # 2.5 rounds to 2: half to even
    (48, 48, 1024): (64, 64),  # 1.5 rounds to 2
    (16, 16, 1024): (32, 32),  # upscaled 2x to the 1024 floor
    (1, 5000, 1024): (32, 4992),  # at least one multiple per side
}


@pytest.mark.parametrize(("request_", "canvas"), sorted(TABLE.items()))
def test_released_canvas_table(request_, canvas) -> None:
    plan = cv.resolve_canvas(*request_)
    assert plan.canvas == canvas
    assert plan.latent == (canvas[0] // 16, canvas[1] // 16)
    assert plan.canvas[0] % 32 == 0 and plan.canvas[1] % 32 == 0
    assert plan.needs_resize == (plan.canvas != request_[:2])


def test_half_to_even_can_shrink_the_canvas() -> None:
    plan = cv.resolve_canvas(80, 80, 1024)
    assert plan.canvas == (64, 64) and plan.needs_resize
    assert plan.resize_rule() == "pil.lanczos(uint8, reducing_gap=None)"


def test_default_request_is_2048_square_without_resize() -> None:
    plan = cv.resolve_canvas()
    assert plan.requested == plan.canvas == (2048, 2048)
    assert plan.latent == (128, 128)
    assert plan.num_target_rows == 64 * 64
    assert plan.resize_rule() == "none"
    assert plan.artifact()["pillow"] is None


def test_same_request_same_plan_and_digest() -> None:
    a, b = cv.resolve_canvas(1000, 700, 1024), cv.resolve_canvas(1000, 700, 1024)
    assert a == b and a.artifact() == b.artifact()
    assert cv.resolve_canvas(1000, 701, 1024).artifact()["sha256"] != a.artifact()["sha256"]


def test_scale_is_recorded_exactly() -> None:
    plan = cv.resolve_canvas(1080, 1920)
    assert float.fromhex(plan.artifact()["scale_hex"]) == plan.scale
    assert plan.scale > 1.0


def test_evidence_block_has_the_trajectory_canvas_fields() -> None:
    block = cv.resolve_canvas(80, 80, 1024).evidence_block()
    assert set(block) >= {"requested", "canvas", "latent", "min_generation_area", "resize"}
    assert block["plan_sha256"] == cv.resolve_canvas(80, 80, 1024).artifact()["sha256"]


@pytest.mark.parametrize(
    ("profile", "canvas"),
    [("round16", (80, 80)), ("round64", (64, 64)), ("no_min_area", (64, 64))],
)
def test_probe_profiles_are_labelled_and_not_strict(profile: str, canvas) -> None:  # I5 / I6
    plan = cv.resolve_canvas(80, 80, 1024, profile=profile)
    assert plan.canvas == canvas
    with pytest.raises(ValueError, match="probe"):
        cv.check_strict(plan)


def test_no_min_area_probe_differs_on_small_requests() -> None:  # I6
    released = cv.resolve_canvas(1024, 1024)
    probe = cv.resolve_canvas(1024, 1024, profile="no_min_area")
    assert released.canvas == (2048, 2048) and probe.canvas == (1024, 1024)


def test_check_strict_accepts_released_and_rejects_edited_plans() -> None:
    plan = cv.resolve_canvas(1080, 1920)
    cv.check_strict(plan)
    edited = cv.CanvasPlan(
        plan.requested,
        plan.min_generation_area,
        32,
        plan.scale,
        (1536, 2688),
        (96, 168),
        "released",
    )
    with pytest.raises(ValueError, match="does not match"):
        cv.check_strict(edited)


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ((0, 64), "height"),
        ((64, -1), "width"),
        ((64.0, 64), "height"),
        ((True, 64), "height"),
        ((64, 64, 1023), "within"),
        ((64, 64, 4096 * 4096 + 1), "within"),
    ],
)
def test_invalid_requests_fail_closed(args, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        cv.resolve_canvas(*args)


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="profile"):
        cv.resolve_canvas(64, 64, profile="round8")


# --- resize -------------------------------------------------------------------------------


def _decoded(plan: cv.CanvasPlan, seed: int = 0, batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(batch, 3, *plan.canvas, generator=generator)
    return (pixels * 255).clamp(0, 255).to(torch.uint8)


def test_resize_is_identity_when_canvas_equals_request() -> None:
    plan = cv.resolve_canvas(4096, 1024)
    images = _decoded(plan, batch=1)
    assert cv.resize_output(images, plan) is images


def test_resize_returns_the_requested_uint8_size() -> None:
    pytest.importorskip("PIL")
    plan = cv.resolve_canvas(80, 120, 1024)
    out = cv.resize_output(_decoded(plan), plan)
    assert out.dtype == torch.uint8 and tuple(out.shape) == (2, 3, 80, 120)


def test_resize_is_per_image_and_deterministic() -> None:  # I-C1 analogue for the resize
    pytest.importorskip("PIL")
    plan = cv.resolve_canvas(80, 120, 1024)
    images = _decoded(plan)
    batched = cv.resize_output(images, plan)
    assert torch.equal(batched, cv.resize_output(images, plan))
    assert torch.equal(batched[1:], cv.resize_output(images[1:], plan))


@pytest.mark.parametrize(
    ("images", "needle"),
    [
        (torch.zeros(1, 3, 64, 64), "uint8"),
        (torch.zeros(1, 4, 64, 64, dtype=torch.uint8), "uint8"),
        (torch.zeros(1, 3, 64, 96, dtype=torch.uint8), "plan canvas"),
    ],
)
def test_resize_rejects_wrong_images(images: torch.Tensor, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        cv.resize_output(images, cv.resolve_canvas(80, 80, 1024))


# --- pinned Diffusers ---------------------------------------------------------------------


def test_canvas_matches_pinned_diffusers() -> None:
    module = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    for (h, w, area), canvas in TABLE.items():
        assert module.get_image_canvas_size(h, w, area, 32) == canvas
    import random

    rng = random.Random(0)
    for _ in range(5000):
        h, w = rng.randint(1, 6000), rng.randint(1, 6000)
        area = rng.randint(*cv.MIN_GENERATION_AREA_BOUNDS)
        assert cv.resolve_canvas(h, w, area).canvas == module.get_image_canvas_size(h, w, area, 32)


@pytest.mark.parametrize("request_", [(80, 80, 1024), (1000, 700, 1024), (333, 517, 1024 * 1024)])
def test_resize_matches_the_pinned_decode_step(request_) -> None:
    image_processor = pytest.importorskip("diffusers.image_processor")
    processor = image_processor.VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    plan = cv.resolve_canvas(*request_)
    images = _decoded(plan)
    # MiniMaxH3ImageDecodeStep after quantisation: float k / 255 -> PIL -> resize -> back.
    height, width = plan.requested
    image = images.float() / 255
    pil = processor.postprocess(image, output_type="pil")
    pil = [processor.resize(item, height, width) for item in pil]
    expected = processor.numpy_to_pt(processor.pil_to_numpy(pil))
    expected_u8 = (expected * 255).round().to(torch.uint8)
    assert torch.equal(cv.resize_output(images, plan), expected_u8)
