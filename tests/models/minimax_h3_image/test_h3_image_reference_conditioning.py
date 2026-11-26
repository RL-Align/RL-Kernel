# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_reference_conditioning`` (RFC #738): noised reference rows and their draw order.

Pins:

* the mix is ``t * z + (1 - t) * noise`` with ``t = fp32(0.999)`` through the registered
  ``conditioning_noise_mix`` operator; a Python-float level (``1 - t`` in FP64) moves the bytes;
* each reference keeps its own grid; rows go in reference order; the layout must agree;
* the generator is consumed as posterior[0..N-1] -> target noise -> reference noise[0..N-1],
  and any other sequence (RFC #738 probe I12) fails closed;
* reference rows are byte-identical after every step of the released six-step loop, and the
  I13 / I14 arms (resampled or written reference rows) are caught at the step they happen;
* rows, target rows and the final generator state equal the pinned Diffusers edit blocks
  ``EditVaeEncoderStep -> PrepareLatentsStep -> EditPrepareLatentsStep`` (Diffusers-gated),
  on CPU and on CUDA with CPU and CUDA generators (CUDA-gated), with the PyTorch reference
  and with the CUDA ``conditioning_noise_mix`` kernel (extension-gated).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import reference_conditioning as rc

GRIDS = [(4, 4), (2, 6), (30, 30)]
RELEASE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724]
RELEASE_SIGMAS += [0.23200887441635132, 0.0]
# Reference image sizes: not whole 16 px cells, and one with an odd latent side.
IMAGE_HW = [(72, 104), (40, 200), (150, 150)]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _latents(grids=GRIDS, seed: int = 0, device="cpu") -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 24, h, w, generator=generator).to(device) for h, w in grids]


def _ledger_with_posteriors(n: int, seed: int = 0, device="cpu") -> rc.DrawLedger:
    ledger = rc.DrawLedger(torch.Generator(device).manual_seed(seed))
    for i in range(n):
        ledger.draw("posterior", i, (1, 24, 4, 4), device)
    return ledger


def _layout(grids=GRIDS, target=(6, 8)) -> pl.PackedLayout:
    tags = torch.ones(9, dtype=torch.long)
    tags[1:4] = 0
    return pl.build_layout(tags, grids, target)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


# --- mix -----------------------------------------------------------------------------------


def test_mix_is_the_fp32_tensor_level_form() -> None:
    (z,) = _latents([(8, 8)])
    noise = torch.randn(z.shape, generator=torch.Generator().manual_seed(9))
    out = rc.condition_references([z], noise=[noise])
    t = torch.tensor(0.999, dtype=torch.float32)
    expected = pp.ImagePatchPacker().pack_image(t * z + (1.0 - t) * noise)[0]
    assert torch.equal(_bits(out.rows), _bits(expected))
    # A Python-float level keeps 1 - t in FP64: a different (probe) contract.
    double = pp.ImagePatchPacker().pack_image(0.999 * z + (1.0 - 0.999) * noise)[0]
    assert not torch.equal(_bits(out.rows), _bits(double))
    art = out.artifact()
    assert art["level_f32_hex"] == "0x1.ff7cee0000000p-1"
    assert art["one_minus_level_f32_hex"] == "0x1.0624000000000p-10"
    assert float((1.0 - t)).hex() == "0x1.0624000000000p-10"  # 1.3e-5 rel. below FP64 1 - 0.999


def test_level_one_returns_the_latents_and_other_levels_move_rows() -> None:
    lat = _latents()
    noise = _latents(seed=5)
    clean = rc.condition_references(lat, noise=noise, level=1.0)
    assert torch.equal(clean.rows, pp.ImagePatchPacker().pack_references(lat)[0])
    released = rc.condition_references(lat, noise=noise).digests["rows"]
    for level in (1.0, float(1.0 - RELEASE_SIGMAS[1]), 0.99):  # I13: clean / schedule t / other
        assert rc.condition_references(lat, noise=noise, level=level).digests["rows"] != released


def test_mix_equals_the_registered_gold() -> None:
    from rl_engine.reference.conditioning.conditioning_noise_mix import NativeConditioningNoiseMixOp

    lat, noise = _latents(), _latents(seed=3)
    out = rc.condition_references(lat, noise=noise)
    gold = NativeConditioningNoiseMixOp()
    t = torch.tensor(rc.REFERENCE_NOISE_LEVEL, dtype=torch.float32)
    rows = [
        pp.ImagePatchPacker().pack_image(gold.forward_fp32(z, t, n))[0] for z, n in zip(lat, noise)
    ]
    assert torch.equal(out.rows, torch.cat(rows))
    assert out.provenance["mix"]["semantic_op"] == "conditioning_noise_mix"


# --- rows ----------------------------------------------------------------------------------


def test_each_reference_keeps_its_grid_and_order() -> None:
    lat, noise = _latents(), _latents(seed=1)
    out = rc.condition_references(lat, noise=noise, layout=_layout())
    assert out.rows_per_reference == (4, 3, 225)
    assert out.num_reference_rows == 232 == _layout().num_reference_rows
    assert out.latent_grids == tuple(GRIDS)
    start = 0
    for i, n in enumerate(out.rows_per_reference):
        alone = rc.condition_references([lat[i]], noise=[noise[i]])
        assert torch.equal(out.rows[start : start + n], alone.rows)
        start += n
    # Swapping two references (I9 / I10) changes the rows.
    swapped = rc.condition_references(
        [lat[1], lat[0], lat[2]], noise=[noise[1], noise[0], noise[2]]
    )
    assert swapped.digests["rows"] != out.digests["rows"]


def test_given_noise_equals_the_ledger_draw() -> None:
    lat = _latents()
    ledger = _ledger_with_posteriors(3)
    target = rc.draw_target_noise(ledger, (6, 8), "cpu")
    drawn = rc.condition_references(lat, ledger=ledger)
    replay = torch.Generator().manual_seed(0)
    for _ in range(3):
        torch.randn(1, 24, 4, 4, generator=replay)
    assert torch.equal(torch.randn(target.shape, generator=replay), target)
    noise = [torch.randn(z.shape, generator=replay) for z in lat]
    given = rc.condition_references(lat, noise=noise)
    assert torch.equal(drawn.rows, given.rows)
    assert drawn.digests == given.digests
    assert ledger.generator.get_state().equal(replay.get_state())


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"grids": [(4, 4), (2, 6)]}, "layout has 3 references"),
        ({"grids": [(4, 4), (6, 2), (30, 30)]}, "layout grid"),
    ],
)
def test_layout_disagreement_fails_closed(bad, match) -> None:
    lat = _latents(bad["grids"])
    with pytest.raises(ValueError, match=match):
        rc.condition_references(lat, noise=_latents(bad["grids"], seed=1), layout=_layout())


def test_layout_row_count_disagreement_fails_closed() -> None:
    layout = dataclasses.replace(_layout(), num_reference_rows=233)
    with pytest.raises(ValueError, match="reserved 233 reference rows"):
        rc.condition_references(_latents(), noise=_latents(seed=1), layout=layout)


@pytest.mark.parametrize(
    "latents, kwargs, err",
    [
        ([], {"noise": []}, ValueError),
        ([torch.zeros(1, 24, 4, 4)], {}, ValueError),  # neither ledger nor noise
        ([torch.zeros(1, 24, 4, 4)], {"noise": [], "ledger": None}, ValueError),
        ([torch.zeros(2, 24, 4, 4)], {"noise": [torch.zeros(2, 24, 4, 4)]}, ValueError),
        ([torch.zeros(1, 16, 4, 4)], {"noise": [torch.zeros(1, 16, 4, 4)]}, ValueError),
        ([torch.zeros(1, 24, 3, 4)], {"noise": [torch.zeros(1, 24, 3, 4)]}, ValueError),
        ([torch.zeros(1, 24, 4, 4, dtype=torch.bfloat16)], {"noise": [torch.zeros(4)]}, TypeError),
        ([torch.zeros(1, 24, 4, 4)], {"noise": [torch.zeros(24 * 16 - 1)]}, ValueError),
    ],
)
def test_unsupported_inputs_fail_closed(latents, kwargs, err) -> None:
    with pytest.raises(err):
        rc.condition_references(latents, **kwargs)


# --- draw order ------------------------------------------------------------------------------


def test_released_draw_order_passes() -> None:
    ledger = _ledger_with_posteriors(3)
    rc.draw_target_noise(ledger, (6, 8), "cpu")
    rc.condition_references(_latents(), ledger=ledger)
    ledger.check_order(3)
    art = ledger.artifact()
    assert [d["stage"] for d in art["draws"]] == ["posterior"] * 3 + ["target_noise"] + [
        "reference_noise"
    ] * 3
    assert art["draws"][3]["shape"] == [1, 24, 1, 6, 8]


def test_t2i_order_is_the_target_draw_alone() -> None:
    ledger = rc.DrawLedger(torch.Generator().manual_seed(0))
    rc.draw_target_noise(ledger, (4, 4), "cpu")
    ledger.check_order(0)


def test_reference_noise_before_target_fails_closed() -> None:  # I12
    ledger = _ledger_with_posteriors(3)
    rc.condition_references(_latents(), ledger=ledger)
    rc.draw_target_noise(ledger, (6, 8), "cpu")
    with pytest.raises(rc.DrawOrderError, match="released order"):
        ledger.check_order(3)


def test_reference_first_order_moves_every_row() -> None:  # I12, value side
    lat = _latents()
    released = _ledger_with_posteriors(3)
    rc.draw_target_noise(released, (6, 8), "cpu")
    a = rc.condition_references(lat, ledger=released)
    h3_video_order = _ledger_with_posteriors(3)
    b = rc.condition_references(lat, ledger=h3_video_order)
    assert a.digests["per_reference_rows"][0] != b.digests["per_reference_rows"][0]


def test_hidden_draw_between_stages_fails_closed() -> None:
    ledger = _ledger_with_posteriors(3)
    rc.draw_target_noise(ledger, (6, 8), "cpu")
    torch.randn(3, generator=ledger.generator)  # someone else consumed the generator
    rc.condition_references(_latents(), ledger=ledger)
    with pytest.raises(rc.DrawOrderError, match="does not start where"):
        ledger.check_order(3)


def test_a_draw_that_does_not_consume_the_generator_fails_closed() -> None:
    ledger = rc.DrawLedger(torch.Generator().manual_seed(0))
    ledger.observe("posterior", 0, lambda: torch.zeros(1, 24, 4, 4))  # posterior.mode()
    rc.draw_target_noise(ledger, (4, 4), "cpu")
    rc.condition_references(_latents([(4, 4)]), ledger=ledger)
    with pytest.raises(rc.DrawOrderError, match="did not consume"):
        ledger.check_order(1)


def test_missing_and_extra_draws_fail_closed() -> None:
    ledger = _ledger_with_posteriors(2)
    rc.draw_target_noise(ledger, (6, 8), "cpu")
    rc.condition_references(_latents(), ledger=ledger)
    with pytest.raises(rc.DrawOrderError):
        ledger.check_order(3)
    with pytest.raises(rc.DrawOrderError):
        ledger.check_order(2)


def test_external_records_join_the_chain() -> None:
    generator = torch.Generator().manual_seed(4)
    ledger = rc.DrawLedger(generator)
    before = rc.generator_state_digest(generator)
    torch.randn(1, 24, 4, 4, generator=generator)  # the image VAE contract's posterior draw
    ledger.record("posterior", 0, (1, 24, 4, 4), before, rc.generator_state_digest(generator))
    rc.draw_target_noise(ledger, (4, 4), "cpu")
    rc.condition_references(_latents([(4, 4)]), ledger=ledger)
    ledger.check_order(1)


@pytest.mark.parametrize("stage, index", [("noise", 0), ("posterior", -1), ("posterior", True)])
def test_unknown_stage_or_index_is_refused(stage, index) -> None:
    ledger = rc.DrawLedger(torch.Generator())
    with pytest.raises(ValueError):
        ledger.draw(stage, index, (2,), "cpu")


# --- fixed rows --------------------------------------------------------------------------------


def _conditioned() -> rc.ReferenceConditioning:
    return rc.condition_references(_latents(), noise=_latents(seed=1))


def test_check_fixed_accepts_identical_rows() -> None:
    cond = _conditioned()
    rows = cond.image_rows(torch.randn(12, 96))
    assert cond.check_fixed(rows, step=0) == cond.digests["rows"]
    assert cond.check_fixed(rows[None], step=0) == cond.digests["rows"]


@pytest.mark.parametrize("row", [0, 4, 231])
def test_check_fixed_catches_one_bit(row) -> None:
    cond = _conditioned()
    rows = cond.image_rows(torch.randn(12, 96)).clone()
    rows.view(torch.int32)[row, 17] ^= 1
    reference = sum(1 for n in (4, 7, 232) if n <= row)
    with pytest.raises(rc.ReferenceRowsChanged, match=f"row {row} \\(reference {reference}\\)"):
        cond.check_fixed(rows, step=2)


def test_check_fixed_compares_bits_not_values() -> None:
    cond = rc.condition_references([torch.zeros(1, 24, 2, 2)], noise=[torch.zeros(1, 24, 2, 2)])
    rows = cond.image_rows(torch.zeros(1, 96)).clone()
    rows[0, 0] = -0.0
    with pytest.raises(rc.ReferenceRowsChanged):
        cond.check_fixed(rows)
    rows = cond.image_rows(torch.zeros(1, 96)).clone()
    rows.view(torch.int32)[0, :] = 0x7FC00001
    cond.rows.view(torch.int32)[0, :] = 0x7FC00000
    with pytest.raises(rc.ReferenceRowsChanged):
        cond.check_fixed(rows)


def test_check_fixed_refuses_a_dtype_change_and_short_rows() -> None:
    cond = _conditioned()
    with pytest.raises(rc.ReferenceRowsChanged, match="bfloat16"):
        cond.check_fixed(cond.image_rows(torch.zeros(1, 96)).bfloat16())
    with pytest.raises(ValueError):
        cond.check_fixed(cond.rows[:-1])


def test_artifact_is_deterministic() -> None:
    def run() -> tuple[dict, dict]:
        ledger = _ledger_with_posteriors(3, seed=11)
        rc.draw_target_noise(ledger, (6, 8), "cpu")
        return rc.condition_references(_latents(), ledger=ledger).artifact(), ledger.artifact()

    assert run() == run()
    json.dumps(run())  # machine-readable


@CUDA
def test_cuda_rows_equal_cpu_rows_for_a_cpu_generator() -> None:
    def run(device: str) -> str:
        ledger = _ledger_with_posteriors(3, device="cpu")
        rc.draw_target_noise(ledger, (6, 8), device)
        return rc.condition_references(
            _latents(device=device), ledger=ledger, device=device
        ).digests["rows"]

    assert run("cuda") == run("cpu")


# --- pinned Diffusers ------------------------------------------------------------------------


class _StubVAE:
    """``encode`` -> Diffusers ``DiagonalGaussianDistribution`` of a fixed pixel projection.

    The network is not this row's subject (``image_vae_contract``); the posterior object, its
    ``sample(generator)`` and everything after it are the pinned Diffusers code.
    """

    spatial_compression_ratio = 16

    def __init__(self, device, on_sample=None) -> None:
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        stats = torch.Generator().manual_seed(2)
        self.config = SimpleNamespace(
            latents_mean=torch.randn(24, generator=stats).tolist(),
            latents_std=(torch.rand(24, generator=stats) + 0.5).tolist(),
        )
        self.weight = torch.randn(48, 3, generator=torch.Generator().manual_seed(3)).to(device)
        self.on_sample = on_sample
        self._dist = DiagonalGaussianDistribution

    def encode(self, pixels: torch.Tensor, return_dict: bool = False):
        pooled = torch.nn.functional.avg_pool2d(pixels, 16)
        moments = torch.einsum("oc,bchw->bohw", self.weight, pooled) * 0.5
        dist = self._dist(moments)
        if self.on_sample is not None:
            sample = dist.sample
            dist.sample = lambda generator=None: self.on_sample(lambda: sample(generator=generator))
        return (dist,)


def _components(device, on_sample=None):
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as P,
    )
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    return SimpleNamespace(
        _execution_device=torch.device(device),
        vae=_StubVAE(device, on_sample),
        vae_spatial_compression_ratio=16,
        vae_latent_channels=24,
        patch_size=(1, 2, 2),
        pixel_mean=P.pixel_mean.fget(None),
        pixel_std=P.pixel_std.fget(None),
        reference_noise_aug=P.reference_noise_aug.fget(None),
        scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity"),
    )


def _images():
    from PIL import Image

    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in IMAGE_HW]


TARGET_HW = (6, 8)


def _encode(device, generator, on_sample=None):
    """Steps 2-8 (``image_vae_contract``): the pinned VAE-encode block itself."""

    from diffusers.modular_pipelines.minimax_h3_image import encoders as enc
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    components = _components(device, on_sample)
    state = PipelineState()
    state.set("reference_images", _images())
    state.set("generator", generator)
    components, state = enc.MiniMaxH3ImageEditVaeEncoderStep()(components, state)
    return components, state


def _diffusers_rows(device, generator):
    from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd

    components, state = _encode(device, generator)
    refs = state.get("reference_latents")
    num_rows = sum((r.shape[-2] // 2) * (r.shape[-1] // 2) for r in refs)
    state.set("latent_height", TARGET_HW[0])
    state.set("latent_width", TARGET_HW[1])
    components, state = bd.MiniMaxH3ImagePrepareLatentsStep()(components, state)
    target_rows = state.get("latents").clone()
    state.set("num_reference_rows", num_rows)
    components, state = bd.MiniMaxH3ImageEditPrepareLatentsStep()(components, state)
    return components, refs, target_rows, state.get("latents"), num_rows


def _ours(device, generator):
    ledger = rc.DrawLedger(generator)
    counter = iter(range(len(IMAGE_HW)))
    _, state = _encode(
        device, generator, on_sample=lambda fn: ledger.observe("posterior", next(counter), fn)
    )
    refs = state.get("reference_latents")
    target = rc.draw_target_noise(ledger, TARGET_HW, device)
    target_rows = pp.ImagePatchPacker().pack_image(target)[0]
    grids = [tuple(r.shape[-2:]) for r in refs]
    layout = _layout(grids, TARGET_HW)
    cond = rc.condition_references(refs, ledger=ledger, layout=layout)
    return ledger, cond, target_rows


def _generator(kind: str, device: str) -> torch.Generator:
    return torch.Generator("cpu" if kind == "cpu" else device).manual_seed(1234)


DEVICES = [
    pytest.param("cpu", "cpu", id="cpu"),
    pytest.param("cuda", "cpu", id="cuda-cpu_generator", marks=CUDA),
    pytest.param("cuda", "cuda", id="cuda-cuda_generator", marks=CUDA),
]


@pytest.mark.parametrize("device, gen", DEVICES)
def test_rows_and_draws_match_the_pinned_edit_blocks(device, gen) -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    g_ref, g_ours = _generator(gen, device), _generator(gen, device)
    _, refs, target_ref, image_rows, num_rows = _diffusers_rows(device, g_ref)
    ledger, cond, target_ours = _ours(device, g_ours)
    ledger.check_order(len(IMAGE_HW))
    assert [tuple(r.shape[-2:]) for r in refs] == [(4, 6), (2, 12), (8, 8)]
    assert cond.num_reference_rows == num_rows == 6 + 6 + 16
    assert torch.equal(_bits(image_rows[:num_rows]), _bits(cond.rows))
    assert torch.equal(_bits(image_rows[num_rows:]), _bits(target_ours))
    assert torch.equal(_bits(target_ref), _bits(target_ours))
    assert g_ref.get_state().equal(g_ours.get_state())


def _loop(components, cond, image_rows, *, write_all=False, resample=None):
    from diffusers.modular_pipelines.minimax_h3_image.denoise import MiniMaxH3ImageLoopSchedulerStep

    sched = components.scheduler
    timesteps = torch.tensor([1.0 - s for s in RELEASE_SIGMAS[:-1]], dtype=torch.float32)
    sched.set_timesteps(
        sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device=image_rows.device
    )
    state = SimpleNamespace(
        latents=image_rows,
        noise_pred=None,
        num_reference_rows=0 if write_all else cond.num_reference_rows,
    )
    velocity = torch.Generator().manual_seed(77)
    digests = []
    for i, t in enumerate(sched.timesteps):
        state.noise_pred = torch.randn(1, *image_rows.shape, generator=velocity).to(
            image_rows.device
        )
        MiniMaxH3ImageLoopSchedulerStep()(components, state, i, t)
        if resample is not None and i == resample:
            fresh = rc.condition_references(
                [torch.zeros(1, 24, h, w, device=image_rows.device) for h, w in cond.latent_grids],
                noise=[torch.ones(1, 24, h, w) for h, w in cond.latent_grids],
            )
            state.latents[: cond.num_reference_rows] = fresh.rows
        digests.append(cond.check_fixed(state.latents, step=i))
    return digests


@pytest.mark.parametrize("device, gen", DEVICES)
def test_reference_rows_are_fixed_over_the_released_loop(device, gen) -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.denoise")
    components, _, _, image_rows, _ = _diffusers_rows(device, _generator(gen, device))
    _, cond, _ = _ours(device, _generator(gen, device))
    before_target = image_rows[cond.num_reference_rows :].clone()
    digests = _loop(components, cond, image_rows)
    assert len(digests) == 6 and set(digests) == {cond.digests["rows"]}
    assert not torch.equal(image_rows[cond.num_reference_rows :], before_target)


def test_writing_every_image_row_is_caught_at_step_zero() -> None:  # I14
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.denoise")
    components, _, _, image_rows, _ = _diffusers_rows("cpu", _generator("cpu", "cpu"))
    _, cond, _ = _ours("cpu", _generator("cpu", "cpu"))
    with pytest.raises(rc.ReferenceRowsChanged, match="step 0"):
        _loop(components, cond, image_rows, write_all=True)


def test_resampling_the_references_is_caught_at_that_step() -> None:  # I13
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.denoise")
    components, _, _, image_rows, _ = _diffusers_rows("cpu", _generator("cpu", "cpu"))
    _, cond, _ = _ours("cpu", _generator("cpu", "cpu"))
    with pytest.raises(rc.ReferenceRowsChanged, match="step 3"):
        _loop(components, cond, image_rows, resample=3)


def test_pinned_constants_and_source_lines() -> None:
    bd = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    from diffusers.modular_pipelines.minimax_h3_image import encoders as enc
    from diffusers.modular_pipelines.minimax_h3_image import modular_blocks_minimax_h3_image as mb
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as P,
    )

    assert P.reference_noise_aug.fget(None) == rc.REFERENCE_NOISE_LEVEL
    mix = inspect.getsource(bd.MiniMaxH3ImageEditPrepareLatentsStep.__call__)
    for line in (
        "randn_tensor(latents.shape, generator=block_state.generator, device=device, "
        "dtype=torch.float32)",
        "components.scheduler.scale_noise(latents, "
        "torch.tensor(components.reference_noise_aug), noise)",
        "block_state.latents = torch.cat([condition_rows, block_state.latents])",
    ):
        assert line in mix, line
    assert "posterior.sample(generator=block_state.generator)" in inspect.getsource(
        enc.MiniMaxH3ImageEditVaeEncoderStep.__call__
    )
    # The draw order comes from the block order: VAE encode, target latents, reference noise.
    names = [b.__name__ for b in mb.MiniMaxH3ImageEditCoreDenoiseStep.block_classes]
    assert names.index("MiniMaxH3ImagePrepareLatentsStep") < names.index(
        "MiniMaxH3ImageEditPrepareLatentsStep"
    )
    top = [b.__name__ for b in mb.MiniMaxH3ImageBlocks.block_classes]
    assert top.index("MiniMaxH3ImageAutoVaeEncoderStep") < top.index(
        "MiniMaxH3ImageAutoDenoiseStep"
    )


# --- CUDA kernel ----------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("gen", ["cpu", "cuda"])
def test_cuda_kernel_matches_the_reference_op(gen) -> None:
    try:
        from rl_engine.backends.cuda.conditioning.conditioning_noise_mix import (
            ConditioningNoiseMixCudaOp,
        )

        op = ConditioningNoiseMixCudaOp()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    lat = _latents([(4, 6), (2, 12), (8, 8), (62, 98)], device="cuda")
    generator = _generator(gen, "cuda")
    noise = [rc.draw_normal(z.shape, generator, "cuda") for z in lat]
    a = rc.condition_references(lat, noise=noise)
    b = rc.condition_references(lat, noise=noise, mix_op=op)
    assert torch.equal(_bits(a.rows), _bits(b.rows))
    assert b.provenance["mix"]["backend_id"] == "ConditioningNoiseMixCudaOp"
