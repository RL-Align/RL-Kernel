# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #738 ``image_vae_contract`` and ``image_vae_tiling_contract``.

The identity, geometry, tiling, normalization and output-conversion tests are
self-contained. Tests that build a small random ``AutoencoderKLMiniMaxH3Image`` skip when
Diffusers with MiniMax-H3-Image (commit 875373ac or later) is not importable; the ones that
compare with the Diffusers pipeline blocks also need CUDA, because the release normalizes
pixels with CUDA division semantics. Real-checkpoint tests read the released
``vae_diffusers/`` folder from ``$RL_KERNEL_H3_IMAGE_VAE_DIR`` and skip when it is unset.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3 import visual_vae as vv
from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import image_vae as iv
from tools.validation.models import h3_image_vae_evidence as evidence

VAE_DIR_ENV = "RL_KERNEL_H3_IMAGE_VAE_DIR"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


# --- identity -----------------------------------------------------------------------------


def test_constants_follow_the_fingerprint() -> None:
    config = fp.transformer_config()
    assert iv.PATCH_SIZE == config["patch_size"][1] == config["patch_size"][2] == 2
    assert iv.LATENT_CHANNELS == config["in_channels"] == iv.release_config()["latent_channels"]
    assert iv.SPATIAL_FACTOR * iv.PATCH_SIZE == iv.CANVAS_MULTIPLE == 32
    facts = fp.vae_facts()
    assert (facts["tile_size"], facts["tile_overlap_min"]) == (iv.TILE_SIZE, iv.TILE_MIN_OVERLAP)
    assert facts["pixel_mean"] == list(iv.PIXEL_MEAN) == list(vv.PIXEL_MEAN)
    assert facts["pixel_std"] == list(iv.PIXEL_STD) == list(vv.PIXEL_STD)


def test_release_identity() -> None:
    identity = iv.release_identity()
    assert identity["repo"] == "MiniMaxAI/MiniMax-H3-Image"
    assert identity["subfolder"] == "vae_diffusers"
    assert identity["reference"] == {
        "library": "diffusers",
        "commit": "875373ac7f5856071e345a3e394c74efccc6b9e6",
        "class": "AutoencoderKLMiniMaxH3Image",
    }
    assert identity["tensor_table"]["tensors"] == 511
    weights = [f for f in identity["files"] if f["path"].endswith(".safetensors")]
    assert weights[0]["sha256"] == (
        "5bb2809698c3989033f719ece2243f277fefedef1dd9a1d05f3d4136b2f93b35"
    )
    # A caller cannot change the pinned identity through the returned copy.
    identity["files"].clear()
    identity["tensor_table"]["tensors"] = 0
    assert iv.release_identity()["tensor_table"]["tensors"] == 511


def test_release_latent_stats() -> None:
    mean, std = iv.release_latent_stats()
    config = iv.release_config()
    assert mean.dtype == std.dtype == torch.float32 and mean.shape == (24,)
    assert mean.tolist() == config["latents_mean"] and std.tolist() == config["latents_std"]
    assert bool((std > 0).all())


def test_manifest_matches_golden_file() -> None:
    assert evidence.MANIFEST_PATH.read_text() == evidence.manifest_text()


def _fake_checkpoint(tmp_path, *, dtype: str = "F32", extra: dict | None = None):
    """A folder with the pinned config and a tiny safetensors stand-in (wrong size)."""

    root = tmp_path / "vae_diffusers"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({**iv.release_config(), **(extra or {})}))
    header = {"w": {"dtype": dtype, "shape": [2], "data_offsets": [0, 8 if dtype == "F32" else 4]}}
    raw = json.dumps(header).encode()
    (root / "diffusion_pytorch_model.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + b"\0" * 8
    )
    return root


def test_checkpoint_dir_checks_fail_closed(tmp_path) -> None:
    root = _fake_checkpoint(tmp_path, extra={"latent_channels": 16})
    with pytest.raises(ValueError, match="config.json differs"):
        iv.check_checkpoint_dir(root)
    root = _fake_checkpoint(tmp_path / "b")
    with pytest.raises(ValueError, match="size differs"):
        iv.check_checkpoint_dir(root)


def test_tensor_table_check_fails_closed() -> None:
    with pytest.raises(ValueError, match="differs from the pinned table"):
        iv.check_tensor_table({"w": ("F32", (2,))})


# --- geometry -----------------------------------------------------------------------------


def test_canvas_and_cell_checks() -> None:
    iv.check_canvas(2048, 2048)
    iv.check_canvas(1536, 2688)
    for bad in ((2048, 2040), (1040, 1024), (0, 32), (-32, 32)):
        with pytest.raises(ValueError):
            iv.check_canvas(*bad)
    iv.check_cells(1040, 1072)
    with pytest.raises(ValueError):
        iv.check_cells(1100, 1248)


def test_reference_crops_are_top_left() -> None:
    pixels = torch.arange(3 * 37 * 50, dtype=torch.int64).view(1, 3, 37, 50)
    cropped = iv.crop_to_cells(pixels)
    assert cropped.shape == (1, 3, 32, 48)
    assert torch.equal(cropped, pixels[..., :32, :48])
    latents = torch.randn(1, 24, 65, 67)
    assert iv.crop_to_patches(latents).shape == (1, 24, 64, 66)
    with pytest.raises(ValueError):
        iv.crop_to_cells(torch.zeros(1, 3, 15, 64))
    with pytest.raises(ValueError):
        iv.crop_to_patches(torch.zeros(1, 24, 1, 8))


# --- tiling -------------------------------------------------------------------------------


def test_tile_plans_are_the_h3_plans() -> None:
    """Diffusers marks the image VAE's _split_tiles "Copied from" the H3 VAE: same digests."""

    for h, w in ((256, 256), (1024, 1024), (2048, 2048), (1536, 2688), (1088, 1248)):
        assert iv.plan_tiles(h, w).digest() == vv.plan_tiles(h, w).digest()


def test_release_canvas_tile_plan_golden() -> None:
    plan = iv.plan_tiles(2048, 2048)
    assert plan.tiles == 121
    # 128 px of slack spread round-robin in 16 px steps over 10 overlaps of 64 px.
    assert plan.rows.starts == (0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1600, 1792)
    assert plan.rows.overlaps == (80,) * 8 + (64, 64)
    assert plan.cols == plan.rows
    assert sum(plan.rows.sizes) - sum(plan.rows.overlaps) == 2048
    assert min(plan.rows.overlaps) >= 64 and all(o % 16 == 0 for o in plan.rows.overlaps)
    assert plan.covers_exactly()
    assert iv.plan_tiles(1024, 1024).tiles == 25
    assert iv.plan_tiles(256, 256).tiles == 1  # one tile: tiled equals untiled bytes here


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_overlap_height": 0},
        {"min_overlap_width": 0},
        {"tile_height": 250},
        {"min_overlap_width": 60},
        {"tile_width": 256, "min_overlap_width": 256},
    ],
)
def test_unsupported_tile_geometry_fails_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        iv.plan_tiles(1024, 1024, **kwargs)


def test_off_grid_images_fail_closed() -> None:
    with pytest.raises(ValueError):
        iv.plan_tiles(1100, 1024)


def test_tiled_and_untiled_identities_differ() -> None:
    plan = iv.plan_tiles(1024, 1024)
    tiled = iv.tiling_identity(plan, iv.RELEASE_TILE_GEOMETRY)
    untiled = iv.tiling_identity(None, None)
    assert tiled["mode"] == "tiled-256x256-o64x64" and untiled["mode"] == "untiled"
    assert tiled["plan_sha256"] == plan.digest()
    assert tiled["latent_overlaps"] == [[4, 4, 4, 4], [4, 4, 4, 4]]
    # The same plan placed with other overlap minima is another identity.
    other = iv.plan_tiles(1024, 1024, min_overlap_height=96, min_overlap_width=96)
    assert iv.tiling_identity(other, (256, 256, 96, 96))["plan_sha256"] != tiled["plan_sha256"]


# --- pixel normalization and output conversion --------------------------------------------


def _release_cuda_semantics(pixels: torch.Tensor) -> torch.Tensor:
    """``(x.float().div(255.0) - mean) / std`` as CUDA computes it (multiply by fl32(1/255))."""

    inv255 = torch.tensor(1.0) / torch.tensor(255.0)
    mean = torch.tensor(iv.PIXEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(iv.PIXEL_STD).view(1, 3, 1, 1)
    return (pixels.float() * inv255 - mean) / std


def test_normalize_pixels_is_the_release_formula() -> None:
    gen = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (2, 3, 17, 23), generator=gen, dtype=torch.uint8)
    out = iv.normalize_pixels(pixels)
    assert out.dtype == torch.float32 and out.shape == pixels.shape and out.is_contiguous()
    assert torch.equal(out, _release_cuda_semantics(pixels))
    # The table is the MiniMax-H3 one, reused as is.
    assert iv.pixel_lut_digest() == vv.pixel_lut_digest()


@needs_cuda
def test_normalize_pixels_matches_cuda_division() -> None:
    values = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 256).expand(1, 3, 1, 256)
    mean = torch.tensor(iv.PIXEL_MEAN, device="cuda").view(1, -1, 1, 1)
    std = torch.tensor(iv.PIXEL_STD, device="cuda").view(1, -1, 1, 1)
    release = (values.cuda().to(torch.float32).div(255.0) - mean) / std
    assert torch.equal(iv.normalize_pixels(values.cuda()), release)
    assert torch.equal(iv.normalize_pixels(values), release.cpu())


def test_normalize_pixels_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        iv.normalize_pixels(torch.zeros(1, 3, 4, 4))
    with pytest.raises(ValueError):
        iv.normalize_pixels(torch.zeros(1, 3, 1, 4, 4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        iv.normalize_pixels(torch.zeros(1, 4, 4, 4, dtype=torch.uint8))


def test_output_conversion_truncates() -> None:
    """I18: the release truncates to uint8; rounding gives different bytes."""

    unit = torch.tensor([0.0, 0.999 / 255, 1.999 / 255, 254.9 / 255, 1.0, 1.5, -0.2])
    unit = unit.view(1, 1, 1, -1).expand(1, 3, 1, -1).contiguous()
    q = iv.quantize_uint8(unit)
    assert q[0, 0, 0].tolist() == [0, 0, 1, 254, 255, 255, 0]
    rounded = (unit * 255).round().clamp(0, 255).to(torch.uint8)
    assert not torch.equal(q, rounded)
    with pytest.raises(TypeError):
        iv.quantize_uint8(unit.double())


def test_denormalize_pixels_formula() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, 8, 8, generator=gen) * 3
    mean = torch.tensor(iv.PIXEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(iv.PIXEL_STD).view(1, 3, 1, 1)
    assert torch.equal(iv.denormalize_pixels(x), (x * std + mean).clamp(0, 1))
    assert torch.equal(iv.denormalize_pixels(x.half()), (x.half().float() * std + mean).clamp(0, 1))
    # Round trip of the pixel normalization for every uint8 value.
    values = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 256).expand(1, 3, 1, 256)
    back = iv.quantize_uint8(iv.denormalize_pixels(iv.normalize_pixels(values)))
    assert int((back.long() - values.long()).abs().max()) <= 1


def test_processor_rounding_reproduces_truncated_bytes() -> None:
    """The pipeline passes q / 255 to VaeImageProcessor, which rounds x * 255 back: q again."""

    q = torch.arange(256, dtype=torch.uint8)
    as_float = (q.float() / 255).numpy()
    assert np.array_equal((as_float * 255).round().astype("uint8"), q.numpy())


# --- latent normalization -----------------------------------------------------------------


def test_latent_normalization_formula_and_op_order() -> None:
    mean, std = iv.release_latent_stats()
    gen = torch.Generator().manual_seed(4)
    z = torch.randn(2, 24, 6, 8, generator=gen) * 2
    m, s = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    assert torch.equal(iv.normalize_latents(z, mean, std), (z - m) / s)
    assert torch.equal(iv.denormalize_latents(z, mean, std), z * s + m)
    # I15: scalar scaling or no normalization is a different transformer input.
    assert not torch.equal(iv.normalize_latents(z, mean, std), (z - mean.mean()) / std.mean())


def test_latent_normalization_rejects_bad_inputs() -> None:
    mean, std = iv.release_latent_stats()
    with pytest.raises(ValueError):
        iv.normalize_latents(torch.zeros(1, 24, 1, 4, 4), mean, std)
    with pytest.raises(ValueError):
        iv.normalize_latents(torch.zeros(1, 16, 4, 4), mean, std)
    with pytest.raises(TypeError):
        iv.normalize_latents(torch.zeros(1, 24, 4, 4, dtype=torch.int32), mean, std)
    with pytest.raises(TypeError):
        iv.normalize_latents(torch.zeros(1, 24, 4, 4), mean.double(), std)
    with pytest.raises(ValueError):
        iv.normalize_latents(torch.zeros(1, 24, 4, 4), mean.requires_grad_(), std)


@needs_cuda
def test_latent_normalization_cpu_cuda_bitwise() -> None:
    mean, std = iv.release_latent_stats()
    z = torch.randn(1, 24, 32, 48, generator=torch.Generator().manual_seed(5)) * 3
    for fn in (iv.normalize_latents, iv.denormalize_latents):
        assert torch.equal(fn(z, mean, std), fn(z.cuda(), mean, std).cpu())


def test_reference_fp16_round_trip() -> None:
    """I11: the release rounds the sampled reference latent to FP16 and back."""

    z = torch.randn(1, 24, 8, 8, generator=torch.Generator().manual_seed(6))
    r = iv.round_reference_latents(z)
    assert r.dtype == torch.float32
    assert torch.equal(r, z.half().float())
    assert not torch.equal(r, z)
    assert torch.equal(iv.round_reference_latents(r), r)


def test_reference_recipe_is_rfc_steps_2_to_8() -> None:
    steps = iv.reference_recipe()
    assert len(steps) == 7
    assert "VAE cells" in steps[0] and "float16" in steps[4] and "patches" in steps[5]


# --- encode/decode with a small Diffusers model -------------------------------------------


def _diffusers():
    return pytest.importorskip(
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image",
        reason="needs diffusers with MiniMax-H3-Image (875373ac+)",
    )


def _tiny_vae(device: str, tile: bool = True):
    module = _diffusers()
    mean, std = iv.release_latent_stats()
    torch.manual_seed(0)
    vae = module.AutoencoderKLMiniMaxH3Image(
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        latents_mean=tuple(mean.tolist()),
        latents_std=tuple(std.tolist()),
    )
    with torch.no_grad():
        gen = torch.Generator().manual_seed(1)
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    if tile:
        vae.enable_tiling(32, 32, 16, 16)
    else:
        vae.disable_tiling()
    return vae.to(device).eval()


def _pixels(batch: int, height: int = 64, width: int = 96, seed: int = 5) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, 3, height, width), generator=gen, dtype=torch.uint8)


def _gen(seed: int = 7, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def test_split_tiles_matches_the_image_vae() -> None:
    vae = _tiny_vae("cpu")
    for length in range(16, 4097, 16):
        ours = iv.split_tiles(length, 256, 64, 16)
        starts, sizes, overlaps = vae._split_tiles(length, 256, 64)
        assert (list(ours.starts), list(ours.sizes), list(ours.overlaps)) == (
            starts,
            sizes,
            overlaps,
        )


def test_zero_overlap_breaks_the_diffusers_stitch() -> None:
    """Why plan_tiles refuses a zero overlap: ``tile[..., :-0]`` empties the tile, and the
    next blend fails. (``enable_tiling(..., 0)`` cannot set it, ``0 or default`` keeps 64.)"""

    vae = _tiny_vae("cpu")
    vae.tile_sample_min_overlap_height = 0
    vae.tile_sample_min_overlap_width = 0
    with torch.no_grad(), pytest.raises(RuntimeError, match="must match the size"):
        vae.decode(torch.zeros(1, 24, 4, 4), return_dict=False)
    contract = iv.ImageVAEContract(vae, require_release=False)
    with pytest.raises(ValueError, match="overlaps must be positive"):
        contract.decode(torch.zeros(1, 24, 4, 4))


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_encode_and_decode_are_bitwise_identical(device: str) -> None:
    """I-C10: tiled repeated execution gives identical latent and byte artifacts."""

    contract = iv.ImageVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(2)
    first = contract.encode(pixels, generator=_gen())
    again = contract.encode(pixels, generator=_gen())
    assert again.digest == first.digest and torch.equal(again.tensor, first.tensor)
    assert first.digest == tensor_digest(first.tensor)
    assert first.row_digests == [tensor_digest(r) for r in first.tensor.split(1)]
    assert first.tensor.shape == (2, 24, 4, 6)
    assert first.tile_plan is not None and first.tile_plan.tiles == 3 * 5
    assert first.record["tiling"]["plan_sha256"] == first.tile_plan.digest()
    decoded = contract.decode(first.tensor)
    assert decoded.digests == contract.decode(first.tensor).digests
    assert decoded.pixels.dtype == torch.uint8 and decoded.pixels.shape == (2, 3, 64, 96)
    assert decoded.digests["uint8"] == tensor_digest(decoded.pixels)
    assert torch.equal(decoded.pixels, iv.quantize_uint8(decoded.unit))
    assert set(decoded.digests) == {"unit", "uint8"}


@pytest.mark.parametrize("device", DEVICES)
def test_batch_rows_are_independent(device: str) -> None:
    contract = iv.ImageVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(3)
    mode = contract.encode(pixels, posterior="mode")
    for i in range(3):
        assert contract.encode(pixels[i : i + 1], posterior="mode").row_digests[0] == (
            mode.row_digests[i]
        )
    # Sampled with one shared request generator: row i alone, after the draws of rows < i.
    sampled = contract.encode(pixels, generator=_gen(9))
    gen = _gen(9)
    for i in range(3):
        assert contract.encode(pixels[i : i + 1], generator=gen).row_digests[0] == (
            sampled.row_digests[i]
        )
    draws = sampled.record["generator"]["draws"]
    assert [d["after"] for d in draws[:-1]] == [d["before"] for d in draws[1:]]
    decoded = contract.decode(mode.tensor)
    assert contract.decode(mode.tensor[1:2]).row_digests["uint8"][0] == (
        decoded.row_digests["uint8"][1]
    )


def test_sample_requires_the_request_generator() -> None:
    contract = iv.ImageVAEContract(_tiny_vae("cpu"), require_release=False)
    with pytest.raises(ValueError, match="generator"):
        contract.encode(_pixels(1))
    with pytest.raises(ValueError):
        contract.encode(_pixels(1), generator=_gen(), posterior="mean")
    with pytest.raises(ValueError):
        contract.encode(_pixels(1).float(), generator=_gen())


def test_reference_recipe_crops_and_records_geometry() -> None:
    contract = iv.ImageVAEContract(_tiny_vae("cpu"), require_release=False)
    image = _pixels(1, 70, 90)  # -> 64x80 pixels -> 4x5 latents -> 4x4 after patch crop
    out = contract.encode(image, generator=_gen())
    assert out.tensor.shape == (1, 24, 4, 4)
    assert out.record["geometry"] == {
        "pixels_in": [70, 90],
        "pixels_encoded": [64, 80],
        "latent": [4, 5],
        "latent_out": [4, 4],
    }
    assert out.record["recipe"] == iv.reference_recipe()
    # The crop happens before encoding: the cropped image gives the same latent.
    assert contract.encode(image[..., :64, :80], generator=_gen()).digest == out.digest
    with pytest.raises(ValueError):
        contract.encode(image, generator=_gen(), crop=False)
    # I11 on the contract: keeping the FP32 sample is a different latent.
    assert contract.encode(image, generator=_gen(), round_fp16=False).digest != out.digest


def test_encode_references_draws_in_order() -> None:
    contract = iv.ImageVAEContract(_tiny_vae("cpu"), require_release=False)
    a, b = _pixels(1, 64, 96, seed=1)[0], _pixels(1, 80, 48, seed=2)[0]
    outs = contract.encode_references([a, b], _gen(3))
    gen = _gen(3)
    assert contract.encode(a[None], generator=gen).digest == outs[0].digest
    assert contract.encode(b[None], generator=gen).digest == outs[1].digest
    swapped = contract.encode_references([b, a], _gen(3))
    assert swapped[1].digest != outs[0].digest  # the draw order is part of the contract


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_encode_equals_stitched_tile_encodes(device: str) -> None:
    vae = _tiny_vae(device)
    contract = iv.ImageVAEContract(vae, require_release=False)
    plan = contract.tile_plan(64, 96)
    assert plan is not None
    x = iv.normalize_pixels(_pixels(1).to(device))
    with torch.no_grad(), iv.pinned_runtime():
        tiled = vae._encode(x)
        rows = []
        for top, h in zip(plan.rows.starts, plan.rows.sizes, strict=True):
            row = []
            for left, w in zip(plan.cols.starts, plan.cols.sizes, strict=True):
                tile = x[..., top : top + h, left : left + w]
                row.append(vae.quant_conv(vae.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)
        ry, rx = plan.latent_overlaps()
        stitched = vae._stitch_tiles(rows, list(ry), list(rx))
    assert torch.equal(tiled, stitched)


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_and_untiled_runs_have_distinct_identities(device: str) -> None:
    vae = _tiny_vae(device)
    tiled = iv.ImageVAEContract(vae, require_release=False)
    latents = tiled.encode(_pixels(1), posterior="mode").tensor
    t_out = tiled.decode(latents)
    vae.disable_tiling()
    untiled = iv.ImageVAEContract(vae, tiled=False, require_release=False)
    u_out = untiled.decode(latents)
    assert t_out.record["tiling"]["mode"] == "tiled-32x32-o16x16"
    assert u_out.record["tiling"] == {
        "contract_id": iv.TILING_CONTRACT_ID,
        "mode": "untiled",
        "tiles": 1,
    }
    assert u_out.tile_plan is None
    assert u_out.digests["unit"] != t_out.digests["unit"]  # I16: tiling changes the output
    assert untiled.decode(latents).digests == u_out.digests


def test_contract_rejects_unsupported_vae_modes() -> None:
    vae = _tiny_vae("cpu")
    with pytest.raises(ValueError, match="config differs|release tiling"):
        iv.ImageVAEContract(vae)  # tiny config / tile geometry differ from the release
    with pytest.raises(ValueError, match="declared tiled=False"):
        iv.ImageVAEContract(vae, tiled=False, require_release=False)
    vae.use_slicing = True
    with pytest.raises(ValueError, match="slicing"):
        iv.ImageVAEContract(vae, require_release=False)
    vae.use_slicing = False
    vae.decoder.transformer_blocks.to(torch.bfloat16)
    with pytest.raises(TypeError, match="FP32"):
        iv.ImageVAEContract(vae, require_release=False)


def test_contract_rechecks_the_mode_before_every_call() -> None:
    vae = _tiny_vae("cpu")
    contract = iv.ImageVAEContract(vae, require_release=False)
    latents = torch.zeros(1, 24, 4, 4)
    vae.disable_tiling()
    with pytest.raises(ValueError, match="tiling off"):
        contract.encode(_pixels(1), generator=_gen())
    with pytest.raises(ValueError, match="tiling off"):
        contract.decode(latents)


def test_decode_requires_a_canvas_multiple_of_32() -> None:
    contract = iv.ImageVAEContract(_tiny_vae("cpu"), require_release=False)
    with pytest.raises(ValueError, match="multiples of 32"):
        contract.decode(torch.zeros(1, 24, 4, 5))


@pytest.mark.parametrize("device", DEVICES)
def test_ambient_autocast_does_not_reach_the_vae(device: str) -> None:
    contract = iv.ImageVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(1)
    expected = contract.encode(pixels, posterior="mode")
    decoded = contract.decode(expected.tensor)
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device_type=device, dtype=dtype):
        assert contract.encode(pixels, posterior="mode").digest == expected.digest
        assert contract.decode(expected.tensor).digests == decoded.digests


@needs_cuda
@pytest.mark.parametrize("generator_device", ["cpu", "cuda"])
def test_encode_matches_the_diffusers_edit_vae_encoder_step(generator_device: str) -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    vae = _tiny_vae("cuda")
    contract = iv.ImageVAEContract(vae, require_release=False)
    refs = [_pixels(1, 70, 90, seed=1)[0], _pixels(1, 64, 96, seed=2)[0]]
    ours = contract.encode_references(refs, _gen(3, generator_device))
    theirs = evidence.diffusers_encode_references(vae, refs, _gen(3, generator_device))
    for o, t in zip(ours, theirs, strict=True):
        assert torch.equal(o.tensor, t)


@needs_cuda
def test_decode_matches_the_diffusers_decode_step() -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    vae = _tiny_vae("cuda")
    contract = iv.ImageVAEContract(vae, require_release=False)
    latents = contract.encode(_pixels(1), posterior="mode").tensor
    out = contract.decode(latents)
    assert torch.equal(out.pixels, evidence.diffusers_decode(vae, latents))


# --- released checkpoint ------------------------------------------------------------------


@pytest.fixture(scope="module")
def release_vae():
    root = os.environ.get(VAE_DIR_ENV)
    if not root:
        pytest.skip(f"{VAE_DIR_ENV} not set (path to the released vae_diffusers/ folder)")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    module = _diffusers()
    return iv.load_release_vae(module.AutoencoderKLMiniMaxH3Image, root, "cuda")


def test_release_checkpoint_identity(release_vae) -> None:
    summary = iv.check_checkpoint_dir(os.environ[VAE_DIR_ENV])
    assert summary["tensors"] == 511 and summary["dtypes"] == ["F32"]
    iv.ImageVAEContract(release_vae).check_release()


def test_release_checkpoint_parity_and_determinism(release_vae) -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    contract = iv.ImageVAEContract(release_vae)
    refs = [evidence.synthetic_image(300, 410, 1), evidence.synthetic_image(272, 304, 2)]
    ours = contract.encode_references(refs, _gen(7))
    assert [o.digest for o in contract.encode_references(refs, _gen(7))] == [o.digest for o in ours]
    theirs = evidence.diffusers_encode_references(release_vae, refs, _gen(7))
    for o, t in zip(ours, theirs, strict=True):
        assert torch.equal(o.tensor, t)
    target = evidence.synthetic_image(512, 768, 3)[None]
    latents = contract.encode(target, posterior="mode", round_fp16=False).tensor
    decoded = contract.decode(latents)
    assert contract.decode(latents).digests == decoded.digests
    assert torch.equal(decoded.pixels, evidence.diffusers_decode(release_vae, latents))


def test_release_checkpoint_batch_rows_and_tiling(release_vae) -> None:
    contract = iv.ImageVAEContract(release_vae)
    batch = torch.stack([evidence.synthetic_image(512, 512, 10 + i) for i in range(2)])
    together = contract.encode(batch, posterior="mode")
    alone = [contract.encode(batch[i : i + 1], posterior="mode").row_digests[0] for i in range(2)]
    assert alone == together.row_digests
    decoded = contract.decode(together.tensor)
    assert contract.decode(together.tensor[1:]).row_digests["uint8"][0] == (
        decoded.row_digests["uint8"][1]
    )
    release_vae.disable_tiling()
    try:
        untiled = iv.ImageVAEContract(release_vae, tiled=False).decode(together.tensor[:1])
    finally:
        release_vae.enable_tiling(*iv.RELEASE_TILE_GEOMETRY)
    assert untiled.record["tiling"]["mode"] != decoded.record["tiling"]["mode"]
    assert untiled.digests["unit"] != decoded.row_digests["unit"][0]
