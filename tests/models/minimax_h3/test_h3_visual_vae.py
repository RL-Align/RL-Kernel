# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``visual_vae_contract``: MiniMax-H3 VisualVAE layout, tiling,
preprocessing, normalization and encode/decode determinism.

The layout/tiling/normalization tests are self-contained. Tests marked ``diffusers`` build a
small random ``AutoencoderKLMiniMaxH3`` and skip when Diffusers with MiniMax-H3 (commit
f53d5520 or later) is not importable. Real-checkpoint tests read the released ``vae/``
folder from ``$RL_KERNEL_H3_VAE_DIR`` and skip when it is unset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import visual_vae as vv
from rl_engine.models.minimax_h3.trajectory import tensor_digest

VAE_DIR_ENV = "RL_KERNEL_H3_VAE_DIR"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --- layout -------------------------------------------------------------------------------


def test_release_layout_is_f16t4d24() -> None:
    layout = vv.RELEASE_LAYOUT
    assert layout.name == "f16t4d24"
    assert (layout.latent_channels, layout.spatial_factor, layout.temporal_factor) == (24, 16, 4)
    assert (layout.clip_length, layout.token_drop) == (17, 3)
    # Derived chunk geometry (AutoencoderKLMiniMaxH3.__init__ at diffusers f53d5520).
    assert layout.frame_pre_padding == 3
    assert layout.tokens_chunk_size == 5
    assert layout.token_overlap == 2
    assert layout.frame_overlap == 5


def test_layout_matches_pinned_config_and_transformer() -> None:
    config = vv.release_config()
    assert vv.VisualVAELayout.from_config(config) == vv.RELEASE_LAYOUT
    assert config["_class_name"] == "AutoencoderKLMiniMaxH3"
    # The transformer's video head consumes 24 latent channels.
    pinned = json.loads((Path(vv.__file__).with_name("pinned_checkpoint.json")).read_text())
    assert pinned["configs"]["transformer"]["in_channels"] == vv.LATENT_CHANNELS


@pytest.mark.parametrize("n", range(0, 15))
def test_frame_map_17n_plus_5(n: int) -> None:
    layout = vv.RELEASE_LAYOUT
    assert layout.latent_frames(17 * n + 5) == 5 * n + 2
    if n >= 1:
        assert layout.pixel_frames(5 * n + 2) == 17 * n + 5
        assert layout.is_release_frame_count(17 * n + 5)


def test_frame_map_edges() -> None:
    layout = vv.RELEASE_LAYOUT
    assert layout.latent_frames(1) == 1  # a keyframe bypasses temporal chunking
    assert layout.latent_frames(17) == 2  # one clip, padded nowhere, 3 latents dropped
    assert layout.latent_frames(18) == 7  # a second clip is padded by frame repetition
    assert [f for f in range(1, 80) if layout.is_release_frame_count(f)] == [1, 22, 39, 56, 73]
    assert not layout.decodable(1) and not layout.decodable(2)
    with pytest.raises(ValueError):
        layout.latent_frames(0)
    with pytest.raises(ValueError):
        layout.pixel_frames(2)


def test_latent_shape_and_spatial_checks() -> None:
    layout = vv.RELEASE_LAYOUT
    assert layout.latent_shape(2, 121, 480, 832) == (2, 24, 37, 30, 52)
    with pytest.raises(ValueError):
        layout.latent_shape(1, 1, 480, 830)
    with pytest.raises(ValueError):
        vv.VisualVAELayout(token_drop=5)


# --- tiling -------------------------------------------------------------------------------


def test_split_tiles_golden() -> None:
    # Values from AutoencoderKLMiniMaxH3._split_tiles(length, 256, 64) at diffusers f53d5520.
    golden = {
        256: ((0,), (256,), ()),
        384: ((0, 128), (256, 256), (128,)),
        480: ((0, 112, 224), (256,) * 3, (144, 144)),
        832: ((0, 192, 384, 576), (256,) * 4, (64, 64, 64)),
        1280: ((0, 160, 320, 496, 672, 848, 1024), (256,) * 7, (96, 96, 80, 80, 80, 80)),
    }
    for length, (starts, sizes, overlaps) in golden.items():
        axis = vv.split_tiles(length, 256, 64, 16)
        assert (axis.starts, axis.sizes, axis.overlaps) == (starts, sizes, overlaps), length


@pytest.mark.parametrize("length", [16 * k for k in range(1, 161)])
def test_split_tiles_covers_frame_on_latent_grid(length: int) -> None:
    axis = vv.split_tiles(length, 256, 64, 16)
    assert axis.starts[0] == 0
    assert axis.starts[-1] + axis.sizes[-1] == length or axis.count == 1
    assert all(o >= 64 and o % 16 == 0 for o in axis.overlaps)
    assert all(s % 16 == 0 for s in axis.starts)
    assert list(axis.starts) == sorted(set(axis.starts))
    assert axis == vv.split_tiles(length, 256, 64, 16)


@pytest.mark.parametrize("args", [(264, 256, 64, 16), (400, 250, 64, 16), (480, 256, 70, 16)])
def test_split_tiles_rejects_off_grid_geometry(args: tuple[int, int, int, int]) -> None:
    # Off the grid the remainder is dropped and the last tile overruns the frame edge
    # (264: starts (0, 16), so the second tile ends at 272).
    with pytest.raises(ValueError, match="grid"):
        vv.split_tiles(*args)


def test_tile_plan_digest_is_stable() -> None:
    plan = vv.plan_tiles(480, 832)
    assert plan.tiles == 3 * 4  # 480 -> 3 rows, 832 -> 4 columns
    assert plan.covers_exactly()
    assert plan.latent_overlaps() == (
        tuple(o // 16 for o in plan.rows.overlaps),
        tuple(o // 16 for o in plan.cols.overlaps),
    )
    assert plan.digest() == vv.plan_tiles(480, 832).digest()
    assert plan.digest() != vv.plan_tiles(480, 848).digest()
    with pytest.raises(ValueError):
        vv.plan_tiles(480, 832, tile_height=250)


# --- preprocessing ------------------------------------------------------------------------


def test_pixel_lut_reproduces_cuda_release_semantics_on_cpu() -> None:
    lut = vv.pixel_lut()
    assert lut.shape == (3, 256) and lut.dtype == torch.float32
    values = torch.arange(256, dtype=torch.float32)
    inv = torch.tensor(1.0) / torch.tensor(255.0)
    for c in range(3):
        expected = (values * inv - torch.tensor(vv.PIXEL_MEAN[c])) / torch.tensor(vv.PIXEL_STD[c])
        assert torch.equal(lut[c], expected)
    # CPU true division differs from the release (CUDA) semantics for some values.
    cpu_div = values.div(255.0)
    assert not torch.equal(cpu_div, values * inv)
    assert vv.pixel_lut_digest() == vv.pixel_lut_digest()


def test_preprocess_is_a_gather() -> None:
    gen = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (2, 3, 5, 16, 32), generator=gen, dtype=torch.uint8)
    out = vv.preprocess_pixels(pixels)
    assert out.shape == pixels.shape and out.dtype == torch.float32
    lut = vv.pixel_lut()
    for c in range(3):
        assert torch.equal(out[:, c], lut[c][pixels[:, c].long()])
    with pytest.raises(TypeError):
        vv.preprocess_pixels(pixels.float())
    with pytest.raises(ValueError):
        vv.preprocess_pixels(pixels[:, :2])


def test_preprocess_chunks_match_one_gather() -> None:
    gen = torch.Generator().manual_seed(1)
    frames = 2 * vv.PREPROCESS_FRAME_CHUNK + 3  # a partial last chunk
    pixels = torch.randint(0, 256, (1, 3, frames, 4, 8), generator=gen, dtype=torch.uint8)
    channel = torch.arange(3).view(1, 3, 1, 1, 1)
    assert torch.equal(vv.preprocess_pixels(pixels), vv.pixel_lut()[channel, pixels.long()])


def test_postprocess_pixels_is_fp32_under_any_default_dtype() -> None:
    gen = torch.Generator().manual_seed(3)
    video = torch.randn(1, 3, 2, 4, 8, generator=gen)
    expected = vv.postprocess_pixels(video)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        out = vv.postprocess_pixels(video)
    finally:
        torch.set_default_dtype(previous)
    assert out.dtype == torch.float32
    assert torch.equal(out, expected)


@needs_cuda
def test_preprocess_peak_memory_is_bounded() -> None:
    """The int64 gather index is 8x the input; it must not be built for the whole clip."""

    pixels = torch.randint(0, 256, (1, 3, 64, 256, 256), dtype=torch.uint8, device="cuda")
    out_bytes = pixels.numel() * 4
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = vv.preprocess_pixels(pixels)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert out.shape == pixels.shape
    # The output plus a few frames of index (1.6x here); the whole-clip index made it 5x.
    assert peak < 2 * out_bytes
    del out


@needs_cuda
def test_pixel_lut_matches_diffusers_cuda_preprocessing() -> None:
    values = torch.arange(256, dtype=torch.uint8, device="cuda")
    pixels = values.view(1, 1, 1, 1, 256).expand(1, 3, 1, 1, 256)
    mean = torch.tensor(vv.PIXEL_MEAN, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vv.PIXEL_STD, device="cuda").view(1, -1, 1, 1, 1)
    # The exact op sequence of diffusers modular_pipelines/minimax_h3/encoders.py.
    reference = (pixels.to(torch.float32).div(255.0) - mean) / std
    assert torch.equal(vv.preprocess_pixels(pixels).cpu(), reference.cpu())
    assert torch.equal(vv.preprocess_pixels(pixels.cpu()), reference.cpu())


# --- latent normalization -----------------------------------------------------------------


def test_release_latent_stats() -> None:
    mean, std = vv.release_latent_stats()
    assert mean.shape == std.shape == (24,)
    assert mean.dtype == std.dtype == torch.float32
    assert bool((std > 0).all())
    config = vv.release_config()
    assert torch.equal(mean, torch.tensor(config["latents_mean"]))


def test_normalization_formula_and_op_order() -> None:
    mean, std = vv.release_latent_stats()
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(2, 24, 3, 4, 5, generator=gen)
    m, s = mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1)
    assert torch.equal(vv.normalize_latents(z, mean, std), (z - m) / s)
    assert torch.equal(vv.denormalize_latents(z, mean, std), z * s + m)
    # Division by std is not multiplication by its reciprocal.
    assert not torch.equal(vv.normalize_latents(z, mean, std), (z - m) * (1 / s))
    bf16 = z.to(torch.bfloat16)
    assert vv.normalize_latents(bf16, mean, std).dtype == torch.float32
    rounded = vv.round_condition_latents(z)
    assert torch.equal(rounded, z.half().float())


def test_normalization_rejects_bad_inputs() -> None:
    mean, std = vv.release_latent_stats()
    z = torch.zeros(1, 24, 1, 2, 2)
    with pytest.raises(ValueError):
        vv.normalize_latents(torch.zeros(1, 16, 1, 2, 2), mean, std)
    with pytest.raises(ValueError):
        vv.normalize_latents(z, mean[:16], std)
    with pytest.raises(TypeError):
        vv.normalize_latents(z, mean.double(), std)
    with pytest.raises(ValueError):
        vv.normalize_latents(z, mean, std.clone().requires_grad_(True))
    with pytest.raises(TypeError):
        vv.normalize_latents(z.long(), mean, std)


@needs_cuda
def test_normalization_cpu_cuda_bitwise() -> None:
    mean, std = vv.release_latent_stats()
    gen = torch.Generator().manual_seed(4)
    z = torch.randn(2, 24, 7, 8, 12, generator=gen) * 3
    for fn in (vv.normalize_latents, vv.denormalize_latents):
        assert torch.equal(fn(z, mean, std), fn(z.cuda(), mean, std).cpu())


def test_pinned_runtime_restores_flags() -> None:
    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32)
    policy = vv.RuntimePolicy(cudnn_deterministic=not before[0], cudnn_benchmark=False)
    with vv.pinned_runtime(policy) as readback:
        assert cudnn.deterministic is (not before[0])
        assert readback["cudnn_deterministic"] is (not before[0])
    assert (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32) == before


def _fake_gpus(monkeypatch) -> None:
    """CUDA present, current device 0, names and capabilities that tell the indices apart."""

    # cuDNN checks the real devices on its first version() call; make that call first.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.version()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"gpu{index}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, index))


def test_runtime_readback_names_the_given_device(monkeypatch) -> None:
    _fake_gpus(monkeypatch)
    assert vv.runtime_readback()["device_name"] == "gpu0"
    assert vv.runtime_readback("cuda")["device_name"] == "gpu0"
    one = vv.runtime_readback(torch.device("cuda", 1))
    assert (one["device_name"], one["compute_capability"]) == ("gpu1", [9, 1])
    cpu = vv.runtime_readback("cpu")
    assert "device_name" not in cpu and "compute_capability" not in cpu
    with vv.pinned_runtime(device="cuda:1") as readback:
        assert readback["device_name"] == "gpu1"


def test_pinned_runtime_serializes_threads() -> None:
    """A second thread's scope must not start inside the first one's: interleaved scopes
    save each other's pinned flags and leave the process pinned after both return."""

    import threading

    cudnn = torch.backends.cudnn
    before = cudnn.deterministic
    policy = vv.RuntimePolicy(cudnn_deterministic=not before)
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()

    def first() -> None:
        with vv.pinned_runtime(policy):
            first_in.set()
            second_in.wait(timeout=0.5)  # times out when the second scope is held back
        first_out.set()

    def second() -> None:
        first_in.wait()
        with vv.pinned_runtime(policy):
            second_in.set()
            first_out.wait(timeout=5)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert cudnn.deterministic is before
    finally:
        cudnn.deterministic = before


def test_manifest_is_reproducible() -> None:
    a, b = vv.contract_manifest(), vv.contract_manifest()
    assert a == b
    assert a["frame_map"] == {"1": 1, "22": 7, "121": 37, "241": 72}
    assert a["checkpoint"]["revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_output_digests_equal_tensor_digest(dtype: torch.dtype, batch: int) -> None:
    g = torch.Generator().manual_seed(batch)
    rows = [(torch.rand(1, 3, 2, 5, 7, generator=g) * 200).to(dtype) for _ in range(batch)]
    digest, row_digests = vv.output_digests(rows)
    assert digest == tensor_digest(torch.cat(rows))
    assert row_digests == [tensor_digest(r) for r in rows]


def test_output_digests_strided_and_empty_rows() -> None:
    base = torch.arange(2 * 4 * 6, dtype=torch.float32).view(2, 4, 6)
    rows = [base[:1].transpose(1, 2), base[1:].transpose(1, 2)]  # non-contiguous
    assert vv.output_digests(rows) == (
        tensor_digest(torch.cat(rows)),
        [tensor_digest(r) for r in rows],
    )
    empty = [torch.empty(1, 0), torch.empty(1, 0)]
    assert vv.output_digests(empty)[0] == tensor_digest(torch.cat(empty))
    with pytest.raises(ValueError):
        vv.output_digests([])


def test_load_release_vae_reads_only_the_pinned_shards(tmp_path, monkeypatch) -> None:
    from safetensors.torch import save_file

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4))

        @classmethod
        def from_config(cls, config):
            return cls()

    (tmp_path / "config.json").write_text(json.dumps({"k": 1}))
    save_file({"w": torch.ones(4)}, str(tmp_path / "model.safetensors"))
    # Same key, not in the fingerprint, sorted after the pinned shard.
    save_file({"w": torch.full((4,), 7.0)}, str(tmp_path / "zzz.safetensors"))
    files = [
        {"path": f"vae/{name}", "size": (tmp_path / name).stat().st_size}
        for name in ("config.json", "model.safetensors")
    ]
    fingerprint = {"config": {"k": 1}, "checkpoint": {"files": files}}
    monkeypatch.setattr(vv, "release_fingerprint", lambda: fingerprint)
    vae = vv.load_release_vae(Tiny, tmp_path)
    assert torch.equal(vae.w.detach(), torch.ones(4))


class _TwoParams(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(4))
        self.b = torch.nn.Parameter(torch.zeros(4))

    @classmethod
    def from_config(cls, config):
        return cls()


def _pinned_folder(tmp_path, monkeypatch, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    """A release-like folder with these shards, all listed in a patched fingerprint."""

    from safetensors.torch import save_file

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"k": 1}))
    for name, tensors in shards.items():
        save_file(tensors, str(tmp_path / name))
    files = [
        {"path": f"vae/{name}", "size": (tmp_path / name).stat().st_size}
        for name in ("config.json", *shards)
    ]
    fingerprint = {"config": {"k": 1}, "checkpoint": {"files": files}}
    monkeypatch.setattr(vv, "release_fingerprint", lambda: fingerprint)


def test_load_release_vae_holds_one_shard_at_a_time(tmp_path, monkeypatch) -> None:
    """The host must not keep a second full copy of the weights while loading."""

    import weakref

    import safetensors.torch

    shards = {
        "s1.safetensors": {"a": torch.ones(4)},
        "s2.safetensors": {"b": torch.full((4,), 2.0)},
    }
    _pinned_folder(tmp_path, monkeypatch, shards)
    real_load = safetensors.torch.load_file
    earlier: list[weakref.ref] = []

    def load_file(path, device="cpu"):
        assert all(ref() is None for ref in earlier), "an earlier shard is still alive"
        out = real_load(path, device=device)
        earlier.extend(weakref.ref(t) for t in out.values())
        return out

    monkeypatch.setattr(safetensors.torch, "load_file", load_file)
    vae = vv.load_release_vae(_TwoParams, tmp_path)
    assert torch.equal(vae.a.detach(), torch.ones(4))
    assert torch.equal(vae.b.detach(), torch.full((4,), 2.0))


def test_load_release_vae_rejects_missing_and_repeated_keys(tmp_path, monkeypatch) -> None:
    _pinned_folder(tmp_path / "missing", monkeypatch, {"s1.safetensors": {"a": torch.ones(4)}})
    with pytest.raises((ValueError, RuntimeError), match="b"):
        vv.load_release_vae(_TwoParams, tmp_path / "missing")
    shards = {
        "s1.safetensors": {"a": torch.ones(4), "b": torch.ones(4)},
        "s2.safetensors": {"b": torch.full((4,), 2.0)},
    }
    _pinned_folder(tmp_path / "repeated", monkeypatch, shards)
    with pytest.raises(ValueError, match="more than one shard"):
        vv.load_release_vae(_TwoParams, tmp_path / "repeated")


def test_release_fingerprint_cannot_be_changed_by_callers() -> None:
    vv.release_fingerprint()["config"]["latent_channels"] = 16
    vv.release_config()["latents_mean"][0] = 9.0
    vv.contract_manifest()["checkpoint"]["files"].clear()
    assert vv.release_config()["latent_channels"] == 24
    assert vv.release_latent_stats()[0][0].item() != 9.0
    assert len(vv.release_fingerprint()["checkpoint"]["files"]) == 5
    assert vv.contract_manifest() == vv.contract_manifest()


# --- encode/decode with a small Diffusers model -------------------------------------------


def _diffusers():
    return pytest.importorskip(
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3",
        reason="needs diffusers with MiniMax-H3 (f53d5520+)",
    )


def _tiny_vae(device: str, tile: bool = True):
    module = _diffusers()
    mean, std = vv.release_latent_stats()
    torch.manual_seed(0)
    vae = module.AutoencoderKLMiniMaxH3(
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


DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


def _pixels(batch: int, frames: int, height: int = 64, width: int = 48, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, 3, frames, height, width)
    return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)


def test_tiny_layout_and_tiles_match_diffusers() -> None:
    vae = _tiny_vae("cpu")
    layout = vv.VisualVAELayout.from_config(vae.config)
    assert layout == vv.RELEASE_LAYOUT
    for name in ("frame_pre_padding", "tokens_chunk_size", "token_overlap", "frame_overlap"):
        assert getattr(layout, name) == getattr(vae, name)
    for length in range(16, 2049, 16):
        ours = vv.split_tiles(length, 256, 64, 16)
        starts, sizes, overlaps = vae._split_tiles(length, 256, 64)
        assert (list(ours.starts), list(ours.sizes), list(ours.overlaps)) == (
            list(starts),
            list(sizes),
            list(overlaps),
        )


def test_tiny_pixel_frames_match_diffusers_decode() -> None:
    vae = _tiny_vae("cpu", tile=False)
    layout = vv.RELEASE_LAYOUT
    gen = torch.Generator().manual_seed(2)
    for frames in range(3, 18):
        z = torch.randn(1, 24, frames, 1, 1, generator=gen)
        with torch.no_grad():
            video = vae.decode(z, return_dict=False)[0]
        assert video.shape[2] == layout.pixel_frames(frames), frames


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("frames", [1, 22])
def test_repeated_encode_is_bitwise_identical(device: str, frames: int) -> None:
    contract = vv.VisualVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(2, frames)
    first = contract.encode(pixels)
    for _ in range(2):
        again = contract.encode(pixels)
        assert again.digest == first.digest
        assert torch.equal(again.tensor, first.tensor)
    assert first.digest == tensor_digest(first.tensor)
    assert first.row_digests == [tensor_digest(r) for r in first.tensor.split(1)]
    assert tuple(first.tensor.shape) == vv.RELEASE_LAYOUT.latent_shape(2, frames, 64, 48)
    assert first.tile_plan is not None and first.tile_plan.tiles == 6
    assert first.record["tile_plan_sha256"] == first.tile_plan.digest()


@pytest.mark.parametrize("device", DEVICES)
def test_batch_rows_are_independent(device: str) -> None:
    contract = vv.VisualVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(3, 22)
    batched = contract.encode(pixels)
    for i in range(3):
        alone = contract.encode(pixels[i : i + 1])
        assert alone.row_digests[0] == batched.row_digests[i]
    decoded = contract.decode(batched.tensor)
    alone = contract.decode(batched.tensor[1:2])
    assert alone.row_digests[0] == decoded.row_digests[1]
    assert alone.digest == tensor_digest(alone.tensor)
    assert decoded.digest == tensor_digest(decoded.tensor)


def test_record_names_the_vae_device(monkeypatch) -> None:
    """A CPU run must not record the GPU that happens to be current."""

    contract = vv.VisualVAEContract(_tiny_vae("cpu"), require_release=False)
    _fake_gpus(monkeypatch)
    encoded = contract.encode(_pixels(1, 22))
    decoded = contract.decode(encoded.tensor)
    for out in (encoded, decoded):
        assert "device_name" not in out.record["runtime"]
        assert "compute_capability" not in out.record["runtime"]


@needs_cuda
@pytest.mark.parametrize("frames", [1, 5, 22])
def test_encode_matches_diffusers_condition_recipe(frames: int) -> None:
    encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
    vae = _tiny_vae("cuda")
    contract = vv.VisualVAEContract(vae, require_release=False)
    pixels = _pixels(2, frames)
    ours = contract.encode(pixels)
    for i in range(2):
        # The pipeline block runs under no_grad; encode_vae_condition itself does not.
        with torch.no_grad():
            reference = encoders.encode_vae_condition(
                vae, pixels[i : i + 1].cuda(), vv.PIXEL_MEAN, vv.PIXEL_STD
            )
        assert torch.equal(ours.tensor[i : i + 1], reference)


@needs_cuda
def test_decode_matches_diffusers_decode_block() -> None:
    vae = _tiny_vae("cuda")
    contract = vv.VisualVAEContract(vae, require_release=False)
    latents = contract.encode(_pixels(1, 22)).tensor
    ours = contract.decode(latents)
    # The arithmetic of diffusers modular_pipelines/minimax_h3/decoders.py (decode step).
    config = vae.config
    mean = torch.tensor(config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    with torch.no_grad():
        z = latents.cuda() * std + mean
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            video = vae.decode(z, return_dict=False)[0]
        pixel_mean = torch.tensor(vv.PIXEL_MEAN, device="cuda").view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(vv.PIXEL_STD, device="cuda").view(1, -1, 1, 1, 1)
        reference = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    assert torch.equal(ours.tensor, reference.cpu())
    assert ours.tensor.shape[2] == 22


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_encode_equals_stitched_tile_encodes(device: str) -> None:
    """Tiling is a deterministic function of the plan: per-tile encodes plus the release
    blend reproduce the tiled encode bit for bit."""

    vae = _tiny_vae(device)
    contract = vv.VisualVAEContract(vae, require_release=False)
    pixels = _pixels(1, 1)
    plan = contract.tile_plan(64, 48)
    assert plan is not None
    x = vv.preprocess_pixels(pixels.to(device))
    with torch.no_grad(), vv.pinned_runtime():
        tiled = vae._encode(x)
        rows = []
        for top, h in zip(plan.rows.starts, plan.rows.sizes, strict=True):
            row = []
            for left, w in zip(plan.cols.starts, plan.cols.sizes, strict=True):
                tile = x[..., top : top + h, left : left + w]
                row.append(vae.quant_conv(vae.encoder(tile)))
            rows.append(row)
        ry, rx = plan.latent_overlaps()
        stitched = vae._stitch_tiles(rows, list(ry), list(rx))
    assert torch.equal(tiled, stitched)


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_tiled_decode_is_bitwise_identical(device: str) -> None:
    contract = vv.VisualVAEContract(_tiny_vae(device), require_release=False)
    latents = contract.encode(_pixels(1, 22)).tensor
    first = contract.decode(latents)
    assert first.tile_plan is not None and first.tile_plan.tiles == 6
    assert contract.decode(latents).digest == first.digest


def test_contract_rejects_non_release_vae() -> None:
    vae = _tiny_vae("cpu")
    with pytest.raises(ValueError):
        vv.VisualVAEContract(vae)  # tile geometry differs from the release
    vae.enable_tiling(256, 256, 64, 64)
    vv.VisualVAEContract(vae)  # same layout, stats and tiles as released
    vae.use_slicing = True
    with pytest.raises(ValueError):
        vv.VisualVAEContract(vae)


def test_contract_rechecks_the_release_before_every_call() -> None:
    vae = _tiny_vae("cpu")
    vae.enable_tiling(256, 256, 64, 64)
    contract = vv.VisualVAEContract(vae)
    latents = torch.zeros(1, 24, 2, 4, 3)
    vae.disable_tiling()
    with pytest.raises(ValueError, match="release tiling"):
        contract.encode(_pixels(1, 1))
    with pytest.raises(ValueError, match="release tiling"):
        contract.decode(latents)
    # Opting out still allows untiled runs (the evidence tool's tiled-vs-untiled check).
    untiled = vv.VisualVAEContract(vae, require_release=False)
    assert untiled.encode(_pixels(1, 1)).tile_plan is None


def test_contract_follows_a_non_release_latent_width() -> None:
    """``require_release=False`` takes the layout from the VAE config, latent channels too."""

    module = _diffusers()
    torch.manual_seed(0)
    vae = module.AutoencoderKLMiniMaxH3(
        latent_channels=8,
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        latents_mean=(0.1,) * 8,
        latents_std=(2.0,) * 8,
    ).eval()
    contract = vv.VisualVAEContract(vae, require_release=False)
    assert contract.layout.latent_channels == 8
    latents = contract.encode(_pixels(1, 22, 32, 32)).tensor
    assert latents.shape == (1, 8, 7, 2, 2)
    assert contract.decode(latents).tensor.shape == (1, 3, 22, 32, 32)


@pytest.mark.parametrize("device", DEVICES)
def test_encode_ignores_ambient_autocast(device: str) -> None:
    """The release encodes in FP32; a caller's autocast must not reach the encoder."""

    contract = vv.VisualVAEContract(_tiny_vae(device), require_release=False)
    pixels = _pixels(1, 1)
    expected = contract.encode(pixels, round_fp16=False)
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device_type=device, dtype=dtype):
        ours = contract.encode(pixels, round_fp16=False)
    assert torch.equal(ours.tensor, expected.tensor)


# --- released checkpoint ------------------------------------------------------------------


@pytest.fixture(scope="module")
def release_vae():
    root = os.environ.get(VAE_DIR_ENV)
    if not root:
        pytest.skip(f"{VAE_DIR_ENV} not set (path to the released vae/ folder)")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    module = _diffusers()
    return vv.load_release_vae(module.AutoencoderKLMiniMaxH3, root, "cuda")


def test_release_checkpoint_repeated_encode_decode(release_vae) -> None:
    encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
    contract = vv.VisualVAEContract(release_vae)
    keyframe = _pixels(1, 1, 256, 384, seed=11)
    video = _pixels(1, 22, 256, 384, seed=12)
    for pixels in (keyframe, video):
        first = contract.encode(pixels)
        assert contract.encode(pixels).digest == first.digest
        with torch.no_grad():  # as in the pipeline block; autograd would hold ~55 GB
            reference = encoders.encode_vae_condition(
                release_vae, pixels.cuda(), vv.PIXEL_MEAN, vv.PIXEL_STD
            )
        assert torch.equal(first.tensor, reference)
    latents = contract.encode(video).tensor
    decoded = contract.decode(latents)
    assert decoded.tensor.shape == (1, 3, 22, 256, 384)
    assert contract.decode(latents).digest == decoded.digest
