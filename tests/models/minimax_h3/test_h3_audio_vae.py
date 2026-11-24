# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``audio_vae_contract``: stereo AudioVAE encode/decode and the 40 Hz layout.

CPU tests use a reduced-width random module. Diffusers parity tests run when the installed (or
``PYTHONPATH``-provided) diffusers has ``AutoencoderKLMiniMaxH3Audio`` and drive Diffusers' own
MiniMax-H3 pipeline blocks. Released-weight tests read ``$RL_KERNEL_H3_AUDIO_VAE`` (the Hub
``FL2VA/audio_vae`` directory) and need CUDA.

Without those, pytest reports the parity and released-weight tests as *skipped* (with the reason),
not passed: a CPU-only CI run checks the layout, the contract guards and the execution scope only.
Diffusers bitwise parity and the released-weight checks are established by running this file with
both set on a GPU host (the PR's B200 / H100 evidence).
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import types
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import audio_vae as av
from rl_engine.models.minimax_h3.trajectory import tensor_digest

AUDIO_VAE_ENV = "RL_KERNEL_H3_AUDIO_VAE"
TINY = dict(encoder_dim=4, latent_dim=64, decoder_dim=128)
LIBRARY = av.ExecutionPolicy("library", None, cudnn_enabled=True, sdpa_backend=None)


def stereo(num_samples: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.3 * torch.randn(2, num_samples, generator=g)


@pytest.fixture(scope="module")
def tiny_stats() -> tuple[list[float], list[float]]:
    return [0.1 * i - 1.0 for i in range(32)], [0.5 + 0.05 * i for i in range(32)]


@pytest.fixture(scope="module")
def tiny_model() -> av.H3AudioVAE:
    torch.manual_seed(0)
    model = av.H3AudioVAE(**TINY)
    with torch.no_grad():  # non-trivial Snake parameters, so channels and stereo sides differ
        for name, p in model.named_parameters():
            if name.endswith(("alpha", "beta")):
                p.uniform_(0.2, 1.5) if p.ndim == 3 else p.uniform_(-0.5, 0.5)
    return model.eval()


@pytest.fixture
def contract(tiny_model, tiny_stats) -> av.H3AudioVAEContract:
    return av.H3AudioVAEContract(tiny_model, *tiny_stats)


def diffusers_or_skip():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLMiniMaxH3Audio"):
        pytest.skip("this diffusers has no AutoencoderKLMiniMaxH3Audio (needs f53d5520 or later)")
    return diffusers


# --- layout -------------------------------------------------------------------------------


def test_layout_constants_match_pinned_checkpoint():
    pinned = json.loads(Path(av.__file__).with_name("pinned_checkpoint.json").read_text())
    for name in ("transformer", "transformer_ref"):
        config = pinned["configs"][name]
        assert config["audio_in_channels"] == av.LATENT_CHANNELS
        assert (config["audio_in_channels"], config["hidden_size"]) == av.AUDIO_INPUT_HEAD
    assert pinned["configs"]["audio_scheduler"]["shift"] == 3.0
    assert math.prod(av.ENCODER_RATES) == math.prod(av.DECODER_RATES) == av.HOP_LENGTH == 800
    assert av.SAMPLE_RATE / av.HOP_LENGTH == av.LATENTS_PER_SECOND == 40
    assert av.STEREO_ORDER == ("left", "right")


@pytest.mark.parametrize(
    ("frames", "latents"), [(1, 2), (22, 37), (121, 202), (241, 402), (481, 802)]
)
def test_audio_latent_num_frames(frames, latents):
    assert av.audio_latent_num_frames(frames) == latents


def test_audio_latent_num_frames_matches_diffusers():
    diffusers_or_skip()
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames

    for frames in range(1, 1000):
        assert av.audio_latent_num_frames(frames) == audio_latent_num_frames(frames)


def test_latent_frames_for_samples_and_bad_lengths():
    assert [av.latent_frames_for_samples(n) for n in (1, 800, 801, 161_600)] == [1, 1, 2, 202]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            av.latent_frames_for_samples(bad)
        with pytest.raises(ValueError):
            av.audio_latent_num_frames(bad)


def test_rows_are_channel_major():
    channel, t = av.row_channel_and_time(3)
    assert channel.tolist() == [0, 0, 0, 1, 1, 1]
    assert t.tolist() == [0, 1, 2, 0, 1, 2]


def test_pack_unpack_roundtrip(tiny_stats):
    mean, std = (torch.tensor(v) for v in tiny_stats)
    latents = torch.randn(2, 32, 5, generator=torch.Generator().manual_seed(1))
    rows = av.pack_audio_rows(latents, mean, std)
    assert rows.shape == (10, 32) and rows.dtype == torch.float32
    expected = (latents.transpose(1, 2) - mean.view(1, 1, 32)) / std.view(1, 1, 32)
    assert torch.equal(rows[:5], expected[0]) and torch.equal(rows[5:], expected[1])
    assert torch.equal(av.unpack_audio_rows(rows, 5), expected.transpose(1, 2))


def test_pack_unpack_reject_bad_shapes(tiny_stats):
    mean, std = (torch.tensor(v) for v in tiny_stats)
    with pytest.raises(ValueError):
        av.pack_audio_rows(torch.zeros(1, 32, 4), mean, std)
    with pytest.raises(TypeError):
        av.pack_audio_rows(torch.zeros(2, 32, 4, dtype=torch.float64), mean, std)
    with pytest.raises(ValueError):
        av.unpack_audio_rows(torch.zeros(9, 32), 5)
    with pytest.raises(ValueError, match=r"\[2 \* 3, 32\]"):
        av.unpack_audio_rows(torch.zeros(6, 1), 3)  # one column would broadcast in denormalize
    assert av.unpack_audio_rows(torch.zeros(6, 32), 3).shape == (2, 32, 3)


# --- preprocessing --------------------------------------------------------------------------


def test_prepare_stereo_waveform_upmixes_and_truncates_at_native_rate():
    mono = torch.arange(10, dtype=torch.float64)[None]
    out = av.prepare_stereo_waveform(mono, av.SAMPLE_RATE, max_duration=7 / av.SAMPLE_RATE)
    assert out.dtype == torch.float32 and out.shape == (2, 7) and out.is_contiguous()
    assert torch.equal(out[0], out[1]) and out[0].tolist() == list(range(7))
    pair = stereo(100)
    assert torch.equal(av.prepare_stereo_waveform(pair, av.SAMPLE_RATE), pair)


def test_prepare_stereo_waveform_rejects():
    with pytest.raises(ValueError, match="32000 Hz"):
        av.prepare_stereo_waveform(stereo(100), 44_100)
    for bad in (torch.zeros(3, 10), torch.zeros(10), torch.zeros(2, 0)):
        with pytest.raises(ValueError):
            av.prepare_stereo_waveform(bad, av.SAMPLE_RATE)
    with pytest.raises(TypeError):
        av.prepare_stereo_waveform(torch.zeros(2, 10, dtype=torch.int16), av.SAMPLE_RATE)
    with pytest.raises(ValueError, match="non-finite"):
        av.prepare_stereo_waveform(torch.tensor([[0.0, float("nan")]]), av.SAMPLE_RATE)
    with pytest.raises(ValueError, match="max_duration"):
        av.prepare_stereo_waveform(stereo(100), av.SAMPLE_RATE, max_duration=-1.0)


@pytest.mark.parametrize(("channels", "rate"), [(1, 32_000), (2, 32_000), (1, 44_100), (2, 48_000)])
def test_prepare_stereo_waveform_matches_diffusers(channels, rate):
    diffusers_or_skip()
    pytest.importorskip("torchaudio")
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

    wave = torch.randn(channels, rate // 3, generator=torch.Generator().manual_seed(rate))
    expected = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        wave, rate, av.SAMPLE_RATE, 121 / 24
    )
    out = av.prepare_stereo_waveform(
        wave, rate, max_duration=121 / 24, allow_resample=rate != av.SAMPLE_RATE
    )
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prepare_stereo_waveform_resamples_a_cuda_waveform_on_the_cpu():
    pytest.importorskip("torchaudio")
    wave = torch.randn(2, 48_000 // 3, generator=torch.Generator().manual_seed(3))
    out = av.prepare_stereo_waveform(wave.cuda(), 48_000, allow_resample=True)
    assert out.is_cuda
    assert torch.equal(out.cpu(), av.prepare_stereo_waveform(wave, 48_000, allow_resample=True))


# --- contract on a reduced-width module ------------------------------------------------------


def test_repeated_encode_and_decode_are_identical(contract):
    wave = stereo(800 * 6 - 37)
    runs = [contract.encode_stereo(wave) for _ in range(3)]
    assert len({r.latents_sha256 for r in runs}) == 1 and len({r.rows_sha256 for r in runs}) == 1
    assert all(torch.equal(r.rows, runs[0].rows) for r in runs)
    assert runs[0].num_latents == 6 and runs[0].rows.shape == (12, 32)
    assert runs[0].latents_sha256 == tensor_digest(runs[0].latents)
    decoded = [contract.decode_rows(runs[0].rows, 6) for _ in range(3)]
    assert len({d.waveform_sha256 for d in decoded}) == 1
    assert decoded[0].waveform.shape == (2, 6 * av.HOP_LENGTH)
    assert decoded[0].waveform.abs().max() <= 1.0


def test_channel_bytes_do_not_depend_on_batch(contract):
    target = stereo(800 * 5 + 11, seed=3)
    others = torch.cat([stereo(800 * 5 + 11, seed=s) for s in (4, 5, 6)])
    ref = contract.encode_latents(target)
    assert torch.equal(contract.encode_latents(target[:1])[0], ref[0])
    assert torch.equal(contract.encode_latents(target[1:])[0], ref[1])
    assert torch.equal(contract.encode_latents(torch.cat([target, others]))[:2], ref)
    assert torch.equal(contract.encode_latents(torch.cat([others, target]))[6:], ref)
    z = ref * 0.5
    zr = contract.decode_latents(z)
    assert torch.equal(contract.decode_latents(z[1:])[0], zr[1])
    assert torch.equal(contract.decode_latents(torch.cat([z.flip(-1), z.roll(1, -1), z]))[4:], zr)


def test_stereo_order_is_left_then_right(contract):
    wave = stereo(800 * 4, seed=7)
    enc = contract.encode_stereo(wave)
    swapped = contract.encode_stereo(wave.flip(0))
    assert not torch.equal(enc.latents[0], enc.latents[1])
    assert torch.equal(swapped.latents, enc.latents.flip(0))
    t = enc.num_latents
    assert torch.equal(swapped.rows[:t], enc.rows[t:]) and torch.equal(
        swapped.rows[t:], enc.rows[:t]
    )
    dec = contract.decode_rows(enc.rows, t).waveform
    assert torch.equal(contract.decode_rows(swapped.rows, t).waveform, dec.flip(0))


def test_encode_matches_explicit_mono_calls(contract):
    wave = stereo(800 * 3 + 5)
    with torch.inference_mode(), av.execution_scope(av.STRICT_ENCODE):
        left = contract.model.encode_mean(wave[:1, None])
    assert torch.equal(contract.encode_latents(wave)[:1], left)


def test_contract_rejects_bad_inputs(tiny_model, tiny_stats, contract):
    with pytest.raises(TypeError, match="FP32"):
        av.H3AudioVAEContract(av.H3AudioVAE(**TINY).to(torch.bfloat16), *tiny_stats)
    mean, std = tiny_stats
    for bad_mean, bad_std in (
        ([0.0], std),  # one value would broadcast over all 32 channels
        (mean, std[:31]),
        (mean[:31] + [float("nan")], std),
        (mean, std[:31] + [0.0]),
    ):
        with pytest.raises(ValueError, match="latents_"):
            av.H3AudioVAEContract(tiny_model, bad_mean, bad_std)
    with pytest.raises(ValueError):
        contract.encode_stereo(stereo(800)[:1])
    with pytest.raises(ValueError):
        contract.encode_latents(stereo(800).double())
    with pytest.raises(ValueError):
        contract.decode_latents(torch.zeros(2, 32, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"\[B, 32, T\]"):
        contract.decode_latents(torch.zeros(2, 16, 3))
    with pytest.raises(ValueError):
        contract.decode_rows(torch.zeros(7, 32), 3)
    with pytest.raises(ValueError, match="32"):
        contract.decode_rows(torch.zeros(6, 1), 3)
    other_hop = av.H3AudioVAE(
        **TINY, encoder_rates=(2, 4, 4, 4, 5), decoder_rates=(5, 4, 2, 2, 2, 2, 2)
    )
    assert other_hop.hop_length == 640
    with pytest.raises(ValueError, match="800-sample hop"):
        av.H3AudioVAEContract(other_hop.eval(), *tiny_stats)
    # the right hop, but kernel 8 / stride 5 gives 5L + 1 samples in the first upsample
    odd = av.H3AudioVAE(**TINY, decoder_kernel_sizes=(8, 9, 4, 4, 4, 4, 4))
    assert odd.hop_length == 800
    assert odd.decode(torch.zeros(1, 32, 3)).shape[-1] == (5 * 3 + 1) * 160 != 3 * 800
    with pytest.raises(ValueError, match=r"ups\.0\.0 \(kernel 8, stride 5"):
        av.H3AudioVAEContract(odd.eval(), *tiny_stats)
    with pytest.raises(ValueError):
        tiny_model.encode_mean(torch.zeros(2, 800))


def test_execution_scope_sets_and_restores_flags():
    b = torch.backends
    before = (
        b.cuda.matmul.allow_tf32,
        b.cudnn.allow_tf32,
        b.cudnn.enabled,
        b.cudnn.deterministic,
        b.cudnn.benchmark,
    )
    with av.execution_scope(av.STRICT_ENCODE):
        assert not b.cuda.matmul.allow_tf32 and not b.cudnn.allow_tf32
        assert not b.cudnn.enabled and b.cudnn.deterministic and not b.cudnn.benchmark
    with av.execution_scope(av.STRICT_DECODE):
        assert b.cudnn.enabled and b.cudnn.deterministic and not b.cudnn.allow_tf32
    assert (
        b.cuda.matmul.allow_tf32,
        b.cudnn.allow_tf32,
        b.cudnn.enabled,
        b.cudnn.deterministic,
        b.cudnn.benchmark,
    ) == before


@pytest.fixture
def restore_precision():
    """Undo any precision setting a test makes, whichever API it used."""

    state = av._save_tf32_state()
    yield
    av._restore_tf32_state(state)


def _precision_state() -> tuple:
    m, c = torch.backends.cuda.matmul, torch.backends.cudnn
    if av._has_fp32_precision_api():
        return (m.fp32_precision, c.conv.fp32_precision, c.rnn.fp32_precision)
    return (torch.get_float32_matmul_precision(), c.allow_tf32)


@pytest.mark.parametrize("start", ["highest", "high", "medium"])
@pytest.mark.parametrize("raises", [False, True])
def test_execution_scope_restores_float32_matmul_precision(restore_precision, start, raises):
    """A caller's precision string survives the scope, including ``"medium"`` and an exception.

    ``matmul.allow_tf32`` reads True for both "high" and "medium", so restoring that flag
    turned "medium" into "high" (PyTorch 2.4) or into a mixed legacy/new state in which
    ``get_float32_matmul_precision()`` raises (PyTorch 2.13).
    """

    torch.set_float32_matmul_precision(start)
    before = _precision_state()
    raw = av._raw_fp32_precision()
    entered = []
    # only the body's own error is expected; an error from the scope's setup or teardown fails
    expect = pytest.raises(RuntimeError, match="^inside the scope$") if raises else None
    with expect or contextlib.nullcontext():
        with av.execution_scope(av.STRICT_ENCODE):
            assert not torch.backends.cuda.matmul.allow_tf32
            entered.append(True)
            if raises:
                raise RuntimeError("inside the scope")
    assert entered == [True]
    assert torch.get_float32_matmul_precision() == start
    assert _precision_state() == before
    assert av._raw_fp32_precision() == raw


@pytest.mark.skipif(not av._has_fp32_precision_api(), reason="needs the fp32_precision API")
@pytest.mark.parametrize(
    "setup",
    [
        "",  # PyTorch's defaults: matmul inherits ("none")
        "torch.backends.cudnn.conv.fp32_precision = 'none'",
        "torch.backends.fp32_precision = 'tf32'; torch.backends.cudnn.rnn.fp32_precision = 'none'",
    ],
)
def test_execution_scope_keeps_inheriting_backends_inheriting(restore_precision, setup):
    """Restoring the legacy ``"highest"`` writes ``"ieee"`` into backends that inherited; the
    scope puts every backend's own ``fp32_precision`` (``"none"`` included) back."""

    exec(setup)
    raw = av._raw_fp32_precision()
    for policy in (av.STRICT_ENCODE, av.STRICT_DECODE):
        with av.execution_scope(policy):
            pass
    assert av._raw_fp32_precision() == raw


@pytest.mark.skipif(not av._has_fp32_precision_api(), reason="needs the fp32_precision API")
def test_execution_scope_accepts_and_restores_per_backend_precision(restore_precision):
    """cuDNN conv and RNN precision set separately (new API) are accepted and restored."""

    cudnn = torch.backends.cudnn
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    cudnn.conv.fp32_precision = "ieee"
    cudnn.rnn.fp32_precision = "tf32"
    before = _precision_state()
    with av.execution_scope(av.STRICT_DECODE):
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert cudnn.conv.fp32_precision == "ieee" and cudnn.rnn.fp32_precision == "ieee"
    assert _precision_state() == before


def _encode_decode_digests(contract: av.H3AudioVAEContract, wave: torch.Tensor) -> tuple[str, str]:
    enc = contract.encode_stereo(wave)
    return enc.rows_sha256, contract.decode_rows(enc.rows, enc.num_latents).waveform_sha256


def test_outer_cpu_autocast_does_not_change_hashes(contract):
    wave = stereo(1600)
    expected = _encode_decode_digests(contract, wave)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")  # the outer autocast is really on
        assert _encode_decode_digests(contract, wave) == expected
    assert torch.is_autocast_enabled("cpu") is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_outer_cuda_autocast_does_not_change_hashes(tiny_model, tiny_stats, dtype):
    contract = av.H3AudioVAEContract(copy.deepcopy(tiny_model).cuda(), *tiny_stats)
    wave = stereo(1600)
    expected = _encode_decode_digests(contract, wave)
    with torch.autocast("cuda", dtype=dtype):
        assert torch.is_autocast_enabled("cuda")  # the outer autocast is really on
        assert _encode_decode_digests(contract, wave) == expected


def test_provenance_records_policies(contract):
    info = contract.provenance()
    assert info["contract"] == av.CONTRACT and info["stereo_order"] == ["left", "right"]
    assert info["encode_policy"]["micro_batch"] == 1 and info["decode_policy"]["micro_batch"] == 1
    assert (
        info["encode_policy"]["cudnn_enabled"] is False
        and info["encode_policy"]["sdpa_backend"] == "math"
    )
    assert info["latents_mean_sha256"] == tensor_digest(contract.latents_mean)


# --- Diffusers parity ----------------------------------------------------------------------


@pytest.fixture
def diffusers_pair(tiny_model, tiny_stats):
    diffusers = diffusers_or_skip()
    mean, std = tiny_stats
    ref = diffusers.AutoencoderKLMiniMaxH3Audio(**TINY, latents_mean=mean, latents_std=std).eval()
    ref.load_state_dict(av.checkpoint_state_dict(tiny_model), strict=True)  # checkpoint names
    ours = av.H3AudioVAEContract(
        tiny_model, mean, std, encode_policy=LIBRARY, decode_policy=LIBRARY
    )
    return ref, ours


def test_module_is_bitwise_diffusers(diffusers_pair):
    ref, ours = diffusers_pair
    wave = stereo(800 * 4 - 9)
    with torch.inference_mode():
        latents = ref.encode(wave[:, None]).latent_dist.mode()
        decoded = ref.decode(latents).sample
    assert torch.equal(ours.encode_latents(wave), latents)
    assert torch.equal(ours.decode_latents(latents), decoded[:, 0])


def test_rows_and_waveform_match_diffusers_pipeline_blocks(diffusers_pair):
    """Drive Diffusers' own reference-encode, unpack and audio-decode blocks on the same module."""

    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.minimax_h3 import decoders, encoders, references

    ref, ours = diffusers_pair
    wave = stereo(800 * 5 + 123, seed=11)
    components = types.SimpleNamespace(
        audio_vae=ref,
        _execution_device=torch.device("cpu"),
        audio_latent_channels=32,
        audio_channels=2,
        audio_sampling_rate=av.SAMPLE_RATE,
        patch_size=(1, 2, 2),
        vae_latent_channels=24,
    )
    state = PipelineState()
    state.set(
        "normalized_references",
        [references.MiniMaxH3AudioReference(audio=wave, sample_rate=av.SAMPLE_RATE)],
    )
    _, state = encoders.MiniMaxH3Ref2VAReferenceEncoderStep()(components, state)
    diffusers_rows = state.get("audio_condition_latents")[0]
    enc = ours.encode_stereo(wave)
    assert torch.equal(diffusers_rows, enc.rows)

    # Generated rows after two reference rows: unpack (after-denoise), then decode.
    t = enc.num_latents
    packed = torch.cat([torch.zeros(2, 32), enc.rows])
    for key, value in {
        "latents": torch.zeros(1, 24 * 4),
        "audio_latents": packed,
        "num_condition_video_rows": 0,
        "num_condition_audio_rows": 2,
        "num_latent_frames": 1,
        "latent_height": 2,
        "latent_width": 2,
        "num_audio_latents": t,
    }.items():
        state.set(key, value)
    _, state = decoders.MiniMaxH3AfterDenoiseStep()(components, state)
    assert torch.equal(state.get("audio_latents"), av.unpack_audio_rows(enc.rows, t))
    _, state = decoders.MiniMaxH3AudioDecodeStep()(components, state)
    diffusers_audio = state.get("audio")  # (1, 2, samples)
    assert diffusers_audio.shape == (1, 2, t * av.HOP_LENGTH)
    assert torch.equal(diffusers_audio[0], ours.decode_rows(enc.rows, t).waveform)


def test_strict_policy_is_close_to_diffusers(diffusers_pair, contract):
    ref, _ = diffusers_pair
    wave = stereo(800 * 6, seed=5)
    strict = contract.encode_stereo(wave)
    with torch.inference_mode():
        latents = ref.encode(wave[:, None]).latent_dist.mode()
        audio = ref.decode(latents).sample[:, 0]
    torch.testing.assert_close(strict.latents, latents, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(contract.decode_latents(latents), audio, rtol=1e-4, atol=1e-5)


# --- checkpoint reader ----------------------------------------------------------------------


def _fake_component(root: Path, meta_overrides: dict | None = None, **overrides) -> Path:
    """Write a fake component; ``overrides`` go to ``config.json``, ``meta_overrides`` to the
    metadata kwargs (``sample_rate`` is in both)."""

    meta = {
        "attn_proj": True,
        "decoder_dim": 1024,
        "decoder_rates": list(av.DECODER_RATES),
        "decoder_type": "bigvgan",
        "encoder_dim": 64,
        "encoder_rates": list(av.ENCODER_RATES),
        "latent_dim": 2048,
        "sample_rate": 32000,
        "vae_latent_channels": 32,
    }
    config = {
        "sample_rate": 32000,
        "latent_channels": 32,
        "latents_mean": [0.0] * 32,
        "latents_std": [1.0] * 32,
        "source_metadata_path": "metadata.json",
        "source_safetensors_path": "model.safetensors",
    }
    unknown = (set(overrides) - set(config)) | (set(meta_overrides or {}) - set(meta))
    assert not unknown, f"no such fake key: {sorted(unknown)}"
    config.update(overrides)
    meta.update(meta_overrides or {})
    (root / "metadata.json").write_text(json.dumps({"metadata": {"kwargs": meta}}))
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors").write_bytes(b"not weights")
    return root


def test_checkpoint_reader_validates_contract(tmp_path):
    ck = av.AudioVAECheckpoint.read(_fake_component(tmp_path))
    assert ck.latents_std == (1.0,) * 32 and ck.kwargs["latent_dim"] == 2048 and ck.sha256 is None
    for bad_config, bad_meta in (
        ({"sample_rate": 44100}, {}),
        ({}, {"sample_rate": 44100}),
        ({"latent_channels": 64}, {}),
        ({}, {"encoder_rates": [2, 4, 8, 8]}),
        ({"latents_std": [0.0] * 32}, {}),
        ({"latents_mean": [0.0] * 31 + [float("nan")]}, {}),
        ({"latents_mean": [0.0] * 31 + [1e39]}, {}),  # finite in JSON, inf in FP32
        ({"latents_std": [1.0] * 31}, {}),
        ({}, {"decoder_type": "hifigan"}),
    ):
        with pytest.raises(ValueError, match="not the H3 audio VAE contract"):
            av.AudioVAECheckpoint.read(_fake_component(tmp_path, bad_meta, **bad_config))
    with pytest.raises(ValueError, match="sha256"):
        av.AudioVAECheckpoint.read(_fake_component(tmp_path), verify_sha256="0" * 64)


def test_checkpoint_state_dict_round_trips(tiny_model):
    sd = av.checkpoint_state_dict(tiny_model)
    assert "encoder.block.0.weight_g" in sd and not any("parametrizations" in k for k in sd)
    fresh = av.H3AudioVAE(**TINY)
    fresh.load_state_dict(sd, strict=True)
    assert all(
        torch.equal(a, b)
        for a, b in zip(fresh.state_dict().values(), tiny_model.state_dict().values(), strict=True)
    )


def test_pinned_weights_hash():
    assert av.pinned_audio_vae_sha256() == av.pinned_audio_vae_sha256("Ref2VA")
    assert (
        av.pinned_audio_vae_sha256()
        == "37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2"
    )
    with pytest.raises(KeyError):
        av.pinned_audio_vae_sha256("T2VA")


def test_benchmark_no_remote_encode_runs_on_the_port(tiny_model):
    """``--no-remote`` feeds the rl_engine port to the engines' encode: it has no ``preprocess``."""

    import importlib.util

    script = Path(__file__).resolve().parents[3] / "benchmarks/models/benchmark_h3_audio_vae.py"
    spec = importlib.util.spec_from_file_location("benchmark_h3_audio_vae_under_test", script)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    x = torch.randn(2, 3 * tiny_model.hop_length - 123, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        got = bench.dac_encode_mean(tiny_model, x)
        want = tiny_model.encode_mean(x[:, None])
    assert got.shape[-1] == 3 and torch.equal(got, want)


def test_benchmark_log_bars_raise_the_axis_over_the_largest_value():
    pytest.importorskip("matplotlib")
    import importlib.util

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    script = Path(__file__).resolve().parents[3] / "benchmarks/models/benchmark_h3_audio_vae.py"
    spec = importlib.util.spec_from_file_location("benchmark_h3_audio_vae_under_test", script)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    fig, ax = plt.subplots()
    bench._grouped_log_bars(ax, ["encode", "decode"], {"ours": [0.0, 5e-6]}, 1e-3)
    assert ax.get_ylim()[1] == 1e-3  # values under the cap keep it
    bench._grouped_log_bars(ax, ["encode", "decode"], {"ours": [0.0, 5e-3]}, 1e-3)
    assert ax.get_ylim()[1] > 5e-3
    plt.close(fig)


def _load_benchmark():
    import importlib.util

    script = Path(__file__).resolve().parents[3] / "benchmarks/models/benchmark_h3_audio_vae.py"
    spec = importlib.util.spec_from_file_location("benchmark_h3_audio_vae_under_test", script)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def test_benchmark_rejects_negative_counts_before_any_work(tmp_path, capsys):
    """The count validator itself refuses (argparse, exit code 2, the flag named), not a later
    prerequisite such as the CUDA check, which also exits."""

    bench = _load_benchmark()
    for flag in ("--repeats", "--warmup"):
        with pytest.raises(SystemExit) as exit_info:
            bench.main(["--checkpoint", str(tmp_path), flag, "-1", "--out", str(tmp_path / "x")])
        assert exit_info.value.code == 2
        assert f"argument {flag}: must be >= 0, got -1" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_benchmark_cold_start_input_is_per_run_and_removed(tmp_path, monkeypatch):
    """The workers' input lives in a directory of its own run, under an ``--out`` parent that
    may not exist yet, and is removed when a worker fails."""

    import subprocess

    bench = _load_benchmark()
    written = []

    def fake_inputs(strict, frames, offset, path):
        torch.save({"encode": torch.zeros(1), "decode": torch.zeros(1)}, path)
        written.append(path)
        return path

    def worker(cmd, **kwargs):
        assert Path(cmd[cmd.index("--cold-start-input") + 1]) == written[-1]
        assert torch.load(written[-1], weights_only=True)["encode"].numel() == 1
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="worker failed")

    monkeypatch.setattr(bench, "cold_start_inputs", fake_inputs)
    monkeypatch.setattr(subprocess, "run", worker)
    scratch = tmp_path / "new" / "out"  # does not exist yet
    for _ in range(2):
        with pytest.raises(RuntimeError, match="worker failed"):
            bench.cold_start([], ["ours"], None, 121, 0, scratch)
    assert written[0] != written[1] and written[0].parent != written[1].parent
    assert scratch.is_dir() and not any(scratch.iterdir())


def test_benchmark_batch8_last_reads_the_final_slot():
    """A path that changes only the last batch slot must fail ``batch8_last``."""

    bench = _load_benchmark()
    target, others = stereo(64), torch.cat([stereo(64, seed=s) for s in (1, 2, 3)])

    def last_slot_differs(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        if x.shape[0] == 8:
            out[-1] += 1.0
        return out

    res = bench.invariance(last_slot_differs, target, others)
    assert res["max_abs"]["batch8_last"] > 0.5 and not res["bitwise_equal"]["batch8_last"]
    assert res["max_abs"]["batch8_first"] == 0.0 and not res["batch_invariant"]


def test_benchmark_invariance_compares_bytes_not_values():
    """``-0.0`` vs ``+0.0`` is a zero ``max_abs`` but not byte equality, so not batch invariant."""

    bench = _load_benchmark()
    target, others = torch.zeros(2, 8), torch.zeros(6, 8)

    def negative_zero_alone(x: torch.Tensor) -> torch.Tensor:
        return -x if x.shape[0] == 1 else x.clone()

    res = bench.invariance(negative_zero_alone, target, others)
    assert res["max_abs"]["mono_alone"] == 0.0 and not res["bitwise_equal"]["mono_alone"]
    assert not res["batch_invariant"] and res["repeat_equal"]


def test_benchmark_log_cap_covers_the_largest_value_in_either_panel():
    bench = _load_benchmark()
    assert bench.log_cap([2.0, 0.1], 1.0) == 10.0
    assert bench.log_cap([0.1], 1.0) == 1.0 and bench.log_cap([], 1e-3) == 1e-3


def test_benchmark_two_panel_charts_share_one_cap(tmp_path):
    """The larger peak in the first panel must survive the second panel's render."""

    pytest.importorskip("matplotlib")
    bench = _load_benchmark()

    def entry(drift: float, err: float) -> dict:
        return {
            "max_abs": {"repeat": 0.0, "mono_alone": drift},
            "vs_fp64_max_abs": err,
            "latency": {"median_ms": 1.0},
        }

    report = {
        "paths": {"ours": "x"},
        "provenance": {"gpu": "GPU"},
        "cold_start": {
            "ours": {"encode_first_vs_third_max_abs": 0.0, "decode_first_vs_third_max_abs": 0.0}
        },
        "cases": [
            {
                "samples": 32_000,
                "latents_per_channel": 40,
                "paths": {"ours": {"encode": entry(2.0, 3.0), "decode": entry(0.1, 0.2)}},
            }
        ],
    }
    import matplotlib.pyplot as plt

    seen = []
    real_close = plt.close

    def close(fig=None):
        if hasattr(fig, "axes"):  # a figure; ``matplotlib.use`` may also close("all")
            seen.append([ax.get_ylim()[1] for ax in fig.axes])
        real_close(fig)

    plt.close = close
    try:
        bench.plot(report, tmp_path / "chart")
    finally:
        plt.close = real_close
    bi, accuracy = seen[0], seen[1]
    assert bi == [10.0, 10.0] and accuracy == [10.0, 10.0]


def test_checkpoint_weights_path_is_the_verified_file(tmp_path):
    import hashlib

    root = _fake_component(tmp_path, source_safetensors_path="weights.safetensors")
    (root / "weights.safetensors").write_bytes(b"other weights")
    ck = av.AudioVAECheckpoint.read(root)
    assert ck.weights_path == root / "weights.safetensors"
    digest = hashlib.sha256(b"other weights").hexdigest()
    assert av.AudioVAECheckpoint.read(root, verify_sha256=digest).sha256 == digest
    with pytest.raises(ValueError, match="sha256"):  # model.safetensors is not what is verified
        av.AudioVAECheckpoint.read(root, verify_sha256=hashlib.sha256(b"not weights").hexdigest())


def test_benchmark_verify_remote_rejects_unpinned_weights(tmp_path):
    bench = _load_benchmark()
    with pytest.raises(ValueError, match="sha256"):
        bench.verify_remote(_fake_component(tmp_path))


def test_native_forward_keeps_fp64(tiny_model):
    model = copy.deepcopy(tiny_model).double()
    x = stereo(1600).double()[:, None]
    with torch.inference_mode():
        z = model.encode_mean_native(x)
        assert z.dtype == torch.float64 and model.encode_mean(x).dtype == torch.float32
        assert torch.equal(model.encode_mean(x), z.float())
        w = model.decode_native(z)
        assert w.dtype == torch.float64 and torch.equal(model.decode(z), w.float())


# --- released weights on CUDA ----------------------------------------------------------------


@pytest.fixture(scope="module")
def released():
    root = os.environ.get(AUDIO_VAE_ENV)
    if not root:
        pytest.skip(f"{AUDIO_VAE_ENV} not set (path to the Hub FL2VA/audio_vae directory)")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    ck = av.AudioVAECheckpoint.read(root, verify_sha256=av.pinned_audio_vae_sha256())
    return ck, av.H3AudioVAEContract.from_checkpoint(ck, "cuda")


def released_clip(seed: int) -> torch.Tensor:
    num = round(121 / 24 * av.SAMPLE_RATE) - 123  # 121 frames, not a hop multiple
    t = torch.arange(num, dtype=torch.float64) / av.SAMPLE_RATE
    tone = torch.stack([torch.sin(2 * math.pi * (220 + 55 * seed + 30 * c) * t) for c in range(2)])
    noise = torch.randn(2, num, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (0.25 * tone + 0.03 * noise).float()


def test_released_encode_repeats_and_is_batch_invariant(released):
    _, contract = released
    target = released_clip(0)
    enc = contract.encode_stereo(target)
    assert enc.latents.shape == (2, 32, av.audio_latent_num_frames(121)) and enc.rows.shape == (
        404,
        32,
    )
    assert contract.encode_stereo(target).rows_sha256 == enc.rows_sha256
    others = torch.cat([released_clip(s) for s in (1, 2, 3)])
    for batch, idx in (
        (target[:1], 0),
        (torch.cat([target, others]), 0),
        (torch.cat([others, target]), 6),
    ):
        assert torch.equal(contract.encode_latents(batch)[idx].cpu(), enc.latents[0])


def test_released_decode_repeats_and_is_batch_invariant(released):
    _, contract = released
    enc = contract.encode_stereo(released_clip(0))
    dec = contract.decode_rows(enc.rows, enc.num_latents)
    assert dec.waveform.shape == (2, enc.num_latents * av.HOP_LENGTH)
    assert contract.decode_rows(enc.rows, enc.num_latents).waveform_sha256 == dec.waveform_sha256
    z = contract.denormalize(enc.rows, enc.num_latents)
    batch = torch.cat([z.roll(3, -1), z, z * 0.9])
    assert torch.equal(contract.decode_latents(batch)[2:4].cpu(), dec.waveform)
    assert torch.equal(contract.decode_latents(z[1:])[0].cpu(), dec.waveform[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_released_hashes_ignore_outer_autocast(released, dtype):
    """Released weights: an outer CUDA autocast leaves the strict row and waveform hashes."""

    _, contract = released
    wave = released_clip(0)
    expected = _encode_decode_digests(contract, wave)
    with torch.autocast("cuda", dtype=dtype):
        assert _encode_decode_digests(contract, wave) == expected


def test_released_matches_diffusers(released):
    """Bitwise at equal flags and per-channel shapes; default-flag Diffusers within 1% rel L2."""

    diffusers = diffusers_or_skip()
    from safetensors.torch import load_file

    ck, contract = released
    ref = diffusers.AutoencoderKLMiniMaxH3Audio(
        latents_mean=list(ck.latents_mean), latents_std=list(ck.latents_std)
    )
    ref.load_state_dict(load_file(str(ck.weights_path)), strict=True)
    ref = ref.cuda().eval()
    wave = released_clip(0)
    enc = contract.encode_stereo(wave)
    with torch.inference_mode():
        with av.execution_scope(av.STRICT_ENCODE):
            same_flags = torch.cat(
                [ref.encode(wave[i : i + 1].cuda()[:, None]).latent_dist.mode() for i in (0, 1)]
            )
        with av.execution_scope(av.STRICT_DECODE):
            same_flags_audio = torch.cat(
                [ref.decode(enc.latents[i : i + 1].cuda()).sample for i in (0, 1)]
            )
        default_latents = ref.encode(wave.cuda()[:, None]).latent_dist.mode().cpu()
        default_audio = ref.decode(enc.latents.cuda()).sample[:, 0].cpu()
    assert torch.equal(same_flags.cpu(), enc.latents)
    ours_audio = contract.decode_latents(enc.latents).cpu()
    assert torch.equal(same_flags_audio[:, 0].cpu(), ours_audio)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.double() - b.double()).norm() / b.double().norm())

    assert rel(enc.latents, default_latents) < 1e-2
    assert rel(ours_audio, default_audio) < 1e-2
