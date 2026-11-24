# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Batch invariance, determinism, accuracy and latency of H3 audio VAE paths (RFC #420 Phase B).

Paths, all on the released FP32 weights (``<ckpt>/FL2VA/audio_vae``):

* ``diffusers``  — ``AutoencoderKLMiniMaxH3Audio`` (``--diffusers-src``), torch default flags,
  stereo as one batch-2 call (the Diffusers pipeline);
* ``sglang``     — (``--sglang``) SGLang's own ``DacAudioVAE`` port under its own encode context
  (TF32 off, cuDNN off, math SDPA) and decode context (TF32 off, deterministic cuDNN);
* ``vllm-omni``  — the checkpoint's remote code (``--remote-src``, what vLLM-Omni loads) under
  vLLM-Omni's recipe: the same encode flags as SGLang, decode with default flags, both directions
  inside ``torch.jit.optimized_execution(False)`` (its ``_AudioVAEDeterminismContext`` enters it
  for encode, ``decode_latent`` for decode);
* ``ours``       — :class:`rl_engine.models.minimax_h3.audio_vae.H3AudioVAEContract` strict
  policies (micro-batch 1 per mono channel).

With ``--no-remote`` the vLLM-Omni recipe runs on the rl_engine port (op-identical).
Per clip length the script records, for encode and decode separately: repeat equality, the max
absolute difference of one logical mono channel across batch compositions (alone, stereo pair,
first/last of an 8-item batch) together with byte equality of those outputs, error against an
FP64 run of the same graph, bitwise parity between paths (sha256), and CUDA-synchronised wall
latency of a stereo call. The cold start is measured in a fresh process per path and direction:
the first call after loading against the third. ``--plot`` renders the batch-invariance,
accuracy, cold-start and latency charts from the JSON; ``--markdown`` prints the same numbers as
tables. Every loader opens the weight file that was verified against the pinned sha256.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.models.minimax_h3 import audio_vae as av  # noqa: E402
from rl_engine.models.minimax_h3.trajectory import tensor_digest  # noqa: E402

DEFAULT_FRAMES = (22, 121, 241, 481)  # 17n+5 video lengths: 0.9 s, 5.0 s, 10.0 s, 20.0 s
# vLLM-Omni's _AudioVAEDeterminismContext (TF32 off, cuDNN off, math SDPA), re-stated here:
# vllm_omni itself is not imported.
ENGINE_ENCODE = av.ExecutionPolicy("engine-encode", None, cudnn_enabled=False, sdpa_backend="math")


@contextlib.contextmanager
def torch_default_flags():
    """Torch's process defaults, whatever earlier code set (cuDNN TF32 on, matmul TF32 off)."""

    b = torch.backends
    saved = (
        b.cuda.matmul.allow_tf32,
        b.cudnn.allow_tf32,
        b.cudnn.benchmark,
        b.cudnn.deterministic,
        b.cudnn.enabled,
    )
    b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32 = False, True
    b.cudnn.benchmark, b.cudnn.deterministic, b.cudnn.enabled = False, False, True
    try:
        yield
    finally:
        (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
        ) = saved


def stereo_clip(index: int, num_samples: int) -> torch.Tensor:
    """Deterministic, distinct left/right test signal: two tones plus seeded noise per channel."""

    out = []
    for channel in range(2):
        seed = 1000 * index + channel
        g = torch.Generator().manual_seed(seed)
        t = torch.arange(num_samples, dtype=torch.float64) / av.SAMPLE_RATE
        f0, f1 = 110.0 * (1 + seed % 7), 1375.0 + 37.0 * (seed % 11)
        tone = 0.25 * torch.sin(2 * math.pi * f0 * t) + 0.1 * torch.sin(
            2 * math.pi * f1 * t + channel
        )
        out.append(
            (tone + 0.03 * torch.randn(num_samples, generator=g, dtype=torch.float64)).float()
        )
    return torch.stack(out)


# --- paths ----------------------------------------------------------------------------------


class Path_:
    def __init__(
        self,
        name: str,
        encode: Callable[[torch.Tensor], torch.Tensor],
        decode: Callable[[torch.Tensor], torch.Tensor],
        source: str,
    ) -> None:
        self.name, self.encode, self.decode, self.source = name, encode, decode, source


def load_remote(remote_src: Path) -> torch.nn.Module:
    """Import the checkpoint's remote-code bundle (no ``__init__.py``) as a namespace package."""

    package = types.ModuleType("h3_remote_audio_vae")
    package.__path__ = [str(remote_src)]
    sys.modules[package.__name__] = package
    entry = importlib.import_module(f"{package.__name__}.minimax_h3_audio_vae")
    return entry.MiniMaxH3AudioVAE.from_pretrained(str(remote_src)).model


def dac_encode_mean(model: Any, x: torch.Tensor) -> torch.Tensor:
    """The engines' encode on ``[B, samples]``: preprocess, encoder, pre_block, mean_proj."""

    if isinstance(model, av.H3AudioVAE):  # --no-remote: the port has no preprocess; same ops
        return model.encode_mean(x[:, None])
    audio = model.preprocess(x[:, None], av.SAMPLE_RATE)
    z = model.encoder(audio)
    z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
    return model.mean_proj(z).float()


def sglang_path(ckpt: av.AudioVAECheckpoint, device: str) -> Path_:
    """SGLang 0.5.21's own audio VAE module and its own encode/decode determinism contexts."""

    from sglang.multimodal_gen.runtime.server_args.server_args import set_global_server_args

    class _ServerArgsStub:  # only the attention selector and SP-mode resolver read it
        attention_backend = None
        kv_gather_degree = 1

        def __getattr__(self, name: str) -> None:
            return None

    set_global_server_args(_ServerArgsStub())  # type: ignore[arg-type]
    from safetensors.torch import load_file
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import DacAudioVAE

    stages = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3"
    reference_encoding = importlib.import_module(f"{stages}.reference_encoding")
    decoding = importlib.import_module(f"{stages}.stages.decoding")

    kw = ckpt.kwargs
    model = DacAudioVAE(
        encoder_dim=int(kw["encoder_dim"]),
        encoder_rates=list(kw["encoder_rates"]),
        latent_dim=int(kw["latent_dim"]),
        decoder_dim=int(kw["decoder_dim"]),
        decoder_rates=list(kw["decoder_rates"]),
        sample_rate=int(kw["sample_rate"]),
        vae_latent_channels=int(kw["vae_latent_channels"]),
        attn_proj=bool(kw["attn_proj"]),
        decoder_type=str(kw["decoder_type"]),
    )
    model.load_state_dict(load_file(str(ckpt.weights_path)), strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    def enc(x: torch.Tensor) -> torch.Tensor:
        with (
            torch.inference_mode(),
            reference_encoding._AudioVAEDeterminismContext(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            return dac_encode_mean(model, x)

    def dec(z: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), decoding._deterministic_audio_decode_context():
            return model.decode(z).float()[:, 0]

    import sglang

    return Path_("sglang", enc, dec, f"sglang {getattr(sglang, '__version__', '?')} DacAudioVAE")


def verify_remote(remote_src: Path) -> str:
    """sha256 of the weight file the remote bundle's loader opens, checked against the pin.

    The remote code loads ``<remote_src>/<config.json source_safetensors_path>``;
    :meth:`AudioVAECheckpoint.read` hashes that same file and rejects a bundle whose weights are
    not the pinned ones, so a vLLM-Omni difference cannot be a weight difference.
    """

    remote = av.AudioVAECheckpoint.read(remote_src, verify_sha256=av.pinned_audio_vae_sha256())
    assert remote.sha256 is not None
    return remote.sha256


def build_paths(
    args: argparse.Namespace,
    ckpt: av.AudioVAECheckpoint,
    device: str,
    only: str | None = None,
) -> list[Path_]:
    """The paths to measure; ``only`` builds just one (the cold-start worker)."""

    def wanted(name: str) -> bool:
        return only is None or only == name

    paths: list[Path_] = []
    if args.diffusers_src and wanted("diffusers"):
        sys.path.insert(0, str(Path(args.diffusers_src).expanduser()))
        from diffusers import AutoencoderKLMiniMaxH3Audio
        from safetensors.torch import load_file

        ref = AutoencoderKLMiniMaxH3Audio(
            latents_mean=list(ckpt.latents_mean), latents_std=list(ckpt.latents_std)
        )
        ref.load_state_dict(load_file(str(ckpt.weights_path)), strict=True)
        ref = ref.to(device).eval()

        def d_enc(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch_default_flags():
                return ref.encode(x[:, None]).latent_dist.mode().float()

        def d_dec(z: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch_default_flags():
                return ref.decode(z).sample.float()[:, 0]

        paths.append(Path_("diffusers", d_enc, d_dec, f"diffusers {args.diffusers_rev}"))

    if args.sglang and wanted("sglang"):
        paths.append(sglang_path(ckpt, device))

    if wanted("vllm-omni"):
        paths.append(vllm_omni_path(args, ckpt, device))

    if wanted("ours"):
        contract = av.H3AudioVAEContract.from_checkpoint(ckpt, device)
        paths.append(
            Path_(
                "ours", contract.encode_latents, contract.decode_latents, f"rl_engine {av.CONTRACT}"
            )
        )
    return paths


def vllm_omni_path(args: argparse.Namespace, ckpt: av.AudioVAECheckpoint, device: str) -> Path_:
    """vLLM-Omni's recipe on the checkpoint's remote code (or, ``--no-remote``, the port)."""

    if args.remote_src:
        verify_remote(Path(args.remote_src).expanduser())
        remote = (
            load_remote(Path(args.remote_src).expanduser())
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        engine_src = "checkpoint remote code"
    else:
        remote = ckpt.build(device)
        engine_src = "rl_engine port (op-identical)"

    # Both directions inside optimized_execution(False), as vLLM-Omni aae3f386 runs them:
    # encode_waveform's _AudioVAEDeterminismContext enters it, and so does decode_latent.
    def vo_enc(x: torch.Tensor) -> torch.Tensor:
        with (
            torch.inference_mode(),
            av.execution_scope(ENGINE_ENCODE),
            torch.jit.optimized_execution(False),
        ):
            return dac_encode_mean(remote, x)

    def vo_dec(z: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch_default_flags(), torch.jit.optimized_execution(False):
            return remote.decode(z).float()[:, 0]

    return Path_(
        "vllm-omni", vo_enc, vo_dec, f"vllm-omni {args.vllm_omni_rev} recipe on {engine_src}"
    )


# --- measurements ---------------------------------------------------------------------------


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def invariance(
    fn: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, others: torch.Tensor
) -> dict[str, Any]:
    """The target's left channel across batch compositions vs its stereo call.

    ``max_abs`` is the numerical difference; ``bitwise_equal`` compares the bytes (sha256), so a
    ``+0.0`` / ``-0.0`` difference shows there and not in ``max_abs``. ``batch8_last`` puts the
    left channel in the final slot of the 8-item batch (right channel just before it).
    """

    stereo = fn(target)
    variants = {
        "repeat": fn(target)[0],
        "mono_alone": fn(target[:1])[0],
        "batch8_first": fn(torch.cat([target, others]))[0],
        "batch8_last": fn(torch.cat([others, target.flip(0)]))[-1],
    }
    reference = tensor_digest(stereo[0].contiguous())
    diffs = {k: max_abs(v, stereo[0]) for k, v in variants.items()}
    bitwise = {k: tensor_digest(v.contiguous()) == reference for k, v in variants.items()}
    return {
        "max_abs": diffs,
        "bitwise_equal": bitwise,
        "batch_invariant": all(v for k, v in bitwise.items() if k != "repeat"),
        "repeat_equal": bitwise["repeat"],
        "stereo": stereo,
    }


def timed(fn: Callable[[], Any], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "samples_ms": samples}


def source_state() -> dict[str, Any]:
    """Commit and cleanliness of the checkout this script runs from (None outside git)."""

    import subprocess

    def git(*cmd: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(ROOT), *cmd], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "clean": None if status is None else status == ""}


COLD_MARK = "COLD_START_RESULT "


def cold_start_inputs(strict: av.H3AudioVAEContract, frames: int, offset: int, path: Path) -> Path:
    """Write the cold-start encode input and its strict encoding (the decode input) to ``path``.

    Prepared in the parent, so a worker's first model call is the measured one.
    """

    num_samples = round(frames / av.VIDEO_FPS * av.SAMPLE_RATE) + offset
    x = stereo_clip(99, num_samples)
    enc = strict.encode_stereo(x)
    z = strict.denormalize(enc.rows, enc.num_latents).cpu()
    torch.save({"encode": x, "decode": z}, path)
    return path


def cold_start_worker(args: argparse.Namespace) -> int:
    """One path, one direction, in this (fresh) process: first call against the third."""

    name, kind = args.cold_start_worker
    ckpt = av.AudioVAECheckpoint.read(args.checkpoint, verify_sha256=av.pinned_audio_vae_sha256())
    (p,) = build_paths(args, ckpt, "cuda", only=name)
    x = torch.load(args.cold_start_input, weights_only=True)[kind].to("cuda")
    fn = p.encode if kind == "encode" else p.decode
    outs = [fn(x) for _ in range(3)]
    result = {
        "first_vs_third_max_abs": max_abs(outs[0], outs[2]),
        "first_vs_third_bitwise": tensor_digest(outs[0].contiguous())
        == tensor_digest(outs[2].contiguous()),
    }
    print(COLD_MARK + json.dumps(result), flush=True)
    return 0


def cold_start(
    argv: list[str],
    names: list[str],
    strict: av.H3AudioVAEContract,
    frames: int,
    offset: int,
    scratch: Path,
) -> dict[str, Any]:
    """First call vs third call of each path and direction, each in a fresh process.

    TorchScript's profiling executor re-fuses scripted functions after the first calls and keeps
    that state per process, so a fresh model instance in a used process is not a cold start.

    The workers' input is written to a directory of this invocation alone (created under
    ``scratch``, which is created if needed), removed when this returns or raises.
    """

    import tempfile

    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3-audio-vae-cold-start-", dir=scratch) as tmp:
        inputs = cold_start_inputs(strict, frames, offset, Path(tmp) / "inputs.pt")
        return _cold_start_workers(argv, names, inputs)


def _cold_start_workers(argv: list[str], names: list[str], inputs: Path) -> dict[str, Any]:
    """Run each path's worker on ``inputs``, one fresh process per path and direction."""

    import subprocess

    out: dict[str, Any] = {}
    for name in names:
        out[name] = {}
        for kind in DIRS:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *argv,
                "--cold-start-worker",
                name,
                kind,
                "--cold-start-input",
                str(inputs),
            ]
            done = subprocess.run(cmd, capture_output=True, text=True, check=False)
            lines = [ln for ln in done.stdout.splitlines() if ln.startswith(COLD_MARK)]
            if done.returncode != 0 or len(lines) != 1:
                raise RuntimeError(
                    f"cold-start worker {name}/{kind} failed ({done.returncode}):\n"
                    f"{done.stderr[-2000:]}"
                )
            res = json.loads(lines[0][len(COLD_MARK) :])
            out[name][f"{kind}_first_vs_third_max_abs"] = res["first_vs_third_max_abs"]
            out[name][f"{kind}_first_vs_third_bitwise"] = res["first_vs_third_bitwise"]
        print(f"[cold start] {name:<10} {out[name]}", flush=True)
    return out


def warm_up(paths: list[Path_], strict: av.H3AudioVAEContract, frames: int, offset: int) -> None:
    """Three calls per path and direction before any measurement in this process.

    The first calls of a TorchScript-scripted Snake differ from later ones (the profiling
    executor re-fuses), which the cold start measures in child processes. Batch invariance and
    repeat equality are about the warm state, so they must not include a process's first call.
    """

    num_samples = round(frames / av.VIDEO_FPS * av.SAMPLE_RATE) + offset
    x = stereo_clip(99, num_samples).to(strict.device)
    enc = strict.encode_stereo(x.cpu())
    z = strict.denormalize(enc.rows, enc.num_latents)
    for p in paths:
        for _ in range(3):
            p.encode(x)
            p.decode(z)


def run(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    device = "cuda"
    ckpt = av.AudioVAECheckpoint.read(args.checkpoint, verify_sha256=av.pinned_audio_vae_sha256())
    strict = av.H3AudioVAEContract.from_checkpoint(ckpt, device)
    # Cold start first, in child processes, before this process runs any model.
    path_names = (
        (["diffusers"] if args.diffusers_src else [])
        + (["sglang"] if args.sglang else [])
        + ["vllm-omni", "ours"]
    )
    cold = cold_start(argv, path_names, strict, args.frames[0], args.sample_offset, args.out.parent)
    paths = build_paths(args, ckpt, device)
    assert [p.name for p in paths] == path_names
    warm_up(paths, strict, args.frames[0], args.sample_offset)
    fp64 = ckpt.build(device).double()
    report: dict[str, Any] = {
        "schema": "rlkernel.h3.audio_vae_bench.v2",
        "checkpoint": {
            "repo": "MiniMaxAI/MiniMax-H3",
            "component": "FL2VA/audio_vae",
            "model_safetensors_sha256": ckpt.sha256,
            "remote_model_safetensors_sha256": (
                verify_remote(Path(args.remote_src).expanduser()) if args.remote_src else None
            ),
        },
        "provenance": {**strict.provenance(), "source": source_state()},
        "paths": {p.name: p.source for p in paths},
        "cold_start_mode": "fresh process per path and direction, first call vs third",
        "cold_start": cold,
        "cases": [],
    }
    for frames in args.frames:
        num_samples = round(frames / av.VIDEO_FPS * av.SAMPLE_RATE) + args.sample_offset
        target = stereo_clip(0, num_samples).to(device)
        others = torch.cat([stereo_clip(i, num_samples) for i in range(1, 4)]).to(device)
        with torch.inference_mode():  # FP64 outputs stay FP64 through the subtraction
            z64 = fp64.encode_mean_native(target.double()[:, None])
        # One shared decode input for every path: the strict encoding, denormalised.
        enc = strict.encode_stereo(target.cpu())
        z_dec = strict.denormalize(enc.rows, enc.num_latents)
        z_others = torch.cat([z_dec.roll(k + 1, dims=-1) * (1 + 0.01 * k) for k in range(3)])
        with torch.inference_mode():
            w64 = fp64.decode_native(z_dec.double())[:, 0]
        case: dict[str, Any] = {
            "video_frames": frames,
            "samples": num_samples,
            "latents_per_channel": enc.num_latents,
            "paths": {},
        }
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for p in paths:
            e = invariance(p.encode, target, others)
            d = invariance(p.decode, z_dec, z_others)
            outputs[p.name] = {"encode": e.pop("stereo"), "decode": d.pop("stereo")}
            e["vs_fp64_max_abs"] = max_abs(outputs[p.name]["encode"], z64)
            d["vs_fp64_max_abs"] = max_abs(outputs[p.name]["decode"], w64)
            e["sha256"] = tensor_digest(outputs[p.name]["encode"])
            d["sha256"] = tensor_digest(outputs[p.name]["decode"])
            if args.repeats:
                e["latency"] = timed(lambda p=p, x=target: p.encode(x), args.warmup, args.repeats)
                d["latency"] = timed(lambda p=p, z=z_dec: p.decode(z), args.warmup, args.repeats)
            case["paths"][p.name] = {"encode": e, "decode": d}
            print(
                f"[{frames:>3} frames] {p.name:<10} enc BI={e['batch_invariant']!s:<5} "
                f"{e['max_abs']} dec BI={d['batch_invariant']!s:<5} {d['max_abs']}",
                flush=True,
            )
        for kind in DIRS:  # byte equality: the digests recorded above, not torch.equal
            ours_sha = case["paths"]["ours"][kind]["sha256"]
            case[f"{kind}_bitwise_equal_to_ours"] = {
                n: case["paths"][n][kind]["sha256"] == ours_sha for n in outputs
            }
        case["rows_sha256"] = enc.rows_sha256
        case["latents_sha256"] = enc.latents_sha256
        report["cases"].append(case)
    return report


# --- charts ---------------------------------------------------------------------------------


# Fixed categorical order (blue, orange, aqua, yellow); identity is also carried by direct labels.
COLORS = {"diffusers": "#2a78d6", "sglang": "#eb6834", "vllm-omni": "#1baf7a", "ours": "#eda100"}
# One display name per path, shared by the charts and the --markdown tables.
LABELS = {"diffusers": "Diffusers", "sglang": "SGLang", "vllm-omni": "vLLM-Omni", "ours": "ours"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"
LOG_FLOOR = 1e-12  # bottom of the log axes; exact zeros are drawn there and labelled
ZERO_LABEL = "= 0"  # a max |Δ| of exactly zero; byte equality is reported separately
DIRS = ("encode", "decode")


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _case_label(case: dict[str, Any], sep: str) -> str:
    return f"{case['samples'] / av.SAMPLE_RATE:.1f} s{sep}T={case['latents_per_channel']}"


def batch_drift(case: dict[str, Any], name: str, kind: str) -> float:
    """Max |Δ| of one mono channel across batch compositions (the warm repeat is excluded)."""
    return max(v for k, v in case["paths"][name][kind]["max_abs"].items() if k != "repeat")


def fp64_error(case: dict[str, Any], name: str, kind: str) -> float:
    return float(case["paths"][name][kind]["vs_fp64_max_abs"])


def cold_drift(report: dict[str, Any], name: str, kind: str) -> float:
    return float(report["cold_start"][name][f"{kind}_first_vs_third_max_abs"])


def median_ms(case: dict[str, Any], name: str, kind: str) -> float | None:
    return case["paths"][name][kind].get("latency", {}).get("median_ms")


def _sci(v: float) -> str:
    return "= 0" if v == 0 else f"{v:.1e}"


def log_cap(values: Any, top: float) -> float:
    """``top`` raised by decades until it is above the largest value (never clip a bar)."""

    peak = max(values, default=0.0)
    while peak >= top:
        top *= 10
    return top


def _grouped_log_bars(
    ax: Any, groups: list[str], series: dict[str, list[float]], top: float
) -> None:
    """Grouped bars on a log axis; an exact zero sits on the floor with an explicit label."""
    names = list(series)
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(groups))]
        vals = series[n]
        ax.bar(
            xs,
            [v if v > 0 else LOG_FLOOR for v in vals],
            width,
            color=COLORS.get(n, "#777"),
            label=_label(n),
            edgecolor=SURFACE,
            linewidth=1,
        )
        for x, v in zip(xs, vals, strict=True):
            if v == 0:
                ax.text(
                    x,
                    LOG_FLOOR * 2,
                    ZERO_LABEL,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK,
                )
    top = log_cap((v for vals in series.values() for v in vals), top)
    ax.set_yscale("log")
    ax.set_ylim(LOG_FLOOR, top)
    ax.set_xticks(range(len(groups)), groups)


def _finish(fig: Any, axes: Any, title: str, path: Path, plt: Any) -> Path:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    # No metadata: the PNGs are uploaded as PR evidence.
    fig.savefig(path, dpi=130, metadata={"Software": None})
    plt.close(fig)
    return path


def plot(report: dict[str, Any], out_prefix: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": "#e4e3de",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    names = list(report["paths"])
    cases = report["cases"]
    ticks = [_case_label(c, "\n") for c in cases]
    gpu = report["provenance"].get("gpu", "GPU")

    def out(suffix: str) -> Path:
        return out_prefix.with_name(out_prefix.name + suffix)

    files = []

    # One cap for both panels: they share the y axis, so the second must not lower it.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cap = log_cap((batch_drift(c, n, k) for c in cases for n in names for k in DIRS), 1.0)
    for ax, kind in zip(axes, DIRS, strict=True):
        _grouped_log_bars(
            ax, ticks, {n: [batch_drift(c, n, kind) for c in cases] for n in names}, cap
        )
        ax.set_title(f"{kind}: max |Δ| of one channel across batch compositions", fontsize=10)
        ax.set_xlabel("clip length, latents per channel")
    axes[0].set_ylabel("max abs diff (0 = batch invariant)")
    files.append(
        _finish(fig, axes, f"H3 audio VAE batch invariance on {gpu} (FP32)", out("-bi.png"), plt)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cap = log_cap((fp64_error(c, n, k) for c in cases for n in names for k in DIRS), 1.0)
    for ax, kind in zip(axes, DIRS, strict=True):
        _grouped_log_bars(
            ax, ticks, {n: [fp64_error(c, n, kind) for c in cases] for n in names}, cap
        )
        ax.set_title(f"{kind}: max |error| against an FP64 run of the same graph", fontsize=10)
        ax.set_xlabel("clip length, latents per channel")
    axes[0].set_ylabel("max abs error")
    files.append(
        _finish(
            fig, axes, f"H3 audio VAE accuracy against FP64 on {gpu}", out("-accuracy.png"), plt
        )
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    _grouped_log_bars(
        ax, list(DIRS), {n: [cold_drift(report, n, k) for k in DIRS] for n in names}, 1e-3
    )
    ax.set_title("max |Δ| of a fresh process's first call against its third", fontsize=10)
    # (each path and direction ran in its own process: see report["cold_start_mode"])
    ax.set_ylabel("max abs diff (0 = no first-call drift)")
    files.append(
        _finish(fig, [ax], f"H3 audio VAE cold-start drift on {gpu}", out("-coldstart.png"), plt)
    )

    if not any(median_ms(c, n, k) is not None for c in cases for n in names for k in DIRS):
        return files  # a --repeats 0 run has no latency chart

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, DIRS, strict=True):
        ends: list[tuple[float, str]] = []
        for n in names:
            lat = [v for c in cases if (v := median_ms(c, n, kind)) is not None]
            if lat and len(lat) == len(cases):
                ax.plot(
                    range(len(cases)),
                    lat,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=COLORS.get(n, "#777"),
                    label=_label(n),
                )
                ends.append((math.log10(lat[-1]), _label(n)))
        # End-of-line labels, nudged apart in log space so coincident lines stay readable.
        ends.sort()
        ax.set_yscale("log")
        ax.margins(y=0.12)
        lo, hi = ax.get_ylim()
        gap = 0.06 * (math.log10(hi) - math.log10(lo))
        placed: list[float] = []
        for y, _ in ends:
            placed.append(max(y, placed[-1] + gap) if placed else y)
        shift = (sum(placed) - sum(y for y, _ in ends)) / max(len(ends), 1)
        for (_, n), y in zip(ends, placed, strict=True):
            y -= shift
            ax.annotate(
                n,
                (len(cases) - 1, 10**y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                annotation_clip=False,
                fontsize=8,
                color=INK,
            )
        ax.set_xticks(range(len(cases)), ticks)
        ax.set_xlim(-0.3, len(cases) - 0.2)
        ax.set_title(f"{kind} latency, stereo call (median)", fontsize=10)
        ax.set_ylabel("ms")
        ax.set_xlabel("clip length, latents per channel")
    files.append(_finish(fig, axes, f"H3 audio VAE latency on {gpu}", out("-latency.png"), plt))
    return files


def markdown(report: dict[str, Any], json_name: str) -> str:
    """The numbers each chart plots, as Markdown tables with the same labels and rounding."""
    names = list(report["paths"])
    cases = report["cases"]
    prov = report["provenance"]
    commit = (prov.get("source") or {}).get("commit") or "unknown"
    source = f"Source: `{json_name}`, commit {commit[:8]}, {prov.get('gpu', 'GPU')}."
    head = "| path | dir | " + " | ".join(_case_label(c, ", ") for c in cases) + " |"
    rule = "|---|---|" + "---|" * len(cases)

    def per_case(
        fn: Callable[[dict[str, Any], str, str], float | None], fmt: Callable[[float], str]
    ) -> str:
        rows = [head, rule]
        for n in names:
            for kind in DIRS:
                vals = [fn(c, n, kind) for c in cases]
                if any(v is None for v in vals):
                    continue
                cells = " | ".join(fmt(float(v)) for v in vals if v is not None)
                rows.append(f"| {_label(n)} | {kind} | {cells} |")
        return "\n".join(rows)

    cold = ["| path | encode | decode |", "|---|---|---|"]
    for n in names:
        enc, dec = (_sci(cold_drift(report, n, kind)) for kind in DIRS)
        cold.append(f"| {_label(n)} | {enc} | {dec} |")
    parity = ["| path | encode bitwise = ours | decode bitwise = ours |", "|---|---|---|"]
    for n in names:
        if n == "ours":
            continue
        n_enc = sum(bool(c["encode_bitwise_equal_to_ours"][n]) for c in cases)
        n_dec = sum(bool(c["decode_bitwise_equal_to_ours"][n]) for c in cases)
        parity.append(f"| {_label(n)} | {n_enc}/{len(cases)} | {n_dec}/{len(cases)} |")
    zero = (
        f"`= 0` is a max abs difference of exactly zero, drawn as `{ZERO_LABEL}` on the chart"
        " (byte equality is the sha256 comparison in the JSON)."
    )
    sections = [
        ("bi", per_case(batch_drift, _sci), f"Max abs diff. {zero}"),
        ("accuracy", per_case(fp64_error, _sci), f"Max abs error. {zero}"),
        ("coldstart", "\n".join(cold), f"Max abs diff. {zero}"),
        ("parity", "\n".join(parity), "Cases bitwise equal to the strict path."),
        ("latency", per_case(median_ms, lambda v: f"{v:.1f}"), "Median ms, rounded to 0.1 ms."),
    ]
    return "\n\n".join(
        f"<!-- {key} -->\n{table}\n\n{note} {source}" for key, table, note in sections
    )


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, help="released FL2VA/audio_vae directory")
    ap.add_argument(
        "--diffusers-src", help="diffusers src/ tree providing AutoencoderKLMiniMaxH3Audio"
    )
    ap.add_argument("--diffusers-rev", default="f53d5520")
    ap.add_argument(
        "--remote-src", help="audio_vae directory whose remote code to run (default: --checkpoint)"
    )
    ap.add_argument(
        "--no-remote", action="store_true", help="run the vLLM-Omni recipe on the rl_engine port"
    )
    ap.add_argument(
        "--sglang", action="store_true", help="add SGLang's own module (needs an SGLang env)"
    )
    ap.add_argument("--vllm-omni-rev", default="aae3f386")
    ap.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_FRAMES))
    ap.add_argument(
        "--sample-offset", type=int, default=-123, help="make clips a non-multiple of the hop"
    )
    ap.add_argument("--warmup", type=non_negative_int, default=2)
    ap.add_argument(
        "--repeats", type=non_negative_int, default=5, help="timed samples per case (0 = no timing)"
    )
    ap.add_argument("--cold-start-worker", nargs=2, metavar=("PATH", "DIR"), help=argparse.SUPPRESS)
    ap.add_argument("--cold-start-input", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--plot-only", action="store_true", help="re-render charts from an existing --out"
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="print the charts' numbers from an existing --out as Markdown tables",
    )
    args = ap.parse_args(argv)
    if args.cold_start_worker and args.cold_start_worker[1] not in DIRS:
        ap.error(f"--cold-start-worker DIR must be one of {DIRS}")
    if args.markdown:
        print(markdown(json.loads(args.out.read_text()), args.out.name))
        return 0
    if args.plot_only:
        print("\n".join(map(str, plot(json.loads(args.out.read_text()), args.out.with_suffix("")))))
        return 0
    if args.checkpoint is None or not torch.cuda.is_available():
        ap.error("--checkpoint and a CUDA device are required")
    if args.remote_src is None and not args.no_remote:
        args.remote_src = str(args.checkpoint)
    if args.no_remote:
        args.remote_src = None
    if args.cold_start_worker:
        return cold_start_worker(args)
    report = run(args, argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    if args.plot:
        print("\n".join(map(str, plot(report, args.out.with_suffix("")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
