# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Batch invariance, determinism, accuracy and latency of H3 audio VAE paths (RFC #420 Phase B).

Paths, all on the released FP32 weights (``<ckpt>/FL2VA/audio_vae``):

* ``diffusers``  — ``AutoencoderKLMiniMaxH3Audio`` (``--diffusers-src``), torch default flags,
  stereo as one batch-2 call (the Diffusers pipeline);
* ``sglang``     — (``--sglang``) SGLang's own ``DacAudioVAE`` port under its own encode context
  (TF32 off, cuDNN off, math SDPA) and decode context (TF32 off, deterministic cuDNN);
* ``vllm-omni``  — the checkpoint's remote code (``--remote-src``, what vLLM-Omni loads) under
  vLLM-Omni's recipe: the same encode flags as SGLang, decode with default flags inside
  ``torch.jit.optimized_execution(False)``;
* ``ours``       — :class:`rl_engine.models.minimax_h3.audio_vae.H3AudioVAEContract` strict
  policies (micro-batch 1 per mono channel).

With ``--no-remote`` the vLLM-Omni recipe runs on the rl_engine port (op-identical).
Per clip length the script records, for encode and decode separately: repeat equality, the max
absolute difference of one logical mono channel across batch compositions (alone, stereo pair,
first/last of an 8-item batch), error against an FP64 run of the same graph, bitwise parity
between paths, and CUDA-synchronised wall latency of a stereo call. ``--plot`` renders the
accuracy and performance charts from the JSON.
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
    model.load_state_dict(load_file(str(ckpt.path / "model.safetensors")), strict=True)
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


def build_paths(args: argparse.Namespace, ckpt: av.AudioVAECheckpoint, device: str) -> list[Path_]:
    paths: list[Path_] = []
    if args.diffusers_src:
        sys.path.insert(0, str(Path(args.diffusers_src).expanduser()))
        from diffusers import AutoencoderKLMiniMaxH3Audio
        from safetensors.torch import load_file

        ref = AutoencoderKLMiniMaxH3Audio(
            latents_mean=list(ckpt.latents_mean), latents_std=list(ckpt.latents_std)
        )
        ref.load_state_dict(load_file(str(ckpt.path / "model.safetensors")), strict=True)
        ref = ref.to(device).eval()

        def d_enc(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch_default_flags():
                return ref.encode(x[:, None]).latent_dist.mode().float()

        def d_dec(z: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch_default_flags():
                return ref.decode(z).sample.float()[:, 0]

        paths.append(Path_("diffusers", d_enc, d_dec, f"diffusers {args.diffusers_rev}"))

    if args.sglang:
        paths.append(sglang_path(ckpt, device))

    if args.remote_src:
        remote = (
            load_remote(Path(args.remote_src).expanduser())
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        engine_src = "checkpoint remote code"
    else:
        remote = ckpt.build(device)
        engine_src = "rl_engine port (op-identical)"

    def vo_enc(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), av.execution_scope(ENGINE_ENCODE):
            return dac_encode_mean(remote, x)

    def vo_dec(z: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch_default_flags(), torch.jit.optimized_execution(False):
            return remote.decode(z).float()[:, 0]

    paths.append(
        Path_("vllm-omni", vo_enc, vo_dec, f"vllm-omni {args.vllm_omni_rev} recipe on {engine_src}")
    )

    contract = av.H3AudioVAEContract.from_checkpoint(ckpt, device)
    paths.append(
        Path_("ours", contract.encode_latents, contract.decode_latents, f"rl_engine {av.CONTRACT}")
    )
    return paths


# --- measurements ---------------------------------------------------------------------------


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def invariance(
    fn: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, others: torch.Tensor
) -> dict[str, Any]:
    """Max |diff| of the target's left channel across batch compositions vs its stereo call."""

    stereo = fn(target)
    variants = {
        "repeat": fn(target)[0],
        "mono_alone": fn(target[:1])[0],
        "batch8_first": fn(torch.cat([target, others]))[0],
        "batch8_last": fn(torch.cat([others, target]))[-2],
    }
    diffs = {k: max_abs(v, stereo[0]) for k, v in variants.items()}
    return {
        "max_abs": diffs,
        "batch_invariant": all(v == 0.0 for k, v in diffs.items() if k != "repeat"),
        "repeat_equal": diffs["repeat"] == 0.0,
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


def cold_start(
    paths: list[Path_], strict: av.H3AudioVAEContract, frames: int, offset: int
) -> dict[str, Any]:
    """First call vs third call of each freshly built path (TorchScript profiling re-fuses)."""

    num_samples = round(frames / av.VIDEO_FPS * av.SAMPLE_RATE) + offset
    x = stereo_clip(99, num_samples).to(strict.device)
    enc = strict.encode_stereo(x.cpu())
    z = strict.denormalize(enc.rows, enc.num_latents)
    out = {}
    for p in paths:
        e = [p.encode(x) for _ in range(3)]
        d = [p.decode(z) for _ in range(3)]
        out[p.name] = {
            "encode_first_vs_third_max_abs": max_abs(e[0], e[2]),
            "decode_first_vs_third_max_abs": max_abs(d[0], d[2]),
        }
        print(f"[cold start] {p.name:<10} {out[p.name]}", flush=True)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda"
    ckpt = av.AudioVAECheckpoint.read(args.checkpoint, verify_sha256=av.pinned_audio_vae_sha256())
    paths = build_paths(args, ckpt, device)
    fp64 = ckpt.build(device).double()
    strict = av.H3AudioVAEContract.from_checkpoint(ckpt, device)
    report: dict[str, Any] = {
        "schema": "rlkernel.h3.audio_vae_bench.v1",
        "checkpoint": {
            "repo": "MiniMaxAI/MiniMax-H3",
            "component": "FL2VA/audio_vae",
            "model_safetensors_sha256": ckpt.sha256,
        },
        "provenance": {**strict.provenance(), "source": source_state()},
        "paths": {p.name: p.source for p in paths},
        "cold_start": cold_start(paths, strict, args.frames[0], args.sample_offset),
        "cases": [],
    }
    for frames in args.frames:
        num_samples = round(frames / av.VIDEO_FPS * av.SAMPLE_RATE) + args.sample_offset
        target = stereo_clip(0, num_samples).to(device)
        others = torch.cat([stereo_clip(i, num_samples) for i in range(1, 4)]).to(device)
        with torch.inference_mode():
            z64 = fp64.encode_mean(target.double()[:, None])
        # One shared decode input for every path: the strict encoding, denormalised.
        enc = strict.encode_stereo(target.cpu())
        z_dec = strict.denormalize(enc.rows, enc.num_latents)
        z_others = torch.cat([z_dec.roll(k + 1, dims=-1) * (1 + 0.01 * k) for k in range(3)])
        with torch.inference_mode():
            w64 = fp64.decode(z_dec.double())[:, 0]
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
        for kind in ("encode", "decode"):
            case[f"{kind}_bitwise_equal_to_ours"] = {
                n: bool(torch.equal(o[kind], outputs["ours"][kind])) for n, o in outputs.items()
            }
        case["rows_sha256"] = enc.rows_sha256
        case["latents_sha256"] = enc.latents_sha256
        report["cases"].append(case)
    return report


# --- charts ---------------------------------------------------------------------------------


# Fixed categorical order (blue, orange, aqua, yellow); identity is also carried by direct labels.
COLORS = {"diffusers": "#2a78d6", "sglang": "#eb6834", "vllm-omni": "#1baf7a", "ours": "#eda100"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"


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
    secs = [f"{c['samples'] / av.SAMPLE_RATE:.1f}s" for c in cases]
    gpu = report["provenance"].get("gpu", "GPU")
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind in zip(axes, ("encode", "decode"), strict=True):
        width = 0.8 / len(names)
        for i, n in enumerate(names):
            vals = [
                max(v for k, v in c["paths"][n][kind]["max_abs"].items() if k != "repeat")
                for c in cases
            ]
            xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(cases))]
            shown = [v if v > 0 else 1e-12 for v in vals]
            ax.bar(
                xs,
                shown,
                width,
                color=COLORS.get(n, "#777"),
                label=n,
                edgecolor=SURFACE,
                linewidth=1,
            )
            for x, v in zip(xs, vals, strict=True):
                if v == 0:
                    ax.text(x, 1.5e-12, "0", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 1)
        ax.set_xticks(range(len(cases)), secs)
        ax.set_title(f"{kind}: max |Δ| of one channel across batch compositions")
        ax.set_xlabel("clip length")
    axes[0].set_ylabel("max abs diff (0 = batch invariant)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(f"H3 audio VAE batch invariance on {gpu} (FP32)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    files.append(out_prefix.with_name(out_prefix.name + "-bi.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind in zip(axes, ("encode", "decode"), strict=True):
        ends: list[tuple[float, str]] = []
        for n in names:
            lat = [c["paths"][n][kind].get("latency", {}).get("median_ms") for c in cases]
            if all(v is not None for v in lat):
                ax.plot(
                    secs,
                    lat,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=COLORS.get(n, "#777"),
                    label=n,
                )
                ends.append((math.log10(lat[-1]), n))
        # End-of-line labels, nudged apart in log space so coincident lines stay readable.
        ends.sort()
        lo, hi = ax.get_ylim()
        gap = 0.06 * (math.log10(hi) - math.log10(lo))
        placed: list[float] = []
        for y, n in ends:
            y = max(y, placed[-1] + gap) if placed else y
            placed.append(y)
            ax.annotate(
                n,
                (len(secs) - 1, 10**y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_xlim(-0.3, len(secs) - 0.2)
        ax.set_title(f"{kind} latency, stereo call (median)")
        ax.set_ylabel("ms")
        ax.set_xlabel("clip length")
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(f"H3 audio VAE latency on {gpu}")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    files.append(out_prefix.with_name(out_prefix.name + "-latency.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)
    return files


def main(argv: list[str] | None = None) -> int:
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
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5, help="timed samples per case (0 = no timing)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--plot-only", action="store_true", help="re-render charts from an existing --out"
    )
    args = ap.parse_args(argv)
    if args.plot_only:
        print("\n".join(map(str, plot(json.loads(args.out.read_text()), args.out.with_suffix("")))))
        return 0
    if args.checkpoint is None or not torch.cuda.is_available():
        ap.error("--checkpoint and a CUDA device are required")
    if args.remote_src is None and not args.no_remote:
        args.remote_src = str(args.checkpoint)
    if args.no_remote:
        args.remote_src = None
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    if args.plot:
        print("\n".join(map(str, plot(report, args.out.with_suffix("")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
