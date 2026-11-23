#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for RFC #420 ``visual_vae_contract`` on the released MiniMax-H3 VisualVAE.

    PYTHONPATH=.:<diffusers>/src python tools/validation/models/h3_visual_vae_evidence.py \\
        --vae-dir <MiniMax-H3 snapshot>/vae --out evidence.json [--quick]

Needs one CUDA GPU with ~40 GB free and Diffusers with MiniMax-H3 (commit f53d5520). It
measures, with the released weights:

* run-to-run determinism of the contract's encode (keyframe and 22-frame video) and decode;
* bitwise parity of the contract with the Diffusers release recipe
  (``encode_vae_condition`` and the decode step of ``decoders.py``), with Diffusers run both
  under unmodified PyTorch flags and under the contract's pinned flags;
* batch invariance: the Diffusers batched path (``use_slicing=False``) at batch 1 vs 2, and
  the contract's per-row path;
* tiling: tiled run-to-run equality, tile-plan digests, and tiled-vs-untiled distance;
* preprocessing: CPU vs CUDA disagreement of ``uint8 / 255`` and the LUT's agreement with CUDA;
* latency (CUDA events, median) of the contract and of the Diffusers calls it wraps;
* the attention/convolution kernels that actually ran (``torch.profiler``);
* ``peak_memory_gib``: the CUDA allocator peak over the whole run (model load, every check
  and timing loop), not the memory of one step.

Paths are not written to the report.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import visual_vae as vv  # noqa: E402
from rl_engine.models.minimax_h3.trajectory import tensor_digest  # noqa: E402

SCHEMA = "rlkernel.h3.visual_vae_evidence.v1"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pixels(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Smooth gradients plus noise: realistic enough for every tile to carry signal."""

    gen = torch.Generator().manual_seed(seed)
    b, c, f, h, w = shape
    grid_y = torch.linspace(0, 1, h).view(1, 1, 1, h, 1)
    grid_x = torch.linspace(0, 1, w).view(1, 1, 1, 1, w)
    phase = torch.rand(b, c, 1, 1, 1, generator=gen) * 6.28
    t = torch.arange(f).view(1, 1, f, 1, 1) * 0.15
    base = 0.5 + 0.35 * torch.sin(6.0 * grid_x + 4.0 * grid_y + phase + t)
    noise = torch.rand(shape, generator=gen) * 0.15
    return ((base + noise).clamp(0, 1) * 255).round().to(torch.uint8)


def _element_bits(t: torch.Tensor) -> torch.Tensor:
    """``[numel, itemsize]`` raw bytes of ``t``, one row per element."""

    t = t.detach().contiguous().cpu()
    return t.reshape(-1).view(torch.uint8).view(t.numel(), t.element_size())


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Compare ``a`` with ``b``. ``bitwise_equal`` means same dtype, shape and bytes (so equal
    NaNs count as equal and ``-0.0`` differs from ``0.0``); ``mismatches`` counts elements
    whose bytes differ (values, when the dtypes differ). ``max_abs`` is in FP32, ``0`` when
    bitwise equal, and ``None`` (JSON ``null``) when there is no finite max: a shape mismatch
    (``shape_a``/``shape_b`` record it) or a non-finite difference. The plots print
    "differs" for ``None``."""

    same_dtype = a.dtype == b.dtype
    if a.shape != b.shape:
        return {
            "bitwise_equal": False,
            "same_dtype": same_dtype,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_abs": None,
        }
    fa, fb = a.float().cpu(), b.float().cpu()
    if same_dtype:
        mismatches = int((_element_bits(a) != _element_bits(b)).any(dim=1).sum())
    else:
        mismatches = int((fa != fb).sum())
    bitwise = same_dtype and mismatches == 0
    max_abs = 0.0 if bitwise else float((fa - fb).abs().max())
    return {
        "bitwise_equal": bitwise,
        "same_dtype": same_dtype,
        "max_abs": max_abs if math.isfinite(max_abs) else None,
        "mismatches": mismatches,
        "numel": a.numel(),
    }


def _time(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }


def _sweep(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> dict[str, Any]:
    """Every row of every batch size 2..B against the same row run alone."""

    alone = [fn(x[i : i + 1]) for i in range(x.shape[0])]
    cases = []
    for size in range(2, x.shape[0] + 1):
        out = fn(x[:size])
        for i in range(size):
            cases.append({"batch": size, "row": i, **_cmp(out[i : i + 1], alone[i])})
    values = [c["max_abs"] for c in cases]
    return {
        "bitwise_equal": all(c["bitwise_equal"] for c in cases),
        "max_abs": None if any(v is None for v in values) else max(values),
        "differing_cases": sum(not c["bitwise_equal"] for c in cases),
        "cases": cases,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Strict JSON: a non-finite float raises instead of being written as ``NaN``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")


# PyTorch's default backend flags (the release policy keeps PyTorch's TF32 defaults).
DEFAULT_FLAGS = vv.RuntimePolicy(cudnn_deterministic=False, cudnn_benchmark=False)


def _flags(deterministic: bool) -> contextlib.AbstractContextManager[dict[str, Any]]:
    """Set all four backend flags: the contract's pinned policy, or PyTorch's defaults."""

    return vv.pinned_runtime(vv.RELEASE_POLICY if deterministic else DEFAULT_FLAGS)


def _kernels(fn: Callable[[], Any]) -> list[dict[str, Any]]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows: dict[str, dict[str, Any]] = {}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.name
        keep = any(
            key in name.lower()
            for key in ("attn", "attention", "fmha", "flash", "conv", "cudnn", "sm100", "gemm")
        )
        if not keep:
            continue
        row = rows.setdefault(name[:160], {"name": name[:160], "calls": 0, "cuda_us": 0.0})
        row["calls"] += 1
        row["cuda_us"] += float(getattr(event, "device_time", 0.0) or 0.0)
    return sorted(rows.values(), key=lambda r: -r["cuda_us"])[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--quick", action="store_true", help="256x384 inputs, 2 repeats")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    if args.quick:
        args.height, args.width, args.repeats, args.timing_repeats = 256, 384, 2, 2
    if args.repeats < 2:
        # Run-to-run determinism compares runs; one run has nothing to compare.
        parser.error("--repeats must be at least 2")

    import diffusers
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    t0 = time.time()
    vae = vv.load_release_vae(
        AutoencoderKLMiniMaxH3, args.vae_dir, "cuda", verify_sha256=args.verify_sha256
    )
    load_s = time.time() - t0
    contract = vv.VisualVAEContract(vae)
    h, w = args.height, args.width
    keyframe = _pixels((3, 3, 1, h, w), seed=101)
    video = _pixels((2, 3, 22, h, w), seed=202)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "contract_id": vv.CONTRACT_ID,
        "rl_kernel_commit": _git("rev-parse", "HEAD"),
        "rl_kernel_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "diffusers_version": diffusers.__version__,
        "reference": vv.release_fingerprint()["reference"],
        "checkpoint": vv.release_fingerprint()["checkpoint"],
        "sha256_verified": bool(args.verify_sha256),
        "environment": {
            **vv.runtime_readback(),
            "python": platform.python_version(),
            "driver": _driver(),
        },
        "inputs": {
            "keyframe": list(keyframe.shape),
            "video": list(video.shape),
            "pixel_sha256": {"keyframe": tensor_digest(keyframe), "video": tensor_digest(video)},
        },
        "load_seconds": load_s,
        "manifest": vv.contract_manifest(heights_widths=((h, w),), frame_counts=(1, 22)),
        "checks": {},
    }
    checks = report["checks"]
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    pmean = torch.tensor(vv.PIXEL_MEAN, device="cuda").view(1, -1, 1, 1, 1)
    pstd = torch.tensor(vv.PIXEL_STD, device="cuda").view(1, -1, 1, 1, 1)

    def diffusers_decode(latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = latents.cuda() * std + mean
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = vae.decode(z, return_dict=False)[0]
            return (out.float() * pstd + pmean).clamp(0, 1).cpu()

    def diffusers_encode(pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat(
                [
                    encode_vae_condition(vae, pixels[i : i + 1].cuda(), vv.PIXEL_MEAN, vv.PIXEL_STD)
                    for i in range(pixels.shape[0])
                ]
            )

    # 1. Run-to-run determinism of the contract.
    runs: dict[str, list[str]] = {"encode_keyframe": [], "encode_video": [], "decode_video": []}
    key_out = vid_out = dec_out = None
    for _ in range(args.repeats):
        key_out = contract.encode(keyframe)
        vid_out = contract.encode(video)
        dec_out = contract.decode(vid_out.tensor)
        runs["encode_keyframe"].append(key_out.digest)
        runs["encode_video"].append(vid_out.digest)
        runs["decode_video"].append(dec_out.digest)
    assert key_out is not None and vid_out is not None and dec_out is not None
    checks["run_to_run"] = {
        name: {"digests": d, "bitwise_equal": len(set(d)) == 1} for name, d in runs.items()
    }
    report["outputs"] = {
        "encode_keyframe": {"shape": list(key_out.tensor.shape), "sha256": key_out.digest},
        "encode_video": {"shape": list(vid_out.tensor.shape), "sha256": vid_out.digest},
        "decode_video": {"shape": list(dec_out.tensor.shape), "sha256": dec_out.digest},
        "encode_record": vid_out.record,
        "decode_record": dec_out.record,
    }
    checks["layout"] = {
        "encode_keyframe_shape_ok": tuple(key_out.tensor.shape)
        == vv.RELEASE_LAYOUT.latent_shape(3, 1, h, w),
        "encode_video_shape_ok": tuple(vid_out.tensor.shape)
        == vv.RELEASE_LAYOUT.latent_shape(2, 22, h, w),
        "decode_frames_ok": dec_out.tensor.shape[2] == 22,
    }

    # 2. Parity with the Diffusers release recipe, under default and pinned flags.
    parity: dict[str, Any] = {}
    for label, deterministic in (("diffusers_default_flags", False), ("diffusers_pinned", True)):
        with _flags(deterministic):
            parity[label] = {
                "encode_keyframe": _cmp(key_out.tensor, diffusers_encode(keyframe)),
                "encode_video": _cmp(vid_out.tensor, diffusers_encode(video)),
                "decode_video": _cmp(dec_out.tensor, diffusers_decode(vid_out.tensor)),
            }
    checks["diffusers_parity"] = parity

    # 3. Batch invariance: Diffusers batched path vs the contract's per-row path.
    def batched_encode_mode(pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = vv.preprocess_pixels(pixels.cuda())
            return vae.encode(x, return_dict=False)[0].mode().cpu()

    def batched_decode(latents: torch.Tensor) -> torch.Tensor:
        return diffusers_decode(latents)

    bi: dict[str, Any] = {}
    for label, deterministic in (("default_flags", False), ("pinned_flags", True)):
        with _flags(deterministic):
            bi[f"diffusers_batched_{label}"] = {
                "encode_keyframe": _sweep(batched_encode_mode, keyframe),
                "encode_video": _sweep(batched_encode_mode, video),
                "decode_video": _sweep(batched_decode, vid_out.tensor),
            }
    bi["contract_per_row"] = {
        "encode_keyframe": all(
            contract.encode(keyframe[i : i + 1]).row_digests[0] == key_out.row_digests[i]
            for i in range(keyframe.shape[0])
        ),
        "encode_video": all(
            contract.encode(video[i : i + 1]).row_digests[0] == vid_out.row_digests[i]
            for i in range(video.shape[0])
        ),
        "decode_video": all(
            contract.decode(vid_out.tensor[i : i + 1]).row_digests[0] == dec_out.row_digests[i]
            for i in range(vid_out.tensor.shape[0])
        ),
    }
    # The same cases as the Diffusers sweeps, per batch size and row, for the charts.
    bi["contract_cases"] = {
        "encode_keyframe": _sweep(lambda x: contract.encode(x).tensor, keyframe),
        "encode_video": _sweep(lambda x: contract.encode(x).tensor, video),
        "decode_video": _sweep(lambda x: contract.decode(x).tensor, vid_out.tensor),
    }
    checks["batch_invariance"] = bi

    # 4. Tiling.
    plan = vv.plan_tiles(h, w)
    vae.disable_tiling()
    try:
        untiled = vv.VisualVAEContract(vae, require_release=False)
        u_key = untiled.encode(keyframe[:1])
        u_dec = untiled.decode(vid_out.tensor[:1])
        timing_untiled = {
            "encode_keyframe": _time(lambda: untiled.encode(keyframe[:1]), args.timing_repeats),
            "decode_video": _time(lambda: untiled.decode(vid_out.tensor[:1]), args.timing_repeats),
        }
    finally:
        vae.enable_tiling(256, 256, 64, 64)
    checks["tiling"] = {
        "plan": {"tiles": plan.tiles, "sha256": plan.digest(), **plan.as_dict()},
        "record_plan_sha256_matches": vid_out.record["tile_plan_sha256"] == plan.digest(),
        "tiled_run_to_run_equal": checks["run_to_run"]["decode_video"]["bitwise_equal"]
        and checks["run_to_run"]["encode_keyframe"]["bitwise_equal"],
        "tiled_vs_untiled_encode_keyframe": _cmp(key_out.tensor[:1], u_key.tensor),
        "tiled_vs_untiled_decode_video": _cmp(dec_out.tensor[:1], u_dec.tensor),
    }

    # 5. Preprocessing and normalization across devices.
    values = torch.arange(256, dtype=torch.uint8)
    cpu_div = values.to(torch.float32).div(255.0)
    cuda_div = values.cuda().to(torch.float32).div(255.0).cpu()
    px = values.view(1, 1, 1, 1, 256).expand(1, 3, 1, 1, 256)
    ref_cuda = ((px.cuda().to(torch.float32).div(255.0) - pmean) / pstd).cpu()
    ref_cpu = (px.to(torch.float32).div(255.0) - pmean.cpu()) / pstd.cpu()
    z = vid_out.tensor
    mean_cpu, std_cpu = vv.release_latent_stats()
    checks["preprocessing"] = {
        "div255_cpu_vs_cuda_mismatches": int((cpu_div != cuda_div).sum()),
        "normalized_cpu_vs_cuda_mismatches": int((ref_cpu != ref_cuda).sum()),
        "lut_equals_cuda_release": bool(
            torch.equal(vv.pixel_lut().view(1, 3, 1, 1, 256), ref_cuda)
        ),
        "lut_sha256": vv.pixel_lut_digest(),
    }
    checks["normalization"] = {
        "normalize_cpu_vs_cuda": _cmp(
            vv.normalize_latents(z, mean_cpu, std_cpu),
            vv.normalize_latents(z.cuda(), mean_cpu, std_cpu),
        ),
        "denormalize_cpu_vs_cuda": _cmp(
            vv.denormalize_latents(z, mean_cpu, std_cpu),
            vv.denormalize_latents(z.cuda(), mean_cpu, std_cpu),
        ),
    }

    # 6. Latency (batch 1, tiled release path).
    one_key, one_vid, one_lat = keyframe[:1], video[:1], vid_out.tensor[:1]
    timing = {
        "contract": {
            "encode_keyframe": _time(lambda: contract.encode(one_key), args.timing_repeats),
            "encode_video": _time(lambda: contract.encode(one_vid), args.timing_repeats),
            "decode_video": _time(lambda: contract.decode(one_lat), args.timing_repeats),
        },
    }
    with _flags(False):
        timing["diffusers_default_flags"] = {
            "encode_keyframe": _time(lambda: diffusers_encode(one_key), args.timing_repeats),
            "encode_video": _time(lambda: diffusers_encode(one_vid), args.timing_repeats),
            "decode_video": _time(lambda: diffusers_decode(one_lat), args.timing_repeats),
        }
    with _flags(True):
        timing["diffusers_pinned_flags"] = {
            "encode_keyframe": _time(lambda: diffusers_encode(one_key), args.timing_repeats),
            "encode_video": _time(lambda: diffusers_encode(one_vid), args.timing_repeats),
            "decode_video": _time(lambda: diffusers_decode(one_lat), args.timing_repeats),
        }
    timing["contract_untiled"] = timing_untiled
    report["timing"] = timing

    # 7. Which kernels ran.
    with vv.pinned_runtime():
        report["kernels"] = {
            "encode_keyframe": _kernels(lambda: contract.encode(one_key)),
            "decode_video": _kernels(lambda: contract.decode(one_lat)),
        }
    # Process-wide CUDA peak since start (load, every check and timing loop), not per step.
    report["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30

    verdicts = {
        "run_to_run": all(v["bitwise_equal"] for v in checks["run_to_run"].values()),
        "layout": all(checks["layout"].values()),
        "diffusers_parity_pinned": all(
            v["bitwise_equal"] for v in parity["diffusers_pinned"].values()
        ),
        "diffusers_parity_default_flags": all(
            v["bitwise_equal"] for v in parity["diffusers_default_flags"].values()
        ),
        "contract_batch_invariant": all(bi["contract_per_row"].values())
        and all(v["bitwise_equal"] for v in bi["contract_cases"].values()),
        "diffusers_batched_batch_invariant_default_flags": all(
            v["bitwise_equal"] for v in bi["diffusers_batched_default_flags"].values()
        ),
        "tiling_deterministic": checks["tiling"]["tiled_run_to_run_equal"]
        and checks["tiling"]["record_plan_sha256_matches"],
        "lut_equals_cuda_release": checks["preprocessing"]["lut_equals_cuda_release"],
        "normalization_cpu_cuda_bitwise": all(
            v["bitwise_equal"] for v in checks["normalization"].values()
        ),
    }
    report["verdicts"] = verdicts
    write_report(args.out, report)
    print(json.dumps(verdicts, indent=1))
    required = (
        "run_to_run",
        "layout",
        "diffusers_parity_pinned",
        "contract_batch_invariant",
        "tiling_deterministic",
        "lut_equals_cuda_release",
        "normalization_cpu_cuda_bitwise",
    )
    return 0 if all(verdicts[k] for k in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
