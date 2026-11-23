#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for RFC #738 ``image_vae_contract`` and ``image_vae_tiling_contract``.

    PYTHONPATH=.:<diffusers>/src python tools/validation/models/h3_image_vae_evidence.py \\
        --vae-dir <MiniMax-H3-Image snapshot>/vae_diffusers --out evidence.json [--quick]
    python tools/validation/models/h3_image_vae_evidence.py --write-manifest   # golden file
    python tools/validation/models/h3_image_vae_evidence.py --check-manifest

Needs one CUDA GPU and Diffusers with MiniMax-H3-Image (commit 875373ac). With the released
weights it measures:

* identity: the local folder against the pinned config, sizes, SHA-256 and header table;
* run-to-run determinism of the edit-reference encode and of the decode (I-C10, tiled);
* bitwise parity with the Diffusers release blocks ``MiniMaxH3ImageEditVaeEncoderStep`` and
  ``MiniMaxH3ImageDecodeStep``, run as they are (decode step at the canvas size, so it skips its
  resize; the resize is ``image_canvas_contract``), under
  unmodified PyTorch flags and under the contract's pinned flags, with CPU and CUDA request
  generators;
* batch invariance: the contract per row (posterior mode and sampled with a shared
  generator), and the Diffusers batched ``encode`` / ``decode`` for comparison;
* tiling: tile-plan digest in the record, tiled vs untiled (different mode identity and
  output), untiled run-to-run;
* preprocessing and normalization across devices;
* the image-specific ablation probes I11, I15, I16, I17 and I18 (how far each wrong variant
  moves the latent or the output bytes);
* latency (CUDA events, median), the kernels that ran, and the process CUDA peak.

Paths are not written to the report.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import visual_vae as vv  # noqa: E402
from rl_engine.models.minimax_h3.trajectory import tensor_digest  # noqa: E402
from rl_engine.models.minimax_h3_image import image_vae as iv  # noqa: E402

SCHEMA = "rlkernel.h3_image.vae_evidence.v1"
MANIFEST_PATH = REPO_ROOT / "rl_engine/models/minimax_h3_image/image_vae_contract.json"


# --- Diffusers release blocks, run as they are ---------------------------------------------


def release_components(vae: Any, device: torch.device | str) -> Any:
    """The attributes the two Diffusers blocks read from a ``MiniMaxH3ImageModularPipeline``.

    The pipeline properties are evaluated from the pinned class itself (not re-typed), on a
    stand-in object that carries only the VAE; ``patch_size`` falls back to the pipeline
    default ``(1, 2, 2)`` because no transformer is loaded.
    """

    from diffusers.modular_pipelines.minimax_h3_image import decoders
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as Pipeline,
    )

    class Components:
        pass

    c = Components()
    c.vae = vae
    c.transformer = None
    c._execution_device = torch.device(device)
    for name in ("vae_spatial_compression_ratio", "vae_latent_channels", "patch_size"):
        setattr(c, name, getattr(Pipeline, name).fget(c))
    for name in ("pixel_mean", "pixel_std"):
        setattr(c, name, getattr(Pipeline, name).fget(c))
    spec = next(
        s
        for s in decoders.MiniMaxH3ImageDecodeStep().expected_components
        if s.name == "image_processor"
    )
    c.image_processor = spec.create()
    return c


def _to_pil(image: torch.Tensor) -> Any:
    from PIL import Image

    return Image.fromarray(image.permute(1, 2, 0).contiguous().cpu().numpy())


def _from_pil(images: Sequence[Any]) -> torch.Tensor:
    import numpy as np

    return torch.stack([torch.from_numpy(np.array(i)).permute(2, 0, 1) for i in images])


def diffusers_encode_references(
    vae: Any, images: Sequence[torch.Tensor], generator: torch.Generator, device: str = "cuda"
) -> list[torch.Tensor]:
    """``MiniMaxH3ImageEditVaeEncoderStep`` on ``uint8`` ``(3, H, W)`` references -> CPU latents."""

    from diffusers.modular_pipelines.minimax_h3_image.encoders import (
        MiniMaxH3ImageEditVaeEncoderStep,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    state.set("reference_images", [_to_pil(i) for i in images])
    state.set("generator", generator)
    _, state = MiniMaxH3ImageEditVaeEncoderStep()(release_components(vae, device), state)
    return [z.cpu() for z in state.get("reference_latents")]


def diffusers_decode(
    vae: Any,
    latents: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """``MiniMaxH3ImageDecodeStep`` (``output_type="pil"``) on ``(1, 24, h, w)`` -> ``uint8``.

    No ``output_height`` / ``output_width`` is set, so the step returns the canvas-size image
    and its resize branch does not run.
    """

    from diffusers.modular_pipelines.minimax_h3_image.decoders import MiniMaxH3ImageDecodeStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    state.set("latents", latents)
    state.set("output_type", "pil")
    _, state = MiniMaxH3ImageDecodeStep()(release_components(vae, device), state)
    return _from_pil(state.get("images"))


# --- helpers ---------------------------------------------------------------------------------


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


def synthetic_image(height: int, width: int, seed: int) -> torch.Tensor:
    """``uint8`` ``(3, H, W)``: smooth gradients plus noise, so every tile carries signal."""

    gen = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, height).view(1, height, 1)
    x = torch.linspace(0, 1, width).view(1, 1, width)
    phase = torch.rand(3, 1, 1, generator=gen) * 6.28
    base = 0.5 + 0.35 * torch.sin(7.0 * x + 5.0 * y + phase) * torch.cos(3.0 * x * y + phase)
    noise = torch.rand(3, height, width, generator=gen) * 0.15
    return ((base + noise).clamp(0, 1) * 255).round().to(torch.uint8)


def _element_bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().cpu()
    return t.reshape(-1).view(torch.uint8).view(t.numel(), t.element_size())


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """``bitwise_equal`` = same dtype, shape and bytes; ``max_abs`` in FP32 (``None`` if
    shapes differ or the difference is not finite); ``mismatches`` = elements that differ."""

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
        keys = ("attn", "attention", "fmha", "flash", "conv", "cudnn", "sm100", "gemm", "cutlass")
        if not any(key in name.lower() for key in keys):
            continue
        row = rows.setdefault(name[:160], {"name": name[:160], "calls": 0, "cuda_us": 0.0})
        row["calls"] += 1
        row["cuda_us"] += float(getattr(event, "device_time", 0.0) or 0.0)
    return sorted(rows.values(), key=lambda r: -r["cuda_us"])[:20]


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Strict JSON: a non-finite float raises instead of being written as ``NaN``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")


DEFAULT_FLAGS = vv.RuntimePolicy(cudnn_deterministic=False, cudnn_benchmark=False)


def _flags(pinned: bool) -> contextlib.AbstractContextManager[dict[str, Any]]:
    return vv.pinned_runtime(vv.RELEASE_POLICY if pinned else DEFAULT_FLAGS)


def _gen(kind: str, seed: int) -> torch.Generator:
    return torch.Generator(device=kind).manual_seed(seed)


def _all(values: Any) -> bool:
    if isinstance(values, dict):
        if "bitwise_equal" in values:
            return bool(values["bitwise_equal"])
        return all(_all(v) for v in values.values())
    if isinstance(values, (list, tuple)):
        return all(_all(v) for v in values)
    return bool(values)


# --- manifest ------------------------------------------------------------------------------


def manifest_text() -> str:
    return json.dumps(iv.contract_manifest(), indent=1, sort_keys=True) + "\n"


# --- main ----------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    import diffusers
    from diffusers import AutoencoderKLMiniMaxH3Image

    canvas = (args.canvas, args.canvas)
    refs_hw = [(1100, 1250), (1040, 1072)]  # short edge in [1024, 2048]: kept at own size
    batch_hw = (1024, 1024)
    if args.quick:
        canvas, refs_hw, batch_hw = (512, 768), [(300, 410), (272, 304)], (256, 384)
        args.repeats, args.timing_repeats = 2, 2

    t0 = time.time()
    identity = iv.check_checkpoint_dir(args.vae_dir, verify_sha256=args.verify_sha256)
    vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, args.vae_dir, "cuda")
    load_s = time.time() - t0
    contract = iv.ImageVAEContract(vae)
    target = synthetic_image(*canvas, seed=101)
    refs = [synthetic_image(h, w, seed=200 + i) for i, (h, w) in enumerate(refs_hw)]
    batch = torch.stack([synthetic_image(*batch_hw, seed=300 + i) for i in range(3)])

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "contract_id": iv.CONTRACT_ID,
        "tiling_contract_id": iv.TILING_CONTRACT_ID,
        "rl_kernel_commit": _git("rev-parse", "HEAD"),
        "rl_kernel_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "diffusers_version": diffusers.__version__,
        "release": iv.release_identity(),
        "checkpoint_check": identity,
        "environment": {
            **vv.runtime_readback(),
            "python": platform.python_version(),
            "driver": _driver(),
        },
        "inputs": {
            "canvas": list(canvas),
            "references": [list(r.shape) for r in refs],
            "batch": list(batch.shape),
            "pixel_sha256": {
                "target": tensor_digest(target),
                "references": [tensor_digest(r) for r in refs],
                "batch": tensor_digest(batch),
            },
        },
        "load_seconds": load_s,
        "manifest": iv.contract_manifest(canvases=(canvas, batch_hw)),
        "checks": {},
    }
    checks = report["checks"]

    # Generated-image latents to decode: the encoded target (posterior mode, normalized).
    latents = contract.encode(target[None], posterior="mode", round_fp16=False).tensor

    # 1. Run-to-run determinism (I-C10: tiled repeated execution).
    runs: dict[str, list[str]] = collections.defaultdict(list)
    ref_out = dec_out = None
    for _ in range(args.repeats):
        ref_out = contract.encode_references(refs, _gen("cpu", 7))
        dec_out = contract.decode(latents)
        runs["encode_references"].append(json.dumps([o.digest for o in ref_out]))
        runs["decode_unit"].append(dec_out.digests["unit"])
        runs["decode_uint8"].append(dec_out.digests["uint8"])
    assert ref_out is not None and dec_out is not None
    checks["run_to_run"] = {
        name: {"digests": d, "bitwise_equal": len(set(d)) == 1} for name, d in runs.items()
    }
    report["outputs"] = {
        "references": [
            {"shape": list(o.tensor.shape), "sha256": o.digest, "geometry": o.record["geometry"]}
            for o in ref_out
        ],
        "decode": {
            "shape": list(dec_out.pixels.shape),
            **dec_out.digests,
        },
        "encode_record": ref_out[0].record,
        "decode_record": dec_out.record,
    }
    checks["geometry"] = {
        "reference_latents": [
            list(o.tensor.shape[-2:]) == [h // 16 // 2 * 2, w // 16 // 2 * 2]
            for o, (h, w) in zip(ref_out, refs_hw, strict=True)
        ],
        "decode_canvas": list(dec_out.pixels.shape[-2:]) == list(canvas),
    }

    # 2. Parity with the Diffusers release blocks.
    parity: dict[str, Any] = {}
    for label, pinned in (("default_flags", False), ("pinned_flags", True)):
        arm: dict[str, Any] = {}
        for gen_kind in ("cpu", "cuda"):
            ours = contract.encode_references(refs, _gen(gen_kind, 7))
            with _flags(pinned):
                theirs = diffusers_encode_references(vae, refs, _gen(gen_kind, 7))
            arm[f"encode_references_{gen_kind}_generator"] = [
                compare(o.tensor, t) for o, t in zip(ours, theirs, strict=True)
            ]
        with _flags(pinned):
            arm["decode_uint8"] = compare(dec_out.pixels, diffusers_decode(vae, latents))
        parity[label] = arm
    checks["diffusers_parity"] = parity

    # 3. Batch invariance.
    def sweep(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> dict[str, Any]:
        alone = [fn(x[i : i + 1]) for i in range(x.shape[0])]
        cases = []
        for size in range(2, x.shape[0] + 1):
            out = fn(x[:size])
            for i in range(size):
                cases.append({"batch": size, "row": i, **compare(out[i : i + 1], alone[i])})
        values = [c["max_abs"] for c in cases]
        return {
            "bitwise_equal": all(c["bitwise_equal"] for c in cases),
            "max_abs": None if any(v is None for v in values) else max(values),
            "differing_cases": sum(not c["bitwise_equal"] for c in cases),
            "cases": cases,
        }

    batch_lat = contract.encode(batch, posterior="mode", round_fp16=False).tensor

    def diffusers_batched_mode(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.encode(iv.normalize_pixels(x.cuda()), return_dict=False)[0].mode().cpu()

    def diffusers_batched_decode(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.decode(z.cuda(), return_dict=False)[0].float().cpu()

    bi: dict[str, Any] = {}
    for label, pinned in (("default_flags", False), ("pinned_flags", True)):
        with _flags(pinned):
            bi[f"diffusers_batched_{label}"] = {
                "encode_mode": sweep(diffusers_batched_mode, batch),
                "decode": sweep(diffusers_batched_decode, batch_lat),
            }
    sampled = contract.encode(batch, generator=_gen("cpu", 11))
    draws = sampled.record["generator"]["draws"]
    alone_sampled = []
    g = _gen("cpu", 11)
    for i in range(batch.shape[0]):
        alone_sampled.append(contract.encode(batch[i : i + 1], generator=g).row_digests[0])
    bi["contract"] = {
        "encode_mode": sweep(
            lambda x: contract.encode(x, posterior="mode", round_fp16=False).tensor, batch
        ),
        "decode_uint8": sweep(lambda z: contract.decode(z).pixels, batch_lat),
        "encode_sample_shared_generator": {
            "bitwise_equal": alone_sampled == sampled.row_digests,
            "draw_chain_sealed": all(
                draws[i]["after"] == draws[i + 1]["before"] for i in range(len(draws) - 1)
            ),
        },
    }
    checks["batch_invariance"] = bi

    # 4. Tiling.
    plan = iv.plan_tiles(*canvas)
    vae.disable_tiling()
    try:
        untiled = iv.ImageVAEContract(vae, tiled=False)
        u_ref = untiled.encode_references(refs[:1], _gen("cpu", 7))[0]
        u_dec = untiled.decode(latents)
        u_dec2 = untiled.decode(latents)
        timing_untiled = {
            "encode_reference": _time(
                lambda: untiled.encode_references(refs[:1], _gen("cpu", 7)), args.timing_repeats
            ),
            "decode": _time(lambda: untiled.decode(latents), args.timing_repeats),
        }
    finally:
        vae.enable_tiling(*iv.RELEASE_TILE_GEOMETRY)
    checks["tiling"] = {
        "plan": iv.tiling_identity(plan, iv.RELEASE_TILE_GEOMETRY),
        "record_plan_sha256_matches": dec_out.record["tiling"]["plan_sha256"] == plan.digest(),
        "h3_plan_sha256_equal": plan.digest() == vv.plan_tiles(*canvas).digest(),
        "modes_differ": dec_out.record["tiling"]["mode"] != u_dec.record["tiling"]["mode"],
        "untiled_mode": u_dec.record["tiling"]["mode"],
        "tiled_mode": dec_out.record["tiling"]["mode"],
        "untiled_run_to_run_equal": u_dec.digests == u_dec2.digests,
        "tiled_vs_untiled_encode_reference": compare(ref_out[0].tensor, u_ref.tensor),
        "tiled_vs_untiled_decode_unit": compare(dec_out.unit, u_dec.unit),
        "tiled_vs_untiled_decode_uint8": compare(dec_out.pixels, u_dec.pixels),
    }

    # 5. Preprocessing, normalization and output conversion across devices.
    values = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 256).expand(1, 3, 1, 256)
    pmean = torch.tensor(iv.PIXEL_MEAN, device="cuda").view(1, -1, 1, 1)
    pstd = torch.tensor(iv.PIXEL_STD, device="cuda").view(1, -1, 1, 1)
    release_cuda = ((values.cuda().to(torch.float32).div(255.0) - pmean) / pstd).cpu()
    release_cpu = (values.to(torch.float32).div(255.0) - pmean.cpu()) / pstd.cpu()
    mean_cpu, std_cpu = iv.release_latent_stats()
    unit_raw = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(3)) * 1.2 - 0.1
    checks["preprocessing"] = {
        "release_cpu_vs_cuda_mismatches": int((release_cpu != release_cuda).sum()),
        "lut_cpu_equals_cuda_release": bool(torch.equal(iv.normalize_pixels(values), release_cuda)),
        "lut_cuda_equals_cuda_release": bool(
            torch.equal(iv.normalize_pixels(values.cuda()).cpu(), release_cuda)
        ),
        "lut_sha256": iv.pixel_lut_digest(),
    }
    checks["normalization"] = {
        "normalize_cpu_vs_cuda": compare(
            iv.normalize_latents(latents, mean_cpu, std_cpu),
            iv.normalize_latents(latents.cuda(), mean_cpu, std_cpu),
        ),
        "denormalize_cpu_vs_cuda": compare(
            iv.denormalize_latents(latents, mean_cpu, std_cpu),
            iv.denormalize_latents(latents.cuda(), mean_cpu, std_cpu),
        ),
        "denormalize_pixels_cpu_vs_cuda": compare(
            iv.denormalize_pixels(unit_raw), iv.denormalize_pixels(unit_raw.cuda())
        ),
        "quantize_cpu_vs_cuda": compare(
            iv.quantize_uint8(iv.denormalize_pixels(unit_raw)),
            iv.quantize_uint8(iv.denormalize_pixels(unit_raw.cuda())),
        ),
    }

    # 6. Ablation probes: how far each wrong variant moves the output.
    ref0 = refs[0][None]
    contract_ref = contract.encode(ref0, generator=_gen("cpu", 7))
    no_round = contract.encode(ref0, generator=_gen("cpu", 7), round_fp16=False)
    with torch.no_grad(), vv.pinned_runtime():
        raw = vae.encode(iv.normalize_pixels(iv.crop_to_cells(ref0).cuda()), return_dict=False)[0]
        z = iv.crop_to_patches(
            iv.round_reference_latents(raw.sample(generator=_gen("cpu", 7)))
        ).cpu()
        x01 = iv.crop_to_cells(ref0).cuda().float() / 255 * 2 - 1
        z_pm1 = vae.encode(x01, return_dict=False)[0].mode().cpu()
        z_imnet = (
            vae.encode(iv.normalize_pixels(iv.crop_to_cells(ref0).cuda()), return_dict=False)[0]
            .mode()
            .cpu()
        )
    scalar_std = float(std_cpu.mean())
    overlap96 = None
    vae.enable_tiling(256, 256, 96, 96)
    try:
        overlap96 = iv.ImageVAEContract(vae, require_release=False).decode(latents)
    finally:
        vae.enable_tiling(*iv.RELEASE_TILE_GEOMETRY)
    rounded = (dec_out.unit * 255).round().clamp(0, 255).to(torch.uint8)
    checks["ablations"] = {
        "I11_reference_fp32_sample_vs_fp16_round_trip": compare(
            no_round.tensor, contract_ref.tensor
        ),
        "I15_scalar_normalization_vs_per_channel": compare(
            (z - float(mean_cpu.mean())) / scalar_std, contract_ref.tensor
        ),
        "I15_no_normalization_vs_per_channel": compare(z, contract_ref.tensor),
        "I16_untiled_vs_tiled_decode_uint8": checks["tiling"]["tiled_vs_untiled_decode_uint8"],
        "I16_overlap96_vs_overlap64_decode_uint8": compare(overlap96.pixels, dec_out.pixels),
        "I17_pm1_pixels_vs_imagenet_latent_mode": compare(z_pm1, z_imnet),
        "I18_round_vs_truncate_uint8": compare(rounded, dec_out.pixels),
    }

    # 7. Timing (batch 1, tiled release path).
    one_ref = refs[:1]
    timing = {
        "contract": {
            "encode_reference": _time(
                lambda: contract.encode_references(one_ref, _gen("cpu", 7)), args.timing_repeats
            ),
            "decode": _time(lambda: contract.decode(latents), args.timing_repeats),
        },
        "contract_untiled": timing_untiled,
    }
    for label, pinned in (("diffusers_default_flags", False), ("diffusers_pinned_flags", True)):
        with _flags(pinned):
            timing[label] = {
                "encode_reference": _time(
                    lambda: diffusers_encode_references(vae, one_ref, _gen("cpu", 7)),
                    args.timing_repeats,
                ),
                "decode": _time(lambda: diffusers_decode(vae, latents), args.timing_repeats),
            }
    report["timing"] = timing

    # 8. Kernels.
    with vv.pinned_runtime():
        report["kernels"] = {
            "encode_reference": _kernels(
                lambda: contract.encode_references(one_ref, _gen("cpu", 7))
            ),
            "decode": _kernels(lambda: contract.decode(latents)),
        }

    report["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30

    verdicts = {
        "identity": identity["digest"] == report["release"]["tensor_table"]["digest"],
        "run_to_run": _all({k: v["bitwise_equal"] for k, v in checks["run_to_run"].items()}),
        "geometry": _all(checks["geometry"]),
        "diffusers_parity_pinned": _all(parity["pinned_flags"]),
        "diffusers_parity_default_flags": _all(parity["default_flags"]),
        "contract_batch_invariant": _all(bi["contract"]),
        "diffusers_batched_batch_invariant_default_flags": _all(
            bi["diffusers_batched_default_flags"]
        ),
        "tiling": checks["tiling"]["record_plan_sha256_matches"]
        and checks["tiling"]["h3_plan_sha256_equal"]
        and checks["tiling"]["modes_differ"]
        and checks["tiling"]["untiled_run_to_run_equal"],
        "lut_equals_cuda_release": checks["preprocessing"]["lut_cpu_equals_cuda_release"]
        and checks["preprocessing"]["lut_cuda_equals_cuda_release"],
        "normalization_cpu_cuda_bitwise": _all(checks["normalization"]),
        "ablations_all_detected": not any(v["bitwise_equal"] for v in checks["ablations"].values()),
    }
    report["verdicts"] = verdicts
    write_report(args.out, report)
    print(json.dumps(verdicts, indent=1))
    required = (
        "identity",
        "run_to_run",
        "geometry",
        "diffusers_parity_pinned",
        "contract_batch_invariant",
        "tiling",
        "lut_equals_cuda_release",
        "normalization_cpu_cuda_bitwise",
    )
    return 0 if all(verdicts[k] for k in required) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--canvas", type=int, default=2048, help="square release canvas")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="small inputs, 2 repeats")
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        MANIFEST_PATH.write_text(manifest_text())
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        return 0
    if args.check_manifest:
        same = MANIFEST_PATH.read_text() == manifest_text()
        print("manifest up to date" if same else "manifest differs; run --write-manifest")
        return 0 if same else 1
    if args.vae_dir is None or args.out is None:
        parser.error("--vae-dir and --out are required for an evidence run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
