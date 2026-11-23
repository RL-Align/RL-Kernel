#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for RFC #738 ``image_reference_conditioning``: fixture digests and probe arms.

    PYTHONPATH=<checkout> python tools/validation/models/h3_image_reference_conditioning.py \
        --out reference-conditioning.json [--device cuda]

Writes one JSON with no paths. For each fixture (1, 2 and 3 references with heterogeneous
grids, and a 1024^2 + 2048x1536 high-resolution pair) and each execution arm (CPU; CUDA with a
CPU or CUDA generator; the CUDA ``conditioning_noise_mix`` kernel when the extension has it)
it records the conditioning artifact, the draw ledger and whether the arm equals the CPU
reference. The probe arms change one declared factor and report how far the reference rows
move (rows changed, elements changed, max |diff|) and whether the contract's guard trips:

* ``I12``: reference noise drawn before the target noise (the H3 Ref2VA order);
* ``I13_clean`` / ``I13_schedule_t`` / ``I13_resample``: level 1.0, the first schedule
  timestep, fresh noise at step 3;
* ``I14``: the scheduler writes every image row, not only the generated suffix;
* ``python_float_level``: ``1 - t`` in FP64 (Diffusers' ``scale_noise`` with a float).

Latents are synthetic (seeded normals); the VAE is ``image_vae_contract``'s subject.
"""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import reference_conditioning as rc

SCHEMA = "rlkernel.h3_image.reference_conditioning_evidence.v1"
RELEASE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75]
RELEASE_SIGMAS += [0.5327739119529724, 0.23200887441635132, 0.0]
FIXTURES: dict[str, dict[str, Any]] = {
    "edit1": {"refs": [(64, 64)], "target": (64, 64)},
    "edit2": {"refs": [(4, 6), (2, 12)], "target": (6, 8)},
    "edit3": {"refs": [(4, 6), (2, 12), (30, 98)], "target": (62, 98)},
    "hires2": {"refs": [(64, 64), (128, 96)], "target": (128, 128)},
}
SEED = 1234


def _latents(grids, device) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return [torch.randn(1, 24, h, w, generator=generator).to(device) for h, w in grids]


def _layout(grids, target) -> pl.PackedLayout:
    tags = torch.ones(12, dtype=torch.long)
    tags[1:5] = pl.IMAGE_TAG
    return pl.build_layout(tags, grids, target)


def _run(fixture, device, gen_device, *, mix_op=None, order="released", level=None):
    """One request: posterior draws, target noise, reference noise; returns everything."""

    grids, target = fixture["refs"], fixture["target"]
    ledger = rc.DrawLedger(torch.Generator(gen_device).manual_seed(SEED))
    for i, (h, w) in enumerate(grids):  # stand-ins for the VAE's posterior draws
        ledger.draw("posterior", i, (1, 24, h, w), device)
    kwargs: dict[str, Any] = {"mix_op": mix_op, "layout": _layout(grids, target)}
    if level is not None:
        kwargs["level"] = level
    if order == "released":
        noise = rc.draw_target_noise(ledger, target, device)
        cond = rc.condition_references(_latents(grids, device), ledger=ledger, **kwargs)
    else:  # I12: reference noise first
        cond = rc.condition_references(_latents(grids, device), ledger=ledger, **kwargs)
        noise = rc.draw_target_noise(ledger, target, device)
    target_rows = pp.ImagePatchPacker().pack_image(noise)[0]
    return cond, ledger, target_rows


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a, b = a.float().cpu(), b.float().cpu()
    bits = a.view(torch.int32) != b.view(torch.int32)
    return {
        "rows_changed": int(bits.any(dim=1).sum()),
        "rows": int(a.shape[0]),
        "elements_changed": int(bits.sum()),
        "elements": int(a.numel()),
        "max_abs": float((a - b).abs().max()),
    }


def _loop(cond, target_rows, *, write_all=False, resample_at=None) -> dict[str, Any]:
    """Six seeded updates of the generated suffix on the released sigma grid (a stand-in for
    ``image_velocity_step``); the reference prefix is checked after each one."""

    image = cond.image_rows(target_rows).clone()
    r = 0 if write_all else cond.num_reference_rows
    sigmas = torch.tensor(RELEASE_SIGMAS, dtype=torch.float32)
    sigmas[:-1] = 1.0 - (1.0 - sigmas[:-1])  # the released f32 round trip
    velocity = torch.Generator().manual_seed(77)
    for i in range(6):
        v = torch.randn(image.shape, generator=velocity).to(image.device)
        image[r:] = (image[r:].float() + (sigmas[i + 1] - sigmas[i]) * v[r:]).to(image.dtype)
        if resample_at == i:
            image[: cond.num_reference_rows] = torch.randn(
                cond.rows.shape, generator=torch.Generator().manual_seed(i)
            ).to(image.device)
        try:
            cond.check_fixed(image, step=i)
        except rc.ReferenceRowsChanged as exc:
            return {"guard": "tripped", "step": i, "message": str(exc)}
    return {"guard": "passed", "steps": 6, "reference_rows_sha256": tensor_digest(image[:r])}


def _cuda_kernel():
    try:
        from rl_engine.backends.cuda.conditioning.conditioning_noise_mix import (
            ConditioningNoiseMixCudaOp,
        )

        return ConditioningNoiseMixCudaOp()
    except RuntimeError as exc:
        return str(exc)


def collect(device: str) -> dict[str, Any]:
    arms: list[tuple[str, str, str, Any]] = [("cpu", "cpu", "cpu", None)]
    kernel: Any = None
    if device == "cuda":
        arms += [("cuda-cpu_generator", "cuda", "cpu", None)]
        arms += [("cuda-cuda_generator", "cuda", "cuda", None)]
        kernel = _cuda_kernel()
        if not isinstance(kernel, str):
            arms += [("cuda_kernel-cpu_generator", "cuda", "cpu", kernel)]
            arms += [("cuda_kernel-cuda_generator", "cuda", "cuda", kernel)]
    fixtures: dict[str, Any] = {}
    for name, fixture in FIXTURES.items():
        ref_cpu, _, _ = _run(fixture, "cpu", "cpu")
        out: dict[str, Any] = {"refs": fixture["refs"], "target": list(fixture["target"])}
        for arm, dev, gen, op in arms:
            cond, ledger, target_rows = _run(fixture, dev, gen, mix_op=op)
            ledger.check_order(len(fixture["refs"]))
            art = cond.artifact()
            out[arm] = {
                "rows_sha256": art["digests"]["rows"],
                "artifact_sha256": art["artifact_sha256"],
                "num_reference_rows": art["num_reference_rows"],
                "draw_order_sha256": ledger.artifact()["order_sha256"],
                "mix_backend": art["provenance"]["mix"]["backend_id"],
                "equals_cpu_cpu_generator": gen == "cpu"
                and torch.equal(cond.rows.cpu().view(torch.int32), ref_cpu.rows.view(torch.int32)),
                "loop": _loop(cond, target_rows),
            }
        fixtures[name] = out
    probes = _probes()
    return {
        "schema": SCHEMA,
        "environment": _environment(kernel),
        "contract": rc.contract_manifest(),
        "fixtures": fixtures,
        "probes": probes,
    }


def _probes() -> dict[str, Any]:
    fixture = FIXTURES["edit3"]
    released, ledger, target_rows = _run(fixture, "cpu", "cpu")
    out: dict[str, Any] = {}
    i12, i12_ledger, _ = _run(fixture, "cpu", "cpu", order="reference_first")
    try:
        i12_ledger.check_order(len(fixture["refs"]))
        guard = "passed"
    except rc.DrawOrderError:
        guard = "tripped"
    out["I12"] = {"ledger": guard, **_diff(i12.rows, released.rows)}
    t0 = float(1.0 - torch.tensor(RELEASE_SIGMAS[1], dtype=torch.float32))
    for name, level in (("I13_clean", 1.0), ("I13_schedule_t", t0)):
        moved, _, _ = _run(fixture, "cpu", "cpu", level=level)
        out[name] = {"level": level, **_diff(moved.rows, released.rows)}
    out["I13_resample"] = _loop(released, target_rows, resample_at=3)
    out["I14"] = _loop(released, target_rows, write_all=True)
    lat = _latents(fixture["refs"], "cpu")
    noise = [
        torch.randn(z.shape, generator=torch.Generator().manual_seed(k)) for k, z in enumerate(lat)
    ]
    f32 = rc.condition_references(lat, noise=noise)
    packer = pp.ImagePatchPacker()
    f64 = torch.cat(
        [packer.pack_image(0.999 * z + (1.0 - 0.999) * n)[0] for z, n in zip(lat, noise)]
    )
    out["python_float_level"] = _diff(f64, f32.rows)
    out["released_loop"] = _loop(released, target_rows)
    return out


def _environment(kernel: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_kernel": (
            "available" if kernel is not None and not isinstance(kernel, str) else kernel
        ),
    }
    if torch.cuda.is_available():
        env.update(
            device_name=torch.cuda.get_device_name(0),
            compute_capability=list(torch.cuda.get_device_capability(0)),
            cuda=torch.version.cuda,
        )
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda needs a visible GPU")
    result = collect(args.device)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
        fh.write("\n")
    bad = [
        f"{name}/{arm}"
        for name, fixture in result["fixtures"].items()
        for arm, row in fixture.items()
        if isinstance(row, dict)
        and (row["loop"]["guard"] != "passed" or not row["equals_cpu_cpu_generator"])
        and not arm.endswith("cuda_generator")
    ]
    print(json.dumps(result["probes"], indent=1, default=str)[:4000])
    print(f"arms not equal to the CPU reference or failing the fixed-row loop: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
