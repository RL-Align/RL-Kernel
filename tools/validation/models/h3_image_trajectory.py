#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Released-weight evidence for RFC #738 ``ws1_full_h3_image_t2i`` and ``ws1_full_h3_image_edit``.

    PYTHONPATH=<checkout> python tools/validation/models/h3_image_trajectory.py \
        --root <MiniMax-H3-Image> --text-encoder-root <MiniMax-H3 checkpoint> --out <DIR> \
        [--fixtures t2i_upscaled,edit_two_refs] [--controls t2i_upscaled,edit_two_refs] \
        [--train-steps 0,5] [--strict-only] [--against <DIR of another run>]

One GPU. Phase 1 runs the text contract on the released Qwen3-VL and frees it. Phase 2 loads the
VAE and the Diffusers transformer, whose tensors the strict transformer reads, and per fixture:

* ``strict``: ``full_chain.run_chain`` with the strict transformer and the Triton velocity step,
  from the prompt to the decoded image. Checks six model calls with heads 0..5 in order, and every
  edit reference row equal to its conditioned value after every update.
* ``provider``: the same chain with the pinned Diffusers transformer, and again under each
  Diffusers kernel choice (for the transformer calls; the FP32 VAE keeps its default kernels).
  Both arms must have the same inputs, heads, sigmas and timestep tables at every step; the velocity and latent distance per step, and the PSNR of the decoded
  images, are reported next to the distance between Diffusers kernel choices.
* controls (fixtures in ``--controls``): the six steps run again as one loop, alone, batched with
  another request (the same request with seed + 1) at both positions, and in a physically
  permuted layout, each compared with the chain's step records at every step and field; a
  training step (checkpointed, twice) on the trajectory's own state at each ``--train-steps``.
* ``--against``: the strict step records and output equal those of another process's run.

``--strict-only`` runs only the strict chain (for the second process). Writes ``summary.json``,
``<fixture>.<arm>.json`` artifacts, ``<fixture>.<arm>.png`` and ``<fixture>.records.json`` (no
paths). Exit 1 if a check fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.minimax_h3_image import full_chain as fc
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import trajectory as tj
from rl_engine.models.minimax_h3_image import velocity_step as vs
from rl_engine.validation.models import h3_full_model as hfm
from rl_engine.validation.models import h3_image_step as hs
from rl_engine.validation.models import h3_image_trajectory as ht

DEFAULT_FIXTURES = "t2i_upscaled,edit_two_refs"


def _tool(name: str) -> Any:
    """A sibling tool script as a module (tools are scripts, not a package)."""

    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP_TOOL = _tool("h3_image_step")
_timed = STEP_TOOL._timed


def _sampler() -> vs.ImageVelocitySampler:
    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp

    return vs.ImageVelocitySampler(op=H3ImageVelocityStepTritonOp(), device="cuda")


def _save_png(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image

    Image.fromarray(tensor[0].permute(1, 2, 0).contiguous().numpy()).save(path)


def _texts(args: argparse.Namespace, fixtures: dict[str, Any], names: list[str]) -> dict:
    from transformers import AutoProcessor, AutoTokenizer

    from rl_engine.models.minimax_h3 import qwen3vl_text as q
    from rl_engine.models.minimax_h3_image import text_presentation as tp

    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(args.root / "processor"))
    tp.verify_presentation_files(args.root)
    full, _ = q.load_release_text_encoder(
        args.text_encoder_root / "text_encoder",
        device=torch.device("cuda"),
        num_layers=q.RELEASE_NUM_LAYERS,
    )
    encoder = q.Qwen3VLLayer50Encoder(full)
    out = {}
    for name in names:
        request = fixtures[name]["request"]
        prepared = [fc.prepare_reference(r) for r in request.references]
        text = fc.contract_text(request, prepared, encoder, tokenizer, processor)
        text.prompt_embeds = text.prompt_embeds.cpu()
        out[name] = (prepared, text)
        print(name, "text rows", len(text.token_ids), flush=True)
    del encoder, full
    gc.collect()
    torch.cuda.empty_cache()
    return out


class _Context:
    def __init__(self, args: argparse.Namespace) -> None:
        from diffusers import AutoencoderKLMiniMaxH3Image

        from rl_engine.models.minimax_h3_image import image_vae as iv

        device = torch.device("cuda")
        vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, args.root / "vae_diffusers", device)
        self.vae = iv.ImageVAEContract(vae)
        self.model = STEP_TOOL._transformer(args.root, device)
        self.heads = ph.from_diffusers(self.model)
        self.strict = hs.StrictImageTransformer(
            dict(self.model.named_parameters()), self.heads, require_attention_row=True
        )

    def chain(self, request, prepared, text, denoiser, label):
        return fc.run_chain(
            request,
            text,
            vae_contract=self.vae,
            denoiser=denoiser,
            device=torch.device("cuda"),
            sampler=_sampler(),
            prepared=prepared,
            label=label,
            keep_tensors=True,
        )


def _strict_chain(ctx: _Context, request, prepared, text, label):
    counted = ht.CountingDenoiser(hs.StrictImageDenoiser(ctx.strict))
    torch.cuda.reset_peak_memory_stats()
    run, seconds = _timed(lambda: ctx.chain(request, prepared, text, counted, label))
    fc.verify(run.artifact)
    return run, {
        "seconds": round(seconds, 1),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "denoiser": counted.readback(),
        "fixed_reference_rows": ht.fixed_reference_rows(run.artifact),
    }


def _arm_distance(lhs: fc.ChainRun, rhs: fc.ChainRun) -> dict[str, Any]:
    r = lhs.artifact["trajectory"]["inputs"]["layout"]["num_reference_rows"]
    return {
        "per_step": ht.trajectory_distance(lhs.tensors, rhs.tensors, r),
        "output_psnr_db": ht.psnr(lhs.output, rhs.output),
        "first_drift": tj.first_drift(lhs.artifact["trajectory"], rhs.artifact["trajectory"]),
    }


class _VariantDenoiser(fc.DiffusersTransformerDenoiser):
    """The Diffusers transformer under another kernel choice; the VAE (FP32) keeps the default."""

    def __init__(self, transformer: Any, variant: str) -> None:
        super().__init__(transformer)
        self.variant = variant

    @property
    def backend_id(self) -> str:
        return f"{super().backend_id}:{self.variant}"

    def __call__(self, *args: Any) -> torch.Tensor:
        with hfm.provider_variant_context(self.variant):
            return super().__call__(*args)


def _provider(ctx: _Context, request, prepared, text, strict: fc.ChainRun, name: str, out: Path):
    counted = ht.CountingDenoiser(fc.DiffusersTransformerDenoiser(ctx.model))
    run, seconds = _timed(lambda: ctx.chain(request, prepared, text, counted, f"{name}:diffusers"))
    fc.verify(run.artifact)
    ours, theirs = strict.artifact["trajectory"], run.artifact["trajectory"]
    row: dict[str, Any] = {
        "seconds": round(seconds, 1),
        "denoiser": counted.readback(),
        "input_mismatches": tj.input_mismatches(ours, theirs),
        "schedule_equal": ht.schedule_records(strict.artifact) == ht.schedule_records(run.artifact),
        "strict_vs_provider": _arm_distance(strict, run),
        "variants": {},
    }
    _write(out, name, "diffusers", run)
    for variant in hfm.PROVIDER_VARIANTS[1:]:
        try:
            other = ctx.chain(
                request, prepared, text, _VariantDenoiser(ctx.model, variant), variant
            )
        except RuntimeError as exc:  # a kernel this build or sequence length does not have
            row["variants"][variant] = {"unavailable": str(exc).splitlines()[0][:120]}
            continue
        if ht.chain_records(other.artifact) == ht.chain_records(run.artifact):
            row["variants"][variant] = {"equal_to_default": True}
        else:
            row["variants"][variant] = _arm_distance(other, run)
        del other
        gc.collect()
    row["against_variants"] = _against_variants(row)
    del run
    return row


def _against_variants(row: dict[str, Any]) -> dict[str, Any]:
    """Strict distance over the largest Diffusers-variant distance, per step, and the PSNRs."""

    variants = [v for v in row["variants"].values() if "per_step" in v]
    if not variants:
        return {"variants": 0}
    ours = row["strict_vs_provider"]["per_step"]
    out: dict[str, Any] = {"variants": len(variants), "per_step": []}
    for i, step in enumerate(ours):
        entry = {"step": i}
        for key in ("velocity_target", "velocity_reference", "latent"):
            if key in step:
                largest = max(v["per_step"][i][key] for v in variants)
                entry[f"{key}_ratio"] = step[key] / largest if largest else None
        out["per_step"].append(entry)
    out["output_psnr_db"] = {
        "strict": row["strict_vs_provider"]["output_psnr_db"],
        "variants_min": min(v["output_psnr_db"] for v in variants),
    }
    return out


def _training(ctx: _Context, start: ht.TrajectoryStart, loop: ht.LoopRun, step: int):
    from rl_engine.models.minimax_h3_image import pdd_schedule as ps

    unique, indices = ps.row_timestep_plan(start.layout)[step]
    inputs = hs.StepInputs(
        loop.step_inputs[step], start.prompt_embeds, start.layout, unique, indices, step
    )
    _, base = hs.run_strict(ctx.strict, inputs)
    return STEP_TOOL._training(ctx.strict, inputs, base, False)


def _controls(ctx, fixture, prepared, text, strict: fc.ChainRun, train_steps, name):
    ours = ht.chain_records(strict.artifact)
    start = ht.chain_start(strict, text)
    out: dict[str, Any] = {}
    loop, seconds = _timed(
        functools.partial(ht.run_loop, ctx.strict, start, sampler=_sampler, keep_inputs=True)
    )
    out["loop_alone"] = {
        "first_drift": ht.first_record_drift(loop.records[0], ours),
        "calls": loop.calls,
        "seconds": round(seconds, 1),
    }
    request = fixture["request"]
    partner_request = dataclasses.replace(request, seed=request.seed + 1)
    partner, _ = _strict_chain(ctx, partner_request, prepared, text, f"{name}:partner")
    theirs = ht.chain_records(partner.artifact)
    other = ht.chain_start(partner, text)
    out["partner"] = {
        "seed": partner_request.seed,
        "differs_from_request": ht.first_record_drift(ours, theirs) is not None,
        "fixed_reference_rows": ht.fixed_reference_rows(partner.artifact)["fixed"],
    }
    for position, pair in ((0, [start, other]), (1, [other, start])):
        torch.cuda.reset_peak_memory_stats()
        batched, seconds = _timed(
            functools.partial(ht.run_loop, ctx.strict, ht.stack(pair), sampler=_sampler)
        )
        out[f"batch_row_{position}"] = {
            "request_first_drift": ht.first_record_drift(batched.records[position], ours),
            "partner_first_drift": ht.first_record_drift(batched.records[1 - position], theirs),
            "seconds": round(seconds, 1),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
        del batched
    g = torch.Generator().manual_seed(3)
    order = torch.randperm(start.layout.sequence_length, generator=g)
    moved = ht.run_loop(ctx.strict, start, sampler=_sampler, order=order)
    out["permuted_first_drift"] = ht.first_record_drift(moved.records[0], ours)
    out["training"] = {str(s): _training(ctx, start, loop, s) for s in train_steps}
    drifts = [
        out["loop_alone"]["first_drift"],
        out["batch_row_0"]["request_first_drift"],
        out["batch_row_0"]["partner_first_drift"],
        out["batch_row_1"]["request_first_drift"],
        out["batch_row_1"]["partner_first_drift"],
        out["permuted_first_drift"],
    ]
    out["ok"] = (
        all(d is None for d in drifts)
        and out["loop_alone"]["calls"] == list(range(ph.NUM_HEADS))
        and out["partner"]["differs_from_request"]
        and out["partner"]["fixed_reference_rows"] in (True, None)
        and all(t["ok"] and t["head"] == int(s) for s, t in out["training"].items())
    )
    return out


def _write(out: Path, name: str, arm: str, run: fc.ChainRun) -> None:
    (out / f"{name}.{arm}.json").write_text(json.dumps(run.artifact, indent=1) + "\n")
    _save_png(run.output, out / f"{name}.{arm}.png")


def _records_file(run: fc.ChainRun) -> dict[str, Any]:
    traj = run.artifact["trajectory"]
    return {
        "inputs_sha256": traj["inputs_sha256"],
        "step_records": ht.chain_records(run.artifact),
        "final_latent_sha256": traj["final_latent_sha256"],
        "decoded_sha256": traj["decoded"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--text-encoder-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--controls", default=DEFAULT_FIXTURES)
    parser.add_argument("--train-steps", default="0,5")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--against", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fixtures = STEP_TOOL._chain_fixtures(args.root)
    names = args.fixtures.split(",")
    controls = set(args.controls.split(",")) if args.controls and not args.strict_only else set()
    train_steps = [int(s) for s in args.train_steps.split(",")] if args.train_steps else []
    texts = _texts(args, fixtures, names)
    ctx = _Context(args)
    summary: dict[str, Any] = {
        "schema": "rlkernel.h3_image.ws1_full_evidence.v1",
        "contract_id": ht.CONTRACT_ID,
        "environment": STEP_TOOL._environment(),
        "heads_released": ctx.heads.digests == ph.released_digests(),
        "bindings": ctx.strict.bindings,
        "binding_counts": {},
        "sampler": _sampler().provenance(),
        "fixtures": {},
    }
    for b in ctx.strict.bindings.values():
        counts = summary["binding_counts"]
        counts[b["status"]] = counts.get(b["status"], 0) + 1
    ok = bool(summary["heads_released"])
    for name in names:
        fixture = fixtures[name]
        request = fixture["request"]
        prepared, text = texts[name]
        strict, strict_row = _strict_chain(ctx, request, prepared, text, f"{name}:strict")
        traj = strict.artifact["trajectory"]
        row: dict[str, Any] = {
            "source": fixture["source"],
            "task": request.task,
            "size": [request.height, request.width],
            "canvas": traj["inputs"]["canvas"]["canvas"],
            "reference_latents": [r["latent"] for r in traj["inputs"]["references"]],
            "sequence_length": traj["inputs"]["layout"]["sequence_length"],
            "num_reference_rows": traj["inputs"]["layout"]["num_reference_rows"],
            "strict": strict_row,
            "final_latent_sha256": traj["final_latent_sha256"],
            "output_sha256": traj["decoded"]["sha256"],
        }
        fixed = strict_row["fixed_reference_rows"]["fixed"]
        ok &= strict_row["denoiser"]["six_in_order"] and fixed in (True, None)
        ok &= (fixed is True) == (request.task == "edit")
        _write(args.out, name, "strict", strict)
        records = _records_file(strict)
        if not args.strict_only:
            row["provider"] = _provider(ctx, request, prepared, text, strict, name, args.out)
            p = row["provider"]
            ok &= p["denoiser"]["six_in_order"] and not p["input_mismatches"]
            ok &= p["schedule_equal"]
        if name in controls:
            row["controls"] = _controls(ctx, fixture, prepared, text, strict, train_steps, name)
            ok &= bool(row["controls"]["ok"])
        if args.against is not None:
            other = json.loads((args.against / f"{name}.records.json").read_text())
            row["equal_to_other_process"] = other == records
            ok &= row["equal_to_other_process"]
        (args.out / f"{name}.records.json").write_text(json.dumps(records, indent=1) + "\n")
        del strict
        gc.collect()
        torch.cuda.empty_cache()
        summary["fixtures"][name] = row
        summary["ok"] = ok
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        print(name, json.dumps(strict_row["denoiser"]), strict_row["seconds"], flush=True)
    print("ok", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
