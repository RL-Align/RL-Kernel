#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Released-weight evidence for RFC #738 ``ws1_one_h3_image_step``.

    PYTHONPATH=<checkout> python tools/validation/models/h3_image_step.py capture \
        --root <MiniMax-H3-Image> --text-encoder-root <MiniMax-H3 checkpoint> --out <INPUTS>
    PYTHONPATH=<checkout> python tools/validation/models/h3_image_step.py replay \
        --root <MiniMax-H3-Image> --inputs <INPUTS> --out <DIR> [--against <DIR of another run>]

``capture`` runs the real preprocessing of each fixture (Qwen3-VL text, VAE references, noise,
packed layout: the ``phase_b_image_full_chain`` code) and the six released transformer calls, and
saves the inputs of every call. ``replay`` runs the steps from those files, so it needs neither
the text encoder nor the VAE:

* ``steps``: strict and provider arm on the same inputs at every step: distance per node, per row
  group and per block on the provider's own inputs, and the same distances between Diffusers
  kernel choices;
* controls at the control step: strict repeated, batched with another request (both positions),
  physically permuted, with the next PDD head (must drift at ``pdd_proj_out``), and in a training
  step under checkpointing (``--plain-autograd`` also without it);
* ``--against``: node digests equal those of another process's run.

Writes ``summary.json`` and ``<fixture>.digests.json`` (no paths). Exit 1 if a check fails.
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.minimax_h3_image import full_chain as fc
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.validation.models import h3_full_model as hfm
from rl_engine.validation.models import h3_image_step as hs

DEFAULT_FIXTURES = "t2i_upscaled,edit_two_refs"


def _chain_fixtures(root: Path) -> dict[str, dict[str, Any]]:
    """The fixtures of ``h3_image_full_chain.py`` (tools are scripts, not a package)."""

    path = Path(__file__).with_name("h3_image_full_chain.py")
    spec = importlib.util.spec_from_file_location("h3_image_full_chain_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixtures(root)


def _environment() -> dict[str, Any]:
    import diffusers
    import triton

    from rl_engine.backends.extension import _C

    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "extension_in_checkout": Path(_C.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[3]),
    }


def _transformer(root: Path, device: torch.device) -> Any:
    from diffusers import MiniMaxH3ImageTransformer2DModel

    try:
        # keeps the FP32 modules of the checkpoint only on the low_cpu_mem_usage path
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise SystemExit("accelerate is required to load transformer_diffusers") from exc
    model = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        str(root / "transformer_diffusers"), torch_dtype=torch.bfloat16
    )
    return model.to(device).eval().requires_grad_(False)


# --- capture -----------------------------------------------------------------------------------


class _Captured(Exception):
    """All six calls were captured; the VAE decode of the chain is not needed."""


class _CapturingDenoiser:
    """The released transformer, with the inputs of every call kept."""

    def __init__(self, inner: fc.DiffusersTransformerDenoiser) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step):
        self.calls.append(
            {
                "image_rows": image_rows.detach().cpu().clone(),
                "unique_timesteps": unique_timesteps.cpu().clone(),
                "timestep_indices": timestep_indices.cpu().clone(),
                "layout": layout,
                "prompt_embeds": prompt_embeds.detach().cpu().clone(),
            }
        )
        if len(self.calls) == ph.NUM_HEADS:
            raise _Captured
        return self.inner(
            image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step
        )


def capture(args: argparse.Namespace) -> int:
    from diffusers import AutoencoderKLMiniMaxH3Image
    from safetensors.torch import save_file
    from transformers import AutoProcessor, AutoTokenizer

    from rl_engine.models.minimax_h3 import qwen3vl_text as q
    from rl_engine.models.minimax_h3_image import image_vae as iv
    from rl_engine.models.minimax_h3_image import text_presentation as tp

    device = torch.device("cuda")
    fixtures = _chain_fixtures(args.root)
    names = args.fixtures.split(",")
    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(args.root / "processor"))
    tp.verify_presentation_files(args.root)
    full, _ = q.load_release_text_encoder(
        args.text_encoder_root / "text_encoder", device=device, num_layers=q.RELEASE_NUM_LAYERS
    )
    encoder = q.Qwen3VLLayer50Encoder(full)
    texts = {}
    for name in names:
        request = fixtures[name]["request"]
        prepared = [fc.prepare_reference(r) for r in request.references]
        text = fc.contract_text(request, prepared, encoder, tokenizer, processor)
        text.prompt_embeds = text.prompt_embeds.cpu()
        texts[name] = (prepared, text)
    del encoder, full
    gc.collect()
    torch.cuda.empty_cache()

    vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, args.root / "vae_diffusers", device)
    contract = iv.ImageVAEContract(vae)
    inner = fc.DiffusersTransformerDenoiser(_transformer(args.root, device))
    for name in names:
        request = fixtures[name]["request"]
        prepared, text = texts[name]
        denoiser = _CapturingDenoiser(inner)
        try:
            fc.run_chain(
                request,
                text,
                vae_contract=contract,
                denoiser=denoiser,
                device=device,
                prepared=prepared,
                label=name,
            )
        except _Captured:
            pass
        layout = denoiser.calls[0]["layout"]
        tensors = {
            "prompt_embeds": denoiser.calls[0]["prompt_embeds"],
            "text_token_tags": layout.token_tags[layout.text_indices].contiguous(),
        }
        for step, call in enumerate(denoiser.calls):
            for key in ("image_rows", "unique_timesteps", "timestep_indices"):
                tensors[f"{key}.{step}"] = call[key].contiguous()
        meta = {
            "source": fixtures[name]["source"],
            "task": request.task,
            "size": [request.height, request.width],
            "reference_grids": [list(g) for g in layout.reference_grids],
            "target_grid": list(layout.target_grid),
            "sequence_length": layout.sequence_length,
            "layout_sha256": pl.layout_digest(layout),
        }
        save_file(tensors, str(args.out / f"{name}.inputs.safetensors"))
        (args.out / f"{name}.inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(name, json.dumps(meta), flush=True)
    return 0


def load_inputs(directory: Path, name: str) -> tuple[dict[str, Any], list[hs.StepInputs]]:
    """The captured metadata and the six step inputs of a fixture (layout rebuilt and checked)."""

    from safetensors.torch import load_file

    meta = json.loads((directory / f"{name}.inputs.json").read_text())
    tensors = load_file(str(directory / f"{name}.inputs.safetensors"))
    layout = pl.build_layout(
        tensors["text_token_tags"],
        [tuple(g) for g in meta["reference_grids"]],
        tuple(meta["target_grid"]),
    )
    if pl.layout_digest(layout) != meta["layout_sha256"]:
        raise ValueError(f"{name}: the rebuilt layout is not the captured one")
    plan = ps.row_timestep_plan(layout)
    steps = []
    for step in range(ph.NUM_HEADS):
        unique, indices = tensors[f"unique_timesteps.{step}"], tensors[f"timestep_indices.{step}"]
        if not torch.equal(unique, plan[step][0]) or not torch.equal(indices, plan[step][1]):
            raise ValueError(f"{name}: step {step} timesteps are not the sealed table")
        steps.append(
            hs.StepInputs(
                tensors[f"image_rows.{step}"][None],
                tensors["prompt_embeds"],
                layout,
                unique,
                indices,
                step,
            )
        )
    return meta, steps


# --- replay ------------------------------------------------------------------------------------


def _timed(fn: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def _summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The nodes a reader needs: boundaries, first / last block and the worst node."""

    picked = [r for r in rows if "block." not in r["node"]]
    blocks = [r for r in rows if r["node"].startswith("block.")]
    worst = max(rows, key=lambda r: r["rel_frobenius"])
    return {
        "first_not_bitwise": next((r["node"] for r in rows if not r["bitwise_equal"]), None),
        "nodes": {r["node"]: r for r in [*picked, blocks[0], blocks[-1]]},
        "worst_node": worst,
    }


def _other_request(inputs: hs.StepInputs, seed: int) -> hs.StepInputs:
    """Another request with the same layout: its own noise rows and text features."""

    g = torch.Generator().manual_seed(seed)
    return hs.StepInputs(
        torch.randn(inputs.image_rows.shape, generator=g),
        inputs.prompt_embeds.flip(1),
        inputs.layout,
        inputs.unique_timesteps,
        inputs.timestep_indices,
        inputs.step,
    )


def _per_node(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "per_node_rel_frobenius": {r["node"]: r["rel_frobenius"] for r in rows},
        "per_node_by_group": {r["node"]: r["by_group"] for r in rows if "by_group" in r},
        "largest_error": {r["node"]: r["largest_error"] for r in rows if "largest_error" in r},
    }


def _spread(model: Any, inputs: hs.StepInputs, default: hs.StepTrace) -> dict[str, Any]:
    """Distance of the provider under other kernel choices from its default run.

    ``chained`` runs the whole step; ``isolated`` runs each block on the default run's inputs.
    A variant whose velocity is bitwise the default's is the default kernel under another name.
    """

    out: dict[str, Any] = {}
    layout = inputs.layout
    for variant in hfm.PROVIDER_VARIANTS[1:]:
        try:
            with hfm.provider_variant_context(variant):
                _, trace = hs.run_provider(model, inputs, keep_tensors=True)
                if trace.digests == default.digests:
                    out[variant] = {"equal_to_default": True}
                    continue
                isolated = [
                    hs.distance(n, v, default.tensors[n], layout)
                    for n, v in hs.provider_isolated_blocks(model, inputs, default)
                ]
        except RuntimeError as exc:  # a kernel this build or sequence length does not have
            out[variant] = {"unavailable": str(exc).splitlines()[0][:120]}
            continue
        rows = hs.compare(trace, default, layout)
        out[variant] = {
            **_summary_rows(rows),
            "chained": _per_node(rows),
            "isolated": _per_node(isolated),
        }
    return out


def _against_spread(
    ours: dict[str, Any], isolated: dict[str, Any], spread: dict[str, Any]
) -> dict[str, Any]:
    """Strict distance over the largest variant distance: velocity, and per isolated block."""

    variants = [v for v in spread.values() if "chained" in v]
    if not variants:
        return {"variants": 0}
    key = "per_node_rel_frobenius"
    largest = {n: max(v["chained"][key][n] for v in variants) for n in ours[key]}
    local = {n: max(v["isolated"][key][n] for v in variants) for n in isolated[key]}
    ratios = {n: isolated[key][n] / local[n] for n in local if local[n]}
    worst = max(ratios, key=lambda n: ratios[n])
    return {
        "variants": len(variants),
        "velocity_ratio": ours[key]["velocity"] / largest["velocity"],
        "chained_nodes_above_variants": sorted(n for n in largest if ours[key][n] > largest[n]),
        "isolated_worst_ratio": {"node": worst, "ratio": ratios[worst]},
        "isolated_max_rel_frobenius": max(isolated[key].values()),
        "isolated_max_rel_frobenius_variants": max(local.values()),
    }


TRAINABLE = (
    "image_rows",
    "prompt_embeds",
    "proj_in.weight",
    "proj_in.bias",
    *(f"pdd_proj_out.{i}.{part}" for i in range(ph.NUM_HEADS) for part in ("weight", "bias")),
)


def _training(
    strict: hs.StrictImageTransformer, inputs: hs.StepInputs, base: hs.StepTrace, plain: bool
) -> dict[str, Any]:
    g = torch.Generator().manual_seed(11)
    upstream = torch.randn((1, inputs.image_rows.shape[1], ph.HEAD_WIDTH), generator=g)
    runs, out = {}, {}
    modes = [("checkpointed", True), ("checkpointed_again", True)]
    modes += [("plain", False)] if plain else []
    for label, checkpointed in modes:
        trace = hs.StepTrace()
        run = functools.partial(
            hs.training_step,
            strict,
            inputs,
            TRAINABLE,
            upstream,
            checkpoint_blocks=checkpointed,
            trace=trace,
        )
        torch.cuda.reset_peak_memory_stats()
        runs[label], seconds = _timed(run)
        out[label] = {
            "forward_first_drift_vs_rollout": hs.first_drift(base, trace),
            "seconds": round(seconds, 2),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
    reference = runs["checkpointed"]["grads"]
    head = inputs.step
    other = [n for n in reference if n.startswith("pdd_proj_out.") and f".{head}." not in n]
    used = {n: g for n, g in reference.items() if n not in other}
    out["head"] = runs["checkpointed"]["head"]["head"]
    out["other_heads_have_no_gradient"] = all(reference[n] is None for n in other)
    out["finite_and_nonzero"] = {
        n: bool(torch.isfinite(g).all()) and bool((g != 0).any()) for n, g in used.items()
    }
    out["gradient_norms"] = {n: float(g.double().norm()) for n, g in used.items()}
    out["gradients_equal"] = {
        label: all(torch.equal(runs[label]["grads"][n], g) for n, g in used.items())
        for label, _ in modes[1:]
    }
    out["ok"] = (
        all(out[label]["forward_first_drift_vs_rollout"] is None for label, _ in modes)
        and out["other_heads_have_no_gradient"]
        and all(out["finite_and_nonzero"].values())
        and all(out["gradients_equal"].values())
        and out["head"] == head
    )
    return out


def _controls(
    strict: hs.StrictImageTransformer, inputs: hs.StepInputs, base: hs.StepTrace, plain: bool
) -> dict[str, Any]:
    layout = inputs.layout
    out: dict[str, Any] = {}
    _, again = hs.run_strict(strict, inputs)
    out["repeat_first_drift"] = hs.first_drift(base, again)
    for position in (0, 1):
        pair = [inputs, _other_request(inputs, 5)]
        batch = hs.stack(pair if position == 0 else pair[::-1])
        _, trace = hs.run_strict(strict, batch, keep_tensors=True)
        out[f"batch_row_{position}_first_drift"] = hs.first_drift(base, trace.row(position), layout)
        del trace
    g = torch.Generator().manual_seed(3)
    order = torch.randperm(layout.sequence_length, generator=g)
    _, moved = hs.run_strict(strict, hs.permute_inputs(inputs, order))
    out["permuted_first_drift"] = hs.first_drift(base, moved)
    wrong = hs.StepInputs(
        inputs.image_rows,
        inputs.prompt_embeds,
        layout,
        inputs.unique_timesteps,
        inputs.timestep_indices,
        inputs.step + 1,
    )
    _, probe = hs.run_strict(strict, wrong, keep_tensors=True)
    out["next_head_first_drift"] = hs.first_drift(base, probe, layout)
    del probe
    gc.collect()
    out["training"] = _training(strict, inputs, base, plain)
    out["ok"] = (
        all(out[k] is None for k in out if k.endswith("_first_drift") and "head" not in k)
        and (out["next_head_first_drift"] or {}).get("node") == "pdd_proj_out"
        and out["training"]["ok"]
    )
    return out


def replay(args: argparse.Namespace) -> int:
    device = torch.device("cuda")
    model = _transformer(args.root, device)
    heads = ph.from_diffusers(model)
    fused = ph.from_fused_file(args.root)
    strict = hs.StrictImageTransformer(
        dict(model.named_parameters()), heads, require_attention_row=True
    )
    summary: dict[str, Any] = {
        "schema": "rlkernel.h3_image.one_step_evidence.v1",
        "contract_id": hs.CONTRACT_ID,
        "environment": _environment(),
        "heads": {
            "artifact_sha256": heads.artifact()["sha256"],
            "fused_file_equals_diffusers_layout": fused.digests == heads.digests,
            "released": heads.digests == ph.released_digests(),
        },
        "bindings": strict.bindings,
        "binding_counts": {},
        "fixtures": {},
    }
    for b in strict.bindings.values():
        counts = summary["binding_counts"]
        counts[b["status"]] = counts.get(b["status"], 0) + 1
    ok = summary["heads"]["fused_file_equals_diffusers_layout"] and summary["heads"]["released"]
    steps = [int(s) for s in args.steps.split(",")]
    for name in args.fixtures.split(","):
        meta, inputs = load_inputs(args.inputs, name)
        row: dict[str, Any] = {**meta, "steps": {}}
        digests: dict[str, Any] = {}
        for step in steps:
            torch.cuda.reset_peak_memory_stats()
            (_, ours), ours_s = _timed(
                lambda: hs.run_strict(strict, inputs[step], keep_tensors=True)
            )
            (_, theirs), theirs_s = _timed(
                lambda: hs.run_provider(model, inputs[step], keep_tensors=True)
            )
            layout = inputs[step].layout
            rows = hs.compare(ours, theirs, layout)
            isolated = _per_node(
                [
                    hs.distance(n, v, theirs.tensors[n], layout)
                    for n, v in strict.isolated_blocks(inputs[step], theirs)
                ]
            )
            spread = _spread(model, inputs[step], theirs)
            digests[str(step)] = {
                "inputs": inputs[step].digest(),
                "strict": ours.digests,
                "provider": theirs.digests,
            }
            row["steps"][str(step)] = {
                "head": ours.head,
                "nodes": len(ours.order),
                "strict_vs_provider": _summary_rows(rows),
                "chained": _per_node(rows),
                "isolated": isolated,
                "provider_spread": spread,
                "against_spread": _against_spread(_per_node(rows), isolated, spread),
                "traced_seconds": {"strict": round(ours_s, 2), "provider": round(theirs_s, 2)},
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            if step == args.control_step:
                (_, _), bare = _timed(lambda: (None, _bare(strict, inputs[step])))
                (_, _), bare_p = _timed(lambda: (None, _bare_provider(model, inputs[step])))
                row["forward_seconds"] = {"strict": round(bare, 2), "provider": round(bare_p, 2)}
                row["controls"] = _controls(strict, inputs[step], ours, args.plain_autograd)
                ok &= bool(row["controls"]["ok"])
            del ours, theirs
            gc.collect()
            torch.cuda.empty_cache()
            print(name, step, json.dumps(row["steps"][str(step)]["traced_seconds"]), flush=True)
        if args.against is not None:
            other = json.loads((args.against / f"{name}.digests.json").read_text())
            row["equal_to_other_process"] = {
                step: other[step]["inputs"] == d["inputs"] and other[step]["strict"] == d["strict"]
                for step, d in digests.items()
            }
            ok &= all(row["equal_to_other_process"].values())
        (args.out / f"{name}.digests.json").write_text(json.dumps(digests, indent=1) + "\n")
        summary["fixtures"][name] = row
        summary["ok"] = ok
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("ok", ok)
    return 0 if ok else 1


def _bare(strict: hs.StrictImageTransformer, inputs: hs.StepInputs) -> None:
    with torch.no_grad():
        strict(inputs)


def _bare_provider(model: Any, inputs: hs.StepInputs) -> None:
    device = next(model.parameters()).device
    with torch.no_grad():
        fc.DiffusersTransformerDenoiser(model)(
            inputs.image_rows[0].to(device),
            inputs.prompt_embeds.to(device),
            inputs.layout,
            inputs.unique_timesteps,
            inputs.timestep_indices,
            inputs.step,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--text-encoder-root", required=True, type=Path)
    rep = commands.add_parser("replay")
    rep.add_argument("--inputs", required=True, type=Path)
    rep.add_argument("--steps", default="0,1,2,3,4,5")
    rep.add_argument("--control-step", type=int, default=0)
    rep.add_argument("--against", type=Path)
    rep.add_argument("--plain-autograd", action="store_true")
    for command in (cap, rep):
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return capture(args) if args.command == "capture" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
