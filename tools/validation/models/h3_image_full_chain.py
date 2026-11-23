#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Released-weight evidence for RFC #738 ``phase_b_image_full_chain``.

    PYTHONPATH=<checkout> python tools/validation/models/h3_image_full_chain.py \
        --root <MiniMax-H3-Image> --text-encoder-root <MiniMax-H3 checkpoint> --out <DIR> \
        [--fixtures t2i_example,edit_example] [--repeat]

One GPU with room for the 64-layer Qwen3-VL (phase 1) and then the transformer + VAE (phase 2).

* Phase 1, text: the released Qwen3-VL (``text_encoder/`` of MiniMax-H3, identical to
  MiniMax-H3-Image's) is loaded once; for each fixture the contract (H3 layer-50 encoder,
  ``image_text_presentation_contract``) and the pinned Diffusers text step both run on it.
  The model is freed before phase 2.
* Phase 2, chain: the released VAE (FP32) and transformer (``transformer_diffusers/``, BF16
  with its FP32 modules), and per fixture :func:`full_chain.run_chain` (RL-Kernel contracts)
  and :func:`full_chain.run_diffusers` (pinned modular pipeline, block by block). ``--repeat``
  runs the contract chain twice for run-to-run determinism.

Fixtures are the release's own examples (``examples/``): the T2I request (4736x3552, seed 42),
the same prompt at 1024x768 (canvas upscale and resize back), the edit request
(``reference1.png``, 4096x4096, seed 1234), and a two-reference edit (``reference1.png`` and
``assets/showcase/photo.jpg``, 2048x2048) for heterogeneous reference geometry.

Writes ``<fixture>.<arm>.json`` artifacts, ``<fixture>.<arm>.png`` outputs and ``summary.json``
(no paths). Exit 1 if any fixture fails the latent gate.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.minimax_h3_image import full_chain as fc

EDIT2_PROMPT = (
    "Place the two hosts from <Picture 1> on the waterfront promenade shown in <Picture 2>, at "
    "golden hour, keeping their faces, clothes and the microphones unchanged."
)


def fixtures(root: Path) -> dict[str, dict[str, Any]]:
    from PIL import Image

    ex = root / "examples"
    t2i = json.loads((ex / "request.t2i.json").read_text())
    width, height = (int(v) for v in t2i["size"].split("x"))
    edit_w, edit_h = (int(v) for v in "4096x4096".split("x"))
    ref1 = Image.open(ex / "reference1.png")
    photo = Image.open(root / "assets/showcase/photo.jpg")
    return {
        "t2i_example": {
            "request": fc.ImageRequest(t2i["prompt"], height, width, seed=int(t2i["seed"])),
            "source": "examples/request.t2i.json",
        },
        "t2i_upscaled": {
            "request": fc.ImageRequest(t2i["prompt"], 768, 1024, seed=int(t2i["seed"])),
            "source": "examples/request.t2i.json at 1024x768",
        },
        "edit_example": {
            "request": fc.ImageRequest(
                (ex / "prompt.edit.txt").read_text(), edit_h, edit_w, seed=1234, references=(ref1,)
            ),
            "source": "examples/edit.py defaults: reference1.png, prompt.edit.txt, "
            "4096x4096, seed 1234",
        },
        "edit_two_refs": {
            "request": fc.ImageRequest(EDIT2_PROMPT, 2048, 2048, seed=7, references=(ref1, photo)),
            "source": "reference1.png + assets/showcase/photo.jpg, 2048x2048, seed 7",
        },
    }


def _sync() -> None:
    torch.cuda.synchronize()


def _timed(fn: Any) -> tuple[Any, float]:
    _sync()
    start = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - start


def _save_png(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image

    Image.fromarray(tensor[0].permute(1, 2, 0).contiguous().numpy()).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--text-encoder-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixtures", default="all")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        # from_pretrained keeps the checkpoint's FP32 modules (proj_in, time_embedder,
        # pdd_proj_out, rope) only on the low_cpu_mem_usage path, which needs accelerate.
        # Check before phase 1 rather than after the text encoder has been loaded.
        import accelerate
    except ImportError:
        parser.error("accelerate is required to load transformer_diffusers (pip install --no-deps)")
    import diffusers
    import transformers
    from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from transformers import AutoProcessor, AutoTokenizer

    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp
    from rl_engine.models.minimax_h3 import qwen3vl_text as q
    from rl_engine.models.minimax_h3_image import image_vae as iv
    from rl_engine.models.minimax_h3_image import text_presentation as tp
    from rl_engine.models.minimax_h3_image import velocity_step as vs

    device = torch.device("cuda")
    all_fixtures = fixtures(args.root)
    names = list(all_fixtures) if args.fixtures == "all" else args.fixtures.split(",")
    summary: dict[str, Any] = {
        "schema": "rlkernel.h3_image.full_chain_evidence.v1",
        "manifest": fc.contract_manifest(),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "python": platform.python_version(),
        },
        "fixtures": {},
    }

    # Phase 1: text, both arms, on one load of the released 64-layer Qwen3-VL.
    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(args.root / "processor"))
    tp.verify_presentation_files(args.root)
    full, report = q.load_release_text_encoder(
        args.text_encoder_root / "text_encoder", device=device, num_layers=q.RELEASE_NUM_LAYERS
    )
    summary["text_encoder_load"] = report
    encoder = q.Qwen3VLLayer50Encoder(full)
    resize_pipe = fc.diffusers_pipeline()
    texts: dict[str, Any] = {}
    for name in names:
        request = all_fixtures[name]["request"]
        prepared = [fc.prepare_reference(r) for r in request.references]
        vision = fc.diffusers_resize(resize_pipe, request)[1] if request.references else None
        # functools.partial binds the models now; both are deleted before phase 2.
        ours, ours_s = _timed(
            functools.partial(fc.contract_text, request, prepared, encoder, tokenizer, processor)
        )
        theirs, theirs_s = _timed(
            functools.partial(fc.diffusers_text, request, vision, full, tokenizer, processor)
        )
        for t in (ours, theirs):
            t.prompt_embeds = t.prompt_embeds.cpu()
        texts[name] = (prepared, ours, theirs)
        summary["fixtures"][name] = {
            "source": all_fixtures[name]["source"],
            "task": request.task,
            "size": [request.height, request.width],
            "text_rows": len(ours.token_ids),
            "text_seconds": {"contract": round(ours_s, 2), "diffusers": round(theirs_s, 2)},
        }
        print(name, "text rows", len(ours.token_ids), flush=True)
    del encoder, full
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: the chain.
    vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, args.root / "vae_diffusers", device)
    contract = iv.ImageVAEContract(vae)
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        str(args.root / "transformer_diffusers"), torch_dtype=torch.bfloat16
    )
    transformer = transformer.to(device).eval()
    dtypes: dict[str, int] = {}
    for p in transformer.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    summary["transformer_parameter_dtypes"] = dtypes

    def scheduler() -> Any:
        return MiniMaxH3Scheduler.from_pretrained(str(args.root / "scheduler"))

    def sampler() -> Any:
        # image_velocity_step on the image_sigma_pdd_schedule grid, Triton kernel on GPU.
        return vs.ImageVelocitySampler(op=H3ImageVelocityStepTritonOp(), device=device)

    denoiser = fc.DiffusersTransformerDenoiser(transformer)
    ok = True
    for name in names:
        request = all_fixtures[name]["request"]
        prepared, text_ours, text_theirs = texts[name]
        torch.cuda.reset_peak_memory_stats(device)
        ours, ours_s = _timed(
            lambda: fc.run_chain(
                request,
                text_ours,
                vae_contract=contract,
                denoiser=denoiser,
                device=device,
                sampler=sampler(),
                prepared=prepared,
                label=f"{name}:rl-kernel",
            )
        )
        pipe = fc.diffusers_pipeline(vae=vae, transformer=transformer, scheduler=scheduler())
        theirs, theirs_s = _timed(
            lambda: fc.run_diffusers(request, text_theirs, pipe, label=f"{name}:diffusers")
        )
        report = fc.compare(ours.artifact, theirs.artifact)
        row = summary["fixtures"][name]
        traj = ours.artifact["trajectory"]
        row.update(
            {
                "canvas": traj["inputs"]["canvas"]["canvas"],
                "sequence_length": traj["inputs"]["layout"]["sequence_length"],
                "num_reference_rows": traj["inputs"]["layout"]["num_reference_rows"],
                "gate": report["gate"],
                "secondary_equal": report["secondary_equal"],
                "first_drift": report["first_drift"],
                "trajectory_first_drift": report["trajectory_first_drift"],
                "stages_equal": {s: v["equal"] for s, v in report["stages"].items()},
                "final_latent_sha256": traj["final_latent_sha256"],
                "output_sha256": traj["decoded"]["sha256"],
                "artifact_sha256": {
                    "rl-kernel": ours.artifact["artifact_sha256"],
                    "diffusers": theirs.artifact["artifact_sha256"],
                },
                "chain_seconds": {"contract": round(ours_s, 1), "diffusers": round(theirs_s, 1)},
                "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 1),
            }
        )
        if args.repeat:
            again = fc.run_chain(
                request,
                text_ours,
                vae_contract=contract,
                denoiser=denoiser,
                device=device,
                sampler=sampler(),
                prepared=prepared,
                label=f"{name}:rl-kernel",
            )
            row["run_to_run_equal"] = again.artifact == ours.artifact
        ok &= bool(report["gate"])
        for arm, run in (("rl-kernel", ours), ("diffusers", theirs)):
            fc.verify(run.artifact)
            (args.out / f"{name}.{arm}.json").write_text(json.dumps(run.artifact, indent=1) + "\n")
            _save_png(run.output, args.out / f"{name}.{arm}.png")
        print(
            name,
            json.dumps(
                {k: row[k] for k in ("gate", "secondary_equal", "first_drift", "chain_seconds")}
            ),
            flush=True,
        )
    summary["all_gates_pass"] = ok
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("all_gates_pass", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
