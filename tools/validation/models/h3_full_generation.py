# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Pinned three-seed FL2VA generation evidence for the full H3 candidate stack.

A candidate seed takes about 20 hours and its partition preempts jobs, so a seed can be
run on its own (``--seeds``) and continued after a preemption (``--resume``): saved steps
are replayed from disk and verified (see ``h3_generation_resume``), never recomputed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rl_engine.validation.models.h3_block import CandidateOps
from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution
from rl_engine.validation.models.h3_generation_resume import ResumableRun, StepStore
from tools.validation.models.h3_full_model import digest, verify_checkpoint


class QueryChunkAttention:
    """Bound attention workspace; every query still attends to every key."""

    def __init__(self, op, chunk):
        self.op, self.chunk = op, chunk

    def __call__(self, q, k, v, **kwargs):
        if torch.is_grad_enabled():
            raise ValueError("generation query chunking is inference only")
        return torch.cat(
            [self.op(part.contiguous(), k, v, **kwargs) for part in q.split(self.chunk, dim=2)],
            dim=2,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference-image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument(
        "--seeds", help="comma list of contract seeds to run (default: all; preflight: first)"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue a preempted run in --output: skip finished seeds, replay saved steps",
    )
    p.add_argument("--preflight-steps", type=int, help="test only: preflight num_inference_steps")
    p.add_argument(
        "--stop-after-step", type=int, help="test only: raise after N calls (preemption)"
    )
    args = p.parse_args()
    if args.preflight_steps is not None and not args.preflight:
        p.error("--preflight-steps applies to --preflight only")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    stale = args.output / failure_name(args.seeds)
    if args.resume and stale.exists():
        # A failure of an earlier attempt is kept, not overwritten or hidden.
        stale.rename(args.output / f"{stale.stem}-{int(time.time())}.json")
    source = Path(__file__).resolve().parents[3]
    config_root = source / "examples/models/minimax_h3"
    contract = json.loads((config_root / "generation-contract.json").read_text())
    identity_inputs = json.loads((config_root / "generation-inputs.json").read_text())
    prompt = config_root / "generation-prompt.txt"
    if (
        digest(prompt) != identity_inputs["prompt_sha256"]
        or digest(args.reference_image) != identity_inputs["image_sha256"]
    ):
        raise ValueError("fixed prompt/reference input identity changed")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("generation requires clean source")
    verify_checkpoint(
        args.checkpoint,
        source / "rl_engine/validation/models/h3_full_manifest.json",
        all_files=True,
    )
    import diffusers

    provider_root = Path(diffusers.__file__).resolve().parents[2]
    if (
        subprocess.check_output(
            ["git", "-C", str(provider_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != contract["provider_revision"]
    ):
        raise ValueError("generation provider revision differs")
    if subprocess.check_output(
        ["git", "-C", str(provider_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("generation provider is dirty")
    import soundfile as sf
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.utils.export_utils import encode_video
    from PIL import Image
    from transformers import Qwen3VLProcessor

    # Load before the large models: ModularPipeline otherwise catches processor
    # dependency failures and leaves None, failing much later in conditioning.
    processor = Qwen3VLProcessor.from_pretrained(
        str(args.checkpoint / "processor"), local_files_only=True
    )
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="20GB")
    pipe = ModularPipeline.from_pretrained(
        str(args.checkpoint), workflow="fl2va", components_manager=manager
    )
    pipe.update_components(processor=processor)
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=str(args.checkpoint),
        local_files_only=True,
    )
    for name in (
        "processor",
        "tokenizer",
        "text_encoder",
        "transformer",
        "vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
    ):
        if getattr(pipe, name, None) is None:
            raise RuntimeError(f"Required FL2VA component failed to load: {name}")
    ops = None
    if args.candidate:
        ops = CandidateOps()
        ops.attention = QueryChunkAttention(ops.attention, contract["attention_query_chunk"])
    settings = contract["preflight"] if args.preflight else contract
    if args.preflight_steps is not None:
        settings = dict(settings, num_inference_steps=args.preflight_steps)
    seeds = contract["seeds"][:1] if args.preflight else contract["seeds"]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        if not set(seeds) <= set(contract["seeds"]):
            raise ValueError(f"seeds {seeds} are not all in the contract {contract['seeds']}")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput

    def make_output(values, kwargs):
        if kwargs.get("return_dict", True):
            return MiniMaxH3TransformerOutput(sample=values[0], audio_sample=values[1])
        return values

    for seed in seeds:
        directory = args.output / str(seed)
        if args.resume and (directory / "report.json").exists():
            print(f"seed {seed} already complete; skipping", flush=True)
            continue
        identity = dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            contract_sha256=digest(config_root / "generation-contract.json"),
            image_sha256=identity_inputs["image_sha256"],
            prompt_sha256=identity_inputs["prompt_sha256"],
            candidate=args.candidate,
            preflight=args.preflight,
            seed=seed,
            settings={k: settings[k] for k in ("num_frames", "height", "width")}
            | {"num_inference_steps": settings["num_inference_steps"]},
        )
        run = ResumableRun(StepStore(directory), identity, resume=args.resume)
        trace = FullModelTrace()
        initial = {}

        def capture(module, positional, kwargs, output, run=run, directory=directory):
            # Preserve the complete first packed denoiser input to prove identical
            # noise and conditioning, then retain every step's predicted velocities.
            values = (output.sample, output.audio_sample) if hasattr(output, "sample") else output
            if run.saved == 0 and run.recorded == 0:
                initial.update(
                    {k: v.detach().cpu() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
                )
                tmp = directory / "initial.pt.tmp"
                torch.save(initial, tmp)
                os.replace(tmp, directory / "initial.pt")
            run.record(positional, kwargs, tuple(values))
            if args.stop_after_step is not None and run.recorded >= args.stop_after_step:
                raise KeyboardInterrupt(f"simulated preemption after {run.recorded} calls")

        uninstall = run.install(pipe.transformer, make_output)
        handle = pipe.transformer.register_forward_hook(capture, with_kwargs=True)
        started = time.perf_counter()
        try:
            with (
                torch.inference_mode(),
                full_model_execution(
                    pipe.transformer, candidate=args.candidate, trace=trace, ops=ops
                ),
            ):
                result = pipe(
                    prompt=prompt.read_text(),
                    image=Image.open(args.reference_image).convert("RGB"),
                    num_frames=settings["num_frames"],
                    height=settings["height"],
                    width=settings["width"],
                    num_inference_steps=settings["num_inference_steps"],
                    generator=torch.Generator().manual_seed(seed),
                    output_type="np",
                    output=["videos", "audio", "sampling_rate"],
                )
        finally:
            handle.remove()
            uninstall()
        video = np.asarray(result["videos"][0])
        audio = result["audio"][0].detach().cpu().numpy()
        if video.shape != (
            settings["num_frames"],
            settings["height"],
            settings["width"],
            3,
        ):
            raise ValueError(f"wrong video dimensions {video.shape}")
        if not np.isfinite(video).all() or not np.isfinite(audio).all():
            raise ValueError("non-finite generated media")
        if (
            result["sampling_rate"] != contract["sampling_rate"]
            or abs(
                audio.shape[-1] / result["sampling_rate"] - settings["num_frames"] / contract["fps"]
            )
            > 0.05
        ):
            raise ValueError("incorrect audio sample rate/duration")
        np.save(directory / "video.npy", video)
        np.save(directory / "audio.npy", audio)
        sf.write(directory / "audio.wav", audio.T, result["sampling_rate"], subtype="FLOAT")
        encode_video(
            video,
            fps=contract["fps"],
            output_path=str(directory / "video.mp4"),
            audio=result["audio"][0],
            audio_sample_rate=result["sampling_rate"],
        )
        for index in (0, len(video) // 2, len(video) - 1):
            Image.fromarray(np.round(video[index] * 255).clip(0, 255).astype("uint8")).save(
                directory / f"frame-{index:03}.png"
            )
        report = dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            model_revision=contract["model_revision"],
            provider_revision=contract["provider_revision"],
            status="generation_complete_comparison_pending",
            candidate=args.candidate,
            seed=seed,
            preflight=args.preflight,
            trace=trace.validate(),
            steps=run.recorded,
            resume=run.summary(),
            seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            contract_sha256=digest(config_root / "generation-contract.json"),
        )
        report["artifacts"] = {
            path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file()
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def failure_name(seeds):
    """Failure marker name: per seed selection, so parallel seed tasks do not collide."""

    return "failure.json" if not seeds else f"failure-seeds-{seeds.replace(',', '-')}.json"


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import sys

        if "--output" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output") + 1])
            seeds = sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else None
            marker = directory / failure_name(seeds)
            if directory.exists() and not marker.exists():
                marker.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "when": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                    )
                )
        raise
