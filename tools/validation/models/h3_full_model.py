# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Full pinned H3 forward/backward evidence; reference calibration precedes candidates.

The small packed input checks the entire training graph, not generation quality.
FP32 reference and BF16 provider runs never instantiate candidate operators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from rl_engine.validation.models.h3_cases import h3_fl2va_layout
from rl_engine.validation.models.h3_full_model import (
    MODEL_REVISION,
    PROVIDER_REVISION,
    FullModelTrace,
    full_model_execution,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(root, manifest, *, all_files=False):
    spec = json.loads(manifest.read_text())
    if spec["revision"] != MODEL_REVISION:
        raise ValueError("wrong H3 revision")
    needed = [r for r in spec["files"] if (all_files or r["path"].startswith("transformer/"))]
    if not needed:
        raise ValueError("no pinned transformer shards")
    for record in needed:
        path = root / record["path"]
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {path}")
    return spec


def packed_inputs(device):
    layout = h3_fl2va_layout(
        num_text_tokens=4,
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        device=device,
    )
    generator = torch.Generator().manual_seed(420539)
    values = dict(
        hidden_states=(1, 4, 96), audio_hidden_states=(1, 4, 32), encoder_hidden_states=(1, 4, 5120)
    )
    result = {
        name: torch.randn(shape, generator=generator).to(device).requires_grad_()
        for name, shape in values.items()
    }
    tags = layout["token_tags"]
    return dict(
        **result,
        **layout,
        timestep=torch.tensor([0.0, 500.0], device=device),
        video_indices=(tags == 0).nonzero().flatten(),
        text_indices=(tags == 1).nonzero().flatten(),
        audio_indices=(tags == 2).nonzero().flatten(),
        return_dict=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["fp32-reference", "bf16-provider", "candidate"], required=True
    )
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    if args.mode == "candidate" and not args.thresholds:
        parser.error("candidate execution requires previously frozen calibration thresholds")
    if args.thresholds:
        thresholds = json.loads(args.thresholds.read_text())
        if thresholds.get("status") != "frozen_reference_bounds" or not thresholds.get(
            "reference_report_sha256"
        ):
            raise ValueError("candidate requires immutable independent FP32 calibration")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        mode=args.mode,
        job_id=os.environ.get("SLURM_JOB_ID"),
        model_revision=MODEL_REVISION,
        command=list(os.sys.argv),
    )
    report_path = args.output / "report.json"
    started = time.perf_counter()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("evidence requires a clean source commit")
        report["commit"] = commit
        verify_checkpoint(args.checkpoint, args.manifest)
        import diffusers
        from diffusers import MiniMaxH3Transformer3DModel

        provider_root = Path(diffusers.__file__).resolve().parents[2]
        provider_commit = subprocess.check_output(
            ["git", "-C", str(provider_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if provider_commit != PROVIDER_REVISION:
            raise ValueError("diffusers provider revision changed")
        if subprocess.check_output(
            ["git", "-C", str(provider_root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("diffusers provider source is dirty")
        report.update(
            provider_revision=provider_commit,
            torch=torch.__version__,
            diffusers=diffusers.__version__,
            device=torch.cuda.get_device_name(),
            checkpoint_manifest_sha256=digest(args.manifest),
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(420539)
        dtype = torch.float32 if args.mode == "fp32-reference" else torch.bfloat16
        model, loading = MiniMaxH3Transformer3DModel.from_pretrained(
            str(args.checkpoint / "transformer"),
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        report["checkpoint_loading"] = loading
        if any(
            loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("full H3 checkpoint did not load every tensor exactly")
        model = model.to("cuda").eval()
        model.enable_gradient_checkpointing()
        inputs = packed_inputs("cuda")
        trace = FullModelTrace()
        trace.capture = True
        gradients = {}
        handles = []
        # Persist and release each completed parameter gradient. This preserves
        # the complete backward chain while bounding resident FP32 gradient memory.
        directory = args.output / "gradients"
        directory.mkdir()
        for i, (name, parameter) in enumerate(model.named_parameters()):

            def save_grad(p, name=name, i=i):
                trace.record_event("parameter." + name)
                grad = p.grad.detach().cpu()
                if not torch.isfinite(grad).all():
                    raise ValueError(f"non-finite parameter gradient: {name}")
                path = directory / f"{i:04}.pt"
                torch.save(grad, path)
                gradients[name] = dict(
                    path=str(path.relative_to(args.output)),
                    shape=list(grad.shape),
                    dtype=str(grad.dtype),
                    sha256=digest(path),
                )
                p.grad = None

            handles.append(parameter.register_post_accumulate_grad_hook(save_grad))
        for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):

            def record_input(grad, name=name):
                trace.record_event("input_gradients." + name)
                return grad

            handles.append(inputs[name].register_hook(record_input))
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with full_model_execution(model, candidate=args.mode == "candidate", trace=trace):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                video, audio = model(**inputs)
                if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                    raise ValueError("non-finite full-model prediction")
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - begin
                begin = time.perf_counter()
                loss = video.float().square().mean() + audio.float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                backward_seconds = time.perf_counter() - begin
        for h in handles:
            h.remove()
        if set(gradients) != set(dict(model.named_parameters())):
            raise ValueError("missing full-model parameter gradients")
        input_grads = {}
        for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
            grad = inputs[name].grad
            if grad is None or not torch.isfinite(grad).all():
                raise ValueError(f"missing/non-finite input gradient: {name}")
            input_grads[name] = grad.cpu()
        torch.save(
            dict(
                video=video.detach().cpu(),
                audio=audio.detach().cpu(),
                input_gradients=input_grads,
                nodes=trace.snapshots,
            ),
            args.output / "tensors.pt",
        )
        (args.output / "gradients.json").write_text(json.dumps(gradients, indent=2) + "\n")
        report.update(
            trace=trace.validate(backward=True),
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            loss=loss.item(),
            parameter_gradients=len(gradients),
            tensors_sha256=digest(args.output / "tensors.pt"),
        )
        # Numerical acceptance is a separate comparison against immutable reference
        # artifacts; merely finishing all layers is never a numerical PASS.
        report["status"] = "execution_complete_acceptance_pending"
        if args.thresholds:
            report["thresholds_sha256"] = digest(args.thresholds)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
