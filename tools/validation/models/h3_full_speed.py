# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Time one full-model forward + backward step, provider against the strict RL-Kernel graph.

    python tools/validation/models/h3_full_speed.py --checkpoint <dir with transformer/> \
        --sizes 12,925,4253,8349 --out speed.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

SIZES = {
    "12": None,
    "925": dict(text=77, frames=3, side=32, audio=40),
    "4253": dict(text=77, frames=1, side=128, audio=40),
    "8349": dict(text=77, frames=2, side=128, audio=40),
}


def sized_inputs(device, *, text, frames, side, audio):
    from rl_engine.validation.models.h3_cases import h3_fl2va_layout

    layout = h3_fl2va_layout(
        num_text_tokens=text,
        num_latent_frames=frames,
        latent_height=side,
        latent_width=side,
        num_audio_latents=audio,
        device=device,
    )
    tags = layout["token_tags"]
    video, text_idx, audio_idx = ((tags == t).nonzero().flatten() for t in (0, 1, 2))
    gen = torch.Generator().manual_seed(420539)

    def rnd(*shape):
        return torch.randn(shape, generator=gen).to(device).requires_grad_()

    return dict(
        **layout,
        hidden_states=rnd(1, len(video), 96),
        audio_hidden_states=rnd(1, len(audio_idx), 32),
        encoder_hidden_states=rnd(1, len(text_idx), 5120),
        timestep=torch.tensor([0.0, 500.0], device=device),
        video_indices=video,
        text_indices=text_idx,
        audio_indices=audio_idx,
        return_dict=False,
    )


def timed(model, inputs, candidate, repeats, attention="auto"):
    from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution

    grad_inputs = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    backends = {}

    def once():
        fresh = {
            k: (v.detach().clone().requires_grad_() if k in grad_inputs else v)
            for k, v in inputs.items()
        }
        trace = FullModelTrace()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ops = None
        if candidate:
            from rl_engine.validation.models.h3_block import CandidateOps

            ops = CandidateOps(attention=attention)
        with full_model_execution(
            model, candidate=candidate, trace=trace, ops=ops, strict_boundaries=candidate
        ):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                video, audio = model(**fresh)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                (video.float().square().mean() + audio.float().square().mean()).backward()
                torch.cuda.synchronize()
        backends.update(getattr(trace, "actual_backends", None) or {})
        return t1 - t0, time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    once()
    runs = [once() for _ in range(repeats)]
    return dict(
        forward_s=statistics.median(r[0] for r in runs),
        total_s=statistics.median(r[1] for r in runs),
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        attention_backend=backends.get("attention_h3") or backends.get("attention"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sizes", default="12,925,4253,8349")
    p.add_argument("--attention", default="auto", choices=("auto", "h3_full", "interim"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from diffusers import MiniMaxH3Transformer3DModel

    from tools.validation.models.h3_full_model import packed_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.checkpoint / "transformer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval().enable_gradient_checkpointing()
    for q in model.parameters():
        q.register_post_accumulate_grad_hook(lambda t: setattr(t, "grad", None))
    result = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, sizes={})
    for size in args.sizes.split(","):
        inputs = (
            packed_inputs("cuda") if SIZES[size] is None else sized_inputs("cuda", **SIZES[size])
        )
        entry = dict(tokens=int(inputs["position_ids"].shape[0]))
        for name, candidate in (("provider", False), ("strict", True)):
            try:
                entry[name] = timed(model, inputs, candidate, args.repeats, args.attention)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                entry[name] = dict(error="CUDA out of memory")
        if "total_s" in entry["provider"] and "total_s" in entry["strict"]:
            entry["ratio"] = entry["strict"]["total_s"] / entry["provider"]["total_s"]
            entry["fwd_ratio"] = entry["strict"]["forward_s"] / entry["provider"]["forward_s"]
        result["sizes"][size] = entry
        print(size, json.dumps(entry), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
