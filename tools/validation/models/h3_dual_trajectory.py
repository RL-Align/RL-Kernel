# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #420 ``ws1_dual_trajectory``: two layouts of the same sampling run must agree bitwise.

Both trajectories run the same multi-step flow-matching Euler sampler (``x += (s' - s) v``,
FP32 latents, video + audio) on the full pinned 33B H3 and record it with the sealed
trajectory artifact. They differ only in how the request is laid out for the model:

* ``single``  - the request alone in the batch;
* ``batched`` - the request as row 1 of a batch of 2 beside an unrelated request.

Under RL-Kernel strict operators the two are bitwise identical at every step; under the
provider's own kernels the sampler drifts (and the artifact comparison names the first step).
That is the train/rollout-layout control the end-to-end experiment needs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rl_engine.models.minimax_h3 import trajectory as tj
from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution
from tools.validation.models.h3_full_model import packed_inputs

SIGMAS = (1.0, 0.75, 0.5, 0.25, 0.0)


from contextlib import contextmanager


@contextmanager
def diagnostic_video_projections(model, enabled):
    """DIAGNOSTIC ONLY: run ``proj_in``/``proj_out`` through the fixed-order Triton GEMM.

    The video projection rows belong to another contributor; this binding exists only to
    attribute the remaining single-vs-batched drift and is labelled in the artifact metadata.
    """

    if not enabled:
        yield
        return
    from rl_engine.backends.shared.triton.gemm import h3_projection as g

    saved = []
    for name in ("proj_in", "proj_out"):
        module = getattr(model, name)

        def forward(x, module=module):
            flat = x.reshape(-1, x.shape[-1])
            out = g.linear_forward(flat.contiguous(), module.weight, module.bias)
            return out.view(*x.shape[:-1], out.shape[-1])

        saved.append(module)
        module.forward = forward
    try:
        yield
    finally:
        for module in saved:
            module.__dict__.pop("forward", None)


def sample(model, base, noise, *, candidate, strict_boundaries, layout, label, keep_tensors, seed,
           diag_video=False):
    video, audio = noise["video"].clone(), noise["audio"].clone()
    other = {k: torch.randn(v.shape, generator=torch.Generator().manual_seed(77 + i)).to(v.device)
             for i, (k, v) in enumerate(noise.items())}
    recorder = tj.TrajectoryRecorder(
        variant="FL2VA", label=label, seed=seed,
        sigma_grids={"video": SIGMAS, "audio": SIGMAS},
        initial_noise={"video": video, "audio": audio},
        metadata={"layout": layout, "candidate": candidate, "strict_boundaries": strict_boundaries,
                  "sampler": "euler-ode", "diagnostic_video_projections": diag_video},
        keep_tensors=keep_tensors,
    )
    trace = FullModelTrace()
    with torch.no_grad(), diagnostic_video_projections(model, diag_video), full_model_execution(
        model, candidate=candidate, trace=trace, strict_boundaries=strict_boundaries
    ):
        for step in range(len(SIGMAS) - 1):
            sigma, sigma_next = SIGMAS[step], SIGMAS[step + 1]
            enc = base["encoder_hidden_states"].detach()
            if layout == "single":
                hs, ahs, es = video.bfloat16(), audio.bfloat16(), enc
            else:  # the request is row 1, an unrelated one is row 0
                hs = torch.cat([other["video"].bfloat16(), video.bfloat16()])
                ahs = torch.cat([other["audio"].bfloat16(), audio.bfloat16()])
                es = torch.cat([enc * 0.5, enc])
            inputs = {**base, "hidden_states": hs, "audio_hidden_states": ahs,
                      "encoder_hidden_states": es,
                      "timestep": torch.tensor([0.0, sigma], device=hs.device)}
            v_video, v_audio = model(**inputs)
            row = 0 if layout == "single" else 1
            v_video, v_audio = v_video[row : row + 1].float(), v_audio[row : row + 1].float()
            video = video + (sigma_next - sigma) * v_video
            audio = audio + (sigma_next - sigma) * v_audio
            recorder.record_step(step, velocity={"video": v_video, "audio": v_audio},
                                 latent={"video": video, "audio": audio})
    return recorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=420539)
    parser.add_argument("--diag-video-projections", action="store_true",
                        help="diagnostic: bind proj_in/proj_out to the fixed-order GEMM (strict arm only)")
    parser.add_argument("--strict-only", action="store_true")
    args = parser.parse_args()
    from diffusers import MiniMaxH3Transformer3DModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.checkpoint / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    base = packed_inputs("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    noise = {
        "video": torch.randn(base["hidden_states"].shape, generator=gen).cuda(),
        "audio": torch.randn(base["audio_hidden_states"].shape, generator=gen).cuda(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"load_seconds": time.perf_counter() - started, "pairs": {}}
    arms = (("provider", False), ("rlkernel_strict", True))
    if args.strict_only:
        arms = arms[1:]
    for name, candidate in arms:
        recs = {}
        for layout in ("single", "batched"):
            t0 = time.perf_counter()
            rec = sample(model, base, noise, candidate=candidate, strict_boundaries=candidate,
                         layout=layout, label=f"{name}-{layout}", keep_tensors=True, seed=args.seed,
                         diag_video=args.diag_video_projections and candidate)
            recs[layout] = (rec, time.perf_counter() - t0)
        paths = {}
        for layout, (rec, _) in recs.items():
            art = rec.seal()
            (args.output / f"{name}-{layout}.json").write_text(tj.dumps(art))
            rec.save_tensors(args.output / f"{name}-{layout}.safetensors")
            paths[layout] = art
        report = tj.drift_report(
            paths["single"], paths["batched"],
            tensors_lhs=args.output / f"{name}-single.safetensors",
            tensors_rhs=args.output / f"{name}-batched.safetensors",
        )
        report["seconds"] = {k: v[1] for k, v in recs.items()}
        result["pairs"][name] = report
        print(name, {k: report[k] for k in ("steps", "compared", "differing", "bitwise_equal", "first_drift")}, flush=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
