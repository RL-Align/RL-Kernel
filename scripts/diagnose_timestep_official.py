#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate trig, dot and timestep reduction error; diagnostic, not a gate."""

import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_timestep_official import inputs, NAMES  # noqa: E402
from rl_engine.kernels.ops.timestep_embed_mlp import TimestepEmbedMLPOp  # noqa: E402


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    seed = 9386
    v = inputs(16, 3072, torch.float32, "cuda", seed)
    dy = torch.randn(
        (16, 3072), generator=torch.Generator(device="cuda").manual_seed(seed + 1), device="cuda"
    )
    results = {}
    for device in ("cpu", "cuda"):
        for dtype in (torch.float32, torch.float64):
            values = [v[k].to(device).to(dtype).detach().requires_grad_() for k in NAMES]
            t, w1, b1, w2, b2 = values
            # Hold the actual FP32 phase constant, even in the double diagnostic.
            f = torch.exp(-math.log(10000) * torch.arange(128).float() / 128).to(device)
            phase = (t.float()[:, None] * f) * 1000.0
            e = torch.cat((phase.to(dtype).cos(), phase.to(dtype).sin()), 1)
            z = torch.stack([(w1 * row).sum(1) + b1 for row in e])
            h = z * z.sigmoid()
            y = torch.stack([(w2 * row).sum(1) + b2 for row in h])
            dt, de = torch.autograd.grad(y, (t, e), dy.to(device).to(dtype), retain_graph=True)
            results[f"{device}-{dtype}"] = {
                "dt": dt.tolist(),
                "embedding": e.detach().cpu(),
                "de": de.detach().cpu(),
            }
    for backend in ("cuda", "triton"):
        if backend == "cuda" and "--triton-only" in sys.argv:
            continue
        vals = {k: x.detach().requires_grad_() for k, x in v.items()}
        y = TimestepEmbedMLPOp(backend)(**vals)
        results[backend] = {"dt": torch.autograd.grad(y, vals["timestep"], dy)[0].tolist()}
    ref = results["cpu-torch.float32"]
    for r in results.values():
        if "embedding" in r:
            r["embedding_error_vs_cpu"] = (
                (r["embedding"].double() - ref["embedding"].double()).abs().max().item()
            )
            r["de_error_vs_cpu"] = (r["de"].double() - ref["de"].double()).abs().max().item()
    for r in results.values():
        r.pop("embedding", None)
        r.pop("de", None)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
