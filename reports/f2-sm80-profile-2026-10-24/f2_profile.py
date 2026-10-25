"""Separate host-inclusive timings from CUDA kernel activity durations."""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from rl_engine.backends.extension import _C
from rl_engine.ops.rope.qknorm_scaled_rope import (
    _dispatch_backward,
    _validate_inputs,
    qknorm_scaled_rope_forward,
)
from rl_engine.reference.rope.qknorm_scaled_rope import build_rope_table
from probe import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    c, s = build_rope_table(131072, device="cuda")
    for tokens in (1, 128, 2048):
        gen = torch.Generator(device="cuda").manual_seed(501)
        q = torch.randn(tokens, 32, 128, device="cuda", generator=gen).bfloat16()
        k = torch.randn(tokens, 2, 128, device="cuda", generator=gen).bfloat16()
        p = torch.arange(tokens, device="cuda", dtype=torch.int64)
        gq, gk = torch.ones_like(q), torch.ones_like(k)
        validation = measure(lambda: _validate_inputs(q, k, p, c, s, True))
        native_forward = measure(lambda: _C.qknorm_scaled_rope_forward(q, k, p, c, s, True))
        for local in (False, True):
            for backend in ("cuda", "triton"):
                forward = lambda: qknorm_scaled_rope_forward(
                    q, k, p, c, s, is_local=local, backend=backend
                )
                saved = forward()
                backward = lambda: _dispatch_backward(
                    q, k, gq, gk, *saved[2:], p, c, s, is_local=local, backend=backend
                )
                for phase, fn in [("forward", forward), ("backward", backward)]:
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
                    wall = measure(fn)
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(20):
                            fn()
                        torch.cuda.synchronize()
                    trace = args.out / f"T{tokens}-{local}-{backend}-{phase}.json"
                    prof.export_chrome_trace(str(trace))
                    events = json.loads(trace.read_text())["traceEvents"]
                    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
                    target = (
                        ("f2_forward" if phase == "forward" else "f2_backward")
                        if backend == "cuda"
                        else ("_kernel" if phase == "forward" else "_backward_kernel")
                    )
                    fused = [
                        e
                        for e in kernels
                        if (target in e["name"] if backend == "cuda" else e["name"] == target)
                    ]
                    assert len(fused) == 20, (target, len(fused), [e["name"] for e in kernels])
                    totals = {}
                    for e in kernels:
                        totals[e["name"]] = totals.get(e["name"], 0) + e["dur"] / 20
                    rows.append(
                        {
                            "tokens": tokens,
                            "local": local,
                            "backend": backend,
                            "phase": phase,
                            "wall_median_us": wall["median_us"],
                            "fused_kernel_mean_us_profiled": sum(e["dur"] for e in fused) / 20,
                            "all_gpu_kernels_mean_us_profiled": sum(e["dur"] for e in kernels) / 20,
                            "gpu_kernel_breakdown_us": totals,
                            "python_local_validation_wall_us": validation["median_us"],
                            "native_local_forward_wall_us": native_forward["median_us"],
                        }
                    )
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
