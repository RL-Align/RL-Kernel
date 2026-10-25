"""Run from repository root with PYTHONPATH=.; reuse --inputs for cross-device parity."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton

from rl_engine.ops.rope.qknorm_scaled_rope import (
    _dispatch_backward,
    qknorm_scaled_rope_forward,
)
from rl_engine.reference.rope.qknorm_scaled_rope import build_rope_table


def bits_equal(a, b):
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    # Synchronized wall time includes allocation, validation, launch and GPU work.
    # Do not call this isolated kernel latency; local validation synchronizes.
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 10)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.inputs:
        data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    else:
        gen = torch.Generator().manual_seed(501558)
        data = {
            "q": torch.randn(257, 32, 128, generator=gen).bfloat16(),
            "k": torch.randn(257, 2, 128, generator=gen).bfloat16(),
            "gq": torch.randn(257, 32, 128, generator=gen).bfloat16(),
            "gk": torch.randn(257, 2, 128, generator=gen).bfloat16(),
            "p": (torch.arange(257, dtype=torch.int64) * 509) % 131072,
        }
        scales = torch.exp2(torch.linspace(-60, 60, 257))[:, None, None]
        for name in ("q", "k"):
            data[name] = (data[name].float() * scales).bfloat16()
            data[name][0] = 0
        # Generate once on CUDA, serialize the exact BF16 table for future devices.
        data["c"], data["s"] = (x.cpu() for x in build_rope_table(131072, device="cuda"))
    torch.save(data, args.out / "inputs.pt")
    device = {name: x.cuda() for name, x in data.items()}
    q, k, gq, gk, p, c, s = (device[name] for name in ("q", "k", "gq", "gk", "p", "c", "s"))
    outputs = {}
    for local in (False, True):
        for backend in ("cuda", "triton"):
            fwd = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
            grads = _dispatch_backward(
                q, k, gq, gk, *fwd[2:], p, c, s, is_local=local, backend=backend
            )
            outputs[f"{local}/{backend}"] = [x.cpu() for x in (*fwd, *grads)]
        assert all(
            bits_equal(a, b) for a, b in zip(outputs[f"{local}/cuda"], outputs[f"{local}/triton"])
        )
    torch.save(outputs, args.out / "outputs.pt")
    comparison = None
    if args.compare:
        other = torch.load(args.compare, map_location="cpu", weights_only=True)
        comparison = {
            key: [bits_equal(a, b) for a, b in zip(values, other[key])]
            for key, values in outputs.items()
        }
        assert all(all(v) for v in comparison.values()), comparison
    timings = []
    for tokens in (1, 128, 2048):
        gen = torch.Generator(device="cuda").manual_seed(501)
        q = torch.randn(tokens, 32, 128, device="cuda", generator=gen).bfloat16()
        k = torch.randn(tokens, 2, 128, device="cuda", generator=gen).bfloat16()
        gq, gk = torch.ones_like(q), torch.ones_like(k)
        p = torch.arange(tokens, device="cuda", dtype=torch.int64)
        for local in (False, True):
            for backend in ("cuda", "triton"):
                forward = lambda: qknorm_scaled_rope_forward(
                    q, k, p, c, s, is_local=local, backend=backend
                )
                saved = forward()
                backward = lambda: _dispatch_backward(
                    q, k, gq, gk, *saved[2:], p, c, s, is_local=local, backend=backend
                )
                timings.append(
                    {
                        "tokens": tokens,
                        "local": local,
                        "backend": backend,
                        "forward": measure(forward),
                        "backward_saved_state": measure(backward),
                    }
                )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "metric": "synchronized wall time per call; includes host overhead; not kernel-only",
        "comparison": comparison,
        "timings": timings,
        "artifact_sha256": {
            n: hashlib.sha256((args.out / n).read_bytes()).hexdigest()
            for n in ("inputs.pt", "outputs.pt")
        },
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
