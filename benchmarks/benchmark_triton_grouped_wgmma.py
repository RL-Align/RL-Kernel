# SPDX-License-Identifier: Apache-2.0
"""Experimental SM90 prep/compute/end-to-end benchmark; no provider promotion.

Run probe_p5_triton_wgmma.py and sanitizers first. Timing success is not a
numeric gate. All inputs stay packed; GPU/host latency labels are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_engine.moe.mx_format import MXTensor  # noqa: E402
from rl_engine.moe import triton_grouped_gemm_wgmma as impl  # noqa: E402


def measure(fn, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    # Events measure the elapsed stream interval, including gaps if Python cannot
    # feed short kernels fast enough. Wall time includes allocation and launch.
    for _ in range(5):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(
            {
                "stream_us": start.elapsed_time(end) * 1000 / iterations,
                "wall_us": (time.perf_counter() - wall_start) * 1e6 / iterations,
            }
        )
    return {
        "median_stream_us": statistics.median(s["stream_us"] for s in samples),
        "median_wall_us": statistics.median(s["wall_us"] for s in samples),
        "samples": samples,
    }


def make_inputs(sizes, n, k):
    m, e = sum(sizes), len(sizes)
    torch.manual_seed(2026)
    # Encode finite normal/subnormal FP8 bytes directly, avoiding dequantized W.
    a_codes = torch.randint(0, 120, (m, k), dtype=torch.uint8, device="cuda")
    a_codes |= torch.randint(0, 2, (m, k), dtype=torch.uint8, device="cuda") * 128
    a = MXTensor(
        a_codes, torch.full((m, k // 32), 125, dtype=torch.uint8, device="cuda"), "e4m3", (m, k)
    )
    w = MXTensor(
        torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda"),
        torch.full((e, n, k // 32), 125, dtype=torch.uint8, device="cuda"),
        "e2m1",
        (e, n, k),
    )
    offsets = torch.tensor([0, *itertools.accumulate(sizes)], dtype=torch.int32, device="cuda")
    dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    return a, w, offsets, dy


def run_case(label, sizes, args):
    a, w, offsets, dy = make_inputs(sizes, args.n, args.k)
    m, n, k = sum(sizes), args.n, args.k
    c = impl._program_count(a.codes.device, args.programs)
    # Checked warm-up validates contents outside trusted measurements.
    impl.grouped_gemm_fwd(a, w, offsets)
    impl.grouped_gemm_bwd(dy, w, offsets)
    df, pf = impl._prepare_metadata(offsets, n, k)
    db, pb = impl._prepare_metadata(offsets, n, k, backward=True)
    y = torch.empty((m, n), device="cuda", dtype=torch.float32)
    dx = torch.empty((m, k), device="cuda", dtype=torch.float32)
    measurements = {}
    calls = {
        "fwd_prep_with_allocation": lambda: impl._prepare_metadata(offsets, n, k),
        "bwd_prep_with_allocation": lambda: impl._prepare_metadata(offsets, n, k, backward=True),
        "fwd_compute_preallocated": lambda: impl._launch_fwd(a, w, df, pf, y, c),
        "bwd_compute_preallocated": lambda: impl._launch_bwd(dy, w, db, pb, dx, c),
        "fwd_e2e_trusted": lambda: impl.grouped_gemm_fwd(
            a, w, offsets, validate_contents=False, programs=c
        ),
        "bwd_e2e_trusted": lambda: impl.grouped_gemm_bwd(
            dy, w, offsets, validate_contents=False, programs=c
        ),
        "fwd_e2e_checked": lambda: impl.grouped_gemm_fwd(a, w, offsets, programs=c),
        "bwd_e2e_checked": lambda: impl.grouped_gemm_bwd(dy, w, offsets, programs=c),
    }
    # For the common one-CTA scan, isolate device prep using preallocated buffers.
    if len(sizes) <= impl.SCAN_BLOCK:
        sums = torch.empty((1,), dtype=torch.int64, device="cuda")
        for backward, desc, prefix, name in ((False, df, pf, "fwd"), (True, db, pb, "bwd")):
            calls[f"{name}_prep_preallocated"] = (
                lambda backward=backward, desc=desc, prefix=prefix: impl._prep_kernel[(1,)](
                    offsets,
                    desc,
                    prefix,
                    sums,
                    N=n,
                    K=k,
                    E=len(sizes),
                    BLOCK_M=impl.BM,
                    B=impl.triton.next_power_of_2(len(sizes)),
                    BACKWARD=backward,
                    num_warps=4,
                )
            )
    for alias in args.compare:
        from rl_engine.moe.provider import resolve_provider

        # Explicitly requested comparisons fail if their backend is unavailable.
        provider = resolve_provider(alias)
        calls[f"{alias}_fwd_e2e"] = lambda p=provider: p.mxfp8_mxfp4_grouped_gemm_fwd(a, w, offsets)
        calls[f"{alias}_bwd_e2e"] = lambda p=provider: p.mxfp8_mxfp4_grouped_gemm_bwd(
            dy, w, offsets
        )
    for name, call in calls.items():
        measurements[name] = measure(call, args.iterations)
        print(label, name, measurements[name]["median_wall_us"], "wall us", flush=True)
    return {
        "case": label,
        "sizes": sizes,
        "n": n,
        "k": k,
        "programs": c,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--programs", type=int)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--compare",
        nargs="*",
        choices=("wgmma",),
        default=[],
        help="optional comparison with a separately installed CUDA WGMMA provider",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/triton-wgmma-benchmark.json")
    args = parser.parse_args()
    if min(args.n, args.k, args.iterations) <= 0 or args.k % 32:
        parser.error("n/iterations must be positive and k a positive multiple of 32")
    if impl.triton is None or not torch.cuda.is_available():
        raise RuntimeError("SM90 CUDA and Triton are required")
    impl._require_sm90(torch.empty(0, device="cuda"))
    result = {
        "status": "timing-only; numeric and sanitizer evidence required separately",
        "torch": torch.__version__,
        "triton": impl.triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "numeric_profile_candidate": "p5-wgmma-triton-v1",
        "iterations": args.iterations,
        "warmup": 5,
        "samples": 5,
        "source_sha256": {
            name: hashlib.sha256((ROOT / "rl_engine/moe" / name).read_bytes()).hexdigest()
            for name in (
                "triton_grouped_gemm_wgmma.py",
                "mx_format.py",
            )
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for label, sizes in (
            ("tiny_empty", [0, 1, 0, 1, 0, 1, 0, 1]),
            ("balanced", [128] * 8),
            ("skew", [10000, 1, 0, 1, 0, 1, 0, 1]),
        ):
            result["cases"].append(run_case(label, sizes, args))
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
