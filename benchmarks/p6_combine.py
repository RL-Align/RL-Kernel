# SPDX-License-Identifier: Apache-2.0
"""Compare T05 with ordered eager PyTorch; emits JSON to stdout.

Run from the checkout: python -m benchmarks.p6_combine --tokens 256 --hidden-size 4096
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from rl_engine.p6.combine import canonical_row_lookup, prepare_combine, tensor_bytes
from rl_engine.p6.contract import CONTRACT_VERSION, PROFILE, CombinePlan, digest
from rl_engine.p6.fixtures import make_case


def timing(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    wall_samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        wall_samples.append((time.perf_counter() - wall_start) * 1e6)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "wall_median_us": statistics.median(wall_samples),
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if min(args.tokens, args.hidden_size, args.repeats) <= 0 or args.warmup < 0:
        parser.error("tokens, hidden-size and repeats must be positive; warmup nonnegative")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA is required; there is no CPU timing fallback")
    import triton

    device = torch.device("cuda", torch.cuda.current_device())
    # Small deterministic fixture values avoid overflow/subnormal in the timed profile.
    case = make_case("t05-benchmark", n=args.tokens, h=1, mode="mixed")
    plan = replace(CombinePlan.from_dict(case["plan"]), hidden_size=args.hidden_size)

    def values(n, shift):
        sequence = torch.arange(n * args.hidden_size, dtype=torch.int32, device=device)
        return (
            (((sequence + shift) % 4097 - 2048).float() / 64)
            .to(torch.bfloat16)
            .reshape(n, args.hidden_size)
        )

    rows = values(len(plan.inverse_map), 0)
    shared = values(args.tokens, 127)
    residual = values(args.tokens, 991)
    lookup = torch.tensor(
        canonical_row_lookup(plan, plan.context), dtype=torch.int64, device=device
    )
    # Eager arithmetic oracle, not the timed scalar oracle or native T02/T03 provider.
    padded_rows = torch.cat(
        (rows, torch.zeros((1, args.hidden_size), dtype=torch.bfloat16, device=device))
    )
    safe_lookup = torch.where(lookup >= 0, lookup, len(plan.inverse_map))
    valid = lookup >= 0

    def reference():
        acc = torch.zeros((args.tokens, args.hidden_size), dtype=torch.float32, device=device)
        seen = torch.zeros((args.tokens, 1), dtype=torch.bool, device=device)
        for slot in range(6):
            value = padded_rows[safe_lookup[:, slot]].float()
            mask = valid[:, slot : slot + 1]
            acc = torch.where(mask, torch.where(seen, acc + value, value), acc)
            seen = seen | mask
        return ((acc + shared.float()) + residual.float()).to(torch.bfloat16)

    prepared = prepare_combine(plan, plan.context, device)
    buffers = prepared.allocate()

    def launch():
        prepared.launch(rows, shared, residual, buffers)

    wanted = tensor_bytes(reference())
    launch()
    buffers.check_status()
    if tensor_bytes(buffers.output) != wanted:
        raise RuntimeError("T05/reference byte mismatch: performance run refused")
    debug = prepared.forward(rows, shared, residual, debug=True)
    if tensor_bytes(debug.output) != wanted:
        raise RuntimeError("debug-on/off mismatch: performance run refused")
    eager = timing(reference, args.warmup, args.repeats)
    kernel = timing(launch, args.warmup, args.repeats)
    checked = timing(lambda: prepared.forward(rows, shared, residual), args.warmup, args.repeats)
    buffers.check_status()
    if tensor_bytes(buffers.output) != wanted:
        raise RuntimeError("T05 bytes changed during timing")
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        source, dirty = "unavailable", None
    root = Path(__file__).resolve().parents[1]
    implementation = {
        path: (root / path).read_text()
        for path in ("rl_engine/p6/combine.py", "rl_engine/kernels/ops/triton/moe/combine.py")
    }
    print(
        json.dumps(
            {
                "scope": "synthetic-local-WS1; not live Foundation/WS2 certification",
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": torch.cuda.get_device_capability(device),
                "source_commit": source,
                "working_tree_dirty": dirty,
                "implementation_sha256": digest(implementation),
                "contract_version": CONTRACT_VERSION,
                "profile": PROFILE,
                "tokens": args.tokens,
                "hidden_size": args.hidden_size,
                "debug": False,
                "dtype": "bf16",
                "block_size": 256,
                "num_warps": 4,
                "warmup": args.warmup,
                "plan_fingerprint": plan.fingerprint,
                "order_hash": plan.order_hash,
                "input_sha256": digest([tensor_bytes(v) for v in (rows, shared, residual)]),
                "output_hash": digest(wanted),
                "correctness": "BYTE_EQUAL",
                "eager_reference": eager,
                "prepared_launch": kernel,
                "checked_eager": checked,
                "prepared_speedup": eager["median_us"] / kernel["median_us"],
                "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
