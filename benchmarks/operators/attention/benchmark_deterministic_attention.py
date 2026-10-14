# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for CUDA deterministic standard-softmax attention (issue #147).

Reports latency and peak memory for Qwen3-8B representative shapes,
including the cost of full scores/P materialization.

Use the ``h3`` subcommand for the strict-native/CUDA all-gradient matrix.
"""

import argparse
import math
import sys
import time

import torch
from torch.autograd.function import once_differentiable


def benchmark_attention(
    B: int, Hq: int, Hkv: int, Sq: int, Skv: int, D: int, dtype, warmup: int = 5, iters: int = 20
):
    from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionOp

    op = DeterministicAttentionOp()
    device = "cuda"

    q = torch.randn(B, Hq, Sq, D, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, Skv, D, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, Skv, D, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            op.forward(q, k, v, causal=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            op.forward(q, k, v, causal=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    scores_mem_mb = (B * Hq * Sq * Skv * 4) / (1024 * 1024)

    return {
        "latency_ms": elapsed * 1000,
        "peak_memory_mb": peak_mem_mb,
        "scores_materialization_mb": scores_mem_mb,
    }


QWEN3_8B_SHAPES = [
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1, "Skv": 128, "D": 128, "label": "decode-128"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1, "Skv": 1024, "D": 128, "label": "decode-1k"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 128, "Skv": 128, "D": 128, "label": "prefill-128"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 512, "Skv": 512, "D": 128, "label": "prefill-512"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1024, "Skv": 1024, "D": 128, "label": "prefill-1k"},
    {"B": 4, "Hq": 32, "Hkv": 8, "Sq": 128, "Skv": 128, "D": 128, "label": "batch4-prefill-128"},
    {"B": 8, "Hq": 32, "Hkv": 8, "Sq": 64, "Skv": 64, "D": 128, "label": "batch8-prefill-64"},
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark deterministic attention")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"{'Shape':<25} {'Latency(ms)':>12} {'PeakMem(MB)':>12} {'Scores(MB)':>11}")
    print("-" * 65)

    for shape in QWEN3_8B_SHAPES:
        kwargs = shape.copy()
        label = kwargs.pop("label")
        try:
            result = benchmark_attention(
                **kwargs,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(
                f"{label:<25} {result['latency_ms']:>12.3f} "
                f"{result['peak_memory_mb']:>12.1f} "
                f"{result['scores_materialization_mb']:>11.1f}"
            )
        except RuntimeError as exc:
            print(f"{label:<25} {'OOM' if 'out of memory' in str(exc) else 'ERROR':>12}")
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()


def h3_value_grad(op, xs, dy):
    out = op(*xs)
    return [out.detach(), *[g.detach() for g in torch.autograd.grad(out, xs, dy)]]


def h3_bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class H3StrictNative(torch.autograd.Function):
    """Unoptimized native baseline, compiled with the same strict arithmetic flags."""

    @staticmethod
    def forward(ctx, q, k, v):
        from rl_engine.backends.cuda.attention.h3_full_attention import _extension

        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        out, lse, p = _extension().deterministic_attention_forward(
            q, k, v, False, 1.0 / math.sqrt(128), None
        )
        ctx.save_for_backward(q, k, v, p)
        lse = lse.transpose(1, 2)
        ctx.mark_non_differentiable(lse)
        return out.transpose(1, 2), lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, unused_lse):
        from rl_engine.backends.cuda.attention.h3_full_attention import _extension

        grads = _extension().deterministic_attention_backward(
            grad.transpose(1, 2).contiguous(), *ctx.saved_tensors, False, 1.0 / math.sqrt(128), None
        )
        return tuple(g.transpose(1, 2) for g in grads)


def h3_main(argv=None):
    """Paired strict native/CUDA forward and all-QKV-gradient matrix."""
    import hashlib
    import json
    import statistics
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))
    from rl_engine.backends.cuda.attention.h3_full_attention import H3FullAttentionOp, _extension

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1x129", "1x512", "1x1025", "1x4096", "2x1024", "4x1024"],
    )
    parser.add_argument("--chunks", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        shapes = [tuple(map(int, x.split("x"))) for x in args.shapes]
    except ValueError:
        parser.error("shapes must be positive BxS pairs")
    if any(len(s) != 2 or min(s) <= 0 for s in shapes):
        parser.error("shapes must be positive BxS pairs")
    if args.samples < 3 or args.warmups < 1 or min(args.chunks) <= 0:
        parser.error("use >=3 samples, >=1 warmup and positive chunks")
    args.output.mkdir(parents=True, exist_ok=True)
    ops = {"strict_native": lambda *xs: H3StrictNative.apply(*xs)[0]}
    for chunk in dict.fromkeys(args.chunks):
        ops[f"cuda-c{chunk}"] = H3FullAttentionOp(chunk_rows=chunk)
    sources = [Path(__file__), ROOT / "csrc/cuda/attention/deterministic_attention.cu"]
    sources += [
        ROOT / "rl_engine/backends/cuda/attention/h3_full_attention.py",
        ROOT / "rl_engine/reference/attention/standard_attn.py",
    ]
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension_sha256": hashlib.sha256(Path(_extension().__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "provenance": {k: v.provenance() for k, v in ops.items() if k != "strict_native"},
        "measurement": (
            "CUDA events; eager forward + backward, including layout copies and allocations"
        ),
        "accuracy": "output/dQ/dK/dV raw-bit equality against strict native on identical inputs",
        "warmups": args.warmups,
        "samples": args.samples,
        "cases": [],
    }

    def save():
        temporary = args.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "report.json")

    save()
    for batch, seq in shapes:
        # The native baseline materializes quadratic FP32 intermediates.
        free, _ = torch.cuda.mem_get_info()
        estimate = batch * 56 * seq * seq * 4 * 3 + batch * seq * 56 * 128 * 2 * 20
        if estimate > free * 0.75:
            raise RuntimeError(f"{batch}x{seq}: insufficient free memory for safe paired baseline")
        torch.manual_seed(42)
        xs = [
            torch.randn(batch, seq, 56, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for _ in range(3)
        ]
        dy = torch.randn_like(xs[0])
        golden = h3_value_grad(ops["strict_native"], xs, dy)
        for name, op in ops.items():
            actual = h3_value_grad(op, xs, dy)
            assert all(h3_bits_equal(a, g) for a, g in zip(actual, golden)), (batch, seq, name)
            del actual
        del golden
        for op in ops.values():
            for _ in range(args.warmups):
                h3_value_grad(op, xs, dy)
        torch.cuda.synchronize()
        rows = {name: [] for name in ops}
        names = list(ops)
        for sample in range(args.samples):
            order = names[sample % len(names) :] + names[: sample % len(names)]
            if sample % 2:
                order = list(reversed(order))
            for name in order:
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = h3_value_grad(ops[name], xs, dy)
                end.record()
                end.synchronize()
                rows[name].append(
                    {
                        "ms": start.elapsed_time(end),
                        "extra_peak_bytes": torch.cuda.max_memory_allocated() - resident,
                    }
                )
                del result
        case = dict(
            shape=[batch, seq, 56, 128],
            seed=42,
            backends={
                name: dict(
                    median_ms=statistics.median(s["ms"] for s in samples),
                    peak_extra_bytes=max(s["extra_peak_bytes"] for s in samples),
                    raw_samples=samples,
                )
                for name, samples in rows.items()
            },
        )
        report["cases"].append(case)
        save()
        print(json.dumps(case), flush=True)
        del xs, dy
    report["status"] = "passed"
    save()
    print("PAIRED_MATRIX_PASSED", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["h3"]:
        h3_main(sys.argv[2:])
    else:
        main()
