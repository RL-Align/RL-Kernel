# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the strict Nemotron-H lm_head_logprob op against the native path.

Shapes are the real Nemotron-3-Nano-30B-A3B boundary (hidden 2688, vocab 131072,
BF16). The native production path is a cuBLAS BF16 ``F.linear`` -> ``.float()``
-> ``log_softmax`` -> gather (autograd backward), and ``F.embedding`` for the
input lookup. Reported phases:

* rollout decode / prefill: strict rollout entry (logp + FP32 logits) vs native;
* training: strict train entry forward and forward+backward vs native;
* embedding lookup.

Usage:
    python benchmarks/operators/logprob/benchmark_nemotron_h_lm_head_logp.py
    python benchmarks/operators/logprob/benchmark_nemotron_h_lm_head_logp.py \
        --decode 1,64 --train 4096 --json /tmp/nemotron_lm_head.json
"""

from __future__ import annotations

import argparse
import json

import torch
from tabulate import tabulate

from rl_engine.models.nemotron_h.lm_head_contract import NEMOTRON_H_LM_HEAD_SPEC

HIDDEN = NEMOTRON_H_LM_HEAD_SPEC.hidden_size
VOCAB = NEMOTRON_H_LM_HEAD_SPEC.vocab_size
DEFAULT_DECODE = (1, 8, 32, 128, 256, 512)
DEFAULT_PREFILL = (2048, 8192)
DEFAULT_TRAIN = (2048, 4096, 8192, 16384)


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_mb(fn) -> float:
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**2)


def _native_logp(hidden, weight, target):
    logits = torch.nn.functional.linear(hidden, weight).float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    return logp, logits


def _inputs(rows: int):
    hidden = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, VOCAB, (rows,), device="cuda")
    return hidden, target


def _rollout_rows(op, weight, sizes, args, phase):
    rows = []
    for n in sizes:
        hidden, target = _inputs(n)

        def native():
            with torch.no_grad():
                _native_logp(hidden, weight, target)

        def strict():
            with torch.no_grad():
                op.logprob_with_logits(hidden, weight, target)

        native_ms = _time_ms(native, args.warmup, args.iters)
        strict_ms = _time_ms(strict, args.warmup, args.iters)
        rows.append(
            {
                "phase": phase,
                "tokens": n,
                "native_ms": native_ms,
                "strict_ms": strict_ms,
                "strict_over_native": strict_ms / native_ms,
                "strict_tokens_per_s": n / strict_ms * 1e3,
                "native_peak_mb": _peak_mb(native),
                "strict_peak_mb": _peak_mb(strict),
            }
        )
    return rows


def _train_rows(op, weight, sizes, args):
    rows = []
    for n in sizes:
        hidden, target = _inputs(n)
        grad = torch.randn(n, device="cuda", dtype=torch.float32)

        def native_fwd():
            with torch.no_grad():
                _native_logp(hidden, weight, target)

        def strict_fwd():
            with torch.no_grad():
                op.logprob(hidden, weight, target)

        def native_fwd_bwd():
            h = hidden.detach().requires_grad_(True)
            w = weight.detach().requires_grad_(True)
            _native_logp(h, w, target)[0].backward(grad)

        def strict_fwd_bwd():
            h = hidden.detach().requires_grad_(True)
            w = weight.detach().requires_grad_(True)
            op.logprob(h, w, target)[0].backward(grad)

        timings = {
            name: _time_ms(fn, args.warmup, args.iters)
            for name, fn in (
                ("native_fwd_ms", native_fwd),
                ("strict_fwd_ms", strict_fwd),
                ("native_fwd_bwd_ms", native_fwd_bwd),
                ("strict_fwd_bwd_ms", strict_fwd_bwd),
            )
        }
        rows.append(
            {
                "phase": "train",
                "tokens": n,
                **timings,
                "fwd_strict_over_native": timings["strict_fwd_ms"] / timings["native_fwd_ms"],
                "fwd_bwd_strict_over_native": timings["strict_fwd_bwd_ms"]
                / timings["native_fwd_bwd_ms"],
                "native_fwd_bwd_peak_mb": _peak_mb(native_fwd_bwd),
                "strict_fwd_bwd_peak_mb": _peak_mb(strict_fwd_bwd),
            }
        )
    return rows


def _embedding_rows(op, embed_weight, sizes, args):
    rows = []
    for n in sizes:
        ids = torch.randint(0, VOCAB, (n,), device="cuda")
        native_ms = _time_ms(
            lambda: torch.nn.functional.embedding(ids, embed_weight), args.warmup, args.iters
        )
        strict_ms = _time_ms(lambda: op.embed(ids, embed_weight), args.warmup, args.iters)
        rows.append(
            {
                "phase": "embedding",
                "tokens": n,
                "native_ms": native_ms,
                "strict_ms": strict_ms,
                "strict_over_native": strict_ms / native_ms,
            }
        )
    return rows


def _print(title, rows):
    if not rows:
        return
    print(f"\n### {title}")
    print(tabulate([list(r.values()) for r in rows], headers=list(rows[0]), floatfmt=".3f"))


def run_benchmark(args) -> dict:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Nemotron-H lm_head_logprob benchmark requires an SM90 GPU")
    from rl_engine.backends.cuda.logprob.nemotron_h_lm_head_logp import NemotronHLMHeadLogprobSM90Op

    op = NemotronHLMHeadLogprobSM90Op()
    weight = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    embed_weight = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "provenance": op.provenance(),
        "decode": _rollout_rows(op, weight, args.decode, args, "decode"),
        "prefill": _rollout_rows(op, weight, args.prefill, args, "prefill"),
        "train": _train_rows(op, weight, args.train, args),
        "embedding": _embedding_rows(op, embed_weight, args.decode + args.train, args),
    }
    print(f"device={report['device']} torch={report['torch']}")
    print(f"det_gemm={report['provenance']['det_gemm_backend_id']}")
    for key in ("decode", "prefill", "train", "embedding"):
        _print(key, report[key])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def _sizes(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    return tuple(int(x) for x in raw.split(",") if x.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--decode", type=str, default=None, help="comma-separated batch sizes")
    parser.add_argument("--prefill", type=str, default=None, help="comma-separated token counts")
    parser.add_argument("--train", type=str, default=None, help="comma-separated token counts")
    parser.add_argument("--json", type=str, default=None, help="optional JSON report path")
    args = parser.parse_args()
    args.decode = _sizes(args.decode, DEFAULT_DECODE)
    args.prefill = _sizes(args.prefill, DEFAULT_PREFILL)
    args.train = _sizes(args.train, DEFAULT_TRAIN)
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
