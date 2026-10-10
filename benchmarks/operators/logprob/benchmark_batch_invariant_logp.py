# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark batch-invariant logp: Native vs Triton vs CUDA (SM90 TMA) vs Ascend.

All backends compute ``logits[t, target[t]] - logsumexp(logits[t, :])`` from a
materialized ``[N, V]`` logits tensor with a locked, per-row reduction order
(batch-invariant). The comparison here is latency and peak device memory across
a vocab sweep:

- Native materializes FP32 ``[N, V]`` intermediates for an explicit row-wise
  max, exp, sum, and log calculation.
- Triton streams the vocab through an online softmax (grid = one program/row).
- CUDA is the Hopper TMA online-softmax kernel (one CTA/row); only present when
  the extension is built with ``KERNEL_ALIGN_FORCE_SM90=1`` on an SM90 device.
- Ascend is the CANN two-pass streaming kernel (one AI core block/row); only
  present when the extension is built with ``KERNEL_ALIGN_FORCE_ASCEND=1``.

By default only the forward pass is timed; pass ``--backward`` to also emit the
forward+backward table (grad w.r.t. logits). Timing and memory accounting
dispatch through the active accelerator (``torch.cuda`` or ``torch.npu``), so
the benchmark runs on CUDA/ROCm/NPU devices.

Usage:
    python benchmarks/operators/logprob/benchmark_batch_invariant_logp.py
    python benchmarks/operators/logprob/benchmark_batch_invariant_logp.py --backward
    python benchmarks/operators/logprob/benchmark_batch_invariant_logp.py \
        --configs "4096,128256;8192,151936"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.platforms.device import device_ctx
from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.utils.logger import logger


def _accel():
    """The active accelerator module (torch.npu on Ascend, torch.cuda otherwise)."""
    if device_ctx.device_type == "npu":
        return torch.npu
    return torch.cuda


def _maybe_triton_op():
    """The Triton op, or None when unavailable (no Triton / NPU-only host)."""
    if device_ctx.device_type == "npu":
        return None
    try:
        from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
    except ImportError:
        return None
    return TritonBatchInvariantLogpOp()


def _maybe_sm90_op():
    """The Hopper TMA op, or None when unavailable (non-Hopper / not built)."""
    if device_ctx.device_type != "cuda":
        return None
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and _EXT_AVAILABLE
        and hasattr(_C, "batch_invariant_logp_sm90")
    ):
        return None
    from rl_engine.backends.cuda.logprob.batch_invariant_logp import BatchInvariantLogpSM90Op

    return BatchInvariantLogpSM90Op()


def _maybe_ascend_op():
    """The Ascend C op, or None when unavailable (no NPU / not built)."""
    if device_ctx.device_type != "npu":
        return None
    try:
        from rl_engine.backends.ascend.logprob.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()
    except (ImportError, OSError, RuntimeError):
        return None


def _maybe_accelerated_op():
    """(label, op) for the active device's hardware kernel, else (None, None)."""
    if device_ctx.device_type == "cuda":
        sm90_op = _maybe_sm90_op()
        if sm90_op is not None:
            return "cuda", sm90_op
    elif device_ctx.device_type == "npu":
        ascend_op = _maybe_ascend_op()
        if ascend_op is not None:
            return "ascend", ascend_op
    return None, None


# (num_tokens, vocab); vocab kept a multiple of 8 so the bf16 TMA path runs.
DEFAULT_CONFIGS = [
    (4096, 32768),
    (4096, 128256),
    (4096, 151936),
    (8192, 128256),
]


def _make_inputs(num_tokens, vocab, device, dtype):
    logits = torch.randn(num_tokens, vocab, device=device, dtype=dtype)
    target = torch.randint(0, vocab, (num_tokens,), device=device)
    return logits, target


def _time_ms(fn, warmup, iters):
    acc = _accel()
    for _ in range(warmup):
        fn()
    acc.synchronize()
    start = acc.Event(enable_timing=True)
    end = acc.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    acc.synchronize()
    return start.elapsed_time(end) / iters


def _peak_vram_gb(fn, warmup, iters):
    acc = _accel()
    for _ in range(warmup):
        fn()
    acc.synchronize()
    acc.empty_cache()
    baseline = acc.memory_allocated()
    acc.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    acc.synchronize()
    return (acc.max_memory_allocated() - baseline) / (1024**3)


def _forward_closure(op, logits, target):
    def run():
        with torch.no_grad():
            op(logits, target, validate=False)

    return run


def _forward_backward_closure(op, logits, target):
    # Fresh leaf each call so the grad is allocated (and freed) per iteration,
    # matching a real training step and giving an honest peak-VRAM reading.
    def run():
        x = logits.detach().requires_grad_(True)
        op(x, target, validate=False).sum().backward()

    return run


def _bench_table(configs, backends, closure_factory, device, dtype, warmup, iters):
    """Build a github-markdown table timing each backend, plus accel speedups.

    ``backends`` is ``(native_op, triton_op_or_None, (accel_label, accel_op))``;
    ``closure_factory`` maps ``(op, logits, target) -> callable`` (forward-only
    or forward+backward).
    """
    native, triton_op, (accel_label, accel_op) = backends
    label = "fwd" if closure_factory is _forward_closure else "fwd+bwd"
    rows = []
    for num_tokens, vocab in configs:
        logits, target = _make_inputs(num_tokens, vocab, device, dtype)
        n_c = closure_factory(native, logits, target)

        n_ms = _time_ms(n_c, warmup, iters)
        n_mb = _peak_vram_gb(n_c, warmup, iters) * 1024

        row = [f"{num_tokens}x{vocab}", f"{n_ms:.3f}"]
        t_ms = None
        if triton_op is not None:
            t_c = closure_factory(triton_op, logits, target)
            t_ms = _time_ms(t_c, warmup, iters)
            t_mb = _peak_vram_gb(t_c, warmup, iters) * 1024
            row += [f"{t_ms:.3f}", f"{n_ms/t_ms:.2f}x", f"{n_mb:.0f}", f"{t_mb:.0f}"]
        else:
            row += [f"{n_mb:.0f}"]
        if accel_op is not None:
            a_c = closure_factory(accel_op, logits, target)
            a_ms = _time_ms(a_c, warmup, iters)
            a_mb = _peak_vram_gb(a_c, warmup, iters) * 1024
            row += [f"{a_ms:.3f}", f"{n_ms/a_ms:.2f}x"]
            if t_ms is not None:
                row += [f"{t_ms/a_ms:.2f}x"]
            row += [f"{a_mb:.0f}"]
        rows.append(row)

    headers = ["shape (N x V)", f"native {label} ms"]
    if triton_op is not None:
        headers += [
            f"triton {label} ms",
            f"{label} speedup",
            f"native {label} MB",
            f"triton {label} MB",
        ]
    else:
        headers += [f"native {label} MB"]
    if accel_op is not None:
        headers += [f"{accel_label} {label} ms", f"{accel_label} vs native"]
        if triton_op is not None:
            headers += [f"{accel_label} vs triton"]
        headers += [f"{accel_label} {label} MB"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def run_benchmark(args):
    if device_ctx.device_type not in ["cuda", "hip", "npu"]:
        raise RuntimeError(
            "batch_invariant_logp benchmark requires a CUDA/ROCm/NPU device "
            "(uses accelerator-event timing)."
        )

    device = device_ctx.device
    dtype = torch.bfloat16
    accel_label, accel_op = _maybe_accelerated_op()
    backends = (NativeBatchInvariantLogpOp(), _maybe_triton_op(), (accel_label, accel_op))

    logger.info(
        f"batch_invariant_logp benchmark on {device} (dtype={dtype}); "
        f"accelerated backend: {accel_label or 'unavailable'}"
    )

    print("Forward")
    _bench_table(args.configs, backends, _forward_closure, device, dtype, args.warmup, args.iters)
    if args.backward:
        print("\nForward + backward")
        _bench_table(
            args.configs,
            backends,
            _forward_backward_closure,
            device,
            dtype,
            args.warmup,
            args.iters,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also time and measure a forward+backward pass (grad w.r.t. logits).",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'tokens,vocab' tuples, e.g. '4096,128256;8192,151936'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
