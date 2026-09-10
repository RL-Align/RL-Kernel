# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark deterministic standard-softmax attention across backends.

All backends compute ``softmax(Q K^T * scale + causal mask) @ V`` with a locked,
per-row reduction order (batch-invariant, no split-K). The comparison here is
latency across a sequence sweep:

- Native is the pure-PyTorch fp32-accumulating ground-truth reference.
- Ascend is the CANN two-pass streaming kernel (one AI core block/row); only
  present when the extension is built with ``KERNEL_ALIGN_FORCE_ASCEND=1``.
- CUDA is the deterministic op (one CTA/row); only present when the extension
  is built with ``KERNEL_ALIGN_FORCE_SM90=1`` on an SM90 device.

Timing dispatch through the active accelerator (``torch.cuda`` or
``torch.npu``), so the benchmark runs on CUDA/ROCm/NPU devices.

Usage:
    python benchmarks/benchmark_attention.py
    python benchmarks/benchmark_attention.py --backward
    python benchmarks/benchmark_attention.py --configs "1,8,512;2,8,2048"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

_D = 128
_DEFAULT_KV_HEADS = 8


def _accel():
    """The active accelerator module (torch.npu on Ascend, torch.cuda otherwise)."""
    if device_ctx.device_type == "npu":
        return torch.npu
    return torch.cuda


def _maybe_ascend_op():
    """The Ascend C op, or None when unavailable (no NPU / not built)."""
    if device_ctx.device_type != "npu":
        return None
    try:
        from rl_engine.kernels.ops.ascend.attention.deterministic_attn import (
            DeterministicAttentionAscendOp,
        )
    except (ImportError, RuntimeError):
        return None
    return DeterministicAttentionAscendOp()


def _maybe_cuda_op():
    """The CUDA deterministic op, or None when unavailable (no CUDA / not built)."""
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not (
        torch.cuda.is_available()
        and _EXT_AVAILABLE
        and hasattr(_C, "deterministic_attention_forward")
    ):
        return None
    from rl_engine.kernels.ops.cuda.attention.deterministic_attn import DeterministicAttentionOp

    return DeterministicAttentionOp()


# (batch, num_q_heads, seqlen); Hq = 32, Hkv = 8 (Qwen3-style GQA, g = 4).
DEFAULT_CONFIGS = [
    (1, 32, 512),
    (1, 32, 1024),
    (1, 32, 2048),
    (4, 32, 2048),
]


def _make_inputs(batch, hq, seq, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(batch, hq, seq, _D, dtype=dtype, generator=generator).to(device)
    k = torch.randn(batch, _DEFAULT_KV_HEADS, seq, _D, dtype=dtype, generator=generator).to(device)
    v = torch.randn(batch, _DEFAULT_KV_HEADS, seq, _D, dtype=dtype, generator=generator).to(device)
    return q, k, v


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


def _forward_closure(op, q, k, v):
    def run():
        with torch.no_grad():
            op(q, k, v, causal=True)

    return run


def _forward_backward_closure(op, q, k, v):
    def run():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        op(qq, kk, vv, causal=True).sum().backward()

    return run


def _bench_table(configs, native_op, other_ops, closure_factory, device, dtype, warmup, iters):
    label = "fwd" if closure_factory is _forward_closure else "fwd+bwd"
    rows = []
    for batch, hq, seq in configs:
        q, k, v = _make_inputs(batch, hq, seq, device, dtype)
        n_c = closure_factory(native_op, q, k, v)
        n_ms = _time_ms(n_c, warmup, iters)
        row = [f"{batch}x{hq}x{seq}", f"{n_ms:.3f}"]
        for _name, op in other_ops:
            if op is None:
                row += ["-"]
                continue
            o_ms = _time_ms(closure_factory(op, q, k, v), warmup, iters)
            row += [f"{o_ms:.3f}", f"{n_ms / o_ms:.2f}x"]
        rows.append(row)

    headers = ["shape (B x Hq x S)", f"native {label} ms"]
    for name, _ in other_ops:
        headers += [f"{name} {label} ms", "vs native"]
    logger.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backward", action="store_true", help="also emit the forward+backward table"
    )
    parser.add_argument(
        "--configs",
        default=";".join(",".join(map(str, c)) for c in DEFAULT_CONFIGS),
        help="semicolon-separated 'batch,hq,seq' triples",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    configs = [tuple(int(x) for x in part.split(",")) for part in args.configs.split(";")]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(device_ctx.device_type)

    native_op = NativeAttentionOp()
    other_ops = [
        ("ascend", _maybe_ascend_op()),
        ("cuda", _maybe_cuda_op()),
    ]

    _bench_table(
        configs, native_op, other_ops, _forward_closure, device, dtype, args.warmup, args.iters
    )
    if args.backward:
        _bench_table(
            configs,
            native_op,
            other_ops,
            _forward_backward_closure,
            device,
            dtype,
            args.warmup,
            args.iters,
        )


if __name__ == "__main__":
    main()
