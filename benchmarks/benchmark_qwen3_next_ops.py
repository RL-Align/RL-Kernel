# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the Qwen3-Next linear-attention reference ops on NPU/CUDA/CPU.

Covers the three Phase-0 golden operators (`sigmoid_gate`, `causal_conv1d`,
`gated_deltanet`) in the two regimes that matter for the hybrid stack:

  - prefill: one forward over [B, T] tokens (throughput-oriented)
  - decode:  single-token steps carrying conv cache / GDN state (latency)

The PyTorch goldens are the baseline the Phase-1 AscendC kernels must beat;
on an NPU host they run through torch_npu eager. GDN prefill is a Python
loop over T (reference, not production shape) — the sweep caps its iters
automatically so the run stays bounded.

Usage:
    python benchmarks/benchmark_qwen3_next_ops.py
    python benchmarks/benchmark_qwen3_next_ops.py --device npu --configs "1,4096;4,4096;8,2048"
    python benchmarks/benchmark_qwen3_next_ops.py --skip-prefill --batches "1,4,16,64"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.platforms.device import _npu_available
from rl_engine.reference.activation.sigmoid_gate import NativeSigmoidGateOp
from rl_engine.reference.linear_attn.causal_conv1d_sequence import NativeCausalConv1dOp
from rl_engine.reference.linear_attn.gated_deltanet import NativeGatedDeltaNetOp
from rl_engine.utils.logger import logger

# Qwen3-Next-80B-A3B GDN geometry (verify against your checkpoint's
# config.json before comparing provider performance).
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
# Only q|k|v enter the short convolution; z bypasses it.
# 16*128 + 16*128 + 32*128 = 8192 channels.
CONV_CHANNELS = (
    LINEAR_KEY_HEADS * HEAD_K_DIM + LINEAR_KEY_HEADS * HEAD_K_DIM + LINEAR_VALUE_HEADS * HEAD_V_DIM
)

DEFAULT_CONFIGS = [(1, 4096), (4, 4096), (8, 2048), (16, 1024)]
DEFAULT_DECODE_BATCHES = [1, 4, 16, 64]


def _resolve_device(name: str) -> torch.device:
    if name == "npu":
        if not _npu_available():
            raise RuntimeError("device npu requested but torch_npu/NPU is unavailable")
        return torch.device("npu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    # auto
    if _npu_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _acc(device: torch.device):
    """Accelerator module for sync/events: torch.npu, torch.cuda or None (cpu)."""
    if device.type == "npu":
        import torch_npu  # noqa: F401  # populates torch.npu

        return torch.npu
    if device.type == "cuda":
        return torch.cuda
    return None


@torch.no_grad()
def _time_ms(fn, warmup: int, iters: int, device: torch.device) -> float:
    acc = _acc(device)
    for _ in range(warmup):
        fn()
    if acc is None:
        import time

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) * 1e3 / iters
    acc.synchronize()
    start = acc.Event(enable_timing=True)
    end = acc.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    acc.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def _peak_mem_gb(fn, warmup: int, iters: int, device: torch.device) -> float:
    acc = _acc(device)
    if acc is None or not hasattr(acc, "max_memory_allocated"):
        return float("nan")
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


def run_prefill(args) -> list:
    device = _resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    gate_op = NativeSigmoidGateOp().to(device)
    conv_op = NativeCausalConv1dOp().to(device)
    gdn_op = NativeGatedDeltaNetOp().to(device)

    rows = []
    for batch, seq in args.configs:
        with torch.no_grad():
            x = torch.randn(batch, seq, CONV_CHANNELS, device=device, dtype=dtype)
            conv_w = torch.randn(CONV_CHANNELS, CONV_KERNEL, device=device, dtype=dtype)
            conv_b = torch.randn(CONV_CHANNELS, device=device, dtype=dtype)
            q = torch.randn(batch, LINEAR_KEY_HEADS, seq, HEAD_K_DIM, device=device, dtype=dtype)
            k = torch.randn(batch, LINEAR_KEY_HEADS, seq, HEAD_K_DIM, device=device, dtype=dtype)
            v = torch.randn(batch, LINEAR_VALUE_HEADS, seq, HEAD_V_DIM, device=device, dtype=dtype)
            beta = torch.randn(batch, LINEAR_VALUE_HEADS, seq, device=device, dtype=dtype)
            g = -torch.rand(batch, LINEAR_VALUE_HEADS, seq, device=device, dtype=dtype)

            def gate_fwd(x=x):
                gate_op(x, x)

            def conv_fwd(x=x, conv_w=conv_w, conv_b=conv_b):
                conv_op(x, conv_w, conv_b, None, return_cache=True)

            def gdn_fwd(q=q, k=k, v=v, beta=beta, g=g):
                gdn_op(q, k, v, beta, g, None, return_final_state=True)

        # The GDN golden is a Python loop over T; keep the run bounded.
        gdn_iters = max(3, min(args.iters, int(2048 / max(seq // 512, 1) / 4)))
        for name, fn, iters in (
            ("sigmoid_gate", gate_fwd, args.iters),
            ("causal_conv1d", conv_fwd, args.iters),
            ("gated_deltanet", gdn_fwd, gdn_iters),
        ):
            ms = _time_ms(fn, args.warmup, iters, device)
            mem = _peak_mem_gb(fn, 2, 3, device)
            rows.append(
                [f"{batch}x{seq}", name, f"{ms:.3f}", f"{batch * seq / ms:.1f}", f"{mem:.2f}"]
            )
            logger.info(f"prefill {name} {batch}x{seq}: {ms:.3f} ms")
    return rows


def run_decode(args) -> list:
    device = _resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    gate_op = NativeSigmoidGateOp().to(device)
    conv_op = NativeCausalConv1dOp().to(device)
    gdn_op = NativeGatedDeltaNetOp().to(device)

    rows = []
    for batch in args.batches:
        with torch.no_grad():
            x = torch.randn(batch, 1, CONV_CHANNELS, device=device, dtype=dtype)
            conv_w = torch.randn(CONV_CHANNELS, CONV_KERNEL, device=device, dtype=dtype)
            conv_b = torch.randn(CONV_CHANNELS, device=device, dtype=dtype)
            _, cache = conv_op(
                torch.randn(batch, 16, CONV_CHANNELS, device=device, dtype=dtype),
                conv_w,
                conv_b,
                None,
                return_cache=True,
            )
            q = torch.randn(batch, LINEAR_KEY_HEADS, 1, HEAD_K_DIM, device=device, dtype=dtype)
            k = torch.randn(batch, LINEAR_KEY_HEADS, 1, HEAD_K_DIM, device=device, dtype=dtype)
            v = torch.randn(batch, LINEAR_VALUE_HEADS, 1, HEAD_V_DIM, device=device, dtype=dtype)
            beta = torch.randn(batch, LINEAR_VALUE_HEADS, 1, device=device, dtype=dtype)
            g = -torch.rand(batch, LINEAR_VALUE_HEADS, 1, device=device, dtype=dtype)
            state = torch.zeros(
                batch,
                LINEAR_VALUE_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device=device,
                dtype=torch.float32,
            )

            def gate_fwd(x=x):
                gate_op(x, x)

            def conv_fwd(x=x, conv_w=conv_w, conv_b=conv_b, cache=cache):
                conv_op(x, conv_w, conv_b, cache, return_cache=True)

            def gdn_fwd(q=q, k=k, v=v, beta=beta, g=g, state=state):
                gdn_op.decode_step(q, k, v, beta, g, state)

        for name, fn in (
            ("sigmoid_gate", gate_fwd),
            ("causal_conv1d", conv_fwd),
            ("gated_deltanet", gdn_fwd),
        ):
            ms = _time_ms(fn, args.warmup, args.iters, device)
            rows.append([f"B={batch}", name, f"{ms:.4f}", f"{batch / ms:.1f}", "-"])
            logger.info(f"decode {name} B={batch}: {ms:.4f} ms/step")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--configs",
        default=";".join(f"{b},{s}" for b, s in DEFAULT_CONFIGS),
        help="prefill sweep as batch,seq;batch,seq;...",
    )
    parser.add_argument(
        "--batches",
        default=",".join(map(str, DEFAULT_DECODE_BATCHES)),
        help="decode sweep as comma-separated batch sizes",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    if args.iters <= 0 or args.warmup < 0:
        parser.error("iters must be positive and warmup must be nonnegative")
    args.configs = [tuple(map(int, c.split(","))) for c in args.configs.split(";") if c]
    args.batches = [int(b) for b in args.batches.split(",") if b]

    if any(len(c) != 2 or min(c) <= 0 for c in args.configs) or any(b <= 0 for b in args.batches):
        parser.error("configs and batches must contain positive dimensions")
    device = _resolve_device(args.device)
    logger.info(
        f"qwen3-next op benchmark on {device} (dtype={args.dtype}, "
        f"Hk={LINEAR_KEY_HEADS}, Hv={LINEAR_VALUE_HEADS}, Dk={HEAD_K_DIM}, "
        f"Dv={HEAD_V_DIM}, conv_ch={CONV_CHANNELS})"
    )

    header = ["config", "op", "latency_ms", "tok_per_ms", "peak_mem_gb"]
    if not args.skip_prefill:
        print("\n=== prefill (one forward over B x T tokens) ===")
        print(tabulate(run_prefill(args), headers=header, tablefmt="github"))
    if not args.skip_decode:
        print("\n=== decode (single token, stateful) ===")
        print(tabulate(run_decode(args), headers=header, tablefmt="github"))


if __name__ == "__main__":
    main()
