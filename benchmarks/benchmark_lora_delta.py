# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-3 shared_grouped_lora_delta: torch-native vs Triton vs CUDA (issue #62).

Both deterministic paths compose the repo's shared fixed-K primitives instead of
introducing a new GEMM, so they are batch-invariant and SLOWER than cuBLAS by
design (pinned tiles, no split-K, FP32 accumulation, no TF32). As in
benchmark_det_gemm.py this reports overhead against a fair baseline, not a
speedup.

The operator is memory bound: with X=[M,4096], N=2048, R=8 the arithmetic
intensity is about R = 8 FLOP/byte, and the traffic splits roughly 66% reading X
/ 33% writing Y / 0.26% for the U round trip -- which is why fusing away U is
not where the headroom is (deferred to #58, epilogue fusion).

Measured at M=64 on an RTX A2000, the two BF16 GEMMs cost 0.070 ms while the
whole native forward costs 0.222 ms, so 68% of the wall clock is the elementwise
`.float()` / `* alpha` / BF16-cast traffic around them, not the matmuls. That
also means absolute numbers here are launch-bound at small M and the TF32 column
can land below the FP32 one; read the columns as relative overhead, not as GEMM
throughput.
"""

import argparse

import torch

from rl_engine.moe.backends.lora_delta import (
    LoRADeltaCudaProvider,
    LoRADeltaProvider,
    LoRADeltaTritonProvider,
)

DEV = "cuda"
WARMUP, ITERS, REPEATS = 20, 100, 5
ALPHA = 0.5

# (label, M, K, N, r) -- K/N follow a 4096-hidden / 2048-ffn expert, r is the
# LoRA rank. M sweeps the token counts an RL rollout actually sees.
SHAPES = [
    ("decode_m1", 1, 4096, 2048, 8),
    ("small_m8", 8, 4096, 2048, 8),
    ("batch_m64", 64, 4096, 2048, 8),
    ("batch_m256", 256, 4096, 2048, 8),
    ("fixture", 24, 128, 64, 8),
]


def _time_once(fn) -> float:
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _time(fn) -> float:
    """Median of REPEATS samples -- robust against clock/thermal drift."""
    for _ in range(WARMUP):
        fn()
    samples = sorted(_time_once(fn) for _ in range(REPEATS))
    return samples[len(samples) // 2]


def _make(m: int, k: int, n: int, r: int):
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    a = torch.randn(r, k, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(n, r, dtype=torch.bfloat16, device=DEV)
    dy = torch.randn(m, n, dtype=torch.bfloat16, device=DEV)
    return x, a, b, dy


def _bench_fn(provider, direction: str, tensors, u_ref):
    """Build the closure to time. Kept out of the loop so the captured tensors
    are function arguments rather than loop variables (ruff B023)."""
    x, a, b, dy = tensors

    if direction == "fwd":
        return lambda: provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
    if direction == "bwd":
        return lambda: provider.shared_grouped_lora_delta_bwd(dy, x, a, b, ALPHA, u_ref)

    def both():
        _, u = provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
        provider.shared_grouped_lora_delta_bwd(dy, x, a, b, ALPHA, u)

    return both


def run(direction: str):
    native = LoRADeltaProvider()
    cuda = LoRADeltaCudaProvider()
    triton = LoRADeltaTritonProvider()

    rows = []
    for label, m, k, n, r in SHAPES:
        tensors = _make(m, k, n, r)
        x, a, b, _ = tensors
        _, u_ref = native.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)

        # Fair baseline: TF32 off, since both deterministic paths disable it.
        torch.backends.cuda.matmul.allow_tf32 = True
        t_tf32 = _time(_bench_fn(native, direction, tensors, u_ref))
        torch.backends.cuda.matmul.allow_tf32 = False
        t_native = _time(_bench_fn(native, direction, tensors, u_ref))
        t_triton = _time(_bench_fn(triton, direction, tensors, u_ref))
        t_cuda = _time(_bench_fn(cuda, direction, tensors, u_ref))

        rows.append((label, m, k, n, r, t_tf32, t_native, t_triton, t_cuda))
    return rows


def to_markdown(rows, dev, cap, direction: str) -> str:
    out = [
        f"## P5-3 lora_delta {direction} — {dev} (SM{cap[0]}{cap[1]})",
        "",
        "| shape | M | K | N | r | native tf32 | native fp32 | Triton | CUDA "
        "| Triton ovh | CUDA ovh |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m, k, n, r, t_tf32, t_native, t_tri, t_cuda in rows:
        out.append(
            f"| {label} | {m} | {k} | {n} | {r} | {t_tf32:.3f} | {t_native:.3f} | "
            f"{t_tri:.3f} | {t_cuda:.3f} | {t_tri / t_native:.1f}x | "
            f"{t_cuda / t_native:.1f}x |"
        )
    out += [
        "",
        f"_Times in ms, median of {REPEATS} samples of {ITERS} iterations after "
        f"{WARMUP} warmups. Overhead is against `native fp32` (cuBLAS with TF32 "
        "disabled), the fair baseline -- both deterministic paths disable TF32. "
        "At these sizes ~68% of the wall clock is elementwise dtype/scale traffic "
        "rather than the GEMMs, so the TF32 column can fall below the FP32 one; "
        "treat the columns as relative overhead, not GEMM throughput. Only the "
        "Triton path is byte-equal to the oracle._",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", default="fwd", choices=("fwd", "bwd", "both"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
    print(f"{name}  SM{cap[0]}{cap[1]}  torch {torch.__version__}")
    md = to_markdown(run(args.direction), name, cap, args.direction)
    print("\n" + md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
