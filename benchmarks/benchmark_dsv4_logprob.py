# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 selected-token logprob (dsv4-logprob-v1) against a naive non-invariant path.

Naive: ``logits = h @ W.T`` (cuBLAS BF16), ``log_softmax(logits.float())``, gather; the
backward is PyTorch autograd through it. Ours: DeepGEMM FP32 logits + tile stats /
merge, backward = recomputed logits + BF16 dL/dlogits + DeepGEMM ``bf16_gemm_nn``.
H = 4096, V = 129280 (DSv4-Flash), median of CUDA-event timings, eager.

    PYTHONPATH=. python benchmarks/benchmark_dsv4_logprob.py
"""

from __future__ import annotations

import statistics

import torch

H, V = 4096, 129280


def _time(fn, iters: int) -> float:
    for _ in range(2):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def main() -> None:
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import head_logits, selected_logprob

    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(V, H, generator=g, device="cuda") / H**0.5).bfloat16()

    def naive(h, t):
        logits = h @ w.t()
        return torch.log_softmax(logits.float(), dim=-1).gather(1, t.view(-1, 1)).view(-1)

    print(f"H200, H={H}, V={V}; microseconds (median); ratio = ours / naive")
    print(
        "     T |   ours fwd  naive fwd ratio |  ours f+b naive f+b ratio | "
        "logits GEMM  stats+merge"
    )
    for t_len in (1, 64, 1024, 4096, 16384):
        iters = 20 if t_len <= 1024 else 5
        h = (torch.randn(t_len, H, generator=g, device="cuda") * 2).bfloat16()
        t = torch.randint(V, (t_len,), generator=g, device="cuda")
        gr = torch.randn(t_len, generator=g, device="cuda")

        with torch.no_grad():
            ours_f = _time(lambda: selected_logprob(h, w, t), iters)
            naive_f = _time(lambda: naive(h, t), iters)
            gemm = _time(lambda: head_logits(h[:4096], w), iters)

        def ours_fb():
            x = h.detach().requires_grad_(True)
            logp, _ = selected_logprob(x, w, t)
            (logp * gr).sum().backward()

        def naive_fb():
            x = h.detach().requires_grad_(True)
            (naive(x, t) * gr).sum().backward()

        ours_b = _time(ours_fb, iters)
        naive_b = _time(naive_fb, iters)
        print(
            f"{t_len:6d} | {ours_f:10.1f} {naive_f:10.1f} {ours_f / naive_f:5.2f} | "
            f"{ours_b:9.1f} {naive_b:9.1f} {ours_b / naive_b:5.2f} | "
            f"{gemm:11.1f} {max(ours_f - gemm * -(-t_len // 4096), 0):12.1f}"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
