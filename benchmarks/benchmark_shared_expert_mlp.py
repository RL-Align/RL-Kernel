#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 shared-expert MLP benchmark: eager BF16 vs the DeepGEMM FP8 shared expert.

torch-native is the non-deterministic cuBLAS/eager reference (speed baseline). The
``deepgemm-*`` rows are ``rl_engine.moe.backends.deepgemm_shared_expert`` (profile
``dsv4-deepgemm-fp8-v1``): FP8 dX by default, BF16 det_gemm dX as the reference.

    python benchmarks/benchmark_shared_expert_mlp.py [--tokens 16,256,2048]
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.moe.contract import SharedBatch  # noqa: E402


def torch_native(batch: SharedBatch, dy: torch.Tensor):
    """Eager BF16 reference (cuBLAS + fused silu): fast but not bit-stable."""
    x = batch.x.detach().requires_grad_(True)
    z = x @ batch.w_fc1.t()
    ffn = z.shape[1] // 2
    gate, up = z[:, :ffn], z[:, ffn:]
    h = torch.nn.functional.silu(gate) * up
    y = h @ batch.w_fc2.t()
    y.backward(dy)
    return y, x.grad


def make_runner(provider):
    def run(batch: SharedBatch, dy: torch.Tensor):
        y, saved = provider.shared_expert_mlp_fwd(batch)
        dx = provider.shared_expert_mlp_bwd(dy, batch, saved)
        return y, dx

    return run


def time_ms(fn, *args, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--tokens", default="16,256,2048")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required")
        return 1

    from rl_engine.moe.provider import resolve_provider

    runners: dict[str, object] = {"torch-native": torch_native}
    # Each provider fails closed on a batch that declares another numeric profile.
    profiles: dict[str, str] = {}
    for label, spec in (
        (
            "deepgemm-fp8",
            "rl_engine.moe.backends.deepgemm_shared_expert:DeepGemmSharedExpertProvider",
        ),
        (
            "deepgemm-bf16bwd",
            "rl_engine.moe.backends.deepgemm_shared_expert:DeepGemmSharedExpertBf16BwdProvider",
        ),
    ):
        try:
            provider = resolve_provider(spec)
            runners[label] = make_runner(provider)
            profiles[label] = provider.numeric_profile
        except (NotImplementedError, RuntimeError) as exc:  # RuntimeError: DeepGemmUnavailable
            print(f"[skip] {label}: {exc}")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(2026)
    header = f"{'T':>6} {'backend':>14} {'fwd+bwd ms':>12} {'vs native':>10}"
    print(f"H={args.hidden} F={args.ffn} ({torch.cuda.get_device_name(0)})")
    print(header)
    for t in [int(v) for v in args.tokens.split(",")]:
        x = torch.randn(t, args.hidden, generator=gen).to(torch.bfloat16).to(device)
        w1 = (
            (torch.randn(2 * args.ffn, args.hidden, generator=gen) / args.hidden**0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        w2 = (
            (torch.randn(args.hidden, args.ffn, generator=gen) / args.ffn**0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        batch = SharedBatch(x=x, w_fc1=w1, w_fc2=w2)
        dy = torch.randn(t, args.hidden, generator=gen).to(torch.bfloat16).to(device)

        base_ms = None
        for label, fn in runners.items():
            run_batch = (
                dataclasses.replace(batch, numeric_profile=profiles[label])
                if label in profiles
                else batch
            )
            ms = time_ms(fn, run_batch, dy, iters=args.iters)
            if label == "torch-native":
                base_ms = ms
            rel = f"{ms / base_ms:8.2f}x" if base_ms else "        -"
            print(f"{t:>6} {label:>14} {ms:12.3f} {rel:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
