# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark NativeGRPOLossOp vs TritonGRPOLossOp.

Both ops consume logits and build on the fused ratio/KL op, so the working set
scales with the logits tensor [B, T, V] (B = num_prompts * samples_per_prompt).
Reports forward and forward+backward latency plus peak extra VRAM for the
forward pass across a range of (groups x samples x completion-length x vocab)
shapes; the vocab axis is where the online-softmax fusion pays off.

Usage:
    python benchmarks/benchmark_grpo_loss.py
    python benchmarks/benchmark_grpo_loss.py --configs "4,8,256,32768;4,8,256,131072"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.loss.grpo_loss import NativeGRPOLossOp
from rl_engine.kernels.ops.triton.loss.grpo_loss import TritonGRPOLossOp
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

# (num_prompts, samples_per_prompt, completion_len, vocab)
DEFAULT_CONFIGS = [
    (4, 8, 256, 32768),
    (4, 8, 256, 50257),
    (4, 8, 256, 131072),
]


def _make_inputs(num_prompts, spp, completion_len, vocab, device, dtype):
    batch = num_prompts * spp
    policy = torch.randn(batch, completion_len, vocab, device=device, dtype=dtype)
    ref = torch.randn(batch, completion_len, vocab, device=device, dtype=dtype)
    action_ids = torch.randint(0, vocab, (batch, completion_len), device=device)
    old = torch.randn(batch, completion_len, device=device, dtype=dtype)
    rewards = torch.randn(batch, device=device, dtype=dtype)
    mask = torch.ones(batch, completion_len, dtype=torch.bool, device=device)
    return policy, ref, action_ids, old, rewards, mask


def _time_ms(fn, warmup, iters):
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


def _peak_vram_gb(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**3)


def run_benchmark(args):
    if device_ctx.device_type != "cuda":
        raise RuntimeError("GRPO loss benchmark requires a CUDA device (Triton op is CUDA-only).")

    device = device_ctx.device
    dtype = torch.float16
    native = NativeGRPOLossOp()
    triton_op = TritonGRPOLossOp()
    kwargs = dict(clip_eps=args.clip_eps, beta=args.beta)

    logger.info(f"GRPO loss benchmark on {device} (dtype={dtype})")

    rows = []
    for num_prompts, spp, comp_len, vocab in args.configs:
        policy, ref, action_ids, old, rewards, mask = _make_inputs(
            num_prompts, spp, comp_len, vocab, device, dtype
        )
        n_tokens = num_prompts * spp * comp_len
        call_args = (ref, action_ids, old, rewards, mask)

        def native_fwd(p=policy):
            with torch.no_grad():
                native.forward(p, *call_args, samples_per_prompt=spp, **kwargs)

        def triton_fwd(p=policy):
            with torch.no_grad():
                triton_op.forward(p, *call_args, samples_per_prompt=spp, **kwargs)

        pol_grad = policy.clone().requires_grad_(True)

        def native_fwd_bwd(p=pol_grad):
            loss, _, _ = native.forward(p, *call_args, samples_per_prompt=spp, **kwargs)
            torch.autograd.grad(loss, p)

        def triton_fwd_bwd(p=pol_grad):
            loss, _, _ = triton_op.forward(p, *call_args, samples_per_prompt=spp, **kwargs)
            torch.autograd.grad(loss, p)

        n_fwd = _time_ms(native_fwd, args.warmup, args.iters)
        t_fwd = _time_ms(triton_fwd, args.warmup, args.iters)
        n_fb = _time_ms(native_fwd_bwd, args.warmup, args.iters)
        t_fb = _time_ms(triton_fwd_bwd, args.warmup, args.iters)
        n_vram = _peak_vram_gb(native_fwd)
        t_vram = _peak_vram_gb(triton_fwd)

        rows.append(
            [
                f"{num_prompts}x{spp}x{comp_len}x{vocab}",
                f"{n_tokens/1e3:.0f}K",
                f"{n_fwd:.3f}",
                f"{t_fwd:.3f}",
                f"{n_fwd/t_fwd:.2f}x",
                f"{n_fb:.3f}",
                f"{t_fb:.3f}",
                f"{n_fb/t_fb:.2f}x",
                f"{n_vram*1024:.0f}",
                f"{t_vram*1024:.0f}",
            ]
        )

    headers = [
        "shape (P x S x L x V)",
        "tokens",
        "native fwd ms",
        "triton fwd ms",
        "fwd speedup",
        "native f+b ms",
        "triton f+b ms",
        "f+b speedup",
        "native fwd MB",
        "triton fwd MB",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'prompts,samples,len,vocab' tuples, "
        "e.g. '4,8,256,32768;4,8,256,131072'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
