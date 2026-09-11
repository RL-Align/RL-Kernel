# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the Triton FlashAttention LSE export + packed varlen path.

Covers the two additions from `rl_engine.kernels.ops.triton.triton_attn`
(docs/operators/attention-varlen.md):

  * `triton_flash_attention(..., return_lse=True)` vs. `torch.nn.functional.
    scaled_dot_product_attention` on dense `[B, H, S, D]` tensors -- reports
    the LSE-export overhead (return_lse=True vs. False) alongside the native
    baseline.
  * `triton_flash_attention_varlen` on packed `[total_tokens, H, D]` tensors
    vs. a pad-mask-unpad SDPA baseline -- the realistic alternative for RL
    rollout/training batches with skewed per-sequence response lengths. This
    is where packing's compute/memory savings actually show up, since the
    padded baseline pays for the longest sequence in the batch on every row.

Reports forward and forward+backward latency plus peak extra VRAM for the
forward pass.

Usage:
    python benchmarks/benchmark_triton_attention_varlen.py
    python benchmarks/benchmark_triton_attention_varlen.py --dense-only
    python benchmarks/benchmark_triton_attention_varlen.py --varlen-only
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.triton.triton_attn import (
    triton_flash_attention,
    triton_flash_attention_varlen,
)
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

# (batch, heads, seq_len, head_dim)
DENSE_CONFIGS = [
    (2, 8, 512, 64),
    (2, 8, 2048, 64),
    (1, 16, 4096, 128),
    (1, 16, 8192, 128),
]

# Per-sequence lengths (query == key/value, self-attention prefill), one
# entry per batch config. Deliberately skewed / not block-aligned, mimicking
# RL rollout batches where responses stop at wildly different lengths (and
# occasionally immediately, e.g. an empty completion).
VARLEN_CONFIGS = [
    ("uniform_512", [512] * 8),
    ("skewed_short_long", [32, 64, 96, 128, 1536, 2048, 3072, 4000]),
    ("many_short_few_long", [16, 24, 40, 48, 64, 80, 96, 4096]),
    ("with_empty_seq", [0, 128, 256, 512, 1024, 2048, 3000, 3500]),
]

HEADS = 8
HEAD_DIM = 64


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


def _sdpa_dense(q, k, v, causal):
    # q, k, v: [B, H, S, D]
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _pad_mask_unpad_sdpa_varlen(q, k, v, seqlens, causal):
    """Pad-mask-unpad SDPA baseline for packed varlen input.

    q, k, v: [total_tokens, H, D] packed along dim 0, split by `seqlens`.
    This is the naive alternative to `triton_flash_attention_varlen`: pad
    every sequence to the batch max, run dense masked SDPA, then gather the
    valid rows back out. Included so the benchmark reflects what packing is
    actually saving, not just kernel-vs-kernel throughput.
    """
    batch = len(seqlens)
    max_len = max(seqlens) if seqlens else 0
    H, D = q.shape[1], q.shape[2]
    device, dtype = q.device, q.dtype

    q_pad = torch.zeros(batch, H, max_len, D, device=device, dtype=dtype)
    k_pad = torch.zeros(batch, H, max_len, D, device=device, dtype=dtype)
    v_pad = torch.zeros(batch, H, max_len, D, device=device, dtype=dtype)
    attn_mask = torch.zeros(batch, 1, max_len, max_len, device=device, dtype=torch.bool)

    offset = 0
    for b, s in enumerate(seqlens):
        if s == 0:
            continue
        q_pad[b, :, :s, :] = q[offset : offset + s].transpose(0, 1)
        k_pad[b, :, :s, :] = k[offset : offset + s].transpose(0, 1)
        v_pad[b, :, :s, :] = v[offset : offset + s].transpose(0, 1)
        valid = torch.ones(s, s, device=device, dtype=torch.bool)
        if causal:
            valid = torch.tril(valid)
        attn_mask[b, 0, :s, :s] = valid
        offset += s

    out_pad = torch.nn.functional.scaled_dot_product_attention(
        q_pad, k_pad, v_pad, attn_mask=attn_mask
    )

    out = torch.zeros_like(q)
    offset = 0
    for b, s in enumerate(seqlens):
        if s == 0:
            continue
        out[offset : offset + s] = out_pad[b, :, :s, :].transpose(0, 1)
        offset += s
    return out


def run_dense_benchmark(args):
    device = device_ctx.device
    dtype = torch.float16
    rows = []

    for batch, heads, seq, head_dim in args.dense_configs:
        q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
        sm_scale = 1.0 / (head_dim**0.5)

        def sdpa_fwd(q=q, k=k, v=v):
            with torch.no_grad():
                _sdpa_dense(q, k, v, causal=True)

        def triton_fwd(q=q, k=k, v=v):
            with torch.no_grad():
                triton_flash_attention(q, k, v, causal=True, sm_scale=sm_scale)

        def triton_fwd_lse(q=q, k=k, v=v):
            with torch.no_grad():
                triton_flash_attention(
                    q, k, v, causal=True, sm_scale=sm_scale, return_lse=True
                )

        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)

        def sdpa_fwd_bwd(q=qg, k=kg, v=vg):
            out = _sdpa_dense(q, k, v, causal=True)
            torch.autograd.grad(out, (q, k, v), torch.ones_like(out))

        def triton_fwd_bwd(q=qg, k=kg, v=vg):
            out = triton_flash_attention(q, k, v, causal=True, sm_scale=sm_scale)
            torch.autograd.grad(out, (q, k, v), torch.ones_like(out))

        s_fwd = _time_ms(sdpa_fwd, args.warmup, args.iters)
        t_fwd = _time_ms(triton_fwd, args.warmup, args.iters)
        t_fwd_lse = _time_ms(triton_fwd_lse, args.warmup, args.iters)
        s_fb = _time_ms(sdpa_fwd_bwd, args.warmup, args.iters)
        t_fb = _time_ms(triton_fwd_bwd, args.warmup, args.iters)
        s_vram = _peak_vram_gb(sdpa_fwd)
        t_vram = _peak_vram_gb(triton_fwd)

        rows.append(
            [
                f"{batch}x{heads}x{seq}x{head_dim}",
                f"{s_fwd:.3f}",
                f"{t_fwd:.3f}",
                f"{t_fwd_lse:.3f}",
                f"{s_fwd/t_fwd:.2f}x",
                f"{s_fb:.3f}",
                f"{t_fb:.3f}",
                f"{s_fb/t_fb:.2f}x",
                f"{s_vram*1024:.0f}",
                f"{t_vram*1024:.0f}",
            ]
        )

    headers = [
        "shape (B x H x S x D)",
        "sdpa fwd ms",
        "triton fwd ms",
        "triton fwd+lse ms",
        "fwd speedup",
        "sdpa f+b ms",
        "triton f+b ms",
        "f+b speedup",
        "sdpa fwd MB",
        "triton fwd MB",
    ]
    print("\nDense attention: triton_flash_attention vs. torch SDPA (causal)")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def run_varlen_benchmark(args):
    device = device_ctx.device
    dtype = torch.float16
    heads, head_dim = args.heads, args.head_dim
    sm_scale = 1.0 / (head_dim**0.5)
    rows = []

    for name, seqlens in args.varlen_configs:
        total = sum(seqlens)
        max_seqlen = max(seqlens)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device
        )

        q = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(total, heads, head_dim, device=device, dtype=dtype)

        def pad_fwd(q=q, k=k, v=v):
            with torch.no_grad():
                _pad_mask_unpad_sdpa_varlen(q, k, v, seqlens, causal=True)

        def triton_fwd(q=q, k=k, v=v):
            with torch.no_grad():
                triton_flash_attention_varlen(
                    q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                    causal=True, sm_scale=sm_scale,
                )

        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)

        def pad_fwd_bwd(q=qg, k=kg, v=vg):
            out = _pad_mask_unpad_sdpa_varlen(q, k, v, seqlens, causal=True)
            torch.autograd.grad(out, (q, k, v), torch.ones_like(out))

        def triton_fwd_bwd(q=qg, k=kg, v=vg):
            out = triton_flash_attention_varlen(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                causal=True, sm_scale=sm_scale,
            )
            torch.autograd.grad(out, (q, k, v), torch.ones_like(out))

        p_fwd = _time_ms(pad_fwd, args.warmup, args.iters)
        t_fwd = _time_ms(triton_fwd, args.warmup, args.iters)
        p_fb = _time_ms(pad_fwd_bwd, args.warmup, args.iters)
        t_fb = _time_ms(triton_fwd_bwd, args.warmup, args.iters)
        p_vram = _peak_vram_gb(pad_fwd)
        t_vram = _peak_vram_gb(triton_fwd)

        padded_tokens = len(seqlens) * max_seqlen
        waste_pct = 100.0 * (1.0 - total / padded_tokens)

        rows.append(
            [
                name,
                f"{len(seqlens)}",
                f"{total} ({waste_pct:.0f}% pad waste)",
                f"{p_fwd:.3f}",
                f"{t_fwd:.3f}",
                f"{p_fwd/t_fwd:.2f}x",
                f"{p_fb:.3f}",
                f"{t_fb:.3f}",
                f"{p_fb/t_fb:.2f}x",
                f"{p_vram*1024:.0f}",
                f"{t_vram*1024:.0f}",
            ]
        )

    headers = [
        "config",
        "batch",
        "total tokens (vs. padded)",
        "pad+sdpa fwd ms",
        "triton varlen fwd ms",
        "fwd speedup",
        "pad+sdpa f+b ms",
        "triton varlen f+b ms",
        "f+b speedup",
        "pad+sdpa fwd MB",
        "triton varlen fwd MB",
    ]
    print(
        f"\nPacked varlen attention: triton_flash_attention_varlen vs. "
        f"pad-mask-unpad SDPA (causal, H={heads}, D={head_dim})"
    )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--varlen-only", action="store_true")
    parser.add_argument(
        "--dense-configs",
        type=str,
        default=None,
        help="Semicolon-separated 'batch,heads,seq_len,head_dim' tuples.",
    )
    args = parser.parse_args()
    if args.dense_configs:
        args.dense_configs = [
            tuple(int(x) for x in tup.split(",")) for tup in args.dense_configs.split(";")
        ]
    else:
        args.dense_configs = DENSE_CONFIGS
    args.varlen_configs = VARLEN_CONFIGS
    return args


def main():
    args = parse_args()
    if device_ctx.device_type != "cuda":
        raise RuntimeError(
            "Triton attention benchmark requires a CUDA device (Triton kernels are CUDA-only)."
        )

    logger.info(f"Triton attention benchmark on {device_ctx.device}")

    if not args.varlen_only:
        run_dense_benchmark(args)
    if not args.dense_only:
        run_varlen_benchmark(args)


if __name__ == "__main__":
    main()
