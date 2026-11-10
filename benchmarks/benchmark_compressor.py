# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer compressor (C4, head 128) against vLLM's Triton kernel (D7).

``--vllm-src`` points at vLLM's
``vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py``
(``ISEEKYAN/vllm@1a6542c``); its ``_fused_kv_compress_norm_rope_insert_indexer_attn``
is extracted and JIT-compiled with Triton. Both kernels read the same paged
state cache and write the same paged FP8 K cache; the script also reports how
many FP8 bytes and scales agree (information only: the contract is our own
fixed-order arithmetic, design doc T02). CUDA-graph replay, median.

Usage:
    python benchmarks/benchmark_compressor.py [--vllm-src PATH]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import statistics
import tempfile

import torch

from rl_engine.kernels.ops.cuda.compressor import compress_indexer_cuda

D, ROPE, RATIO, WIDTH, STATE_BS, K_BS = 128, 64, 4, 256, 16, 64
CASES = (  # (label, requests, tokens per request)
    ("decode b256", 256, 1),
    ("prefill 1 x 4096", 1, 4096),
    ("prefill 4 x 4096", 4, 4096),
    ("prefill 1 x 32768", 1, 32768),
)


def _load_vllm(path):
    import triton  # noqa: F401

    text = open(path).read()
    fn = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_fused_kv_compress_norm_rope_insert_indexer_attn"
    )
    lines = text.splitlines()
    body = "\n".join(lines[fn.decorator_list[0].lineno - 1 : fn.end_lineno])
    tmp = os.path.join(tempfile.mkdtemp(), "vllm_indexer_compress.py")
    with open(tmp, "w") as f:
        f.write("import triton\nimport triton.language as tl\n\n" + body + "\n")
    spec = importlib.util.spec_from_file_location("vllm_indexer_compress", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fused_kv_compress_norm_rope_insert_indexer_attn


def _graph_us(fn, iters=30):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-src", default=None)
    args = parser.parse_args()
    vllm = _load_vllm(args.vllm_src) if args.vllm_src else None

    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device="cuda").double() / ROPE))
    cos_sin = torch.cat(
        [
            (torch.arange(1 << 16, device="cuda").double()[:, None] * inv).cos(),
            (torch.arange(1 << 16, device="cuda").double()[:, None] * inv).sin(),
        ],
        -1,
    ).float()
    w = (1 + 0.1 * torch.randn(D, device="cuda")).float()
    print(
        f"{'case':<20}{'ours us':>10}"
        + (f"{'vllm us':>10}{'ratio':>8}{'byte agree':>14}" if vllm else "")
    )
    for label, reqs, per in CASES:
        ctx = 8192 if per == 1 else per  # decode: each request has an 8K-token history
        n_state_blocks = reqs * (ctx // STATE_BS)
        state = torch.randn(n_state_blocks, STATE_BS, 2 * WIDTH, device="cuda")
        table = torch.randperm(n_state_blocks, device="cuda").int().view(reqs, -1)
        if per == 1:
            pos = torch.full((reqs,), ctx - 1, device="cuda", dtype=torch.long)
            req = torch.arange(reqs, device="cuda", dtype=torch.int32)
        else:
            pos = torch.arange(per, device="cuda").repeat(reqs)
            req = torch.arange(reqs, device="cuda", dtype=torch.int32).repeat_interleave(per)
        entries = reqs * (ctx // RATIO)
        k_blocks = -(-entries // K_BS)
        k_slots = torch.where(
            (pos + 1) % RATIO == 0, req.long() * (ctx // RATIO) + pos // RATIO, -1
        )
        ours_cache = torch.zeros(k_blocks, K_BS * (D + 4), dtype=torch.uint8, device="cuda")
        ref_cache = torch.zeros_like(ours_cache)
        token_slots = torch.zeros_like(pos)

        def ours():
            compress_indexer_cuda(
                state, table, req, pos, w, 1e-6, cos_sin, ours_cache, K_BS, k_slots
            )

        line = f"{label:<20}{_graph_us(ours):>10.1f}"
        if vllm:

            def ref():
                vllm[(pos.shape[0],)](
                    state,
                    state.stride(0),
                    state.stride(1),
                    req,
                    pos,
                    token_slots,
                    table,
                    table.stride(0),
                    STATE_BS,
                    w,
                    1e-6,
                    cos_sin,
                    cos_sin.stride(0),
                    ref_cache,
                    k_slots,
                    K_BS,
                    HEAD_SIZE=D,
                    TRITON_BLOCK_SIZE=D,
                    STATE_WIDTH=WIDTH,
                    COMPRESS_RATIO=RATIO,
                    OVERLAP=True,
                    ROPE_HEAD_DIM=ROPE,
                    FP8_MAX=448.0,
                    QUANT_BLOCK=128,
                    TOKEN_STRIDE=128,
                    SCALE_DIM=4,
                    KV_BLOCK_STRIDE=ref_cache.stride(0),
                    num_warps=1,
                )

            ref_us = _graph_us(ref)
            ours()
            ref()
            torch.cuda.synchronize()
            written = k_slots[k_slots >= 0]
            q_ours = ours_cache[written // K_BS].view(-1, K_BS * (D + 4))
            q_ref = ref_cache[written // K_BS].view(-1, K_BS * (D + 4))
            agree = (q_ours == q_ref).float().mean().item()
            line += f"{ref_us:>10.1f}{_graph_us(ours) / ref_us:>8.2f}{agree:>14.7f}"
        print(line)


if __name__ == "__main__":
    main()
