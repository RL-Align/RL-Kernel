# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark the batch-invariant DSv4 router against the non-invariant path.

Forward: DeepGEMM ``bf16_gemm_nt`` (FP32 out) + our top-6 kernel, against cuBLAS
``torch.mm(out_dtype=float32)`` + vLLM's Triton ``_dsv4_topk_kernel`` (loaded from
``--vllm-src``, the fork's file; without it, PyTorch eager routing). Backward:
our six-row gather against the dense dgrad ``scatter + torch.mm``. Shapes are
DSv4-Flash (H 4096, 256 experts, top-6). CUDA-graph replay, median, microseconds.

Usage:
    python benchmarks/benchmark_dsv4_router.py \
        [--vllm-src ~/dsv4-work/vllm-src/vllm/model_executor/layers/fused_moe/router/dsv4_topk.py]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import statistics
import tempfile

import torch

from rl_engine.kernels.ops.cuda.router import route_backward_dx_cuda, route_topk_cuda
from rl_engine.models.deepseek_v4.router import gate_logits

H, E, SCALE = 4096, 256, 1.5
TOKENS = (1, 64, 256, 1024, 4096, 16384)


def _load_vllm(path, name="_dsv4_topk_kernel"):
    import triton  # noqa: F401

    text = open(path).read()
    nodes = ast.walk(ast.parse(text))
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines()
    body = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in lines[fn.decorator_list[0].lineno - 1 : fn.end_lineno]
    )
    tmp = os.path.join(tempfile.mkdtemp(), f"vllm_{name.strip('_')}.py")
    with open(tmp, "w") as f:
        f.write("import triton\nimport triton.language as tl\n\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(f"vllm_{name.strip('_')}", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, name)

    def run(logits, bias):
        t = logits.shape[0]
        w = torch.empty(t, 6, dtype=torch.float32, device=logits.device)
        ids = torch.empty(t, 6, dtype=torch.int32, device=logits.device)
        kernel[(t,)](
            logits, bias, w, ids, SCALE, NUM_EXPERTS=E, BLOCK_N=E, num_warps=1, launch_pdl=False
        )
        return w, ids

    return run


def _eager_route(logits, bias):
    s = torch.sqrt(torch.nn.functional.softplus(logits))
    ids = torch.topk(s + bias, 6, dim=-1).indices
    w = s.gather(1, ids)
    return w / w.sum(-1, keepdim=True) * SCALE, ids


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
    baseline = _load_vllm(os.path.expanduser(args.vllm_src)) if args.vllm_src else _eager_route
    base_name = "vLLM Triton" if args.vllm_src else "eager"
    g = torch.Generator(device="cuda").manual_seed(0)
    gate = (torch.randn(E, H, generator=g, device="cuda") * 0.02).bfloat16()
    bias = torch.randn(E, generator=g, device="cuda") * 0.1
    print(
        f"{'T':>6} | {'gate DG':>8} {'cuBLAS':>8} | {'topk':>7} {base_name:>11} | "
        f"{'fwd':>7} {'base':>7} {'ratio':>5} | {'bwd dX':>7} {'dense':>7} {'ratio':>5}"
    )
    for t in TOKENS:
        x = torch.randn(t, H, generator=g, device="cuda").bfloat16()
        logits = gate_logits(x, gate)
        d = torch.randn(t, 6, generator=g, device="cuda")
        _, ids = route_topk_cuda(logits, bias, SCALE)
        dense_buf = torch.zeros(t, E, device="cuda")

        gate_dg = _graph_us(lambda: gate_logits(x, gate))
        gate_cb = _graph_us(lambda: torch.mm(x, gate.T, out_dtype=torch.float32))
        topk = _graph_us(lambda: route_topk_cuda(logits, bias, SCALE))
        topk_base = _graph_us(lambda: baseline(logits, bias))
        fwd = _graph_us(lambda: route_topk_cuda(gate_logits(x, gate), bias, SCALE))
        fwd_base = _graph_us(lambda: baseline(torch.mm(x, gate.T, out_dtype=torch.float32), bias))
        bwd = _graph_us(lambda: route_backward_dx_cuda(logits, ids, d, gate, SCALE, torch.bfloat16))
        bwd_base = _graph_us(lambda: torch.mm(dense_buf.scatter(1, ids.long(), d).bfloat16(), gate))
        print(
            f"{t:>6} | {gate_dg:8.1f} {gate_cb:8.1f} | {topk:7.1f} {topk_base:11.1f} | "
            f"{fwd:7.1f} {fwd_base:7.1f} {fwd / fwd_base:5.2f} | "
            f"{bwd:7.1f} {bwd_base:7.1f} {bwd / bwd_base:5.2f}"
        )


if __name__ == "__main__":
    main()
