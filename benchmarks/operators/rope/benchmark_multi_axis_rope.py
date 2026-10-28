# SPDX-License-Identifier: Apache-2.0
"""Standalone RoPE: contiguous/packed QKV, eager/graph, optional pinned FlashAttention."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.ops.rope.multi_axis_rope import (  # noqa: E402
    build_multi_axis_cos_sin,
    get_multi_axis_rope_op,
    qwen_image_positions,
)

FLASH_REVISION = "98c1cee0ed59307833841cdbf567fe336ceec13a"
FLASH_SHA256 = "d2037e623854c702fb075784fe4292d4a89d6b9ac1ae33cab3d6788f8f94f240"
FLASH_LAYER_SHA256 = "a79cb42dfc33e126227bdba27ae7a024d7d5cc5c8fd7f7a5f26c68d501024e0a"


def upstream_complex(x, cis):
    """Diffusers use_real=False arithmetic; table assembly is outside timing."""
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], 64, 2))
    return torch.view_as_real(z * cis.unsqueeze(-2)).flatten(-2).to(x.dtype)


def load_flash_baselines(source_path, layer_path):
    """Load checksum-pinned upstream kernel and autograd class without installing flash-attn."""
    source = source_path.read_text(encoding="utf-8")
    layer = layer_path.read_text(encoding="utf-8")
    for text, expected in ((source, FLASH_SHA256), (layer, FLASH_LAYER_SHA256)):
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError("FlashAttention source checksum mismatch (normalized UTF-8 text)")
    node = next(n for n in ast.parse(layer).body if getattr(n, "name", None) == "ApplyRotaryEmb")
    directory = ROOT / "artifacts/flash_rope_baseline"
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    assert source.count("BLOCK_H=2,") == 1
    for variant in ("default", "no_fma"):
        code = source
        if variant == "no_fma":
            code = code.replace("BLOCK_H=2,", "BLOCK_H=2,\n            enable_fp_fusion=False,")
        path = directory / f"flash_rotary_{variant}.py"
        path.write_text(code, encoding="utf-8")
        name = f"rlk_flash_baseline_{variant}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        namespace = {
            "__name__": name + "_autograd",
            "torch": torch,
            "Tensor": torch.Tensor,
            "Optional": Optional,
            "Union": Union,
            "apply_rotary": module.apply_rotary,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(layer_path), "exec"), namespace)
        cls = namespace["ApplyRotaryEmb"]
        result[f"flash_{variant}"] = lambda x, c, s, cls=cls: cls.apply(x, c, s, True)
    return result


def measure(fn, args):
    for _ in range(args.warmup):
        result = fn()
    torch.cuda.synchronize()
    del result
    wall = []
    for _ in range(args.trials):
        start = time.perf_counter()
        for _ in range(args.iterations):
            result = fn()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1000 / args.iterations)
    del result

    # Collect graph owners before capture, so GC cannot destroy another graph
    # during stream capture. Preserve the caller's GC state.
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            for _ in range(args.graph_calls):
                result = fn()
        for _ in range(args.warmup):
            graph.replay()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iterations)
        ]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) / args.graph_calls for start, end in events]
        del graph, result
    finally:
        if enabled:
            gc.enable()

    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    del result
    return {
        "graph_median_ms": statistics.median(times),
        "graph_min_ms": min(times),
        "graph_max_ms": max(times),
        "eager_wall_median_ms": statistics.median(wall),
        "eager_wall_trials_ms": wall,
        "extra_peak_bytes": extra,
    }


def byte_differences(a, b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["contiguous", "packed_qkv"],
        default=["contiguous", "packed_qkv"],
    )
    parser.add_argument("--flash-source", type=Path)
    parser.add_argument("--flash-layer-source", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.trials, args.graph_calls) < 1:
        parser.error("timing counts must be positive")
    if bool(args.flash_source) != bool(args.flash_layer_source):
        parser.error("both FlashAttention source files are required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(386)
    ours = {name: get_multi_axis_rope_op(name, device="cuda") for name in ("cuda", "triton")}
    flash = (
        load_flash_baselines(args.flash_source, args.flash_layer_source)
        if args.flash_source
        else {}
    )
    reference = get_multi_axis_rope_op("reference")
    rows = []
    correctness = []
    for height, width in ((2, 3), (64, 64), (83, 83), (104, 58)):
        positions, _ = qwen_image_positions(height, width, 0)
        cos, sin = build_multi_axis_cos_sin(positions, device="cuda")
        cis = torch.complex(cos, sin)

        def upstream_op(x, _cos, _sin, cis=cis):
            return upstream_complex(x, cis)

        ops = {"upstream_complex": upstream_op, **ours, **flash}
        for layout in args.layouts:
            if layout == "packed_qkv":
                packed = torch.randn(1, height * width, 3, 24, 128, device="cuda", dtype=dtype)
                x = packed[:, :, 0].detach().requires_grad_()
            else:
                x = torch.randn(
                    1, height * width, 24, 128, device="cuda", dtype=dtype, requires_grad=True
                )
            grad = torch.randn_like(x)
            gold = reference(x, cos, sin)
            gold_dx = torch.autograd.grad(gold, x, grad)[0]
            for name, op in ops.items():
                result = op(x, cos, sin)
                dx = torch.autograd.grad(result, x, grad)[0]
                diff = [byte_differences(result, gold), byte_differences(dx, gold_dx)]
                if name in ("cuda", "triton", "flash_no_fma"):
                    assert diff == [0, 0], (name, layout, x.shape, diff)
                else:
                    tol = {torch.float32: 2e-6, torch.float16: 0.004, torch.bfloat16: 0.04}[dtype]
                    torch.testing.assert_close(result, gold, atol=tol, rtol=tol)
                    torch.testing.assert_close(dx, gold_dx, atol=tol, rtol=tol)
                correctness.append(
                    {
                        "backend": name,
                        "layout": layout,
                        "shape": list(x.shape),
                        "different_bytes_y_dx": diff,
                    }
                )
            del result, dx, gold, gold_dx
            for name, op in ops.items():
                for phase in ("forward", "forward_backward"):

                    def call(op=op, phase=phase, x=x, cos=cos, sin=sin, grad=grad):
                        if phase == "forward":
                            with torch.no_grad():
                                return op(x, cos, sin)
                        return torch.autograd.grad(op(x, cos, sin), x, grad)[0]

                    stats = measure(call, args)
                    rows.append(
                        {
                            "backend": name,
                            "phase": phase,
                            "layout": layout,
                            "shape": list(x.shape),
                            "input_stride": list(x.stride()),
                            **stats,
                        }
                    )
                    print(
                        f"{height}x{width} {layout:11s} {name:16s} {phase:16s} "
                        f"graph={stats['graph_median_ms']:.5f}ms "
                        f"eager={stats['eager_wall_median_ms']:.5f}ms",
                        flush=True,
                    )
    sources = [
        "csrc/cuda/rope/multi_axis_rope.cu",
        "rl_engine/backends/shared/triton/rope/_flash_attn_rotary.py",
        "rl_engine/backends/shared/triton/rope/multi_axis_rope.py",
        "rl_engine/backends/cuda/rope/multi_axis_rope.py",
        "rl_engine/ops/rope/_registration.py",
        "rl_engine/ops/rope/multi_axis_rope.py",
        "rl_engine/reference/rope/multi_axis_rope.py",
        "benchmarks/operators/rope/benchmark_multi_axis_rope.py",
    ]
    import triton

    evidence = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "os": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "dtype": args.dtype,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "timing": "CUDA graph ms/call and synchronized eager wall ms/call; warm allocation; "
        "excludes tables, compilation, and initial allocation of inputs",
        "memory": "extra peak allocated bytes excluding inputs/tables/grad; includes returned "
        "output or gradient; packed QKV backing storage allocated before timing",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "trials": args.trials,
        "graph_calls": args.graph_calls,
        "flash_baseline": {
            "included": bool(flash),
            "revision": FLASH_REVISION,
            "kernel_sha256_normalized": FLASH_SHA256,
            "layer_sha256_normalized": FLASH_LAYER_SHA256,
            "mode": "upstream rotary_kernel, apply_rotary and ApplyRotaryEmb, interleaved=True",
            "no_fma_change": "only add enable_fp_fusion=False at the original kernel launch",
        },
        "trace": {name: op.trace() for name, op in ours.items()},
        "qualification": "Exact bytes vs unfused FP32 gold for CUDA, Triton and flash_no_fma; "
        "declared tolerance for upstream complex and default FlashAttention",
        "correctness": correctness,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
