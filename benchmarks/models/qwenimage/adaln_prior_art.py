#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native, reused fixed GEMV, and an optional existing QKV cuBLASLt build."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.models.qwenimage.adaln_evidence import equal_bytes, load_weights, source_state
from benchmarks.models.qwenimage.streams import STREAMS
from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import projection_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fast-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(29)
    fast = None
    if args.fast_extension:
        spec = importlib.util.spec_from_file_location("qwenimage_prior_art._C", args.fast_extension)
        fast = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast)
    report = {"source": source_state(), "cases": [], "tf32": torch.backends.cuda.matmul.allow_tf32}
    if args.fast_extension:
        report["existing_qkv_extension_sha256"] = hashlib.sha256(
            args.fast_extension.read_bytes()
        ).hexdigest()
    report["environment"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    for label, stem in STREAMS:
        weights, _ = load_weights(args.model, stem)
        for dtype in (torch.bfloat16, torch.float32):
            w, b = [weights[k].to(device="cuda", dtype=dtype) for k in ("weight", "bias")]
            t = torch.randn(64, 3072, device="cuda", dtype=dtype)
            funcs = {
                "native": lambda z: F.linear(F.silu(z), w, b),
                "fixed_gemv": lambda z: projection_forward(z, w, b)[0],
            }
            if fast:
                funcs["existing_qkv_cublaslt"] = lambda z: (
                    fast.qwenimage_fast_gemm_fp32_cuda(F.silu(z).contiguous(), w) + b.float()
                ).to(dtype)
            timings = {name: [] for name in funcs}
            orders = []
            for fn in funcs.values():
                for _ in range(4):
                    fn(t[:1])
            for iteration in range(15):
                order = list(funcs)
                if iteration % 2:
                    order.reverse()
                orders.append(order)
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    funcs[name](t[:1])
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end))
            for name, fn in funcs.items():
                base = fn(t[:1])
                violations = []
                for batch in [1, 2, 3, 4, 7, 8, 9, 16, 33, 64]:
                    for pos in sorted(set([0, batch // 2, batch - 1])):
                        z = t[:batch].clone()
                        z[pos] = t[0]
                        y = fn(z)[pos : pos + 1]
                        if not equal_bytes(y, base):
                            violations.append(
                                {
                                    "batch": batch,
                                    "position": pos,
                                    "differing_bytes": (
                                        y.contiguous().view(torch.uint8)
                                        != base.contiguous().view(torch.uint8)
                                    )
                                    .sum()
                                    .item(),
                                }
                            )
                report["cases"].append(
                    {
                        "stream": label,
                        "dtype": str(dtype),
                        "implementation": name,
                        "batch_invariance_violations": violations,
                        "forward_b1_ms": timings[name],
                        "timing_order": orders,
                    }
                )
                print(label, dtype, name, "violations", len(violations), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
