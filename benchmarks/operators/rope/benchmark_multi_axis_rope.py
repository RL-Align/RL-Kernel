# SPDX-License-Identifier: Apache-2.0
"""Eager forward/VJP timings for the Qwen-Image RoPE reference geometries."""

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from rl_engine.ops.rope.multi_axis_rope import (
    build_multi_axis_cos_sin,
    get_multi_axis_rope_op,
    qwen_image_positions,
)


def elapsed_ms(fn, args):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.trials):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.iters)
    return statistics.median(samples)


def different_bytes(actual, expected):
    return int(
        (actual.contiguous().view(torch.uint8) != expected.contiguous().view(torch.uint8))
        .sum()
        .item()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cuda", "triton", "reference"),
        default=("cuda", "triton", "reference"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.trials < 1:
        parser.error("warmup must be nonnegative; iters and trials must be positive")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("a real NVIDIA CUDA GPU is required")
    torch.manual_seed(386)
    dtype = getattr(torch, args.dtype)
    reference = get_multi_axis_rope_op("reference")
    candidates = {name: get_multi_axis_rope_op(name, device="cuda") for name in args.backends}
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iters,
        "trials": args.trials,
        "timing": "median CUDA-event elapsed ms/call; eager dispatch and submission gaps included",
        "excluded": "table construction, input allocation and first-use compilation",
        "limits": "shared GPU, unpinned clocks; operator timings do not establish model speedup",
        "trace": {name: op.trace() for name, op in candidates.items()},
        "results": [],
    }
    for height, width in ((2, 3), (64, 64), (83, 83), (104, 58)):
        positions, _ = qwen_image_positions(height, width, 0)
        cos, sin = build_multi_axis_cos_sin(positions, device="cuda")
        seq = height * width
        for layout in ("contiguous", "packed_qkv"):
            if layout == "packed_qkv":
                backing = torch.randn(1, seq, 3, 24, 128, device="cuda", dtype=dtype)
                x = backing[:, :, 0].detach().requires_grad_()
            else:
                x = torch.randn(1, seq, 24, 128, device="cuda", dtype=dtype, requires_grad=True)
            grad = torch.randn_like(x)
            expected = reference(x.detach(), cos, sin)
            expected_dx = reference(grad, cos, -sin)
            for name, op in candidates.items():
                y = op(x, cos, sin)
                dx = torch.autograd.grad(y, x, grad)[0]
                differences = [different_bytes(y, expected), different_bytes(dx, expected_dx)]
                if name != "reference":
                    assert differences == [0, 0], (name, layout, seq, differences)
                else:
                    # Generic slice-autograd can normalize signed zeros in the reference VJP.
                    torch.testing.assert_close(dx, expected_dx, atol=0, rtol=0)

                def forward():
                    with torch.no_grad():
                        return op(x, cos, sin)

                def forward_backward():
                    return torch.autograd.grad(op(x, cos, sin), x, grad)[0]

                row = {
                    "backend": name,
                    "grid": [height, width],
                    "layout": layout,
                    "shape": list(x.shape),
                    "stride": list(x.stride()),
                    "different_bytes_y_dx": differences,
                    "forward_ms": elapsed_ms(forward, args),
                    "forward_backward_ms": elapsed_ms(forward_backward, args),
                }
                report["results"].append(row)
                print(
                    f"{height}x{width} {layout:10s} {name:9s} "
                    f"fwd={row['forward_ms']:.5f}ms fwd+bwd={row['forward_backward_ms']:.5f}ms"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
