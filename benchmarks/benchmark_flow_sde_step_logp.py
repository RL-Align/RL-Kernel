# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image SDE benchmark; explicitly invoked only, never a test side effect.

Compares standard PyTorch mean, canonical PyTorch reference, native CUDA.
Reports wall/device latency and incremental allocated memory, actual backend,
hardware/runtime, warmup/iterations and exact workload. No model download.
"""

import argparse
import json
import math
import platform
import time

import torch

from rl_engine.kernels.ops.cuda.diffusion.flow_sde_step_logp import CUDAFlowSDEStepLogpOp
from rl_engine.kernels.ops.pytorch.diffusion.flow_sde_step_logp import (
    NativeFlowSDEStepLogpOp,
    prepare_inputs,
    reference_coefficients,
)


def torch_mean_step(**inputs):
    """Production-style arithmetic; torch.mean has no canonical tree guarantee."""
    x, v = inputs["sample"], inputs["model_output"]
    params, noise = prepare_inputs(**inputs)
    coeff = reference_coefficients(params)
    a, b, tau, _std = (value[:, None, None] for value in coeff.unbind(1))
    mean = x.float() * a + v.float() * b
    target = mean + tau * noise
    density = -(target.detach() - mean).square() / (2 * tau.square())
    density = density - tau.log() - math.log(math.sqrt(2 * math.pi))
    return target, density.flatten(1).mean(1)


def measure(op, data, backward, warmup, iterations):
    def run():
        if backward:
            call = dict(data)
            call["model_output"] = data["model_output"].detach().requires_grad_(True)
            result = op(**call)
            torch.autograd.grad(result[1].sum(), call["model_output"])
        else:
            with torch.no_grad():
                op(**data)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
    return {
        "wall_ms": wall_ms,
        "device_ms": start.elapsed_time(end) / iterations,
        "peak_extra_mb": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+", default=[7, 4096, 6889, 6032])
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cuda = CUDAFlowSDEStepLogpOp()  # Missing CUDA kernel must fail, never fallback.
    backends = (
        ("torch-mean", torch_mean_step),
        ("canonical-reference", NativeFlowSDEStepLogpOp()),
        ("cuda", cuda),
    )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "workload": vars(args),
                "cuda_trace": cuda.execution_trace(),
                "timing_note": (
                    "wall includes metadata validation/copies; device includes launch gaps"
                ),
            }
        )
    )
    generator = torch.Generator(device="cuda").manual_seed(386)
    for tokens in args.tokens:
        shape = (args.batch, tokens, 64)
        data = {
            "sample": torch.randn(shape, device="cuda", dtype=dtype, generator=generator),
            "model_output": torch.randn(shape, device="cuda", dtype=dtype, generator=generator),
            "noise": torch.randn(shape, device="cuda", generator=generator),
            "sigma": 0.75,
            "sigma_next": 0.5,
            "sigma_max": 0.98,
            "noise_level": 0.7,
        }
        for name, op in backends:
            result = measure(op, data, args.backward, args.warmup, args.iterations)
            print(json.dumps({"backend": name, "shape": shape, **result}))


if __name__ == "__main__":
    main()
