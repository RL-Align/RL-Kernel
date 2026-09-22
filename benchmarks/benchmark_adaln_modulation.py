# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare AdaLN backends on Qwen-Image image-token lengths.

Run with --real for S=4096/6889/6032, H=3072. These S values are derived
from 8x VAE downsampling and 2x2 latent packing, not specified by #386.
"""

import argparse
import platform
import statistics

import torch

from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import NativeAdaLNModulationOp


def _bench(op, x, modulation, *, backward, warmup, repeat):
    def run():
        x.grad = modulation.grad = None
        if backward:
            y, gate = op(x, modulation)
            torch.autograd.backward((y, gate), (torch.ones_like(y), torch.ones_like(gate)))
        else:
            with torch.no_grad():
                op(x, modulation)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="run all three Qwen-Image resolutions")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    backends = [("pytorch", NativeAdaLNModulationOp()), ("triton", TritonAdaLNModulationOp())]
    try:
        backends.append(("cuda", CudaAdaLNModulationOp()))
    except RuntimeError:
        print("cuda extension unavailable: explicit skip")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(
        f"GPU={torch.cuda.get_device_name()} cuda_runtime={torch.version.cuda} "
        f"torch={torch.__version__} python={platform.python_version()} "
        f"dtype={args.dtype} warmup={args.warmup} repeat={args.repeat}"
    )
    sizes = (4096, 6889, 6032) if args.real else (64,)
    for seq in sizes:
        x = torch.randn(1, seq, 3072, device="cuda", dtype=dtype, requires_grad=True)
        modulation = torch.randn(1, 9216, device="cuda", dtype=dtype, requires_grad=True)
        for name, op in backends:
            for backward in (False, True):
                ms, peak = _bench(
                    op, x, modulation, backward=backward, warmup=args.warmup,
                    repeat=args.repeat,
                )
                print(
                    f"S={seq} backend={name} backward={backward} "
                    f"median_ms={ms:.3f} peak_MB={peak / 2**20:.1f}"
                )


if __name__ == "__main__":
    main()
