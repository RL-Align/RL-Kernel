import argparse
import time

import torch

from rl_engine.kernels.ops.pytorch.norm.rmsnorm_ref import rmsnorm_ref_custom
from rl_engine.kernels.ops.triton.rmsnorm_triton import rmsnorm_triton

try:
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    HAS_CUDA_EXT = True
except Exception:
    HAS_CUDA_EXT = False


def bench(fn, x, w, dy, warmup=20, iters=100):
    for _ in range(warmup):
        x.grad = None
        w.grad = None
        y = fn(x, w)
        y.backward(dy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        x.grad = None
        w.grad = None
        y = fn(x, w)
        y.backward(dy)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    T, H = args.T, args.H

    torch.manual_seed(0)
    x_base = torch.randn((T, H), device=device, dtype=dtype) * 0.2
    w_base = torch.randn((H,), device=device, dtype=dtype) * 0.2
    dy = torch.randn((T, H), device=device, dtype=dtype) * 0.2

    def make_inputs():
        return (
            x_base.detach().clone().requires_grad_(True),
            w_base.detach().clone().requires_grad_(True),
        )

    x, w = make_inputs()
    t_ref = bench(lambda a, b: rmsnorm_ref_custom(a, b), x, w, dy)
    print(f"pytorch ref : {t_ref:.4f} ms")

    x, w = make_inputs()
    t_tri = bench(lambda a, b: rmsnorm_triton(a, b), x, w, dy)
    print(f"triton      : {t_tri:.4f} ms | speedup vs ref: {t_ref / t_tri:.2f}x")

    if HAS_CUDA_EXT:
        x, w = make_inputs()
        t_cuda = bench(lambda a, b: rmsnorm_cuda(a, b), x, w, dy)
        print(f"cuda        : {t_cuda:.4f} ms | speedup vs ref: {t_ref / t_cuda:.2f}x")
    else:
        print("cuda        : skipped, extension is not built")


if __name__ == "__main__":
    main()
