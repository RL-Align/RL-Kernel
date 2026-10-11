import json
import sys

import torch

from rl_engine.kernels.ops.pytorch.ffn.ffn import qwen3_ffn

H, INTERMEDIATE = 4096, 12288
WARMUP = 5
FW_ITERS = 20
FB_ITERS = 10


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "median_ms": float(values.median()),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": samples,
    }


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(2026)
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": {"hidden": H, "intermediate": INTERMEDIATE, "dtype": "bfloat16"},
        "warmup": WARMUP,
        "forward_iters": FW_ITERS,
        "forward_backward_iters": FB_ITERS,
        "rows": [],
    }
    for tokens in map(int, sys.argv[1:]):
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gate = torch.randn(INTERMEDIATE, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn(INTERMEDIATE, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        down = torch.randn(H, INTERMEDIATE, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        for deterministic in (True, False):

            def forward():
                return qwen3_ffn(x, gate, up, down, deterministic=deterministic)

            def forward_backward():
                y = forward()
                torch.autograd.grad(y, (x, gate, up, down), dout)

            row = {
                "tokens": tokens,
                "mode": "det" if deterministic else "prod",
                "forward": timed(forward, WARMUP, FW_ITERS),
                "forward_backward": timed(forward_backward, WARMUP, FB_ITERS),
            }
            result["rows"].append(row)
            print(
                f"tokens={tokens} mode={row['mode']} "
                f"forward_median={row['forward']['median_ms']:.4f} "
                f"fwd_bwd_median={row['forward_backward']['median_ms']:.4f}",
                flush=True,
            )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
