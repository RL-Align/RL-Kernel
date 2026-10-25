"""Compare raw Triton launch geometries using preallocated buffers and CUDA graphs."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from rl_engine.backends.shared.triton.rope.qknorm_scaled_rope import _kernel, _backward_kernel
from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope_forward, _dispatch_backward
from probe import bits_equal


def timing(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(50):
            fn()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 50)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.inputs, map_location="cuda", weights_only=True)
    rows = []
    for tokens in (1, 128, 257, 2048, 8192):
        if tokens == 257:
            q, k, gq, gk = (data[n] for n in ("q", "k", "gq", "gk"))
            p = data["p"]
        else:
            gen = torch.Generator(device="cuda").manual_seed(501)
            q = torch.randn(tokens, 32, 128, device="cuda", generator=gen).bfloat16()
            k = torch.randn(tokens, 2, 128, device="cuda", generator=gen).bfloat16()
            gq = torch.randn(q.shape, device="cuda", generator=gen).bfloat16()
            gk = torch.randn(k.shape, device="cuda", generator=gen).bfloat16()
            p = torch.arange(tokens, device="cuda", dtype=torch.int64)
        c, s = data["c"], data["s"]
        for local in (False, True):
            ref = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend="cuda")
            ref_grad = _dispatch_backward(
                q, k, gq, gk, *ref[2:], p, c, s, is_local=local, backend="cuda"
            )
            oq, ok, rq, rk = (torch.empty_like(x) for x in ref)
            dq, dk = torch.empty_like(q), torch.empty_like(k)
            for warps in (4, 1, 2):
                options = dict(num_warps=warps, enable_fp_fusion=False, enable_reflect_ftz=False)
                forward = lambda: _kernel[(tokens * 34,)](
                    q, k, p, c, s, oq, ok, rq, rk, 32, 2, local, **options
                )
                backward = lambda: _backward_kernel[(tokens * 34,)](
                    q, k, gq, gk, rq, rk, p, c, s, dq, dk, 32, 2, local, **options
                )
                forward()
                backward()
                for a, b in zip((oq, ok, rq, rk, dq, dk), (*ref, *ref_grad)):
                    assert bits_equal(a, b), (tokens, local, warps)
                rows.append(
                    {
                        "tokens": tokens,
                        "local": local,
                        "warps": warps,
                        "bitwise_equal": True,
                        "forward": timing(forward),
                        "backward": timing(backward),
                    }
                )
    args.out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "metric": "CUDA graph replay device time per preallocated raw kernel launch",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
