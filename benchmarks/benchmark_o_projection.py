"""MI300X task-23 timing. Writes only to the caller's explicit output path."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch

from rl_engine.backends.rocm.gemm.o_projection import AiterOProjectionBackend
from rl_engine.contracts.operators.o_projection import OProjectionContract
from rl_engine.models.deepseek_v4.o_projection import OProjection, OProjectionRows


def measure(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    device_us = start.elapsed_time(end) * 1000 / repeats
    begin = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return {"event_us": device_us, "wall_us": (time.perf_counter() - begin) * 1e6 / repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    backend = AiterOProjectionBackend()
    torch.manual_seed(23)
    records = []
    for heads, dim, rank, hidden in ((16, 128, 128, 128), (64, 512, 1024, 4096)):
        for m in (1, 32, 256):
            c = OProjectionContract(heads, dim, 64, rank, hidden)
            op = OProjection(c, layer_id="benchmark23", backend=backend)
            a = torch.randn(c.local_groups, rank, c.group_width, device="cuda").bfloat16()
            b = torch.randn(hidden, c.local_groups * rank, device="cuda").bfloat16()
            w = op.prepare_weights(a, b, version="bench")
            x = torch.randn(m, heads, dim, device="cuda").bfloat16()
            keys = torch.stack(
                (torch.zeros(m, device="cuda", dtype=torch.int64), torch.arange(m, device="cuda")),
                1,
            )
            angle = torch.arange(m, device="cuda")[:, None].float() * (
                10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
            )
            rows = OProjectionRows.bind(
                c, keys, angle.cos(), angle.sin(), head_ids=c.head_ids, version="bench"
            )
            inv = backend.inverse_rope(x, rows.cos, rows.sin)
            grouped = inv.reshape(m, c.local_groups, c.group_width).transpose(0, 1).contiguous()
            z = backend.grouped(grouped, a).transpose(0, 1).contiguous().flatten(1)
            with torch.no_grad():
                timings = {
                    "inverse_rope": measure(
                        partial(backend.inverse_rope, x, rows.cos, rows.sin), args.repeats
                    ),
                    "grouped_a": measure(partial(backend.grouped, grouped, a), args.repeats),
                    "b": measure(
                        partial(backend.linear, z, b, output_dtype=torch.float32, role="o_b"),
                        args.repeats,
                    ),
                    "full_bound_operator": measure(partial(op, x, w, rows=rows), args.repeats),
                }
            records.append(
                {
                    "rows": m,
                    "heads": heads,
                    "head_dim": dim,
                    "rank": rank,
                    "hidden": hidden,
                    "timings": timings,
                    "implementation": op.implementation_fingerprint,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    Path(args.output).write_text(
        json.dumps({"records": records, "provider": backend.provenance}, indent=2)
    )


if __name__ == "__main__":
    main()
