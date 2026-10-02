"""Compare current AITER and fixed CK launches on identical paged attention inputs."""

from __future__ import annotations

import json
import statistics
import time
from functools import partial

import torch

from examples.vime_rocm_attention_ablation.probe_paged_dispatch import cache_for, packed_forward
from rl_engine.kernels.ops.rocm.attention.fixed_paged_ck import fixed_paged_prefill
from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) * 1000 / 50
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(20):
            out = fn()
    measurements = []
    for _ in range(7):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        measurements.append(begin.elapsed_time(end) / 200)
    assert bool(torch.isfinite(out).all())
    return dict(
        eager_ms=eager_ms,
        graph_ms=statistics.median(measurements),
        graph_min_ms=min(measurements),
        graph_max_ms=max(measurements),
    )


@torch.inference_mode()
def main():
    for length, qlen, batch in [(4096, 4096, 1), (7168, 128, 4), (7168, 1, 4)]:
        torch.manual_seed(1234)
        q = torch.randn((batch * qlen, 8, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((batch, 2, length, 128), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        for layout in ("interleaved", "large"):
            kc, vc, table = cache_for(k, v, layout, True)
            cuq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qlen
            indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * table.shape[1]
            seqs = torch.full((batch,), length, device="cuda", dtype=torch.int32)
            core = StrictRocmAiterCKAttentionCore()
            original = core._mha_batch_prefill
            for name, entry in [
                ("aiter_dynamic", original),
                ("fixed_m128", partial(fixed_paged_prefill, tile_m=128)),
                ("fixed_m64", partial(fixed_paged_prefill, tile_m=64)),
            ]:
                core._mha_batch_prefill = entry
                call = partial(
                    packed_forward,
                    core,
                    q,
                    kc,
                    vc,
                    table,
                    seqs,
                    cuq,
                    indptr,
                    qlen,
                    length,
                    causal=qlen > 1,
                    lse=qlen > 1,
                )
                print(
                    json.dumps(
                        dict(
                            route=name,
                            length=length,
                            qlen=qlen,
                            batch=batch,
                            layout=layout,
                            **measure(call),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
