"""Separate the upstream forward, captured core, and checked record overhead."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch

from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.contracts.operators.joint_attention import JointAttentionContract
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata


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
    event = start.elapsed_time(end) * 1000 / repeats
    begin = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return dict(event_us=event, wall_us=(time.perf_counter() - begin) * 1e6 / repeats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    backend = AiterJointAttentionBackend()
    torch.manual_seed(24)
    records = []
    for m, h, d, topk, window in (
        (1, 8, 128, 2, 4),
        (32, 64, 512, 2, 4),
        (128, 64, 512, 2, 4),
        (1, 128, 512, 512, 128),
    ):
        pos = topk * 4 - 1
        kvkeys, selected = [], []
        for request in range(m):
            start = len(kvkeys)
            kvkeys.extend((request, 0, i * 4 + 3) for i in range(topk))
            kvkeys.extend((request, 1, i) for i in range(pos - window + 1, pos + 1))
            selected.append(list(range(start, start + topk)))
        c = JointAttentionContract(h, d, m, len(kvkeys), topk, window)
        keys = torch.tensor([[r, pos] for r in range(m)], device="cuda", dtype=torch.int64)
        meta = JointMetadata.bind(
            c,
            keys,
            torch.tensor(kvkeys, device="cuda"),
            torch.tensor(selected, dtype=torch.int32, device="cuda"),
            version="bench",
        )
        plan = JointAttentionPlan(c, backend=backend)
        q = torch.randn(m, h, d, device="cuda").bfloat16()
        kv = torch.randn(len(kvkeys), d, device="cuda").bfloat16()
        sink = torch.randn(h, device="cuda")
        forward = partial(
            plan.forward_record, q, kv, sink, metadata=meta, version="bench", mode="decode"
        )
        forward()
        plan.capture()
        native = partial(
            backend.forward_into,
            plan._q,
            plan._kv,
            plan._sink,
            plan._indices,
            plan._out,
            plan._lse,
            c,
        )
        timing = {
            "upstream_forward": measure(native, args.repeats),
            "captured_core": measure(plan._graph.replay, args.repeats),
            "eager_forward_record": measure(forward, args.repeats),
            "graph_forward_record": measure(partial(forward, use_graph=True), args.repeats),
        }
        records.append(
            dict(
                m=m,
                heads=h,
                dim=d,
                selected=c.selected_capacity,
                workspace=c.workspace_bytes,
                timing=timing,
                implementation=plan.implementation,
            )
        )
        print(json.dumps(records[-1]), flush=True)
    Path(args.output).write_text(
        json.dumps(dict(records=records, provider=backend.provenance), indent=2)
    )


if __name__ == "__main__":
    main()
