"""Native fixed-record VJP and canonical autograd timings, without hidden setup."""

import argparse
from functools import partial
import json
from pathlib import Path

import torch

from benchmarks.benchmark_joint_attention import measure
from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.contracts.operators.joint_attention import JointAttentionContract
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata
from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd
from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session


def step(adapter, q, kv, sink, meta, dy):
    q.grad = kv.grad = sink.grad = None
    with canonical_backward_session() as session:
        out = adapter(q, kv, sink, metadata=meta, version="bench")
        out.output.backward(dy)
        session.validate_complete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    backend = AiterJointAttentionBackend()
    torch.manual_seed(25)
    records = []
    for m, h, d, topk, window in (
        (7, 13, 128, 2, 4),
        (7, 64, 512, 2, 4),
        (65, 13, 128, 2, 4),
        (1, 128, 512, 512, 128),
    ):
        start = max(64, topk * 4 - 1)
        qkeys = [[0, p] for p in range(start, start + m)]
        kvkeys = [[0, 0, i * 4 + 3] for i in range(topk)]
        kvkeys.extend([0, 1, p] for p in range(start - window + 1, start + m))
        c = JointAttentionContract(h, d, m, len(kvkeys), topk, window)
        meta = JointMetadata.bind(
            c,
            torch.tensor(qkeys, device="cuda"),
            torch.tensor(kvkeys, device="cuda"),
            torch.arange(topk, dtype=torch.int32, device="cuda")[None].repeat(m, 1),
            version="bench",
        )
        q = torch.randn(m, h, d, device="cuda").bfloat16().requires_grad_()
        kv = torch.randn(len(kvkeys), d, device="cuda").bfloat16().requires_grad_()
        sink = torch.randn(h, device="cuda", requires_grad=True)
        dy = torch.randn_like(q)
        plan = JointAttentionPlan(c, backend=backend)
        saved = plan.forward_record(q, kv, sink, metadata=meta, version="bench", mode="train")
        backward = JointBackwardPlan(plan, saved)
        backward.capture()
        adapter = JointAttentionAutograd(plan, layer_id="bench", backend=backward.backend)
        timings = {
            "native_pipeline": measure(backward._launch, args.repeats),
            "graph_core": measure(backward._graph.replay, args.repeats),
            "checked_eager_vjp": measure(partial(backward.backward, dy), args.repeats),
            "checked_graph_vjp": measure(
                partial(backward.backward, dy, use_graph=True), args.repeats
            ),
            "canonical_forward_backward": measure(
                partial(step, adapter, q, kv, sink, meta, dy), args.repeats
            ),
        }
        records.append(
            dict(
                m=m,
                h=h,
                d=d,
                selected=c.selected_capacity,
                native_workspace=backward.backend.workspace_bytes(
                    m, h, d, c.selected_capacity, len(kvkeys)
                ),
                timings=timings,
                backward_implementation=adapter.backward_implementation,
            )
        )
        print(json.dumps(records[-1]), flush=True)
    Path(args.output).write_text(
        json.dumps(dict(records=records, provider=backward.backend.provenance), indent=2)
    )


if __name__ == "__main__":
    main()
