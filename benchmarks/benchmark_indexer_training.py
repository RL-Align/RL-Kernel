"""Checked auxiliary/VJP timing, with quantization and collection costs visible."""

import argparse
from functools import partial
import json
from pathlib import Path

import torch

from benchmarks.benchmark_joint_attention import measure
from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.models.deepseek_v4.indexer_training import (
    IndexerAuxiliary,
    IndexerAuxMetadata,
    canonical_indexer_auxiliary_loss,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session


def train_step(op, q, k, w, tq, tk, meta):
    q.master.grad = k.master.grad = w.grad = None
    with canonical_backward_session() as session:
        result = op.autograd(q, k, w, tq, tk, metadata=meta, version="bench")
        canonical_indexer_auxiliary_loss([result]).backward()
        session.validate_complete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(26)
    records = []
    providers = {}
    for precision in ("mxfp4", "fp8"):
        for m, selected in ((1, 8), (16, 32), (64, 128)):
            c = IndexerAuxContract(
                m,
                0.01,
                64,
                128,
                512,
                selected_capacity=selected,
                max_queries=m,
                max_keys=selected,
                precision=precision,
            )
            op = IndexerAuxiliary(c, layer_id="bench")
            qm = torch.randn(m, 64, 128, device="cuda").bfloat16().requires_grad_()
            km = torch.randn(selected, 128, device="cuda").bfloat16().requires_grad_()
            w = torch.randn(m, 64, device="cuda", requires_grad=True)
            tq = torch.randn(m, 128, 512, device="cuda").bfloat16()
            tk = torch.randn(selected, 512, device="cuda").bfloat16()
            qids = torch.tensor([[7, 4 * (selected + i) - 1] for i in range(m)], device="cuda")
            kids = torch.tensor([[7, g] for g in range(selected)], device="cuda")
            topk = torch.arange(selected, device="cuda")[None].repeat(m, 1)
            meta = IndexerAuxMetadata.bind(c, qids, kids, topk, version="bench")
            q = op.prepare_operand(qm, role="q", version="bench")
            k = op.prepare_operand(km, role="k", version="bench")
            forward = partial(op.forward_record, q, k, w, tq, tk, metadata=meta, version="bench")
            saved = forward()
            dy = torch.ones(m, device="cuda")
            timings = {
                "prepare_query_operand": measure(
                    partial(op.prepare_operand, qm, role="q", version="bench"), args.repeats
                ),
                "checked_forward_record": measure(forward, args.repeats),
                "row_local_vjp": measure(
                    partial(op.backend.backward_parts, saved, dy), args.repeats
                ),
                "canonical_forward_backward": measure(
                    partial(train_step, op, q, k, w, tq, tk, meta), args.repeats
                ),
            }
            records.append(
                dict(
                    precision=precision,
                    queries=m,
                    selected=selected,
                    workspace_estimate=op.workspace_estimate(m, selected),
                    timings=timings,
                    implementation=op.implementation,
                )
            )
            providers[precision] = op.backend.backward_provenance
            print(json.dumps(records[-1]), flush=True)
    Path(args.output).write_text(json.dumps(dict(records=records, providers=providers), indent=2))


if __name__ == "__main__":
    main()
