# SPDX-License-Identifier: Apache-2.0
"""Localize a failed block-0 gradient gate without changing its numerical contract."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from rl_engine.backends.extension import _C
from rl_engine.validation.models import h3_block as block
from rl_engine.validation.models.h3_weights import load_h3_block_weights
from tools.validation.models.h3_full_model import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-job", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("diagnostic requires clean source")
    ops = block.CandidateOps()
    weights = load_h3_block_weights("cuda")
    ctx = block.make_block_context(weights, layout="tiny", seed=0, batch=1)
    upstream = (
        torch.randn((1, ctx["seq_len"], block.HIDDEN), generator=torch.Generator().manual_seed(11))
        .bfloat16()
        .cuda()
    )
    report = dict(
        status="diagnostic_only_not_acceptance",
        failed_job=args.failed_job,
        job_id=os.environ.get("SLURM_JOB_ID"),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        extension_sha256=digest(Path(_C.__file__)),
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        backends=ops.backends(),
        default_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        leaves={},
        nodes=[],
    )
    # Keep FP32 arithmetic strict through backward as well as forward.
    with block.strict_fp32_matmul():
        golden, golden_nodes = block._backward("golden", ctx, None, upstream)
        provider, provider_nodes = block._backward("provider", ctx, None, upstream)
        candidate, candidate_nodes = block._backward("candidate", ctx, ops, upstream)
    for name in golden:
        report["leaves"][name] = dict(
            candidate=block.rel_err(candidate[name], golden[name]),
            provider=block.rel_err(provider[name], golden[name]),
        )
    for node in reversed(block.NODES):
        report["nodes"].append(
            dict(
                node=node.name,
                errors=[
                    dict(candidate=block.rel_err(c, g), provider=block.rel_err(p, g))
                    for g, p, c in zip(
                        golden_nodes[node.name],
                        provider_nodes[node.name],
                        candidate_nodes[node.name],
                        strict=True,
                    )
                ],
            )
        )
    # Same input and upstream gradient isolate the RMSNorm implementation from
    # error accumulated through other candidate nodes.
    with torch.no_grad():
        values = block.run_graph("provider", ctx)
    isolated = {}
    for name, source, weight_name in (
        ("q_norm", "q_proj", "attn.norm_q.weight"),
        ("k_norm", "k_proj", "attn.norm_k.weight"),
    ):
        grads = {}
        node = next(n for n in block.NODES if n.name == name)
        for mode in ("golden", "provider", "candidate"):
            x = values[source].detach().clone()
            w = weights[block.BLOCK + "." + weight_name].detach().clone()
            if mode == "golden":
                x, w = x.float(), w.float()
            x.requires_grad_()
            w.requires_grad_()
            local = {**ctx, "weights": {**weights, block.BLOCK + "." + weight_name: w}}
            y = (
                node.candidate(ops, local, x)
                if mode == "candidate"
                else getattr(node, mode)(local, x)
            )
            torch.autograd.backward(y, provider_nodes[name][0].to(y.dtype))
            grads[mode] = w.grad
        isolated[name] = dict(
            candidate=block.rel_err(grads["candidate"], grads["golden"]),
            provider=block.rel_err(grads["provider"], grads["golden"]),
        )
    report["isolated_norm_weight_gradients"] = isolated
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
