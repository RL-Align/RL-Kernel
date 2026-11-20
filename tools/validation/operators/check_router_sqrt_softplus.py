# SPDX-License-Identifier: Apache-2.0
"""Task-29 score/VJP replay, independent FP64 oracle, and latency disclosure."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import torch

from rl_engine.backends.rocm.moe.router_sqrt_softplus import implementation
from rl_engine.ops.moe.router_sqrt_softplus import router_sqrt_softplus
from rl_engine.reference.moe.router_sqrt_softplus import sqrt_softplus_fp64, sqrt_softplus_vjp_fp64


def checksum(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def measure(call):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def relative_normal_error(actual, reference):
    mask = reference.abs() >= torch.finfo(torch.float32).tiny
    relative = (actual.double()[mask] - reference[mask]).abs() / reference[mask].abs()
    return relative.max().item() if relative.numel() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    generator = torch.Generator().manual_seed(29)
    maximum = torch.finfo(torch.float32).max
    row_cpu = torch.rand(1, 256, generator=generator) * 320 - 160
    edge = [
        -maximum,
        -1000.0,
        -400.0,
        -300.0,
        -220.0,
        -200.0,
        -180.0,
        -160.0,
        -120.0,
        -100.0,
        -80.0,
        -20.0,
        -16.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        20.0,
        80.0,
        1000.0,
        maximum,
    ]
    row_cpu[0, : len(edge)] = torch.tensor(edge)
    grad_cpu = torch.randn(1, 256, generator=generator)
    grad_cpu[0, 3] = maximum
    row = row_cpu.cuda().requires_grad_()
    grad_row = grad_cpu.cuda()

    def project(x):
        return router_sqrt_softplus(x, model_version="task29-synthetic-arithmetic-v1")

    baseline = project(row)
    baseline_dx = torch.autograd.grad(baseline.scores, row, grad_row)[0]
    api, _ = implementation()
    cases = []
    for m in (1, 16, 17, 32, 65, 257):
        inputs = torch.rand(m, 256, generator=generator) * 320 - 160
        grads = torch.randn(m, 256, generator=generator)
        positions = sorted({0, m // 2, m - 1})
        inputs[positions] = row_cpu
        grads[positions] = grad_cpu
        x = inputs.cuda().requires_grad_()
        g = grads.cuda()
        actual = project(x)
        dx = torch.autograd.grad(actual.scores, x, g)[0]
        with torch.inference_mode():
            replay = project(x)
        score_cpu = actual.scores.detach().cpu()
        dx_cpu = dx.cpu()
        reference = sqrt_softplus_fp64(inputs)
        reference_dx = sqrt_softplus_vjp_fp64(inputs, grads)
        torch.testing.assert_close(score_cpu.double(), reference, rtol=5e-7, atol=2.0**-146)
        torch.testing.assert_close(dx_cpu.double(), reference_dx, rtol=2e-5, atol=2.0**-146)
        score_bits = actual.scores.detach().contiguous().view(torch.int32)
        dx_bits = dx.contiguous().view(torch.int32)
        score_mismatches = int(
            (score_bits[positions] != baseline.scores.detach().view(torch.int32)).sum()
        )
        dx_mismatches = int((dx_bits[positions] != baseline_dx.view(torch.int32)).sum())
        mode_mismatches = int((score_bits != replay.scores.view(torch.int32)).sum())
        cases.append(
            {
                "m": m,
                "positions": positions,
                "input_sha256": checksum(x),
                "upstream_sha256": checksum(g),
                "scores_sha256": checksum(actual.scores),
                "vjp_sha256": checksum(dx),
                "row_score_bit_mismatches": score_mismatches,
                "row_vjp_bit_mismatches": dx_mismatches,
                "mode_bit_mismatches": mode_mismatches,
                "max_normal_score_relative_error_vs_fp64": relative_normal_error(
                    score_cpu, reference
                ),
                "max_normal_vjp_relative_error_vs_fp64": relative_normal_error(
                    dx_cpu, reference_dx
                ),
                "median_adapter_forward_us": measure(lambda x=x: project(x)),
                "median_aiter_forward_us": measure(lambda x=x: api.router_sqrt_softplus_fwd(x)),
                "median_aiter_backward_us": measure(
                    lambda x=x, scores=actual.scores, g=g: api.router_sqrt_softplus_bwd(
                        x, scores, g
                    )
                ),
                "execution": actual.execution,
            }
        )
    report = {
        "schema": "rlkernel.router_sqrt_softplus.task29.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "kind": "synthetic_fp32_logits_and_upstream",
            "cpu_seed": 29,
            "experts": 256,
            "model_weights_used": False,
            "tokenizer_used": False,
            "row_sha256": checksum(row),
            "grad_row_sha256": checksum(grad_row),
            "edge_values": edge,
        },
        "accuracy": {
            "score_rtol": 5e-7,
            "vjp_rtol": 2e-5,
            "subnormal_atol": 2.0**-146,
            "oracle": "independent CPU FP64 logaddexp/sqrt and analytic VJP",
        },
        "timing": {
            "warmup": 10,
            "samples": 30,
            "method": "GPU events around Python calls",
            "adapter_includes": "validation, finite checks, synchronization, provenance, autograd",
        },
        "cases": cases,
        "passed": all(
            c["row_score_bit_mismatches"]
            == c["row_vjp_bit_mismatches"]
            == c["mode_bit_mismatches"]
            == 0
            for c in cases
        ),
        "not_validated": ["MI355X", "NVIDIA", "Graph", "full model", "framework integration"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": [
                    {
                        k: c[k]
                        for k in (
                            "m",
                            "row_score_bit_mismatches",
                            "row_vjp_bit_mismatches",
                            "median_adapter_forward_us",
                            "median_aiter_forward_us",
                        )
                    }
                    for c in cases
                ],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
