# SPDX-License-Identifier: Apache-2.0
"""Replay task-28 gate rows across batches and record raw bits and GPU timings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import torch

from rl_engine.ops.moe.router_gate import router_gate_projection


def checksum(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checksum(fixture["weight"]) != fixture["metadata"]["weight_sha256"]:
        raise ValueError("gate weight checksum does not match the replay metadata")
    weight = fixture["weight"].cuda().requires_grad_()
    generator = torch.Generator().manual_seed(28)
    row = torch.randn((1, weight.shape[1]), generator=generator, dtype=torch.bfloat16).cuda()
    version = fixture["metadata"]["revision"] + ":" + fixture["metadata"]["weight_sha256"]

    def project(x):
        return router_gate_projection(x, weight, model_version=version)

    baseline = project(row).logits
    cases = []
    for m in (1, 16, 17, 32, 65):
        x = torch.randn((m, weight.shape[1]), generator=generator, dtype=torch.bfloat16).cuda()
        positions = sorted({0, m // 2, m - 1})
        x[positions] = row
        actual = project(x)
        with torch.inference_mode():
            replay = project(x)
        raw = actual.logits.contiguous().view(torch.int32)
        ref = baseline.contiguous().view(torch.int32)
        mismatches = int((raw[positions] != ref).sum().item())
        mode_mismatches = int((raw != replay.logits.view(torch.int32)).sum().item())
        reference = x.double() @ weight.double().T
        max_error = (actual.logits.double() - reference).abs().max().item()
        torch.testing.assert_close(actual.logits.double(), reference, atol=2e-4, rtol=3e-5)
        for _ in range(10):
            project(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(30):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            project(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        cases.append(
            {
                "m": m,
                "input_sha256": checksum(x),
                "positions": positions,
                "row_bit_mismatches": mismatches,
                "mode_bit_mismatches": mode_mismatches,
                "logits_sha256": checksum(actual.logits),
                "max_abs_error_vs_fp64": max_error,
                "median_adapter_us": statistics.median(samples),
                "samples": 30,
                "warmup": 10,
                "execution": actual.execution,
            }
        )
    report = {
        "schema": "rlkernel.router_gate.task28.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture["metadata"],
        "activation_fixture": {
            "kind": "synthetic_bf16_normal",
            "seed": 28,
            "generator": "torch_cpu",
            "tokenizer": "not_used",
        },
        "cases": cases,
        "passed": all(c["row_bit_mismatches"] == c["mode_bit_mismatches"] == 0 for c in cases),
        "scope": "single-rank gate projection; checkpoint weights, synthetic activations",
        "not_validated": ["MI355X", "CUDA mm/CuTe", "whole model", "verl", "vime"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": [
                    {k: c[k] for k in ("m", "row_bit_mismatches", "median_adapter_us")}
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
