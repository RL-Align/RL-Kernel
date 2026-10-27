# SPDX-License-Identifier: Apache-2.0
"""Task-31 independent exact-order/payload replay and performance disclosure."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from rl_engine.backends.rocm.moe.stable_topk6 import implementation
from rl_engine.ops.moe.stable_topk6 import stable_topk6, tensor_sha256
from rl_engine.reference.moe.stable_topk6 import stable_topk6_reference


def bits(x):
    return x.detach().contiguous().view(torch.int32)


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


def fixture_row(pattern, experts, generator):
    row = torch.randint(-10, 11, (1, experts), generator=generator).float()
    if pattern == "ties":
        row.fill_(1.0)
    if pattern == "near_scores":
        row.fill_(-1.0)
        v = torch.tensor(1.0)
        for expert in range(experts):
            row[0, expert] = v
            v = torch.nextafter(v, torch.tensor(torch.inf))
    if pattern == "zero_subnormal":
        row.fill_(-1.0)
        raw = torch.tensor(
            [0, -2147483648, 1, -2147483647, 2, -2147483646, 3, -2147483645, 4, -2147483644],
            dtype=torch.int32,
        )
        row[0, : min(10, experts)] = raw.view(torch.float32)[: min(10, experts)]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    generator = torch.Generator().manual_seed(31)
    api, _, _, _ = implementation()
    cases = []
    configurations = [
        (256, pattern, m)
        for pattern in ("ties", "random_ties", "near_scores", "zero_subnormal")
        for m in (1, 16, 17, 32, 65, 257)
    ]
    configurations.extend((e, "random_ties", 17) for e in (6, 129, 257, 1025))
    rows = {}
    maps = {}
    baseline = {}
    grad_row = torch.tensor([[-0.0, 1.0, -1.0, torch.finfo(torch.float32).tiny, 3.0, -3.0]])
    for experts, pattern, m in configurations:
        key = (experts, pattern)
        if key not in rows:
            rows[key] = fixture_row(pattern, experts, generator)
            maps[key] = torch.randperm(experts, generator=generator)
        row_cpu = rows[key]
        ids_cpu = maps[key]
        ids = ids_cpu.cuda()

        def project(x, ids=ids, experts=experts):
            return stable_topk6(
                x,
                model_version="task31-synthetic-v1",
                global_expert_ids=ids,
                global_expert_count=experts,
            )

        if key not in baseline:
            bx = row_cpu.cuda().requires_grad_()
            b = project(bx)
            bdx = torch.autograd.grad(b.selected_scores, bx, grad_row.cuda())[0]
            baseline[key] = (b, bdx)
        b, bdx = baseline[key]
        inputs = torch.randn(m, experts, generator=generator)
        gradients = torch.randn(m, 6, generator=generator)
        positions = sorted({0, m // 2, m - 1})
        inputs[positions] = row_cpu
        gradients[positions] = grad_row
        x = inputs.cuda().requires_grad_()
        grad = gradients.cuda()
        result = project(x)
        dx = torch.autograd.grad(result.selected_scores, x, grad)[0]
        with torch.inference_mode():
            replay = project(x)
        ref, ref_ids, ref_columns = stable_topk6_reference(inputs, ids_cpu)
        expected_dx = torch.zeros_like(inputs).scatter_(1, ref_columns, gradients)
        checks = {
            "expert_id_mismatches": int((result.expert_ids.cpu() != ref_ids).sum()),
            "column_mismatches": int((result.column_indices.cpu().long() != ref_columns).sum()),
            "payload_bit_mismatches": int((bits(result.selected_scores).cpu() != bits(ref)).sum()),
            "vjp_bit_mismatches": int((bits(dx).cpu() != bits(expected_dx)).sum()),
            "row_id_mismatches": int((result.expert_ids[positions] != b.expert_ids).sum()),
            "row_payload_bit_mismatches": int(
                (bits(result.selected_scores)[positions] != bits(b.selected_scores)).sum()
            ),
            "row_vjp_bit_mismatches": int((bits(dx)[positions] != bits(bdx)).sum()),
            "mode_bit_mismatches": int(
                (bits(result.selected_scores) != bits(replay.selected_scores)).sum()
            ),
        }
        assert all(value == 0 for value in checks.values()), (experts, pattern, m, checks)
        cases.append(
            {
                "experts": experts,
                "pattern": pattern,
                "m": m,
                "positions": positions,
                **checks,
                "input_sha256": tensor_sha256(x),
                "global_ids_sha256": tensor_sha256(ids),
                "upstream_sha256": tensor_sha256(grad),
                "expert_ids_sha256": tensor_sha256(result.expert_ids),
                "column_indices_sha256": tensor_sha256(result.column_indices),
                "selected_scores_sha256": tensor_sha256(result.selected_scores),
                "vjp_sha256": tensor_sha256(dx),
                "median_adapter_forward_us": measure(lambda x=x: project(x)),
                "median_aiter_forward_us": measure(
                    lambda x=x, ids=ids: api.stable_topk6_fwd(x, ids)
                ),
                "median_aiter_vjp_us": measure(
                    lambda columns=result.column_indices, grad=grad, experts=experts: (
                        api.stable_topk6_bwd(columns, grad, experts)
                    )
                ),
                "execution": result.execution,
            }
        )
    report = {
        "schema": "rlkernel.stable_topk6.task31.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "fixture": {
            "cpu_seed": 31,
            "kind": "synthetic FP32 selection scores",
            "checkpoint_weights_used": False,
            "tokenizer_used": False,
        },
        "timing": {
            "warmup": 10,
            "samples": 30,
            "method": "GPU events around Python calls",
            "adapter_includes": "finite/coverage checks, ID snapshots/hashing, autograd/provenance",
        },
        "oracle": (
            "independent CPU Python lexicographic sort; "
            "original FP32 payload gather; exact local VJP"
        ),
        "cases": cases,
        "not_validated": ["MI355X", "NVIDIA", "Graph", "full model", "framework hooks"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "cases": len(cases),
                "mismatches": 0,
                "shared_source_profile": cases[0]["execution"]["shared_source_profile"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
