# SPDX-License-Identifier: Apache-2.0
"""Pinned three-layer Hash table replay, raw-bit BI and latency disclosure."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from rl_engine.backends.rocm.moe.hash_route import implementation
from rl_engine.contracts.operators.hash_route import HASH_LAYERS, REVISION
from rl_engine.models.deepseek_v4.hash_router import DeepSeekV4HashRouter
from rl_engine.ops.moe.hash_route import prepare_hash_route_table, tensor_sha256
from rl_engine.reference.moe.hash_route import hash_route_fp64, hash_route_vjp_fp64
from rl_engine.validation.operators.hash_route_fixture import load_checkpoint_table


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


def bits(x):
    return x.detach().contiguous().view(torch.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    # Validate every table's identity on CPU before selecting a GPU.
    loaded = {
        layer: load_checkpoint_table(args.tables_dir / f"0731-layer{layer}-tid2eid.pt", layer)
        for layer in HASH_LAYERS
    }
    torch.cuda.set_device(args.device)
    prepared = {
        layer: prepare_hash_route_table(
            table.cuda(),
            absolute_layer=layer,
            model_version=REVISION,
            table_version=REVISION + f":layer{layer}",
            expected_sha256=metadata["sha256"],
        )
        for layer, (table, metadata) in loaded.items()
    }
    router = DeepSeekV4HashRouter(prepared)
    api, _ = implementation()
    generator = torch.Generator().manual_seed(30)
    row_cpu = torch.randn(1, 256, generator=generator) * 20
    row_grad_cpu = torch.randn(1, 6, generator=generator)
    row = row_cpu.cuda().requires_grad_()
    row_grad = row_grad_cpu.cuda()
    cases = []
    for layer in HASH_LAYERS:
        token = torch.tensor([129279], device="cuda")
        table, metadata = loaded[layer]

        def project(x, ids, layer=layer):
            return router(
                x,
                ids,
                absolute_layer=layer,
                model_version=REVISION,
                table_version=REVISION + f":layer{layer}",
            )

        baseline = project(row, token)
        baseline_dx = torch.autograd.grad(baseline.weights, row, row_grad)[0]
        for m in (1, 16, 17, 32, 65, 257):
            inputs = torch.randn(m, 256, generator=generator) * 20
            token_ids = torch.randint(0, 129280, (m,), generator=generator)
            gradients = torch.randn(m, 6, generator=generator)
            positions = sorted({0, m // 2, m - 1})
            inputs[positions] = row_cpu
            token_ids[positions] = 129279
            gradients[positions] = row_grad_cpu
            x = inputs.cuda().requires_grad_()
            ids = token_ids.cuda()
            grad = gradients.cuda()
            actual = project(x, ids)
            dx = torch.autograd.grad(actual.weights, x, grad)[0]
            with torch.inference_mode():
                replay = project(x, ids)
            reference, reference_ids = hash_route_fp64(inputs, token_ids, table)
            reference_dx = hash_route_vjp_fp64(inputs, token_ids, table, gradients)
            torch.testing.assert_close(
                actual.weights.detach().cpu().double(),
                reference,
                rtol=2e-6,
                atol=(2**-146) * 1.5 / 1e-20,
            )
            torch.testing.assert_close(dx.cpu().double(), reference_dx, rtol=3e-4, atol=2e-7)
            mismatched_ids = int((actual.expert_ids.cpu() != reference_ids).sum())
            mismatched_weights = int(
                (bits(actual.weights)[positions] != bits(baseline.weights)).sum()
            )
            mismatched_dx = int((bits(dx)[positions] != bits(baseline_dx)).sum())
            mode_mismatches = int((bits(actual.weights) != bits(replay.weights)).sum())
            cases.append(
                {
                    "absolute_layer": layer,
                    "m": m,
                    "positions": positions,
                    "table_sha256": metadata["sha256"],
                    "input_sha256": tensor_sha256(x),
                    "token_ids_sha256": tensor_sha256(ids),
                    "upstream_sha256": tensor_sha256(grad),
                    "expert_ids_sha256": tensor_sha256(actual.expert_ids),
                    "weights_sha256": tensor_sha256(actual.weights),
                    "vjp_sha256": tensor_sha256(dx),
                    "expert_id_mismatches": mismatched_ids,
                    "row_weight_bit_mismatches": mismatched_weights,
                    "row_vjp_bit_mismatches": mismatched_dx,
                    "mode_bit_mismatches": mode_mismatches,
                    "max_weight_abs_error_vs_fp64": (actual.weights.detach().cpu() - reference)
                    .abs()
                    .max()
                    .item(),
                    "max_vjp_abs_error_vs_fp64": (dx.cpu() - reference_dx).abs().max().item(),
                    "median_adapter_us": measure(lambda x=x, ids=ids: project(x, ids)),
                    "median_aiter_forward_us": measure(
                        lambda x=x, ids=ids, layer=layer: api.hash_route_fwd(
                            x, prepared[layer]._prepared, ids.int(), True, 1.5
                        )
                    ),
                    "median_aiter_vjp_us": measure(
                        lambda x=x, selected=actual.expert_ids, grad=grad: api.hash_route_bwd(
                            x, selected, grad, True, 1.5
                        )
                    ),
                    "execution": actual.execution,
                }
            )
    report = {
        "schema": "rlkernel.hash_route.task30.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "model_revision": REVISION,
            "tables": {str(layer): meta for layer, (_, meta) in loaded.items()},
            "logits": "synthetic FP32 normal times 20",
            "cpu_seed": 30,
            "tokenizer_used": False,
            "weights_used": False,
            "logit_row_sha256": tensor_sha256(row),
            "upstream_row_sha256": tensor_sha256(row_grad),
        },
        "timing": {
            "warmup": 10,
            "samples": 30,
            "method": "GPU events around Python calls",
            "adapter_includes": (
                "table-content hashing and GPU-to-CPU copy, input checks, autograd, provenance"
            ),
        },
        "passed": all(
            c["expert_id_mismatches"]
            == c["row_weight_bit_mismatches"]
            == c["row_vjp_bit_mismatches"]
            == c["mode_bit_mismatches"]
            == 0
            for c in cases
        ),
        "cases": cases,
        "not_validated": [
            "MI355X",
            "NVIDIA",
            "Graph",
            "framework hooks",
            "full model",
            "EP/PP transport",
        ],
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
                            "absolute_layer",
                            "m",
                            "expert_id_mismatches",
                            "row_weight_bit_mismatches",
                            "row_vjp_bit_mismatches",
                            "median_adapter_us",
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
