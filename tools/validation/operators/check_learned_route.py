# SPDX-License-Identifier: Apache-2.0
"""Task-32 normal-route replay with fixed checkpoint bias and synthetic gate logits."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.moe.learned_route import implementation
from rl_engine.ops.moe.learned_route import (
    learned_route,
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.reference.moe.learned_route import learned_route_reference, vjp_fp64
from rl_engine.validation.operators.learned_route_fixture import load_checkpoint_bias


def bits(tensor):
    return tensor.detach().contiguous().view(torch.int32)


def route(x, ids, bias, plan):
    return learned_route(
        x,
        ids,
        bias,
        plan,
        absolute_layer=3,
        model_version="0731.frozen",
        bias_version="0731.layer3.frozen",
        domain_version=plan.domain.domain_version,
        plan_version="task32.normal.v1",
    )


def timing(fn, repeats=30):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    wall, events = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - wall_start) * 1e3)
        events.append(start.elapsed_time(end))
    return {
        "repeats": repeats,
        "warmups": 10,
        "wall_median_ms": statistics.median(wall),
        "gpu_event_median_ms": statistics.median(events),
    }


def run(bias_fixture, raw_output=None):
    bias_cpu, metadata = load_checkpoint_bias(bias_fixture)
    gen = torch.Generator().manual_seed(32)
    x = (torch.randn(257, 256, generator=gen) * 3).cuda().requires_grad_()
    ids = torch.randperm(257, generator=gen).long() * 17 + 1000
    grad = torch.randn(257, 6, generator=gen).cuda()
    bias = bias_cpu.cuda()
    cases, plans, raw_payloads = [], [], {}
    for capacity in (None, 2, 0):
        domain = prepare_capacity_domain(
            ids,
            capacity,
            absolute_layer=3,
            model_version="0731.frozen",
            domain_version=f"task32.N257.capacity{capacity}.v1",
        )

        def prepare(domain=domain):
            return prepare_learned_route_plan(
                x,
                ids,
                bias,
                domain,
                bias_version="0731.layer3.frozen",
                plan_version="task32.normal.v1",
            )

        plan = prepare()
        whole = route(x, ids, bias, plan)
        dx = torch.autograd.grad(whole.weights, x, grad)[0]
        reference, experts, columns, keep, positions = learned_route_reference(
            x,
            ids,
            bias,
            domain.capacities,
        )
        assert torch.equal(whole.expert_ids.cpu(), experts)
        assert torch.equal(whole.surviving_mask.cpu(), keep)
        assert torch.equal(whole.dispatch_positions.cpu(), positions)
        torch.testing.assert_close(
            whole.weights.detach().cpu().double(), reference, rtol=2e-6, atol=1e-8
        )
        math_dx = vjp_fp64(x.detach().cpu(), columns, keep, grad.cpu())
        torch.testing.assert_close(dx.cpu().double(), math_dx, rtol=3e-4, atol=2e-7)
        active = ((keep.sum(1) > 0) & (keep.sum(1) < 6)).nonzero().flatten()
        anchor = int(active[0]) if active.numel() else 128
        for m in (1, 16, 17, 32, 65, 257):
            rows = torch.tensor([anchor] + [i for i in range(257) if i != anchor][: m - 1])
            rows = rows.roll(m // 2)
            device_rows = rows.cuda()
            local = x.detach()[device_rows].clone().requires_grad_()
            result = route(local, ids[rows], bias, plan)
            local_dx = torch.autograd.grad(result.weights, local, grad[device_rows])[0]
            assert torch.equal(bits(result.weights), bits(whole.weights[device_rows]))
            assert torch.equal(bits(local_dx), bits(dx[device_rows]))
            assert torch.equal(result.expert_ids, whole.expert_ids[device_rows])
            assert torch.equal(result.surviving_mask, whole.surviving_mask[device_rows])
            assert torch.equal(result.dispatch_positions, whole.dispatch_positions[device_rows])
            with torch.inference_mode():
                inference = route(local, ids[rows], bias, plan)
            assert torch.equal(bits(result.weights), bits(inference.weights))
            raw_payloads[f"capacity{capacity}.M{m}"] = {
                name: value.detach().cpu().contiguous()
                for name, value in {
                    "logits": local,
                    "logical_ids": ids[rows],
                    "weights": result.weights,
                    "expert_ids": result.expert_ids,
                    "keep": result.surviving_mask,
                    "positions": result.dispatch_positions,
                    "vjp": local_dx,
                }.items()
            }
            cases.append(
                {
                    "capacity": capacity,
                    "M": m,
                    "plan_sha256": plan.sha256,
                    "domain_sha256": domain.sha256,
                    "forward_bi": True,
                    "vjp_bi": True,
                    "inference_train_bi": True,
                    "hashes": {
                        name: tensor_sha256(value)
                        for name, value in {
                            "logits": local,
                            "logical_ids": ids[rows],
                            "weights": result.weights,
                            "expert_ids": result.expert_ids,
                            "keep": result.surviving_mask,
                            "positions": result.dispatch_positions,
                            "vjp": local_dx,
                        }.items()
                    },
                }
            )
        plans.append(
            {
                "execution": whole.execution,
                "normal_oracle_passed": True,
                "max_weight_abs_error": float(
                    (whole.weights.detach().cpu() - reference).abs().max()
                ),
                "max_vjp_abs_error": float((dx.cpu() - math_dx).abs().max()),
            }
        )
        if capacity == 2:
            api, _ = implementation()
            gids = torch.arange(256, device="cuda")
            scores, _, columns_gpu = api.learned_route_select(x, bias, gids)

            def aiter_forward(api=api, gids=gids, whole=whole, plan=plan):
                selected_scores, _, selected_columns = api.learned_route_select(x, bias, gids)
                return api.learned_route_weights(
                    selected_scores,
                    selected_columns,
                    whole.surviving_mask,
                    plan.scale,
                )

            timings = {
                "plan_preparation": timing(prepare, 10),
                "full_rlk_normal_forward": timing(partial(route, x, ids, bias, plan)),
                "aiter_normal_select_and_postmask_weights": timing(aiter_forward),
                "aiter_vjp": timing(
                    partial(
                        api.learned_route_bwd,
                        x,
                        scores,
                        columns_gpu,
                        whole.surviving_mask,
                        grad,
                        plan.scale,
                    )
                ),
            }
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw_payloads, raw_output)
    return {
        "schema": 1,
        "task": 32,
        "fixture": metadata,
        "logits": "synthetic FP32, CPU seed 32, N257 x 256; real checkpoint bias only",
        "normal_selection": True,
        "fixed_route_replay": False,
        "cases": cases,
        "plans": plans,
        "timing_capacity": 2,
        "timings": timings,
        "timing_scope": "isolated operator; full RLK includes host validation, copies, SHA, "
        "normal selection and immutable-plan lookup; plan preparation includes "
        "complete-domain CPU admission; events may include host launch gaps",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-output", type=Path, help="optional ignored artifact for byte comparison"
    )
    args = parser.parse_args()
    report = run(args.bias_fixture, args.raw_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"cases_passed": len(report["cases"]), "timings": report["timings"]}, indent=2)
    )


if __name__ == "__main__":
    main()
