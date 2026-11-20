# SPDX-License-Identifier: Apache-2.0
"""Task-34 real-bias normal-route loss/VJP and canonical microbatch acceptance."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.ops.moe.learned_route import (
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.ops.moe.router_aux_loss import (
    AuxLossConfig,
    prepare_router_aux_context,
    router_aux_loss,
)
from rl_engine.reference.moe.router_aux_loss import router_aux_loss_fp64
from rl_engine.validation.operators.learned_route_fixture import load_checkpoint_bias


def bits(value):
    return value.detach().contiguous().view(torch.int32)


def timing(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(20):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {"warmups": 5, "samples": 20, "wall_median_ms": statistics.median(values)}


def run(args):
    bias_cpu, metadata = load_checkpoint_bias(args.bias_fixture)
    bias = bias_cpu.cuda()
    gen = torch.Generator().manual_seed(34)
    x = (torch.randn(257, 256, generator=gen) * 3).cuda().requires_grad_()
    ids = torch.randperm(257, generator=gen).long() * 17 + 1000
    raw = {"logits": x.detach().cpu(), "logical_ids": ids, "bias": bias_cpu}
    cases = []
    profiles = []
    for name, capacity, scale in [
        ("unbounded", None, 1.5),
        ("capacity2", 2, 1.5),
        ("all_drop", 0, 1.5),
        ("capacity2_scale0", 2, 0.0),
    ]:
        domain = prepare_capacity_domain(
            ids, capacity, absolute_layer=3, model_version="0731.v1", domain_version=name
        )
        plan = prepare_learned_route_plan(
            x,
            ids,
            bias,
            domain,
            bias_version="bias3.v1",
            plan_version="p1",
            routed_scaling_factor=scale,
        )
        config = AuxLossConfig(True, 0.03, "aux.v1")
        context = prepare_router_aux_context(plan, config=config, version="ctx.v1")
        options = dict(
            config=config,
            context=context,
            absolute_layer=3,
            model_version="0731.v1",
            bias_version="bias3.v1",
            domain_version=name,
            plan_version="p1",
        )
        fn = partial(router_aux_loss, bias=bias, plan=plan, **options)
        reference_x = x.detach().cpu().double().requires_grad_()
        reference, experts, _, _, _, counts, masses = router_aux_loss_fp64(
            reference_x, ids, bias_cpu, domain.capacities, config.validate()
        )
        reference_dx = torch.autograd.grad(reference, reference_x)[0]
        baseline = fn(x, ids)
        baseline_dx = torch.autograd.grad(baseline.loss, x)[0]
        assert torch.equal(baseline.expert_ids.cpu(), experts)
        assert torch.equal(baseline.counts.cpu(), counts)
        torch.testing.assert_close(
            baseline.loss.detach().cpu().double(), reference.detach(), rtol=3e-6, atol=2e-8
        )
        torch.testing.assert_close(baseline_dx.cpu().double(), reference_dx, rtol=5e-4, atol=2e-8)
        torch.testing.assert_close(
            baseline.masses.cpu().double(), masses.detach(), rtol=2e-6, atol=2e-7
        )
        for m in (1, 16, 17, 32, 65, 257):
            permutation = torch.randperm(257, generator=gen)
            local = x.detach()[permutation.cuda()].T.contiguous().T.requires_grad_()
            actual = fn(local, ids[permutation], microbatch_size=m)
            dx = torch.autograd.grad(actual.loss, local)[0]
            assert torch.equal(actual.expert_ids.cpu(), experts[permutation])
            assert torch.equal(bits(actual.loss), bits(baseline.loss))
            assert torch.equal(bits(dx), bits(baseline_dx[permutation.cuda()]))
            with torch.inference_mode():
                inference = fn(local, ids[permutation], microbatch_size=m)
            assert torch.equal(bits(inference.loss), bits(actual.loss))
            key = f"{name}.microbatch{m}"
            values = {
                "loss": actual.loss.reshape(1),
                "dlogits": dx,
                "experts": actual.expert_ids,
                "counts": actual.counts,
                "masses": actual.masses,
                "probabilities": actual.probabilities,
            }
            for field, value in values.items():
                raw[key + "/" + field] = value.detach().cpu().contiguous()
            cases.append(
                {
                    "profile": name,
                    "compute_microbatch": m,
                    "global_valid_tokens": 257,
                    "normal_discrete_passed": True,
                    "loss_bytes_equal": True,
                    "vjp_bytes_equal": True,
                    "inference_train_bytes_equal": True,
                    "plan_sha256": plan.sha256,
                    "context_sha256": context.sha256,
                    "hashes": {
                        field: tensor_sha256(value.reshape(-1)) for field, value in values.items()
                    },
                }
            )
        profile = {
            "profile": name,
            "execution": baseline.execution,
            "loss": float(baseline.loss.detach()),
            "mathematical_fp64_loss": float(reference.detach()),
            "max_loss_abs_error": abs(float(baseline.loss.detach()) - float(reference.detach())),
            "max_vjp_abs_error": float((baseline_dx.cpu().double() - reference_dx).abs().max()),
        }
        if name == "capacity2":
            profile["timing_normal_aux_forward"] = timing(partial(fn, x, ids))
            profile["timing_note"] = (
                "N257 includes normal validation, scale-one probabilities, CPU hashes "
                "and fixed reduction; no file I/O or backward"
            )
        profiles.append(profile)
    props = torch.cuda.get_device_properties(0)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, args.raw_output)
    return {
        "schema": "rlkernel.router_aux_loss.acceptance.v1",
        "task": 34,
        "passed": True,
        "gpu": props.name,
        "arch": props.gcnArchName,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "fixture": metadata,
        "logits": "synthetic FP32 N257 x 256, CPU seed34",
        "normal_routing": True,
        "fixed_replay": False,
        "cases": cases,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-fixture", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "cases": len(report["cases"]), "profiles": len(report["profiles"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
