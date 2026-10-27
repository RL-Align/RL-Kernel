# SPDX-License-Identifier: Apache-2.0
"""Task-33 normal-route acceptance first; fixed replay is diagnostic only."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.moe.router_training import implementation
from rl_engine.ops.moe.hash_route import prepare_hash_route_table
from rl_engine.ops.moe.learned_route import (
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.ops.moe.router_training import (
    compare_route_records,
    compare_vjp_records,
    diagnostic_fixed_route,
    load_route_record,
    load_vjp_record,
    normal_hash_route,
    normal_learned_route,
    record_route_vjp,
    save_route_record,
    save_vjp_record,
)
from rl_engine.reference.moe.hash_route import hash_route_fp64, hash_route_vjp_fp64
from rl_engine.reference.moe.learned_route import learned_route_reference, vjp_fp64
from rl_engine.validation.operators.hash_route_fixture import load_checkpoint_table
from rl_engine.validation.operators.learned_route_fixture import load_checkpoint_bias


def bits(value):
    return value.detach().contiguous().view(torch.int32)


def timer(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {"warmups": 5, "samples": 20, "wall_median_ms": statistics.median(samples)}


def run(args):
    bias_cpu, bias_metadata = load_checkpoint_bias(args.bias_fixture)
    tables_cpu = [
        load_checkpoint_table(args.hash_fixture_dir / f"0731-layer{layer}-tid2eid.pt", layer)
        for layer in range(3)
    ]
    gen = torch.Generator().manual_seed(32)
    full = (torch.randn(257, 256, generator=gen) * 3).cuda().requires_grad_()
    ids = torch.randperm(257, generator=gen).long() * 17 + 1000
    all_grad = torch.randn(257, 6, generator=gen).cuda()
    bias = bias_cpu.cuda()
    raw = {
        "input/logits": full.detach().cpu(),
        "input/logical_ids": ids,
        "input/upstream": all_grad.cpu(),
    }
    cases, profiles = [], []
    args.record_dir.mkdir(parents=True, exist_ok=True)
    for label, capacity in [
        ("learned-unbounded", None),
        ("learned-capacity2", 2),
        ("learned-all-drop", 0),
    ]:
        domain = prepare_capacity_domain(
            ids, capacity, absolute_layer=3, model_version="0731.v1", domain_version=label
        )
        plan = prepare_learned_route_plan(
            full, ids, bias, domain, bias_version="0731.bias3.v1", plan_version="plan.v1"
        )
        fn = partial(
            normal_learned_route,
            bias=bias,
            plan=plan,
            record_version="r1",
            absolute_layer=3,
            model_version="0731.v1",
            bias_version="0731.bias3.v1",
            domain_version=label,
            plan_version="plan.v1",
        )
        reference, experts, columns, keep, _ = learned_route_reference(
            full, ids, bias, domain.capacities
        )
        math_dx = vjp_fp64(full.detach().cpu(), columns, keep, all_grad.cpu())
        evaluate(
            label,
            fn,
            full,
            ids,
            all_grad,
            reference,
            experts,
            math_dx,
            cases,
            profiles,
            raw,
            args.record_dir,
        )
    for layer, (table_cpu, metadata) in enumerate(tables_cpu):
        table = prepare_hash_route_table(
            table_cpu.cuda(),
            absolute_layer=layer,
            model_version="0731.v1",
            table_version="0731.table.v1",
            expected_sha256=metadata["sha256"],
        )
        vocab = torch.arange(257, device="cuda", dtype=torch.int64) * 503 % table.vocabulary_size
        for normalize in [True, False]:
            label = f"hash-layer{layer}-norm{normalize}"

            def fn(x, logical_ids, table=table, vocab=vocab, normalize=normalize):
                row_for = {int(token): i for i, token in enumerate(ids)}
                rows = torch.tensor([row_for[int(t)] for t in logical_ids], device="cuda")
                return normal_hash_route(
                    x,
                    logical_ids,
                    vocab[rows],
                    table,
                    record_version="r1",
                    absolute_layer=table.absolute_layer,
                    model_version="0731.v1",
                    table_version="0731.table.v1",
                    renormalize=normalize,
                )

            reference, experts = hash_route_fp64(
                full.detach().cpu(), vocab.cpu(), table_cpu, renormalize=normalize
            )
            math_dx = hash_route_vjp_fp64(
                full.detach().cpu(), vocab.cpu(), table_cpu, all_grad.cpu(), renormalize=normalize
            )
            evaluate(
                label,
                fn,
                full,
                ids,
                all_grad,
                reference,
                experts,
                math_dx,
                cases,
                profiles,
                raw,
                args.record_dir,
            )
    api, manifest = implementation()
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, args.raw_output)
    props = torch.cuda.get_device_properties(0)
    return {
        "schema": "rlkernel.router_training.acceptance.v1",
        "task": 33,
        "passed": True,
        "gpu": props.name,
        "arch": props.gcnArchName,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "source_sha256": manifest["source_sha256"],
        "patch_sha256": manifest["patch_sha256"],
        "fixture": {
            "bias": bias_metadata,
            "tables": [m for _, m in tables_cpu],
            "logits": "synthetic FP32 seed32, N257 x 256",
        },
        "cases": cases,
        "profiles": profiles,
        "normal_acceptance_cases": len(cases),
        "fixed_replay_role": (
            "diagnostic only; normal discrete and FP64 checks complete before replay"
        ),
        "scope": ("local dlogits; BF16 LoRA/frozen Gate chain validated separately in pytest"),
    }


def evaluate(
    label, fn, full, ids, all_grad, reference, experts, math_dx, cases, profiles, raw, record_dir
):
    result = fn(full, ids)
    # No fixed IDs are used to establish correctness of normal selection.
    assert torch.equal(result.expert_ids.cpu(), experts)
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), reference, rtol=2e-6, atol=1e-8
    )
    dx, full_vjp = record_route_vjp(result, all_grad, loss_version="fixed.global257.loss.v1")
    torch.testing.assert_close(dx.cpu().double(), math_dx, rtol=3e-4, atol=2e-7)
    admission = {row["logical_id"]: sum(row["keep"]) for row in result.record.payload()["tokens"]}
    partial_rows = [i for i, token in enumerate(ids.tolist()) if 0 < admission[token] < 6]
    anchor = partial_rows[0] if partial_rows else 0
    for m in (1, 16, 17, 32, 65, 257):
        rows = torch.tensor([anchor] + [i for i in range(257) if i != anchor][: m - 1]).roll(m // 2)
        device_rows = rows.cuda()
        local = full.detach()[device_rows].clone().requires_grad_()
        actual = fn(local, ids[rows])
        local_dx, local_vjp = record_route_vjp(
            actual, all_grad[device_rows], loss_version="fixed.global257.loss.v1"
        )
        assert torch.equal(actual.expert_ids, experts[rows].cuda())
        assert torch.equal(bits(actual.weights), bits(result.weights[device_rows]))
        assert torch.equal(bits(local_dx), bits(dx[device_rows]))
        with torch.inference_mode():
            inference = fn(local, ids[rows])
        assert compare_route_records(inference.record, actual.record)["acceptance"]
        source = actual.record
        if m == 17 and label in {"learned-capacity2", "hash-layer0-normTrue"}:
            path = record_dir / (label + ".route.json")
            save_route_record(source, path)
            source = load_route_record(path)
            path = record_dir / (label + ".vjp.json")
            save_vjp_record(local_vjp, path)
            assert compare_vjp_records(load_vjp_record(path), local_vjp)["acceptance"]
        diagnostic = diagnostic_fixed_route(
            local.detach().clone().requires_grad_(),
            ids[rows],
            source,
            absolute_layer=source.payload()["absolute_layer"],
            versions=source.payload()["versions"],
        )
        replay_dx, replay_vjp = record_route_vjp(
            diagnostic, all_grad[device_rows], loss_version="fixed.global257.loss.v1"
        )
        compared = compare_vjp_records(replay_vjp, local_vjp, require_normal=False)
        assert compared["status"] == "match" and not compared["acceptance"]
        raw[f"{label}.M{m}/weights"] = actual.weights.detach().cpu()
        raw[f"{label}.M{m}/dlogits"] = local_dx.cpu()
        raw[f"{label}.M{m}/experts"] = actual.expert_ids.cpu()
        cases.append(
            {
                "profile": label,
                "M": m,
                "normal_discrete_passed": True,
                "forward_bi": True,
                "vjp_bi": True,
                "inference_train_bytes": True,
                "forward_record_sha256": actual.record.sha256,
                "vjp_record_sha256": local_vjp.sha256,
                "weights_sha256": tensor_sha256(actual.weights),
                "dlogits_sha256": tensor_sha256(local_dx),
                "diagnostic_replay_match": True,
                "diagnostic_acceptance": False,
            }
        )
    profile = {
        "name": label,
        "normal_record_sha256": result.record.sha256,
        "normal_vjp_sha256": full_vjp.sha256,
        "max_weight_abs_error": float((result.weights.detach().cpu() - reference).abs().max()),
        "max_vjp_abs_error": float((dx.cpu() - math_dx).abs().max()),
    }
    if label == "learned-capacity2":
        local = full.detach()[:17]
        profile["timing_recorded_normal_M17"] = timer(partial(fn, local, ids[:17]))
        profile["timing_note"] = (
            "full adapter/normal route/CPU snapshots/backward guard hashes; "
            "isolated wall time, no speedup claim"
        )
    profiles.append(profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-fixture", type=Path, required=True)
    parser.add_argument("--hash-fixture-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "normal_cases": len(report["cases"]),
                "profiles": len(report["profiles"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
