# SPDX-License-Identifier: Apache-2.0
"""Real ROCm operator acceptance; explicitly NOT a four-layer model runner.

Three independent judgments: toleranced reference correctness, byte replay, and
operator autograd-forward/inference-mode parity. Component DP/RCCL smoke is
reported separately from model PP/TP/DP/EP/CP, none of which this runner supplies.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.backends.rocm.attention.v41_compressor_pool import compressor_pool
from rl_engine.backends.rocm.engram.gated_residual import engram_gated_residual
from rl_engine.backends.rocm.gemm.v41_fp8 import V41FP8Linear
from rl_engine.models.deepseek_v41.bitwise import compare_bits
from rl_engine.models.deepseek_v41.checkpoint import LayerCheckpoint
from rl_engine.models.deepseek_v41.config import MODEL_ID, OFFICIAL_REVISION
from rl_engine.models.deepseek_v41.expert import V41Expert, V41FP4Linear
from rl_engine.models.deepseek_v41.reference import (
    compressor_pool_reference,
    dequantize_fp4,
    engram_gated_residual_reference,
    quantized_linear_reference,
)


def validate_case(name, fn, ref, inputs, *, rank, layer, rtol, atol):
    def run():
        xs = [x.detach().clone().requires_grad_() for x in inputs]
        y = fn(*xs)
        # A fixed deterministic cotangent, with both signs and nonuniform entries.
        dy = torch.linspace(-1, 1, y.numel(), device=y.device).reshape(y.shape).to(y.dtype)
        grads = torch.autograd.grad(y, xs, dy)
        return y.detach(), [g.detach() for g in grads], dy

    y, grads, dy = run()
    xs = [x.detach().clone().requires_grad_() for x in inputs]
    expected = ref(*xs)
    eg = torch.autograd.grad(expected, xs, dy)
    torch.testing.assert_close(y, expected, rtol=rtol, atol=atol)
    for a, b in zip(grads, eg, strict=True):
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise AssertionError("nonfinite gradients")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    if not bool(torch.isfinite(y).all() & torch.isfinite(expected).all()):
        raise AssertionError("nonfinite outputs")
    repeated, rg, _ = run()
    replay = []
    for i, (a, b) in enumerate(zip((y, *grads), (repeated, *rg), strict=True)):
        replay.append(
            compare_bits(
                a,
                b,
                token_dims=(2 if a.ndim >= 3 else 1) if i == 0 else None,
                operator=f"{name}/{'output' if i == 0 else 'gradient-' + str(i)}",
                layer=layer,
                rank=rank,
            )
        )
    with torch.inference_mode():
        inference = fn(*[x.detach() for x in inputs])
    parity = compare_bits(
        y, inference, token_dims=2 if y.ndim >= 3 else 1, operator=name, layer=layer, rank=rank
    )
    if any(r["status"] != "PASS" for r in (*replay, parity)):
        raise AssertionError(json.dumps({"replay": replay, "parity": parity}))
    return {
        "operator": name,
        "original_layer_id": layer,
        "scope": "isolated operator",
        "forward_correctness": "PASS",
        "backward_correctness": "PASS",
        "rtol": rtol,
        "atol": atol,
        "forward_max_abs_error": float((y.float() - expected.float()).abs().max()),
        "gradient_max_abs_errors": [
            float((a.float() - b.float()).abs().max()) for a, b in zip(grads, eg, strict=True)
        ],
        "deterministic_replay": replay,
        "operator_train_inference_parity": parity,
        "shared_kernels_for_parity": True,
        "status": "PASS",
    }


def stress_gate(qw, kw, rank):
    """Two real optimizer trajectories, gradient accumulation and post-update parity."""
    torch.manual_seed(39 + rank)
    fixed = [
        torch.randn(1, 3, 4, 5120, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 3, 4, 5120, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 3, 5120, device="cuda", dtype=torch.bfloat16),
    ]
    trajectories = []
    steps = []
    for trajectory in range(2):
        q = torch.nn.Parameter(qw.float().clone())
        k = torch.nn.Parameter(kw.float().clone())
        optimizer = torch.optim.SGD([q, k], lr=1e-4)
        snapshots = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                x, key, value = (t * (micro + 1) for t in fixed)
                y = engram_gated_residual(x, key, value, q, k)
                y.float().square().mean().backward()
            snapshot = [q.grad.detach().clone(), k.grad.detach().clone()]
            if dist.is_initialized():
                for gradient in (q.grad, k.grad):
                    original = gradient.float().clone()
                    reduced = original.clone()
                    repeat = original.clone()
                    dist.all_reduce(reduced)
                    dist.all_reduce(repeat)
                    cmp = compare_bits(
                        reduced, repeat, operator="rccl-gradient-allreduce", layer=1, rank=rank
                    )
                    if cmp["status"] != "PASS":
                        raise AssertionError(json.dumps(cmp))
                    gathered = [torch.empty_like(original) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered, original)
                    expected = torch.stack(gathered).sum(0)
                    torch.testing.assert_close(reduced, expected, rtol=2e-6, atol=1e-7)
                    gradient.copy_(reduced / dist.get_world_size())
                    if trajectory == 0:
                        steps.append(
                            {"step": step, "scope": "component DP/RCCL gradient smoke", **cmp}
                        )
            optimizer.step()
            train = engram_gated_residual(*fixed, q, k)
            with torch.inference_mode():
                infer = engram_gated_residual(*fixed, q.detach(), k.detach())
            cmp = compare_bits(
                train, infer, token_dims=2, layer=1, operator="post-update-gate-parity", rank=rank
            )
            if cmp["status"] != "PASS":
                raise AssertionError(json.dumps(cmp))
            snapshots.append(
                (*snapshot, q.detach().clone(), k.detach().clone(), train.detach().clone())
            )
            if trajectory == 0:
                steps.append({"step": step, "scope": "isolated gate optimizer stress", **cmp})
        trajectories.append(snapshots)
    replay = []
    for step, (a, b) in enumerate(zip(*trajectories, strict=True)):
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            cmp = compare_bits(
                left, right, layer=1, operator=f"optimizer-replay/{step}/{i}", rank=rank
            )
            replay.append(cmp)
            if cmp["status"] != "PASS":
                raise AssertionError(json.dumps(cmp))
    return {
        "optimizer": "SGD",
        "steps": 3,
        "microbatches_per_step": 2,
        "snapshots": steps,
        "trajectory_replay": replay,
        "scope": "Engram gate only, not a model training run",
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rlk-commit", required=True, help="SHA of the mounted, committed host source"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    local = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    report = {
        "scope": "V4.1 isolated operators and component DP/RCCL; no model execution",
        "model": MODEL_ID,
        "checkpoint_revision": OFFICIAL_REVISION,
        "rank": rank,
        "world_size": world,
        "model_layers": None,
        "kernel_backend": "RLK ROCm Triton; fixed per-operator reductions",
        "model_train_inference_parity": "NOT RUN",
        "model_bit_mismatches": None,
        "cases": [],
        "status": "FAIL",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    exit_code = 1
    try:
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("real AMD ROCm GPU is mandatory")
        if torch.cuda.device_count() != world:
            raise RuntimeError("visible GPU count must equal the requested world size")
        torch.cuda.set_device(local)
        if world > 1:
            dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
        report["environment"] = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": str(torch.cuda.get_device_properties(local)),
            "rccl": torch.cuda.nccl.version(),
        }
        import triton

        report["environment"]["triton"] = triton.__version__
        import re

        if re.fullmatch(r"[0-9a-f]{40}", args.rlk_commit) is None:
            raise ValueError("--rlk-commit must be a full immutable SHA")
        report["rlk_commit"] = args.rlk_commit
        report["source_files_sha256"] = {}
        import hashlib

        for root in ("rl_engine/models/deepseek_v41", "rl_engine/backends/rocm/engram"):
            for p in Path(root).glob("*.py"):
                report["source_files_sha256"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            "rl_engine/backends/rocm/norm/v41_rmsnorm.py",
            "rl_engine/backends/rocm/gemm/v41_fp8.py",
            "rl_engine/backends/rocm/attention/v41_compressor_pool.py",
            "rl_engine/backends/rocm/moe/v41_packed_mx.py",
            __file__,
        ):
            report["source_files_sha256"][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        checkpoint = LayerCheckpoint(args.weights)
        report["checkpoint_manifest"] = json.loads((args.weights / "manifest.json").read_text())
        w, s = checkpoint.projection("layers.2.attn.wq_a", device="cuda")
        linear = V41FP8Linear(w, s)
        dense8 = w.float() * s.float().repeat_interleave(32, 0)[: w.shape[0]].repeat_interleave(
            32, 1
        )
        expert_layers = []
        dense4 = []
        for name in ("w1", "w2", "w3"):
            c, s = checkpoint.projection("layers.2.ffn.experts.0." + name, device="cuda")
            expert_layers.append(V41FP4Linear(c, s))
            dense4.append(dequantize_fp4(c, s))
        expert = V41Expert(*expert_layers)

        def expert_reference(x, p):
            g = quantized_linear_reference(x, dense4[0]).float().clamp_max(10)
            u = quantized_linear_reference(x, dense4[2]).float().clamp(-10, 10)
            h = (torch.nn.functional.silu(g) * u * p[:, None]).bfloat16()
            return quantized_linear_reference(h, dense4[1])

        qw = checkpoint.tensor("layers.1.engram.q_weight", device="cuda")
        kw = checkpoint.tensor("layers.1.engram.k_weight", device="cuda")
        for seed, batch, length in ((11, 1, 3), (29, 2, 5), (53, 2, 17)):
            torch.manual_seed(seed)
            x = torch.randn(batch, length, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
            items = [
                (
                    "official-wq_a",
                    linear,
                    lambda x: quantized_linear_reference(x, dense8),
                    [x],
                    2,
                    0.04,
                    0.05,
                ),
                (
                    "official-fp4-expert0",
                    expert,
                    expert_reference,
                    [x.reshape(-1, 5120), torch.rand(batch * length, device="cuda")],
                    2,
                    0.05,
                    0.08,
                ),
            ]
            gate_inputs = [
                torch.randn(batch, length, 4, 5120, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            gate_inputs += [
                torch.randn(batch, length, 5120, device="cuda", dtype=torch.bfloat16),
                qw,
                kw,
            ]
            items.append(
                (
                    "engram-gate-official-gains",
                    engram_gated_residual,
                    engram_gated_residual_reference,
                    gate_inputs,
                    1,
                    0.04,
                    0.05,
                )
            )
            pool_inputs = [torch.randn(batch, length, 2, 512, device="cuda") for _ in range(2)]
            items.append(
                ("c2-pool", compressor_pool, compressor_pool_reference, pool_inputs, 2, 1e-4, 3e-6)
            )
            for name, fn, ref, inputs, layer, rtol, atol in items:
                record = validate_case(
                    name, fn, ref, inputs, rank=rank, layer=layer, rtol=rtol, atol=atol
                )
                record.update(seed=seed, batch=batch, sequence_length=length)
                report["cases"].append(record)
        report["stress"] = stress_gate(qw, kw, rank)
        report["status"] = "PASS"
        exit_code = 0
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output / f"operator-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"rank": rank, "status": report["status"], "cases": len(report["cases"])}),
            flush=True,
        )
        if dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
