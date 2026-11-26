# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate the Muse Glimmer canonical TP tree: TP2/4/8 == TP1, byte for byte (RFC #501).

Two launch modes run the same checks:

  # CPU / Gloo, small shapes, no GPU needed (spawns the ranks itself)
  python tools/validation/models/museglimmer_tp_canonical_tree.py \
      --device cpu --tp 2 4 8 --shapes tiny

  # One 8-GPU node, Muse Glimmer shapes, with benchmark
  torchrun --nproc-per-node 8 tools/validation/models/museglimmer_tp_canonical_tree.py \
      --device cuda --tp 2 4 8 --shapes muse --tokens 4096 --bench \
      --report museglimmer_tp_canonical_tree.json

For every physical TP each rank recomputes the TP1 result locally (collective ``None``, all 8
canonical leaves on one device) and compares its own slices with ``torch.equal``: row-parallel
outputs and input/weight gradients for ``o_proj`` and ``down_proj``, and outputs, input
gradients and weight gradients of the attention (q_proj + gate_proj) and MLP (gate_proj +
up_proj) column groups. TP1 is also checked against an FP64 reference, for repeat-run
determinism, and for batch invariance. Exit status is non-zero on any failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.models.muse_glimmer import tp_canonical_tree as tct  # noqa: E402

DTYPE = torch.bfloat16
TINY = {
    # Same structure as Muse Glimmer at 1/32 scale; every reduced dim splits into 8 leaves.
    "self_attn.o_proj": tct.ProjectionSpec("self_attn.o_proj", tct.ROW_PARALLEL, 128, 208),
    "mlp.down_proj": tct.ProjectionSpec("mlp.down_proj", tct.ROW_PARALLEL, 624, 208),
    "self_attn.q_proj": tct.ProjectionSpec("self_attn.q_proj", tct.COLUMN_PARALLEL, 208, 128),
    "self_attn.gate_proj": tct.ProjectionSpec("self_attn.gate_proj", tct.COLUMN_PARALLEL, 208, 128),
    "mlp.gate_proj": tct.ProjectionSpec("mlp.gate_proj", tct.COLUMN_PARALLEL, 208, 624),
    "mlp.up_proj": tct.ProjectionSpec("mlp.up_proj", tct.COLUMN_PARALLEL, 208, 624),
}
ROW_NAMES = ("self_attn.o_proj", "mlp.down_proj")


def _specs(shapes: str) -> dict[str, tct.ProjectionSpec]:
    return dict(TINY) if shapes == "tiny" else dict(tct.MUSE_PROJECTIONS)


def _randn(shape, seed: int, device, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(DTYPE).to(device)


class Inputs:
    """Identical full tensors on every rank (generated on CPU from fixed seeds)."""

    def __init__(self, specs, tokens: int, device, seed: int) -> None:
        self.specs = specs
        self.weights, self.inputs, self.grads = {}, {}, {}
        for i, (name, spec) in enumerate(sorted(specs.items())):
            scale = spec.in_features**-0.5
            self.weights[name] = _randn(
                (spec.out_features, spec.in_features), seed + 10 * i, device, scale
            )
            self.grads[name] = _randn((tokens, spec.out_features), seed + 10 * i + 1, device)
            if spec.kind == tct.ROW_PARALLEL:
                self.inputs[name] = _randn((tokens, spec.in_features), seed + 10 * i + 2, device)
        hidden = next(iter(specs.values()))
        width = hidden.out_features if hidden.kind == tct.ROW_PARALLEL else hidden.in_features
        self.hidden = _randn((tokens, width), seed + 999, device)


def run_row(inp: Inputs, name: str, collective, tp: int, rank: int, gemm=tct.fp32_reference_gemm):
    spec = inp.specs[name]
    x = tct.shard_activation(inp.inputs[name], spec, rank, tp).requires_grad_(True)
    w = tct.shard_weight(inp.weights[name], spec, rank, tp).requires_grad_(True)
    out = tct.canonical_row_parallel_linear(x, w, spec, collective=collective, gemm=gemm)
    out.backward(inp.grads[name])
    return out.detach(), x.grad, w.grad


def run_group(inp: Inputs, names, collective, tp: int, rank: int, gemm=tct.fp32_reference_gemm):
    specs = [inp.specs[n] for n in names]
    x = inp.hidden.clone().requires_grad_(True)
    ws = [
        tct.shard_weight(inp.weights[n], s, rank, tp).requires_grad_(True)
        for n, s in zip(names, specs, strict=True)
    ]
    outs = tct.canonical_column_parallel_group(x, ws, specs, collective=collective, gemm=gemm)
    grads = [
        tct.shard_activation(inp.grads[n], s, rank, tp) for n, s in zip(names, specs, strict=True)
    ]
    torch.autograd.backward(outs, grads)
    return [o.detach() for o in outs], x.grad, [w.grad for w in ws]


def _collective_bytes(specs, tokens: int) -> int:
    """Largest FP32 payload: a two-lane column group input gradient or a row output."""

    # Row outputs are [tokens, out]; column-group input gradients are [2, tokens, in].
    widest = max(
        s.out_features if s.kind == tct.ROW_PARALLEL else s.in_features for s in specs.values()
    )
    return 2 * tokens * widest * 4


def _make_collective(group, kind: str, device, min_size_bytes: int):
    if kind == "torch" or device.type == "cpu":
        from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective

        return TorchDistributedDeterministicCollective(group=group, device=device)
    from rl_engine.distributed.algorithms.collectives import collective_for_group

    return collective_for_group(group, min_size_bytes=min_size_bytes, device=device)


def _eq(checks: list, label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    ok = actual.dtype == expected.dtype and torch.equal(actual, expected)
    detail = ""
    if not ok and actual.shape == expected.shape:
        diff = (actual.float() - expected.float()).abs()
        detail = f"max_abs={diff.max().item():.3e} mismatched={int((diff != 0).sum())}"
    checks.append({"check": label, "pass": bool(ok), "detail": detail})


def check_tp(inp: Inputs, tp: int, rank: int, collective, reference) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name in ROW_NAMES:
        spec = inp.specs[name]
        out, gx, gw = run_row(inp, name, collective, tp, rank)
        ref_out, ref_gx, ref_gw = reference["row"][name]
        _eq(checks, f"TP{tp} {name} forward", out, ref_out)
        _eq(checks, f"TP{tp} {name} grad_input", gx, tct.shard_activation(ref_gx, spec, rank, tp))
        _eq(checks, f"TP{tp} {name} grad_weight", gw, tct.shard_weight(ref_gw, spec, rank, tp))
    for group, names in tct.MUSE_COLUMN_GROUPS.items():
        outs, gx, gws = run_group(inp, names, collective, tp, rank)
        ref_outs, ref_gx, ref_gws = reference["group"][group]
        for n, o, r in zip(names, outs, ref_outs, strict=True):
            _eq(checks, f"TP{tp} {n} forward", o, tct.shard_activation(r, inp.specs[n], rank, tp))
        _eq(checks, f"TP{tp} {group} group grad_input", gx, ref_gx)
        for n, g, r in zip(names, gws, ref_gws, strict=True):
            _eq(checks, f"TP{tp} {n} grad_weight", g, tct.shard_weight(r, inp.specs[n], rank, tp))
    return checks


def tp1_reference(inp: Inputs) -> dict[str, Any]:
    return {
        "row": {name: run_row(inp, name, None, 1, 0) for name in ROW_NAMES},
        "group": {
            g: run_group(inp, names, None, 1, 0) for g, names in tct.MUSE_COLUMN_GROUPS.items()
        },
    }


def tp1_checks(inp: Inputs, reference) -> list[dict[str, Any]]:
    """FP64 correctness, repeat-run determinism and batch invariance at TP1."""

    checks: list[dict[str, Any]] = []
    again = tp1_reference(inp)
    for name in ROW_NAMES:
        for i, part in enumerate(("forward", "grad_input", "grad_weight")):
            _eq(
                checks,
                f"TP1 repeat {name} {part}",
                again["row"][name][i],
                reference["row"][name][i],
            )
        x, w = inp.inputs[name].double(), inp.weights[name].double()
        g = inp.grads[name].double()
        for part, actual, exact in (
            ("forward", reference["row"][name][0], x @ w.t()),
            ("grad_input", reference["row"][name][1], g @ w),
            ("grad_weight", reference["row"][name][2], g.t() @ x),
        ):
            err = ((actual.double() - exact).norm() / exact.norm()).item()
            # One BF16 rounding of an FP32 result: relative error ~ 2^-9 in the norm.
            checks.append(
                {
                    "check": f"TP1 {name} {part} vs FP64",
                    "pass": err < 4e-3,
                    "detail": f"normalized_error={err:.3e}",
                }
            )
        # Batch invariance: the first rows of a batch equal the same rows run alone.
        half = max(1, inp.inputs[name].size(0) // 2)
        spec = inp.specs[name]
        sub = tct.row_parallel_forward(inp.inputs[name][:half], inp.weights[name], spec)
        _eq(checks, f"TP1 {name} batch invariance", sub, reference["row"][name][0][:half])
    for group, names in tct.MUSE_COLUMN_GROUPS.items():
        _eq(
            checks,
            f"TP1 repeat {group} group grad_input",
            again["group"][group][1],
            reference["group"][group][1],
        )
        x = inp.hidden.double()
        exact = sum(inp.grads[n].double() @ inp.weights[n].double() for n in names)
        err = ((reference["group"][group][1].double() - exact).norm() / exact.norm()).item()
        checks.append(
            {
                "check": f"TP1 {group} group grad_input vs FP64",
                "pass": err < 4e-3,
                "detail": f"normalized_error={err:.3e}",
            }
        )
        for n, out in zip(names, reference["group"][group][0], strict=True):
            exact_out = x @ inp.weights[n].double().t()
            err = ((out.double() - exact_out).norm() / exact_out.norm()).item()
            checks.append(
                {
                    "check": f"TP1 {n} forward vs FP64",
                    "pass": err < 4e-3,
                    "detail": f"normalized_error={err:.3e}",
                }
            )
    return checks


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_tp(inp: Inputs, tp: int, rank: int, group, collective, device, iters: int):
    """Strict canonical forward vs native BF16 linear + native all_reduce, per row projection."""

    rows = []
    for name in ROW_NAMES:
        spec = inp.specs[name]
        x = tct.shard_activation(inp.inputs[name], spec, rank, tp)
        w = tct.shard_weight(inp.weights[name], spec, rank, tp)

        def strict(x=x, w=w, spec=spec):
            return tct.row_parallel_forward(x, w, spec, collective=collective)

        def native(x=x, w=w):
            y = torch.nn.functional.linear(x, w)
            if tp > 1:
                dist.all_reduce(y, group=group)
            return y

        timings = {}
        for label, fn in (("strict_ms", strict), ("native_ms", native)):
            for _ in range(3):
                fn()
            _sync(device)
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            _sync(device)
            timings[label] = 1e3 * (time.perf_counter() - start) / iters
        rows.append(
            {
                "projection": name,
                "tp": tp,
                **timings,
                "strict_over_native": timings["strict_ms"] / timings["native_ms"],
            }
        )
    return rows


def run(rank: int, world: int, args) -> int:
    device = torch.device(
        "cpu" if args.device == "cpu" else f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    specs = _specs(args.shapes)
    inp = Inputs(specs, args.tokens, device, args.seed)
    reference = tp1_reference(inp)
    report: dict[str, Any] = {"checks": [], "bench": [], "provenance": {}}
    if rank == 0:
        report["checks"] += tp1_checks(inp, reference)
    for tp in args.tp:
        if tp > world:
            raise SystemExit(f"TP{tp} needs {tp} ranks, launched {world}")
        group = dist.new_group(list(range(tp)))  # every rank must call new_group
        if rank < tp:
            collective = _make_collective(
                group, args.collective, device, _collective_bytes(specs, args.tokens)
            )
            checks = check_tp(inp, tp, rank, collective, reference)
            report["checks"] += [dict(c, rank=rank) for c in checks]
            report["provenance"][f"TP{tp}"] = tct.tp_canonical_provenance(collective)
            if args.bench:
                bench = bench_tp(inp, tp, rank, group, collective, device, args.iters)
                if rank == 0:
                    report["bench"] += bench
        dist.barrier()

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, report)
    failed = sum(not c["pass"] for r in gathered for c in r["checks"])
    if rank == 0:
        merged = _summary(
            args,
            device,
            world,
            [c for r in gathered for c in r["checks"]],
            gathered[0]["bench"],
            {k: v for r in gathered for k, v in r["provenance"].items()},
        )
        merged["transport"] = "nccl" if device.type == "cuda" else "gloo"
        _emit(merged, args)
    return 1 if failed else 0


def _spawn_entry(rank: int, world: int, args, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=init_file, rank=rank, world_size=world)
    try:
        code = run(rank, world, args)
    finally:
        dist.destroy_process_group()
    if code:
        raise SystemExit(code)


SIMULATED_BACKEND_ID = "threaded_balanced_tree_simulation"


class ThreadedBalancedTreeCollective:
    """In-process stand-in for the fixed-tree collective: one thread per TP rank.

    Ranks exchange their inputs through shared memory and every rank folds them with the
    production ``TorchDistributedDeterministicCollective._balanced_tree_sum``, so the
    reduction expression is the real one; only the transport is simulated. Used where Gloo
    process groups are unavailable (Windows CPU wheels).
    """

    backend_id = SIMULATED_BACKEND_ID

    def __init__(self, shared: dict[str, Any], rank: int) -> None:
        self.shared, self.rank = shared, rank
        self.world_size = shared["world"]

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective

        barrier, slots = self.shared["barrier"], self.shared["slots"]
        slots[self.rank] = input.detach().clone()
        barrier.wait()
        rank_inputs = torch.stack([slots[r] for r in range(self.world_size)])
        barrier.wait()
        return TorchDistributedDeterministicCollective._balanced_tree_sum(rank_inputs).clone()


def run_threads(args) -> int:
    import threading

    tct.FIXED_TREE_COLLECTIVES = tct.FIXED_TREE_COLLECTIVES | {SIMULATED_BACKEND_ID}
    device = torch.device("cpu")
    inp = Inputs(_specs(args.shapes), args.tokens, device, args.seed)
    reference = tp1_reference(inp)
    checks = tp1_checks(inp, reference)
    provenance = {}
    for tp in args.tp:
        shared = {"world": tp, "barrier": threading.Barrier(tp), "slots": {}}
        results: dict[int, Any] = {}

        def worker(rank: int, shared=shared, tp=tp, results=results) -> None:
            try:
                collective = ThreadedBalancedTreeCollective(shared, rank)
                results[rank] = (
                    check_tp(inp, tp, rank, collective, reference),
                    tct.tp_canonical_provenance(collective),
                )
            except BaseException as exc:  # surface the failure, unblock the peers
                shared["barrier"].abort()
                results[rank] = exc

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rank in range(tp):
            if isinstance(results[rank], BaseException):
                raise results[rank]
            rank_checks, prov = results[rank]
            checks += [dict(c, rank=rank) for c in rank_checks]
        provenance[f"TP{tp}"] = prov
    report = _summary(args, device, max(args.tp), checks, [], provenance)
    report["transport"] = SIMULATED_BACKEND_ID
    return _emit(report, args)


def _summary(args, device, world, checks, bench, provenance) -> dict[str, Any]:
    return {
        "row": "tp_canonical_tree",
        "rfc": "RL-Align/RL-Kernel#501",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "shapes": args.shapes,
        "tokens": args.tokens,
        "seed": args.seed,
        "world_size": world,
        "tp": args.tp,
        "checks": checks,
        "bench": bench,
        "provenance": provenance,
        "failed": sum(not c["pass"] for c in checks),
    }


def _emit(report: dict[str, Any], args) -> int:
    for c in report["checks"]:
        if not c["pass"]:
            print(f"FAIL rank{c.get('rank', 0)} {c['check']} {c['detail']}")
    for b in report["bench"]:
        print(
            f"bench TP{b['tp']} {b['projection']}: strict {b['strict_ms']:.3f} ms, "
            f"native {b['native_ms']:.3f} ms ({b['strict_over_native']:.2f}x)"
        )
    total = len(report["checks"])
    print(f"{total - report['failed']}/{total} checks passed")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.report}")
    return 1 if report["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--shapes", choices=("tiny", "muse"), default="muse")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--collective", choices=("auto", "torch"), default="auto")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--launcher",
        choices=("auto", "gloo", "threads"),
        default="auto",
        help="CPU only: Gloo processes, or one thread per rank (auto picks threads on Windows)",
    )
    args = parser.parse_args(argv)
    for tp in args.tp:
        tct.leaves_per_rank(tp)

    if "RANK" in os.environ:  # torchrun
        backend = "nccl" if args.device == "cuda" else "gloo"
        dist.init_process_group(backend)
        try:
            return run(dist.get_rank(), dist.get_world_size(), args)
        finally:
            dist.destroy_process_group()

    if args.device != "cpu":
        parser.error("CUDA runs are launched with torchrun --nproc-per-node 8")
    if args.launcher == "threads" or (args.launcher == "auto" and sys.platform == "win32"):
        return run_threads(args)
    import tempfile

    import torch.multiprocessing as mp

    world = max(args.tp)
    with tempfile.TemporaryDirectory() as tmp:
        init_file = (Path(tmp) / "init").as_uri()
        try:
            mp.spawn(_spawn_entry, args=(world, args, init_file), nprocs=world, join=True)
        except mp.ProcessRaisedException as exc:
            print(exc, file=sys.stderr)
            return 1
        except mp.ProcessExitedException:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
