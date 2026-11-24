# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Full 50-block H3 model on a TP x CP (x SP) grid against its single-GPU canonical run.

Launch one process per GPU, for example on one 8-GPU node::

    srun -N1 --ntasks=8 python tools/validation/models/h3_parallel_full_model.py \\
        --checkpoint <dir with transformer/> --tp 4 --cp 2 --sp --cp-mode ring --size 928 \\
        --out evidence/ws2_full_tp4_cp2_sp

Every rank loads the real checkpoint, runs the model with every block on the canonical
``Grid(1, 1, 0, 0)`` (``h3_parallel_model.parallel_model_execution``), then on the requested grid,
and compares, bit for bit: the video / audio velocity, the gradients of the three inputs, every
block's output and output gradient (first drift with owning rank and logical row, if any), and
this rank's weight-shard gradients of all 350 sharded projections (device checksums). With
``--inject BLOCK:ROW:RANK`` that rank perturbs one row of the block's output: a negative control
that must be reported at exactly that block and row with that owner. The rank must hold the row;
without SP, where the TP ranks of a CP rank hold the same rows, it must be the last of them, whose
copy the assembled output keeps (anything else is refused).

With ``--sharded`` the grid run uses a model loaded by ``h3_sharded_load.load_sharded_model``:
after the canonical run the full model is freed, and each rank loads only its shards of the
sharded projections (the rest of the model in full), so the comparison above holds for a grid
run that never holds those full weights. The canonical run before it still loads the full model
on every GPU, so each GPU needs the full model's memory for that phase. ``--provider-boundaries``
leaves the boundary modules on the provider in both runs, for GPUs the strict boundary rows do not
support (they need SM90).

``--lora-rank R`` puts a seeded LoRA (``h3_lora.make_block_lora``: A normal with std 1/R, B
random or, with ``--lora-b-zero``, zero; scale ``--lora-alpha / R``) on the six targets of every
block, cut for each run's grid (``h3-parallel-lora-v2``), and compares every LoRA shard's
gradient too (``lora_gradients_bitwise``). The two token-refiner blocks are not on the grid and
carry no LoRA
here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_sharded_load as sl
from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_full_model import FullModelTrace

# Layout name -> sized_inputs arguments (None: the 12-token layout of the full-model harness).
SIZES = {
    "12": None,
    "40": dict(text=8, frames=1, side=8, audio=8),
    "925": dict(text=77, frames=3, side=32, audio=40),
    "928": dict(text=80, frames=3, side=32, audio=80),
}
INPUTS = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")


def _inputs(size: str):
    """The packed request of ``size`` on CUDA."""
    from tools.validation.models.h3_full_model import packed_inputs
    from tools.validation.models.h3_runtime_report import sized_inputs

    return packed_inputs("cuda") if SIZES[size] is None else sized_inputs("cuda", **SIZES[size])


def _run(model, base, grid, sp, ops, gradients, digest_leaf, *, strict, inject=None):
    """Forward + backward on ``grid``: outputs, input gradients, snapshots, shard-gradient digests.

    ``digest_leaf(name, grad)`` reduces each weight-shard gradient to its digest as it is
    accumulated (the run never holds more than one block's worth of them).
    """

    inputs = {
        k: (v.detach().clone().requires_grad_() if k in INPUTS else v) for k, v in base.items()
    }
    trace = FullModelTrace()
    trace.capture = True
    began = time.perf_counter()
    with pm.parallel_model_execution(
        model, grid, sp=sp, trace=trace, ops=ops, strict_boundaries=strict
    ) as state:
        state.inject = inject
        state.digest_leaf = digest_leaf
        video, audio = model(**inputs)
        torch.autograd.backward([video, audio], list(gradients))
        torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    return {
        "video": video.detach().cpu(),
        "audio": audio.detach().cpu(),
        "input_grads": {k: inputs[k].grad.detach().cpu() for k in INPUTS},
        "snapshots": dict(trace.snapshots),
        "leaf_digests": state.digests,
        "seconds": seconds,
    }


def shard_gradient_verdict(expected: dict, observed: dict, blocks: int, *, lora: bool) -> dict:
    """Compare the canonical run's shard-gradient digests with the grid's.

    Every block's every sharded projection (and, with ``lora``, every LoRA shard) must be in
    both runs with the same digest: an empty or partial comparison is not a pass, and a key that
    only one run holds is reported as missing. ``base_compared`` / ``lora_compared`` count the
    keys both runs hold (the ones whose digests were actually compared).
    """

    missing = sorted(set(expected) ^ set(observed))
    mismatched = sorted(k for k in set(expected) & set(observed) if expected[k] != observed[k])
    lora_keys = sorted(k for k in expected if ".lora." in k)
    base_keys = [k for k in expected if ".lora." not in k]
    base_mismatched = [k for k in mismatched if ".lora." not in k]
    lora_mismatched = [k for k in mismatched if ".lora." in k]
    base_missing = [k for k in missing if ".lora." not in k]
    lora_missing = [k for k in missing if ".lora." in k]
    both = set(expected) & set(observed)
    return {
        "missing": missing,
        "base_compared": sum(1 for k in both if ".lora." not in k),
        "lora_compared": sum(1 for k in both if ".lora." in k),
        "base_mismatched": base_mismatched,
        "lora_mismatched": lora_mismatched,
        "lora_keys": lora_keys,
        "base_bitwise": not base_mismatched
        and not base_missing
        and len(base_keys) == blocks * len(pm.SHARDED_WEIGHTS),
        "lora_bitwise": lora
        and not lora_mismatched
        and not lora_missing
        and len(lora_keys) == blocks * h3_lora.LEAVES_PER_BLOCK,
    }


def _prepare(model) -> None:
    """Checkpoint the blocks and freeze the parameters: the shard leaves carry the gradients."""
    model.enable_gradient_checkpointing()
    for p in model.parameters():
        p.requires_grad_(False)


def _resident_bytes(model) -> int:
    """Bytes of the parameters and stored projection shards actually on the device."""
    total = sum(p.numel() * p.element_size() for p in model.parameters() if not p.is_meta)
    for block in model.transformer_blocks:
        stored = getattr(block, pm.SHARDS_ATTR, None)
        if stored is not None:
            total += sum(t.numel() * t.element_size() for t in stored["tensors"].values())
    return total


def main() -> None:
    """Run the canonical model and the grid on this rank, compare them, write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--attention", choices=("interim", "h3_full"), default="h3_full")
    parser.add_argument("--cp-mode", choices=("ulysses", "ring"), default="ring")
    parser.add_argument("--size", choices=tuple(SIZES), default="40")
    parser.add_argument("--local-size", type=int, default=8, help="GPUs per node")
    parser.add_argument("--inject", default="", help="BLOCK:ROW:RANK negative control")
    parser.add_argument(
        "--sharded", action="store_true", help="run the grid on a model holding only its shards"
    )
    parser.add_argument(
        "--provider-boundaries", action="store_true", help="boundary modules on the provider"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="seeded LoRA on the six targets (0: none)"
    )
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-seed", type=int, default=0)
    parser.add_argument("--lora-b-zero", action="store_true", help="B = 0, the recipe's init")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29521')}",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local),
    )
    if world != args.tp * args.cp:
        raise SystemExit(f"world {world} != tp {args.tp} x cp {args.cp}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from diffusers import MiniMaxH3Transformer3DModel

    started = time.perf_counter()
    model = (
        MiniMaxH3Transformer3DModel.from_pretrained(
            str(args.checkpoint / "transformer"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    _prepare(model)
    load_seconds = time.perf_counter() - started
    strict = not args.provider_boundaries
    lora_kw = dict(
        rank=args.lora_rank, alpha=args.lora_alpha, seed=args.lora_seed, b_zero=args.lora_b_zero
    )
    lora_bytes = {}
    if args.lora_rank:
        lora_bytes["canonical"] = h3_lora.attach_lora(model, 1, 0, **lora_kw)

    base = _inputs(args.size)
    tags = base["token_tags"].cpu()
    length = int(tags.numel())
    gen = torch.Generator().manual_seed(99)
    ops = hb.CandidateOps()

    def grad_like(shape):
        return torch.randn(shape, generator=gen).bfloat16().cuda()

    with torch.no_grad():
        probe = model(**{k: (v.detach() if k in INPUTS else v) for k, v in base.items()})
    gradients = [grad_like(probe[0].shape), grad_like(probe[1].shape)]
    del probe
    torch.cuda.empty_cache()

    spans = pm.rank_spans(args.tp, args.cp, length, args.sp)
    grid = h3_tp.make_grid(
        args.tp,
        args.cp,
        local_size=args.local_size if args.tp > args.local_size else None,
        attention=args.attention,
        cp_mode=args.cp_mode,
    )
    canonical = h3_tp.Grid(1, 1, 0, 0, attention=args.attention)
    # The canonical run reduces each unsharded gradient to the digest of this rank's shard of it.
    reference = _run(
        model,
        base,
        canonical,
        False,
        ops,
        gradients,
        lambda name, grad: pm.exact_digest(pm.shard_of(grad, name, args.tp, grid.tp_rank)),
        strict=strict,
    )
    expected = reference["leaf_digests"]
    full_param_bytes = _resident_bytes(model)
    sharded = None
    if args.sharded:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = sl.load_sharded_model(
            args.checkpoint / "transformer", grid.tp, grid.tp_rank, "cuda"
        ).eval()
        _prepare(model)
        if args.lora_rank:  # the sharded model starts without a LoRA; give it the grid's
            h3_lora.attach_lora(model, grid.tp, grid.tp_rank, **lora_kw)
        sharded = {
            "load_seconds": time.perf_counter() - started,
            "resident_param_bytes": _resident_bytes(model),
            "full_model_param_bytes": full_param_bytes,
        }
    if args.lora_rank and not args.sharded:
        h3_lora.attach_lora(model, grid.tp, grid.tp_rank, **lora_kw)
    if args.lora_rank:
        lora_bytes["grid"] = sum(
            t.numel() * t.element_size()
            for block in model.transformer_blocks
            for parts in getattr(block, pm.LORA_ATTR)["tensors"].values()
            for t in parts.values()
        )
    torch.cuda.empty_cache()
    inject = None
    if args.inject:
        block, row, victim = (int(v) for v in args.inject.split(":"))
        inject = (block, row, victim)
    observed = _run(
        model,
        base,
        grid,
        args.sp,
        ops,
        gradients,
        lambda name, grad: pm.exact_digest(grad),
        strict=strict,
        inject=inject,
    )
    if sharded is not None:
        sharded["parallel_peak_bytes"] = torch.cuda.max_memory_allocated()

    checks = {
        "video_velocity_bitwise": torch.equal(reference["video"], observed["video"]),
        "audio_velocity_bitwise": torch.equal(reference["audio"], observed["audio"]),
    }
    for k in INPUTS:
        checks[f"input_grad.{k}"] = torch.equal(
            reference["input_grads"][k], observed["input_grads"][k]
        )
    forward_order = [f"block.{i}" for i in range(50)]
    drift_forward = pm.first_drift(
        reference["snapshots"], observed["snapshots"], spans, tags, forward_order
    )
    drift_backward = pm.first_drift(
        reference["snapshots"],
        observed["snapshots"],
        spans,
        tags,
        [f"block.{i}.grad" for i in reversed(range(50))],
    )
    checks["block_outputs_bitwise"] = drift_forward is None
    checks["block_output_gradients_bitwise"] = drift_backward is None
    verdict = shard_gradient_verdict(
        expected, observed["leaf_digests"], len(model.transformer_blocks), lora=bool(args.lora_rank)
    )
    base_mismatched, lora_mismatched = verdict["base_mismatched"], verdict["lora_mismatched"]
    checks["weight_shard_gradients_bitwise"] = verdict["base_bitwise"]
    if args.lora_rank:
        checks["lora_gradients_bitwise"] = verdict["lora_bitwise"]
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "checks": checks,
        "weight_shards_compared": verdict["base_compared"],
        "weight_shards_mismatched": base_mismatched[:10],
        "lora_shards_compared": verdict["lora_compared"],
        "lora_shards_mismatched": lora_mismatched[:10],
        "shards_missing": verdict["missing"][:10],
        "lora_bytes": lora_bytes or None,
        "first_drift_forward": drift_forward,
        "first_drift_backward": drift_backward,
        "seconds": {
            "load": load_seconds,
            "canonical": reference["seconds"],
            "parallel": observed["seconds"],
        },
        "readback": pm.grid_readback(grid, sp=args.sp, ops=ops),
        "sharded": sharded,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        report = dict(
            contract=pm.CONTRACT,
            tp=args.tp,
            cp=args.cp,
            sp=args.sp,
            cp_mode=args.cp_mode,
            attention=args.attention,
            size=args.size,
            packed_rows=length,
            inject=args.inject or None,
            sharded=args.sharded,
            lora=(
                dict(contract=pl.CONTRACT, **lora_kw, scale=args.lora_alpha / args.lora_rank)
                if args.lora_rank
                else None
            ),
            strict_boundaries=strict,
            world=world,
            spans=spans,
            all_bitwise=all(all(r["checks"].values()) for r in gathered),
            ranks=gathered,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
        )
        (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(
            json.dumps({k: report[k] for k in ("tp", "cp", "sp", "cp_mode", "size", "all_bitwise")})
        )
    dist.barrier()
    if rank == 0:
        time.sleep(3)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
