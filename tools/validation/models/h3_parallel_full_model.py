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
that must be reported at exactly that block and row with that owner.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_parallel_model as pm
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


def _run(model, base, grid, sp, ops, gradients, digest_leaf, *, inject=None):
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
    with pm.parallel_model_execution(model, grid, sp=sp, trace=trace, ops=ops) as state:
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
    model.enable_gradient_checkpointing()
    for p in model.parameters():
        p.requires_grad_(False)  # the shard leaves carry the weight gradients
    load_seconds = time.perf_counter() - started

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
    )
    expected = reference["leaf_digests"]
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
        inject=inject,
    )

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
    mismatched = [
        key for key, digest in observed["leaf_digests"].items() if digest != expected.get(key)
    ]
    # every block's every sharded projection, in both runs: an empty comparison is not a pass
    shard_count = len(model.transformer_blocks) * len(pm.SHARDED_WEIGHTS)
    checks["weight_shard_gradients_bitwise"] = (
        not mismatched and len(expected) == len(observed["leaf_digests"]) == shard_count
    )
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "checks": checks,
        "weight_shards_compared": len(expected),
        "weight_shards_mismatched": mismatched[:10],
        "first_drift_forward": drift_forward,
        "first_drift_backward": drift_backward,
        "seconds": {
            "load": load_seconds,
            "canonical": reference["seconds"],
            "parallel": observed["seconds"],
        },
        "readback": pm.grid_readback(grid, sp=args.sp, ops=ops),
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
