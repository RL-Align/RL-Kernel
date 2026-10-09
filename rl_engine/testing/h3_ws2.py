# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 vs WS2 comparisons for the H3 conditioning path (tests and evidence).

The per-rank helpers run inside real multi-process groups (tests and
``scripts/h3_ws2_evidence.py``); the WS1 side runs on one GPU.
"""

from __future__ import annotations

from typing import Any

import torch


def ws1_projection(temb, weight, bias, grad) -> dict[str, torch.Tensor]:
    """The single-GPU ``adaln_projection_3mod`` table and its three gradients."""

    from rl_engine.kernels.ops.cuda.h3.adaln_projection import H3AdaLNProjectionCudaOp

    leaves = [t.detach().clone().requires_grad_(True) for t in (temb, weight, bias)]
    table = H3AdaLNProjectionCudaOp().forward_table(*leaves)
    table.backward(grad)
    return {
        "table": table.detach(),
        "d_temb": leaves[0].grad,
        "d_weight": leaves[1].grad,
        "d_bias": leaves[2].grad,
    }


def tp_projection_rank(collective, temb, weight, bias, grad) -> dict[str, Any]:
    """One TP rank: shard the full weight, run the TP op, return its view of the results."""

    from rl_engine.kernels.ops.cuda.h3.tp_adaln_projection import (
        H3TPAdaLNProjectionCudaOp,
        shard_adaln_projection,
    )

    op = H3TPAdaLNProjectionCudaOp(collective, weight.shape[0])
    w_shard, b_shard = shard_adaln_projection(weight, bias, collective.world_size, collective.rank)
    leaves = [t.detach().clone().requires_grad_(True) for t in (temb, w_shard, b_shard)]
    table = op.forward_table(*leaves)
    table.backward(grad)
    return {
        "table": table.detach(),
        "d_temb": leaves[0].grad,
        "d_weight_shard": leaves[1].grad,
        "d_bias_shard": leaves[2].grad,
        "readback": op.readback(),
    }


def tp_matches_ws1(ws1: dict[str, torch.Tensor], ranks: list[dict[str, Any]]) -> dict[str, bool]:
    """Byte equality of every rank's replicated outputs and of the concatenated shards."""

    return {
        "table": all(torch.equal(r["table"], ws1["table"]) for r in ranks),
        "d_temb": all(torch.equal(r["d_temb"], ws1["d_temb"]) for r in ranks),
        "d_weight": torch.equal(torch.cat([r["d_weight_shard"] for r in ranks]), ws1["d_weight"]),
        "d_bias": torch.equal(torch.cat([r["d_bias_shard"] for r in ranks]), ws1["d_bias"]),
    }


def tp_projection_sweep(collective, temb, weight, bias, grad, iters: int = 100) -> dict[str, Any]:
    """Evidence for one rank: per-T byte equality against WS1 on this GPU, timing, readback.

    ``temb``/``grad`` hold the largest T; every T in ``1..T`` uses their first
    rows. Each rank compares its own replicated outputs and its own shard of
    ``dW``/``db`` with the WS1 result computed here, so nothing large is shipped.
    """

    from rl_engine.kernels.ops.cuda.h3.adaln_projection import H3AdaLNProjectionCudaOp
    from rl_engine.kernels.ops.cuda.h3.tp_adaln_projection import (
        H3TPAdaLNProjectionCudaOp,
        adaln_column_shard,
    )
    from rl_engine.testing.h3_report import time_us

    shard = adaln_column_shard(weight.shape[0], collective.world_size, collective.rank)
    cols = slice(shard.begin, shard.end)
    tp_op, ws1_op = (
        H3TPAdaLNProjectionCudaOp(collective, weight.shape[0]),
        H3AdaLNProjectionCudaOp(),
    )
    equality, perf = [], []
    for num_t in range(1, temb.shape[0] + 1):
        t, g = temb[:num_t], grad[:num_t]
        ws1 = ws1_projection(t, weight, bias, g)
        mine = tp_projection_rank(collective, t, weight, bias, g)
        equality.append(
            {
                "num_timesteps": num_t,
                "table": torch.equal(mine["table"], ws1["table"]),
                "d_temb": torch.equal(mine["d_temb"], ws1["d_temb"]),
                "d_weight_shard": torch.equal(mine["d_weight_shard"], ws1["d_weight"][cols]),
                "d_bias_shard": torch.equal(mine["d_bias_shard"], ws1["d_bias"][cols]),
            }
        )
        leaves = [x.detach().clone().requires_grad_(True) for x in (t, weight[cols], bias[cols])]
        full = [x.detach().clone().requires_grad_(True) for x in (t, weight, bias)]

        def step(op, args):
            for x in args:
                x.grad = None
            op.forward_table(*args).backward(g)

        perf.append(
            {
                "num_timesteps": num_t,
                "tp_forward_us": time_us(
                    lambda: tp_op.forward_table(t, weight[cols], bias[cols]), 30, iters
                ),
                "tp_forward_backward_us": time_us(lambda: step(tp_op, leaves), 10, iters),
                "ws1_forward_us": time_us(lambda: ws1_op.forward_table(t, weight, bias), 30, iters),
                "ws1_forward_backward_us": time_us(lambda: step(ws1_op, full), 10, iters),
            }
        )
    return {"equality": equality, "perf": perf, "readback": tp_op.readback()}


# --- real multi-process groups -------------------------------------------------


def _bootstrap(rank, world, init_method, queue, inputs_path, target, kwargs):
    import traceback

    import torch.distributed as dist

    from rl_engine.distributed.collectives import DeterministicCollective

    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=init_method, rank=rank, world_size=world, device_id=rank
        )
        inputs = {k: v.cuda() for k, v in torch.load(inputs_path).items()}
        collective = DeterministicCollective(device=rank, max_size_bytes=256 << 20)
        out = target(collective, **inputs, **kwargs)
        # Results go through a file: tensors sent over the queue are shared by
        # file descriptor and vanish if this process exits before the parent reads.
        torch.save(
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in out.items()},
            inputs_path.with_name(f"rank{rank}.pt"),
        )
        collective.close()
        dist.destroy_process_group()
        queue.put({"rank": rank})
    except Exception:  # forwarded to the parent
        queue.put({"rank": rank, "error": traceback.format_exc()})


def run_world(world: int, target, inputs: dict[str, torch.Tensor], timeout: float = 900, **kwargs):
    """Run ``target(collective, **inputs, **kwargs)`` on ranks ``0..world-1`` (one GPU each).

    ``target`` must be importable (it is pickled into spawned processes). Each
    rank gets a NCCL group and a ``DeterministicCollective``; results come back
    on the CPU, ordered by rank. Any rank's exception is re-raised here.
    """

    import tempfile
    import time
    from pathlib import Path
    from queue import Empty

    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        raise RuntimeError(f"needs {world} GPUs, found {torch.cuda.device_count()}")
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "inputs.pt"
        torch.save({k: v.cpu() for k, v in inputs.items()}, path)
        queue = ctx.Queue()
        init = (Path(tmp) / "init").as_uri()
        procs = [
            ctx.Process(target=_bootstrap, args=(r, world, init, queue, path, target, kwargs))
            for r in range(world)
        ]
        started = []
        try:
            for proc in procs:
                proc.start()
                started.append(proc)
            deadline = time.monotonic() + timeout
            pending = set(range(world))
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for ranks {sorted(pending)}")
                try:
                    result = queue.get(timeout=min(0.1, remaining))
                except Empty:
                    for rank in pending:
                        if procs[rank].exitcode is not None:
                            raise RuntimeError(
                                f"rank {rank} exited with code {procs[rank].exitcode} "
                                "without reporting a result"
                            ) from None
                    continue
                if "error" in result:
                    raise RuntimeError(f"rank {result['rank']} failed:\n{result['error']}")
                pending.remove(result["rank"])
            for rank, proc in enumerate(procs):
                proc.join(timeout=max(0, deadline - time.monotonic()))
                if proc.is_alive():
                    raise TimeoutError(f"rank {rank} did not exit after reporting a result")
                if proc.exitcode != 0:
                    raise RuntimeError(f"rank {rank} exited with code {proc.exitcode}")
            return [{"rank": r, **torch.load(path.with_name(f"rank{r}.pt"))} for r in range(world)]
        finally:
            for proc in started:
                if proc.is_alive():
                    proc.terminate()
            for proc in started:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            queue.close()
            queue.join_thread()
