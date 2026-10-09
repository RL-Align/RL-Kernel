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


def _region(gate_fn, norm_fn, residual, y, weight, table):
    """One block's MLP-side SP region: ``norm2(residual + gate_msa * y)`` with modulation."""

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = table.chunk(6, dim=-1)
    hidden = gate_fn(residual, y, gate_msa)
    return norm_fn(hidden, weight, shift_mlp, scale_mlp)


def ws1_sp_region(residual, y, weight, table, index, grad) -> dict[str, torch.Tensor]:
    """The SP region on one GPU with the WS1 ops (the reference for ``sp_norm_adaln``)."""

    from rl_engine.kernels.ops.cuda.h3.gate_residual import H3GateResidualCudaOp
    from rl_engine.kernels.ops.cuda.h3.rmsnorm import H3RMSNormCudaOp

    gate_op, norm_op = H3GateResidualCudaOp(), H3RMSNormCudaOp()
    leaves = [t.detach().clone().requires_grad_(True) for t in (residual, y, weight, table)]
    out = _region(
        lambda r, yy, g: gate_op(r, yy, g, index),
        lambda h, w, sh, sc: norm_op.forward_modulated(h, w, sh, sc, index),
        *leaves,
    )
    out.backward(grad)
    names = ("d_residual", "d_y", "d_weight", "d_table")
    return {"out": out.detach(), **{n: leaf.grad for n, leaf in zip(names, leaves)}}


def sp_region_rank(collective, residual, y, weight, table, index, grad) -> dict[str, Any]:
    """One SP rank: its rows of the region and its (replicated) parameter gradients."""

    from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import H3SPNormAdaLNCudaOp

    op = H3SPNormAdaLNCudaOp(collective, seq_len=residual.shape[1], batch=residual.shape[0])
    rows = slice(op.layout.lo, op.layout.hi)
    local = [t[:, rows].contiguous() for t in (residual, y)]
    leaves = [t.detach().clone().requires_grad_(True) for t in (*local, weight, table)]
    out = _region(
        lambda r, yy, g: op.gate_residual(r, yy, g, index),
        lambda h, w, sh, sc: op.norm_modulated(h, w, sh, sc, index),
        *leaves,
    )
    out.backward(grad[:, rows].contiguous())
    names = ("d_residual", "d_y", "d_weight", "d_table")
    return {
        "out": out.detach(),
        **{n: leaf.grad for n, leaf in zip(names, leaves)},
        "readback": op.readback(),
    }


def sp_matches_ws1(ws1: dict[str, torch.Tensor], ranks: list[dict[str, Any]]) -> dict[str, bool]:
    """Rows: each rank's slice equals WS1's; parameters: every rank's copy equals WS1's."""

    def rows(key):
        return all(
            torch.equal(r[key], ws1[key][:, slice(*r["readback"]["positions"])]) for r in ranks
        )

    return {
        "out": rows("out"),
        "d_residual": rows("d_residual"),
        "d_y": rows("d_y"),
        "d_weight": all(torch.equal(r["d_weight"], ws1["d_weight"]) for r in ranks),
        "d_table": all(torch.equal(r["d_table"], ws1["d_table"]) for r in ranks),
    }


def sp_case(batch, seq, hidden=5376, num_t=3, *, layout="block", dtype=torch.bfloat16, seed=0):
    """Inputs of the SP region; identical on every rank (CPU generator)."""

    from rl_engine.testing.h3_cases import h3_block_layout, h3_packed_layout

    g = torch.Generator(device="cpu").manual_seed(seed)
    make = h3_block_layout if layout == "block" else h3_packed_layout
    ti, tags = make(seq, num_t, seed=seed)
    return {
        "residual": torch.randn(batch, seq, hidden, generator=g).to(dtype).cuda(),
        "y": (torch.randn(batch, seq, hidden, generator=g) * 3).to(dtype).cuda(),
        "weight": (torch.rand(hidden, generator=g) + 0.5).to(dtype).cuda(),
        "table": (torch.randn(3 * num_t, 6 * hidden, generator=g) * 0.5).to(dtype).cuda(),
        "index": (ti * 3 + tags).cuda(),
        "grad": torch.randn(batch, seq, hidden, generator=g).to(dtype).cuda(),
    }


def naive_sp_mismatch(case: dict[str, torch.Tensor], sp: int) -> dict[str, float]:
    """Fraction of parameter-gradient elements where a naive SP backward differs from WS1.

    Naive: each rank runs the WS1 backward on its own rows, then the per-rank results
    are summed in rank order (what a plain all-reduce of local gradients does).
    """

    from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import sp_row_layout

    ws1 = ws1_sp_region(**case)
    parts = []
    for rank in range(sp):
        lay = sp_row_layout(case["residual"].shape[1], case["residual"].shape[0], sp, rank)
        part = slice(lay.lo, lay.hi)
        local = {k: (v[:, part] if k in ("residual", "y", "grad") else v) for k, v in case.items()}
        local["index"] = case["index"][part]
        parts.append(ws1_sp_region(**local))
    out = {}
    for key in ("d_weight", "d_table"):
        total = parts[0][key].float()
        for p in parts[1:]:
            total = total + p[key].float()
        out[key] = (total.to(ws1[key].dtype) != ws1[key]).float().mean().item()
    return out


def sp_region_sweep(collective, cases: list[dict[str, Any]], iters: int = 20) -> dict[str, Any]:
    """Evidence for one SP rank: byte equality, rows exchanged and time, per case."""

    from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import _segment_tiles
    from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import _dweight_tiles, _Plan, sp_row_layout

    results = []
    for spec in cases:
        case = sp_case(**spec)
        batch, seq = case["residual"].shape[:2]
        ws1 = ws1_sp_region(**case)
        mine = sp_region_rank(collective, **case)
        equal = sp_matches_ws1(ws1, [mine])
        lay = sp_row_layout(seq, batch, collective.world_size, collective.rank)
        tiles = _segment_tiles(case["index"].repeat(batch), case["table"].shape[0])
        plan = _Plan(lay, [_dweight_tiles(batch * seq, "cuda"), tuple(tiles[:3])], "cuda")
        del ws1, mine
        torch.cuda.empty_cache()
        results.append(
            {
                **spec,
                "equal": equal,
                "local_rows": batch * lay.local_len,
                "rows_sent": int(plan.send_local.numel()),
                "rows_gathered_per_rank": plan.max_send,
                **_region_times(collective, case, lay, iters),
            }
        )
    return {"cases": results}


def _region_times(collective, case, lay, iters) -> dict[str, float]:
    """Forward + backward time of the region (leaves prepared once, grads reset per call)."""

    from rl_engine.kernels.ops.cuda.h3.gate_residual import H3GateResidualCudaOp
    from rl_engine.kernels.ops.cuda.h3.rmsnorm import H3RMSNormCudaOp
    from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import H3SPNormAdaLNCudaOp
    from rl_engine.testing.h3_report import time_us

    index, rows = case["index"], slice(lay.lo, lay.hi)
    sp_op = H3SPNormAdaLNCudaOp(collective, seq_len=lay.seq_len, batch=lay.batch)
    gate_op, norm_op = H3GateResidualCudaOp(), H3RMSNormCudaOp()

    def leaves(local):
        acts = [case[k][:, rows] if local else case[k] for k in ("residual", "y")]
        return [
            t.detach().contiguous().requires_grad_(True)
            for t in (*acts, case["weight"], case["table"])
        ]

    def step(fns, args, grad):
        for leaf in args:
            leaf.grad = None
        _region(*fns, *args).backward(grad)

    sp_fns = (
        lambda r, yy, g: sp_op.gate_residual(r, yy, g, index),
        lambda h, w, sh, sc: sp_op.norm_modulated(h, w, sh, sc, index),
    )
    ws1_fns = (
        lambda r, yy, g: gate_op(r, yy, g, index),
        lambda h, w, sh, sc: norm_op.forward_modulated(h, w, sh, sc, index),
    )
    sp_args, ws1_args = leaves(True), leaves(False)
    sp_grad = case["grad"][:, rows].contiguous()
    return {
        "sp_forward_backward_us": time_us(lambda: step(sp_fns, sp_args, sp_grad), 3, iters),
        "ws1_forward_backward_us": time_us(lambda: step(ws1_fns, ws1_args, case["grad"]), 3, iters),
    }


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
        deadline = time.monotonic() + timeout
        status = {}
        try:
            for proc in procs:
                proc.start()
                started.append(proc)
            while len(status) < world or any(proc.is_alive() for proc in procs):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"rank results timed out after {timeout}s")
                try:
                    result = queue.get(timeout=min(0.1, remaining))
                except Empty:
                    exited = [
                        (rank, proc.exitcode)
                        for rank, proc in enumerate(procs)
                        if proc.exitcode is not None and (proc.exitcode != 0 or rank not in status)
                    ]
                    if not exited:
                        continue
                    # A worker may have flushed its final message between the
                    # timed read and the exit-code check.
                    try:
                        result = queue.get_nowait()
                    except Empty:
                        rank, code = exited[0]
                        raise RuntimeError(
                            f"rank {rank} exited with code {code} without completing"
                        ) from None
                if "error" in result:
                    raise RuntimeError(f"rank {result['rank']} failed:\n{result['error']}")
                status[result["rank"]] = result
            for rank, proc in enumerate(procs):
                proc.join()
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
