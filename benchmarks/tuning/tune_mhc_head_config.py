# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tune aiter's mHC head kernel per M bucket without changing its bits (DSV4 RFC task 8, #508).

aiter's own head config is one config for every M (on gfx942 ``BLOCK_M=16
BLOCK_K=64 BLOCK_C=128 num_warps=4 num_stages=1``), which is batch invariant.
Batch invariance does not need one config for every M, only configs that give
the same bits (docs/operators/mhc.md lists the parameters that change them),
so this script looks for the fastest config per M among
candidates that keep the bits of aiter's head config.

- baseline: ``mhc_head_dsv4`` without a config (aiter's own head config);
- ours: the fastest candidate at that M whose output equals the baseline output
  bit for bit at that M (candidates that differ are excluded, as a safety net
  on top of the measured list of parameters that change the bits);
- comparison: ours / baseline device time.

The kernel takes M as ``tl.constexpr``, so every (candidate, M) compiles once.
Candidates whose ``BLOCK_M`` exceeds ``max(16, next_power_of_2(M))`` are
skipped: they only add masked rows. Compile time dominates, so the sweep is
split into jobs by ``--m-values``; each job rewrites its outputs after every M,
so a job stopped by its time limit keeps the M values it finished. ``--merge``
then combines the per-job JSON files.

Device time from GPU events around a CUDA graph of ITERS calls, so CPU launch
overhead (larger than the kernel at small M) is not counted; median and
min-max over TRIALS graph replays (``--timing eager``: plain launches). Forward only. Shapes follow
DeepSeek-V4-Flash-0731: n=4 streams, C=4096, bf16 residual, FP32 hc_head_fn.

Output JSON uses aiter's ``M_LEQ_x`` keys: a bucket is tuned at its upper bound
M; ``--merge`` renames the largest bucket to ``"any"``. ``--output-json`` gets
the best candidate that ``mhc_head_dsv4``'s public ``config`` can express
(aiter's num_warps and num_stages), ``--output-json-all-parameters`` the best
of all candidates. ``num_warps`` and
``num_stages`` are not accepted by ``mhc_head_dsv4``'s public ``config`` (the
wrapper passes its own), so candidates are launched through
``launch_mhc_head``, which mirrors aiter's wrapper with every launch parameter
taken from the config.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/tuning/tune_mhc_head_config.py --m-values 1 16 64 256 \\
        --output-json part.json --output part.md
    python benchmarks/tuning/tune_mhc_head_config.py --merge part1.json part2.json \\
        --output-json MHC_HEAD-n=4-C=4096.json
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import socket
import statistics
import time

N, C = 4, 4096
EPS = 1e-6
DEV = "cuda"
M_VALUES = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
WARMUP, ITERS, TRIALS = 10, 20, 5
TIMING = "graph"  # or "eager" (--timing): plain launches between the events

# Allowed values per parameter: those measured on MI325X to keep the bits of
# aiter's head config (docs/operators/mhc.md). Every other value of
# BLOCK_M, BLOCK_K, BLOCK_C and num_warps changes the bits or cannot launch.
CANDIDATE_VALUES = {
    "BLOCK_M": [16],
    "BLOCK_K": [64],
    "BLOCK_C": [128],
    "num_warps": [4],
    "num_stages": [1, 2, 3],
    "waves_per_eu": [0, 1, 2, 4],
}

# Triton's AMD launch options that aiter's head wrapper leaves at their default.
TRITON_LAUNCH_DEFAULTS = {"waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1}


def aiter_head_config():
    """The head config aiter's wrapper resolves on this device, as one flat dict.

    Imports torch and aiter lazily, like every function below that needs them,
    so ``--merge`` also runs on a login node without torch.
    """
    aiter_mhc = importlib.import_module("aiter.ops.triton.fusions.mhc")
    cfg = aiter_mhc.get_tuned_kernel_config(
        "fusions", "MHC_DSV4", "_mhc_head_kernel", aiter_mhc._HEAD_FALLBACK
    )
    return dict(cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)


def launch_mhc_head(residual, fn, scale, base, config, *, rms_eps=EPS, hc_pre_eps=EPS):
    """``_mhc_head_dsv4_forward`` of aiter 326cffa, with every launch parameter from ``config``.

    ``config`` must hold BLOCK_M, BLOCK_K and BLOCK_C; every other key
    (num_warps, num_stages, waves_per_eu, ...) is passed to the launch.
    """
    import torch
    import triton
    from aiter.ops.triton._triton_kernels.fusions import _mhc_head_kernel

    m, n, c = residual.shape
    x = residual.view(m, n * c)
    launch = dict(config)
    block_m = max(16, launch.pop("BLOCK_M"))
    block_k = max(16, min(launch.pop("BLOCK_K"), triton.next_power_of_2(n * c)))
    block_c = launch.pop("BLOCK_C")
    out = torch.empty(m, c, dtype=residual.dtype, device=residual.device)
    _mhc_head_kernel[(triton.cdiv(m, block_m),)](
        x,
        fn,
        scale,
        base,
        out,
        M=m,
        K=n * c,
        n=n,
        C=c,
        eps=rms_eps,
        pre_eps=hc_pre_eps,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_fn_n=fn.stride(0),
        stride_fn_k=fn.stride(1),
        stride_om=out.stride(0),
        stride_oc=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_C=block_c,
        N_TILE=max(16, triton.next_power_of_2(n)),
        **launch,
    )
    return out


def head_inputs(m, dtype_name="bfloat16", seed=0):
    """residual [m,n,C], fn [n,n*C] FP32, scale [1] and base [n] with nonzero values."""
    import torch

    gen = torch.Generator(device=DEV).manual_seed(seed)
    fn = torch.randn(N, N * C, generator=gen, device=DEV) / (N * C) ** 0.5
    scale = 0.5 + torch.rand(1, generator=gen, device=DEV)
    base = 0.1 * torch.randn(N, generator=gen, device=DEV)
    residual = torch.randn(m, N, C, generator=gen, device=DEV).to(getattr(torch, dtype_name))
    return residual, fn, scale, base


def bits_differ(y, reference):
    """Number of output elements whose bits differ from the reference (16-bit outputs)."""
    import torch

    return int((y.view(torch.int16) != reference.view(torch.int16)).sum())


def time_ms(call):
    """(median, min, max) device time per call over TRIALS runs of ITERS calls.

    Each run is one replay of a CUDA graph of ITERS calls, or ITERS plain
    launches when TIMING is "eager".
    """
    import torch

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    if TIMING == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(ITERS):
                call()
        run = graph.replay
    else:

        def run():
            for _ in range(ITERS):
                call()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(TRIALS):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / ITERS)
    return statistics.median(samples), min(samples), max(samples)


def candidates(m, values):
    """Every combination of ``values``, without BLOCK_M larger than M needs."""
    import triton

    keys = list(values)
    largest_block_m = max(16, triton.next_power_of_2(m))
    for combo in itertools.product(*(values[key] for key in keys)):
        config = dict(zip(keys, combo))
        if config["BLOCK_M"] <= largest_block_m:
            yield config


def label(config, reference):
    """The parameters where ``config`` differs from aiter's head config."""
    reference = dict(TRITON_LAUNCH_DEFAULTS, **reference)
    changed = [f"{key}={value}" for key, value in config.items() if reference.get(key) != value]
    return " ".join(changed) or "aiter's config"


class IsolatedRunner:
    """Runs ``work(state, item)`` in a child process that is restarted when it dies.

    Compiling some configs aborts the process inside LLVM (an assertion in
    SimplifyCFG for BLOCK_M=256 on Triton 3.7), which no ``except`` catches.
    The child takes the abort, the parent records it and starts a new child.
    ``state`` is a dict the child keeps between items; it is empty after a restart.
    """

    def __init__(self):
        import multiprocessing

        self._ctx = multiprocessing.get_context("spawn")
        self._start()

    def _start(self):
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child_conn,), daemon=True)
        self._proc.start()
        child_conn.close()

    def __call__(self, work, item):
        """``(True, result)``, or ``(False, reason)`` when the child died."""
        self._conn.send((work, item))
        try:
            return True, self._conn.recv()
        except EOFError:
            self._proc.join()
            reason = f"process aborted, exit code {self._proc.exitcode}"
            self._start()
            return False, reason

    def close(self):
        self._conn.send(None)
        self._proc.join()


def _serve(conn):
    state = {}
    while (message := conn.recv()) is not None:
        work, item = message
        conn.send(work(state, item))


def _set_m(state, m, timing):
    """Inputs and baseline output for M in ``state``, built once per M and child."""
    global TIMING
    TIMING = timing
    if state.get("m") != m:
        from aiter.ops.triton.fusions import mhc_head_dsv4

        state.clear()
        state["m"], state["inputs"] = m, head_inputs(m)
        state["baseline_out"] = mhc_head_dsv4(*state["inputs"], rms_eps=EPS, hc_pre_eps=EPS)


def _time_baseline(state, item):
    import torch
    from aiter.ops.triton.fusions import mhc_head_dsv4

    _set_m(state, *item)
    with torch.no_grad():
        return time_ms(lambda: mhc_head_dsv4(*state["inputs"], rms_eps=EPS, hc_pre_eps=EPS))


def _time_candidate(state, item):
    """('timed', result), ('differs', count) or ('cannot launch', exception name)."""
    import torch

    m, timing, config = item
    _set_m(state, m, timing)
    try:
        with torch.no_grad():
            out = launch_mhc_head(*state["inputs"], config)
            torch.cuda.synchronize()
            count = bits_differ(out, state["baseline_out"])
            if count:
                return "differs", count
            return "timed", time_ms(lambda: launch_mhc_head(*state["inputs"], config))
    except Exception as exc:  # a config can exceed LDS or registers
        return "cannot launch", type(exc).__name__


def tune_m(runner, m, values, reference):
    """Time the baseline and every candidate at M; returns the row for the report."""
    ok, baseline = runner(_time_baseline, (m, TIMING))
    if not ok:
        raise RuntimeError(f"baseline at M={m}: {baseline}")
    timed, differing, failed = [], [], []
    for config in candidates(m, values):
        name = label(config, reference)
        ok, result = runner(_time_candidate, (m, TIMING, config))
        if not ok:
            failed.append(f"{name} ({result})")
        elif result[0] == "differs":
            differing.append(f"{name} ({result[1]}/{m * C})")
        elif result[0] == "cannot launch":
            failed.append(f"{name} ({result[1]})")
        else:
            timed.append((result[1], config))
            print(f"M={m} {name}: {result[1][0]:.4f} ms", flush=True)
    timed.sort(key=lambda item: item[0][0])
    return {
        "m": m,
        "baseline": baseline,
        "timed": timed,
        "differing": differing,
        "failed": failed,
    }


def _cell(result):
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def is_public(config, reference):
    """Whether ``mhc_head_dsv4``'s public ``config`` can express ``config``.

    The wrapper always launches with its own num_warps and num_stages.
    """
    return all(config[key] == reference[key] for key in ("num_warps", "num_stages"))


def _best(row, reference, public_only):
    for result, config in row["timed"]:
        if not public_only or is_public(config, reference):
            return result, config
    return None


def to_json(rows, reference, public_only):
    table = {}
    for row in rows:
        best = _best(row, reference, public_only)
        if best:
            config = dict(best[1])
            if public_only:  # the wrapper takes these from aiter, not from the config
                del config["num_warps"], config["num_stages"]
            table[f"M_LEQ_{row['m']}"] = config
    return table


def _best_cells(row, reference, public_only):
    best = _best(row, reference, public_only)
    if best is None:
        return ["no candidate", "n/a: no bit-identical candidate", "-"]
    result, config = best
    ratio = f"{result[0] / row['baseline'][0]:.2f}x"
    return [_cell(result), ratio, f"`{label(config, reference)}`"]


def to_markdown(rows, values, reference, top):
    lines = [
        "# mHC head config tuning",
        "",
        f"- Node: `{socket.gethostname()}`, GPU: {_device_name()}",
        f"- aiter's head config (baseline, `mhc_head_dsv4` without a config): `{reference}`",
        f"- Candidate values: `{values}`",
        f"- Device time per call, ms, forward, n={N}, C={C}, bf16 residual, FP32 hc_head_fn. "
        f"GPU events around {ITERS} calls ({TIMING}: "
        + ("one CUDA graph replay" if TIMING == "graph" else "plain launches")
        + f"); median over {TRIALS} runs (min-max).",
        "- ours = fastest candidate whose output equals the baseline output bit for bit at that "
        "M; `public` = only candidates with aiter's num_warps and num_stages, which "
        "`mhc_head_dsv4`'s `config` cannot change; `all parameters` = any candidate.",
        "",
        "## Best config per M",
        "",
        "| M | baseline: aiter's head config | ours, public | ours public / baseline "
        "| ours public config | ours, all parameters | ours all / baseline "
        "| ours all config | timed / bits differ / cannot launch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        counts = f"{len(row['timed'])} / {len(row['differing'])} / {len(row['failed'])}"
        cells = (
            [str(row["m"]), _cell(row["baseline"])]
            + _best_cells(row, reference, True)
            + _best_cells(row, reference, False)
            + [counts]
        )
        lines.append("| " + " | ".join(cells) + " |")
    for row in rows:
        lines += [
            "",
            f"## M={row['m']}: fastest {top} candidates",
            "",
            "| config | ms (min-max) | / baseline |",
            "|---|---|---|",
        ]
        for result, config in row["timed"][:top]:
            lines.append(
                f"| `{label(config, reference)}` | {_cell(result)} "
                f"| {result[0] / row['baseline'][0]:.2f}x |"
            )
        if row["differing"]:
            lines += ["", "Bits differ (excluded): " + "; ".join(row["differing"])]
        if row["failed"]:
            lines += ["", "Cannot launch: " + "; ".join(row["failed"])]
    return "\n".join(lines) + "\n"


def _device_name():
    import torch

    return torch.cuda.get_device_name()


def merge(paths):
    """Union of per-job JSON files; the largest M_LEQ bucket becomes ``"any"``."""
    merged = {}
    for path in paths:
        with open(path) as f:
            merged.update(json.load(f))
    buckets = sorted(merged, key=lambda key: int(key.removeprefix("M_LEQ_")))
    merged["any"] = merged.pop(buckets[-1])
    return {key: merged[key] for key in buckets[:-1] + ["any"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    for key, default in CANDIDATE_VALUES.items():
        flag = "--" + key.lower().replace("_", "-")
        parser.add_argument(flag, type=int, nargs="+", default=default, dest=key)
    parser.add_argument("--top", type=int, default=5, help="candidates listed per M")
    parser.add_argument(
        "--output-json", help="write the best public config per M bucket (aiter format)"
    )
    parser.add_argument(
        "--output-json-all-parameters",
        help="write the best config per M bucket, num_warps and num_stages included",
    )
    parser.add_argument("--output", help="also write the markdown report to this file")
    parser.add_argument(
        "--merge", nargs="+", metavar="JSON", help="only merge these JSON files into --output-json"
    )
    parser.add_argument(
        "--timing",
        choices=("graph", "eager"),
        default="graph",
        help="eager: time plain launches, if CUDA graph capture fails",
    )
    args = parser.parse_args()

    if args.merge:
        merged = json.dumps(merge(args.merge), indent=4) + "\n"
        print(merged)
        if args.output_json:
            with open(args.output_json, "w") as f:
                f.write(merged)
        return

    global TIMING
    TIMING = args.timing
    values = {key: getattr(args, key) for key in CANDIDATE_VALUES}
    reference = aiter_head_config()
    runner = IsolatedRunner()
    rows = []
    for m in args.m_values:
        rows.append(tune_m(runner, m, values, reference))
        # Rewritten after every M so a job stopped by its time limit keeps what it finished.
        report = to_markdown(rows, values, reference, args.top)
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
        for path, public_only in (
            (args.output_json, True),
            (args.output_json_all_parameters, False),
        ):
            if path:
                with open(path, "w") as f:
                    f.write(json.dumps(to_json(rows, reference, public_only), indent=4) + "\n")
        print(f"{time.strftime('%H:%M:%S')} finished M={m}", flush=True)
    runner.close()
    print(report)


if __name__ == "__main__":
    main()
