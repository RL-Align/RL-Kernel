# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tune aiter gemm_a16w16 per M for lm_head without changing its bits (DSV4 RFC task 9, #510).

Batch invariance does not need one config for every M, only configs that give
the same bits. On gfx942 with Triton 3.7 the parameters of gemm_a16w16 that
change the bits are NUM_KSPLIT, kpack and matrix_instr_nonkdim
(docs/operators/batch-invariant-linear.md). This script keeps them at
FIXED_LINEAR_CONFIG's values (BIT_PARAMS below) and searches the rest, for one
GEMM: the lm_head of DeepSeek-V4-Flash-0731, x [M, 4096] bf16 times
head.weight [129280, 4096] bf16, fp32 output.

Search, for every M bucket of aiter's STANDARD_M_BOUNDS (tuned at M = bound):

1. tile stage: BLOCK_SIZE_M/N/K x num_warps x num_stages, with GROUP_SIZE_M=4,
   waves_per_eu=0, cache_modifier=None. Tiles that cannot fit gfx942's LDS, need
   more than 256 accumulator registers per thread, give a warp less than one
   16x16 MFMA tile, or are far from M are not tried;
2. launch stage: GROUP_SIZE_M x waves_per_eu x cache_modifier for the best
   tile-stage configs;
3. final: the best configs and FIXED_LINEAR_CONFIG are measured again with more
   calls. The fastest whose output bits equal FIXED_LINEAR_CONFIG's wins.

Device time comes from GPU events: the median over trials of the mean per call.
A candidate whose first timed call is more than SKIP_FACTOR times slower than
the best so far is not measured further. Wall-clock time per M is reported only
to plan job lengths. Triton compilation dominates the wall time; --dry-run
prints the number of configs to compile without a GPU.

The sweep can be split over several jobs with --m-values; each job writes the
buckets it tuned (aiter M_LEQ_x JSON) and a markdown report after every M.
--merge joins the job outputs into one table named like aiter's
GEMM-A16W16-N=129280-K=4096.json and needs no torch.

Usage (GPU; aiter with the Triton gemm_a16w16 and AITER_TRITON_ONLY=1):
    python benchmarks/tuning/tune_batch_invariant_linear.py --m-values 1 4 8 \\
        --output tune-part.json --report tune-part.md
    python benchmarks/tuning/tune_batch_invariant_linear.py --dry-run --m-values 1 8192
    python benchmarks/tuning/tune_batch_invariant_linear.py --merge part1.json part2.json \\
        --output GEMM-A16W16-N=129280-K=4096.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time

N, K = 129280, 4096  # DeepSeek-V4-Flash-0731 lm_head: vocab size, hidden size
# aiter/ops/triton/utils/gemm_config_utils.py, aiter 326cffa
STANDARD_M_BOUNDS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
LDS_BYTES = 65536  # gfx942
MFMA_TILE = 16  # matrix_instr_nonkdim

# These decide the bits; every config keeps FIXED_LINEAR_CONFIG's values.
BIT_PARAMS = {"NUM_KSPLIT": 1, "kpack": 1, "matrix_instr_nonkdim": 16}

TILE_VALUES = {
    "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128, 256],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3],
}
TILE_STAGE_LAUNCH = {"GROUP_SIZE_M": 4, "waves_per_eu": 0, "cache_modifier": None}
LAUNCH_VALUES = {
    "GROUP_SIZE_M": [1, 4, 8],
    "waves_per_eu": [0, 1, 2, 4],
    "cache_modifier": [None, ".cg"],
}
LAUNCH_STAGE_TOP = 3  # tile-stage configs whose launch parameters are searched
FINAL_TOP = 5  # configs measured again in the final stage
SKIP_FACTOR = 3.0

# Calls per trial are chosen so a trial lasts about TARGET_MS, within [min, max].
SCREEN = {"warmup": 1, "trials": 3, "target_ms": 5.0, "min_iters": 3, "max_iters": 50}
FINAL = {"warmup": 5, "trials": 7, "target_ms": 50.0, "min_iters": 5, "max_iters": 200}

# Key order of aiter's gemm_a16w16 JSON files.
CONFIG_KEYS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "cache_modifier",
    "NUM_KSPLIT",
    "kpack",
    "SPLITK_BLOCK_SIZE",
]


# ---------------------------------------------------------------------------
# Candidates (pure Python, no torch)
# ---------------------------------------------------------------------------


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _lds_bytes(block_m, block_n, block_k, num_stages):
    """bf16 A and B tiles; aiter's _gemm_lds_bytes without async padding.

    It predicted every OutOfResources of a full candidate sweep on MI325X.
    """
    a, b = block_m * block_k * 2, block_k * block_n * 2
    if num_stages <= 1:
        return max(a, b)
    return (a + b) * (num_stages - 1)


def _ordered(config):
    return {key: config[key] for key in CONFIG_KEYS if key in config}


def tile_candidates(m):
    """Tile-stage configs for M; obviously bad tiles are not tried."""
    m_pow2 = _next_pow2(m)
    # BLOCK_SIZE_M at most the next power of two of M (at least one MFMA tile);
    # large M gets no tiny tiles, which waste bandwidth on a compute-bound GEMM.
    block_m_max = min(256, max(MFMA_TILE, m_pow2))
    block_m_min = min(64, max(MFMA_TILE, m_pow2 // 8))
    block_n_min = 64 if m >= 512 else 32
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(*TILE_VALUES.values()):
        if not block_m_min <= bm <= block_m_max or bn < block_n_min:
            continue
        if _lds_bytes(bm, bn, bk, stages) > LDS_BYTES:
            continue
        if bm * bn > 256 * 64 * warps:  # > 256 fp32 accumulators per thread
            continue
        if bm * bn < MFMA_TILE * MFMA_TILE * warps:  # a warp gets < one MFMA tile
            continue
        config = dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=warps,
            num_stages=stages,
            **TILE_STAGE_LAUNCH,
            **BIT_PARAMS,
        )
        configs.append(_ordered(config))
    return configs


def launch_candidates(tile_config, m):
    """Launch-stage configs: tile_config with other launch parameters."""
    # With one row of tiles, GROUP_SIZE_M does not change the launch order.
    one_row = math.ceil(m / tile_config["BLOCK_SIZE_M"]) == 1
    groups = [1] if one_row else LAUNCH_VALUES["GROUP_SIZE_M"]
    configs = []
    for group, waves, cache in itertools.product(
        groups, LAUNCH_VALUES["waves_per_eu"], LAUNCH_VALUES["cache_modifier"]
    ):
        config = dict(tile_config, GROUP_SIZE_M=group, waves_per_eu=waves, cache_modifier=cache)
        configs.append(_ordered(config))
    return configs


def _key(config):
    return tuple(config.get(key) for key in CONFIG_KEYS if key != "SPLITK_BLOCK_SIZE")


def _label(config):
    short = {
        "BLOCK_SIZE_M": "BM",
        "BLOCK_SIZE_N": "BN",
        "BLOCK_SIZE_K": "BK",
        "GROUP_SIZE_M": "G",
        "num_warps": "warps",
        "num_stages": "stages",
        "waves_per_eu": "wpe",
        "cache_modifier": "cache",
    }
    return " ".join(f"{name}={config[key]}" for key, name in short.items())


def dry_run(m_values):
    """Number of configs per M and compilations for one job over m_values."""
    tile_keys = set()
    print("| M | tile-stage configs | launch-stage configs (at most) |")
    print("|---|---|---|")
    launch_total = 0
    for m in m_values:
        tiles = tile_candidates(m)
        # Triton compiles a config once per process and per value of the kernel's
        # EVEN_MN (M and N multiples of the tile; N=129280 is a multiple of every
        # BLOCK_SIZE_N here), so tile-stage configs are shared between the M of one job.
        tile_keys.update((_key(c), m % c["BLOCK_SIZE_M"] == 0) for c in tiles)
        # Upper bound: the best tiles are known only on the GPU.
        launch = LAUNCH_STAGE_TOP * math.prod(len(v) for v in LAUNCH_VALUES.values())
        launch_total += launch
        print(f"| {m} | {len(tiles)} | {launch} |")
    print(
        f"\nCompilations in one job (upper bound): {len(tile_keys)} tile-stage "
        f"+ {launch_total} launch-stage = {len(tile_keys) + launch_total}"
    )


# ---------------------------------------------------------------------------
# Merge (pure Python, no torch)
# ---------------------------------------------------------------------------


def merge(paths, output):
    table = {}
    for path in paths:
        with open(path) as f:
            part = json.load(f)
        for key, config in part.items():
            if not key.startswith("M_LEQ_"):
                raise ValueError(f"{path}: unexpected key {key}")
            for name, value in BIT_PARAMS.items():
                if config.get(name) != value:
                    raise ValueError(f"{path} {key}: {name}={config.get(name)}, need {value}")
            if key in table and table[key] != config:
                raise ValueError(f"{key} is in more than one file with different configs")
            table[key] = config
    bounds = sorted(int(key.removeprefix("M_LEQ_")) for key in table)
    missing = [b for b in STANDARD_M_BOUNDS if b not in bounds]
    merged = {f"M_LEQ_{b}": table[f"M_LEQ_{b}"] for b in bounds}
    # M above the largest bucket (> 8192 is not tuned) uses the largest bucket's config.
    merged["any"] = dict(merged[f"M_LEQ_{bounds[-1]}"])
    with open(output, "w") as f:
        json.dump(merged, f, indent=4)
        f.write("\n")
    print(f"wrote {output}: buckets {bounds}")
    if missing:
        print(f"not tuned (M there uses the next larger bucket): {missing}")


# ---------------------------------------------------------------------------
# Tuning (GPU)
# ---------------------------------------------------------------------------


class _Tuner:
    def __init__(self):
        import torch
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

        from rl_engine.backends.rocm.gemm.aiter_gemm import FIXED_LINEAR_CONFIG

        self.torch = torch
        self.gemm = gemm_a16w16
        self.fixed = _ordered(FIXED_LINEAR_CONFIG)
        for name, value in BIT_PARAMS.items():
            assert self.fixed[name] == value, f"FIXED_LINEAR_CONFIG {name} is not {value}"
        gen = torch.Generator(device="cuda").manual_seed(0)
        self.weight = (torch.randn(N, K, generator=gen, device="cuda") / K**0.5).to(torch.bfloat16)
        self.x_all = torch.randn(max(STANDARD_M_BOUNDS), K, generator=gen, device="cuda").to(
            torch.bfloat16
        )

    def call(self, config, x, y):
        full = dict(config, SPLITK_BLOCK_SIZE=K)  # NUM_KSPLIT=1: one split, all of K
        return lambda: self.gemm(
            x, self.weight, None, dtype=self.torch.float32, y=y, config=full, backend="triton"
        )

    def time_ms(self, fn, iters):
        torch = self.torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def measure(self, fn, protocol, best_ms=None):
        """('ok', median ms, samples), ('slow', one-call ms) or ('cannot launch', error)."""
        try:
            fn()  # compiles on first use
            self.torch.cuda.synchronize()
        except Exception as exc:  # LDS or registers exceeded
            return "cannot launch", type(exc).__name__, None
        one = self.time_ms(fn, 1)
        if best_ms is not None and one > SKIP_FACTOR * best_ms:
            return "slow", one, None
        iters = protocol["target_ms"] / one
        iters = int(min(protocol["max_iters"], max(protocol["min_iters"], iters)))
        for _ in range(protocol["warmup"]):
            fn()
        samples = [self.time_ms(fn, iters) for _ in range(protocol["trials"])]
        return "ok", statistics.median(samples), samples

    def screen(self, configs, fn_of, results):
        """Screen configs; adds (status, value) per config key to results."""
        best = min((r[1] for r in results.values() if r[0] == "ok"), default=None)
        for config in configs:
            if _key(config) in results:
                continue
            status, value, _ = self.measure(fn_of(config), SCREEN, best)
            results[_key(config)] = (status, value, config)
            if status == "ok" and (best is None or value < best):
                best = value

    def tune(self, m):
        torch = self.torch
        x = self.x_all[:m]
        y = torch.empty(m, N, dtype=torch.float32, device="cuda")
        fn_of = lambda config: self.call(config, x, y)  # noqa: E731
        wall_start = time.perf_counter()

        tiles = tile_candidates(m)
        results = {}
        self.screen(tiles, fn_of, results)
        tile_ok = sorted((r for r in results.values() if r[0] == "ok"), key=lambda r: r[1])
        counts = {
            "tile-stage configs": len(tiles),
            "cannot launch": sum(r[0] == "cannot launch" for r in results.values()),
            "skipped as slow": sum(r[0] == "slow" for r in results.values()),
        }

        launch = []
        for _, _, config in tile_ok[:LAUNCH_STAGE_TOP]:
            launch += launch_candidates(config, m)
        before = len(results)
        self.screen(launch, fn_of, results)
        counts["launch-stage configs"] = len(results) - before

        ranked = sorted((r for r in results.values() if r[0] == "ok"), key=lambda r: r[1])
        finalists = [r[2] for r in ranked[:FINAL_TOP]]
        if _key(self.fixed) not in {_key(c) for c in finalists}:
            finalists.append(self.fixed)
        final = []
        for config in finalists:
            status, median, samples = self.measure(fn_of(config), FINAL)
            if status == "ok":
                final.append((median, min(samples), max(samples), config))
        final.sort(key=lambda r: r[0])

        # The winner must give FIXED_LINEAR_CONFIG's bits.
        y_fixed = torch.empty_like(y)
        self.call(self.fixed, x, y_fixed)()
        winner, differing = None, []
        for median, low, high, config in final:
            self.call(config, x, y)()
            torch.cuda.synchronize()
            count = int((y.view(torch.int32) != y_fixed.view(torch.int32)).sum())
            if count == 0:
                winner = (median, low, high, config)
                break
            differing.append((config, count))
        fixed_final = next((r for r in final if _key(r[3]) == _key(self.fixed)), None)
        if fixed_final is None:
            raise RuntimeError(f"FIXED_LINEAR_CONFIG cannot launch at M={m}")
        y = y_fixed = None  # free before the next M
        torch.cuda.empty_cache()
        return {
            "m": m,
            "winner": winner,
            "fixed": fixed_final,
            "final": final,
            "differing": differing,
            "counts": counts,
            "wall_s": time.perf_counter() - wall_start,
        }


def _cell(median, low, high):
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def report(results, header):
    lines = [header, ""]
    lines += [
        "| M bucket (tuned at M) | ours: tuned config | ours, ms "
        "| baseline: FIXED_LINEAR_CONFIG, ms | ours / baseline (median) "
        "| wall time of this M, s (planning only) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        fixed = _cell(*r["fixed"][:3])
        if r["winner"] is None:
            lines.append(
                f"| M_LEQ_{r['m']} | none: no finalist gave FIXED_LINEAR_CONFIG's bits | n/a "
                f"| {fixed} | n/a | {r['wall_s']:.0f} |"
            )
            continue
        median, low, high, config = r["winner"]
        lines.append(
            f"| M_LEQ_{r['m']} | {_label(config)} | {_cell(median, low, high)} | {fixed} "
            f"| {median / r['fixed'][0]:.2f}x | {r['wall_s']:.0f} |"
        )
    for r in results:
        lines += ["", f"### M = {r['m']}", ""]
        lines.append(", ".join(f"{k}: {v}" for k, v in r["counts"].items()))
        lines += ["", "| final config | ms, median (min-max) |", "|---|---|"]
        for median, low, high, config in r["final"]:
            tag = " (FIXED_LINEAR_CONFIG)" if config == r["fixed"][3] else ""
            lines.append(f"| {_label(config)}{tag} | {_cell(median, low, high)} |")
        for config, count in r["differing"]:
            lines.append(f"\nBits differ from FIXED_LINEAR_CONFIG: {_label(config)}: {count}")
    return "\n".join(lines) + "\n"


def tune(m_values, output, report_path, budget_min):
    import socket

    import torch
    import triton

    tuner = _Tuner()
    header = "\n".join(
        [
            f"# gemm_a16w16 per-M tuning, lm_head N={N} K={K}, bf16 in, fp32 out",
            "",
            f"- Node `{socket.gethostname()}`, {torch.cuda.get_device_name()}, torch "
            f"{torch.__version__}, triton {triton.__version__}",
            f"- Parameters that change the bits, kept fixed: {BIT_PARAMS}",
            f"- Device time per call from GPU events: median over {FINAL['trials']} trials "
            "(min-max); final-stage protocol for every number below.",
        ]
    )
    start = time.perf_counter()
    results, table = [], {}
    for m in m_values:
        elapsed_min = (time.perf_counter() - start) / 60
        if elapsed_min > budget_min:
            print(f"time budget of {budget_min} min used up; not tuned: M >= {m}", flush=True)
            break
        r = tuner.tune(m)
        results.append(r)
        if r["winner"] is not None:
            table[f"M_LEQ_{m}"] = _ordered(dict(r["winner"][3], SPLITK_BLOCK_SIZE=K))
        # Written after every M, so a job that times out keeps the finished buckets.
        with open(output, "w") as f:
            json.dump(table, f, indent=4)
            f.write("\n")
        text = report(results, header)
        if report_path:
            with open(report_path, "w") as f:
                f.write(text)
        print(f"M={m} done in {r['wall_s']:.0f} s", flush=True)
    print(report(results, header))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m-values", type=int, nargs="+", default=list(STANDARD_M_BOUNDS))
    parser.add_argument("--output", help="JSON with the tuned M_LEQ_x buckets")
    parser.add_argument("--report", help="also write the markdown report to this file")
    parser.add_argument(
        "--time-budget-min",
        type=float,
        default=30.0,
        help="do not start another M after this many minutes",
    )
    parser.add_argument("--dry-run", action="store_true", help="print config counts, no GPU")
    parser.add_argument("--merge", nargs="+", help="join tuned JSON files into --output")
    args = parser.parse_args()

    for m in args.m_values:
        if m not in STANDARD_M_BOUNDS:
            parser.error(f"M={m} is not a bound of {STANDARD_M_BOUNDS}")
    if args.merge:
        if not args.output:
            parser.error("--merge needs --output")
        merge(args.merge, args.output)
    elif args.dry_run:
        dry_run(args.m_values)
    else:
        if not args.output:
            parser.error("--output is required")
        tune(args.m_values, args.output, args.report, args.time_budget_min)


if __name__ == "__main__":
    main()
