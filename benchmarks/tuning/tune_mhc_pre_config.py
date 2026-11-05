# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tune per-M launch configs for aiter's mHC pre that keep the output bits (DSV4 RFC task 1, #508).

The mhc_pre op passes its launch config through
rl_engine/backends/rocm/config_utils.py. Batch invariance does not need one
config for all M, only that every M uses a config whose outputs have the same
bits. docs/operators/mhc.md lists the parameters that change the bits; the
allowed values given here must keep them fixed.

``--num-ksplit`` picks the reference config: FIXED_PRE_CONFIG with that
NUM_KSPLIT (aiter's split size is cdiv(K, NUM_KSPLIT), so it is the same for
every M). Several values can be given; each is tuned on its own and gets its
own output files, named with a ``-ksplit<NUM_KSPLIT>`` suffix. Candidates are
all combinations of the allowed values, plus the reference itself, minus the
combinations measured to change the bits (``_keeps_measured_bits`` below). At
each M, every candidate is first run once and its three outputs (post_mix,
comb_mix, layer_input) are compared with the reference's bit for bit; a
candidate whose bits differ is excluded, so a too wide allowed set cannot break
batch invariance. The others are timed, the fastest few are timed again with
more trials, and the fastest of those is kept.

- baseline: FIXED_PRE_CONFIG, one config for every M;
- ours: the fastest candidate with the reference's bits at this M;
- comparison: ours / baseline device time, per M.

Forward only, through the public ``mhc_pre_dsv4(..., config=...)``, so the
time includes the Sinkhorn kernel as in benchmarks/operators/mhc/benchmark_mhc_pre.py.
Device time from GPU events. Shapes follow DeepSeek-V4-Flash-0731: n=4 streams,
C=4096, bf16 residual, fp32 fn, 20 Sinkhorn rounds.

Output: aiter-format JSON with ``M_LEQ_x`` keys (the dispatch uses the smallest
x >= M, so the config tuned at M = x serves the M values above the previous x
up to x) and a markdown table; both are rewritten after each M, so a job that
runs out of time keeps the M values it finished. The sweep can be split into
short jobs with ``--m-values``; ``--merge`` joins their JSON parts and needs no
GPU.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/tuning/tune_mhc_pre_config.py --num-ksplit 4 --m-values 1 16 64 256 \\
        --output-json part.json --output part.md   # writes part-ksplit4.json, part-ksplit4.md
    python benchmarks/tuning/tune_mhc_pre_config.py --merge part1.json part2.json \\
        --output-json MHC_PRE-n=4-C=4096.json
"""

import argparse
import itertools
import json
import socket
import statistics
import time

N, C = 4, 4096
K = N * C
MIX = 2 * N + N * N
SINKHORN_ITERS = 20
M_VALUES = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
WARMUP, ITERS = 10, 50
SCREEN_TRIALS, FINAL_TRIALS, FINALISTS = 3, 5, 5

# Allowed values per parameter, from the measurement in docs/operators/mhc.md:
# BLOCK_K and kpack change the bits, BLOCK_C=256 changes layer_input, and
# BLOCK_M=16 with num_warps=8 changes all three outputs although neither does
# alone. Candidates whose bits differ from the reference are excluded at run
# time anyway.
ALLOWED_VALUES = {
    "BLOCK_M": [16, 32, 64, 128],
    "BLOCK_K": [128],
    "BLOCK_C": [64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2],
    "waves_per_eu": [0, 1, 2],
}
# Key order of the configs written, as in FIXED_PRE_CONFIG.
CONFIG_KEYS = (
    "NUM_KSPLIT",
    "BLOCK_M",
    "BLOCK_K",
    "BLOCK_C",
    "num_warps",
    "num_stages",
    "waves_per_eu",
)


def _time_ms(fn):
    """Mean device time per call over ITERS calls, after WARMUP calls."""
    import torch

    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def _measure(fn, trials):
    """(median, min, max) over trials."""
    samples = [_time_ms(fn) for _ in range(trials)]
    return statistics.median(samples), min(samples), max(samples)


def _cell(result):
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def _label(config):
    return " ".join(f"{key}={config[key]}" for key in CONFIG_KEYS)


def _keeps_measured_bits(config):
    """Whether the config avoids the combinations measured to change the bits
    (docs/operators/mhc.md) that the per-M bit comparison below can miss: at
    M=1 it sees only one row.
    """
    return config["BLOCK_C"] <= 128 and not (config["BLOCK_M"] == 16 and config["num_warps"] == 8)


def _candidates(reference, allowed, m):
    """All combinations of the allowed values that make sense at M, plus the reference."""
    num_ksplit = reference["NUM_KSPLIT"]
    # aiter raises BLOCK_M to 16; a larger BLOCK_M than M rounded up only adds masked rows.
    max_block_m = max(16, 1 << (m - 1).bit_length())
    split_size = -(-K // num_ksplit)
    keys = list(allowed)
    configs = [dict(reference)]
    for values in itertools.product(*(allowed[key] for key in keys)):
        config = dict(reference, **dict(zip(keys, values)))
        if config["BLOCK_M"] > max_block_m or config["BLOCK_K"] > split_size:
            continue
        if not _keeps_measured_bits(config):
            continue
        if config not in configs:
            configs.append(config)
    return [{key: config[key] for key in CONFIG_KEYS} for config in configs]


def _same_bits(outputs, references):
    import torch

    bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
    return all(
        torch.equal(out.view(bits[out.dtype]), ref.view(bits[ref.dtype]))
        for out, ref in zip(outputs, references)
    )


def tune_m(m, reference, allowed, weights, fixed_config):
    """Tune one M. Returns a result dict for the report and the JSON."""
    import torch
    from aiter.ops.triton.fusions import mhc_pre_dsv4

    fn, scale, base = weights
    gen = torch.Generator(device="cuda").manual_seed(m)
    residual = torch.randn(m, N, C, generator=gen, device="cuda").to(torch.bfloat16)

    def call(config):
        return lambda: mhc_pre_dsv4(
            residual, fn, scale, base, sinkhorn_repeat=SINKHORN_ITERS, config=dict(config)
        )

    with torch.no_grad():
        references = call(reference)()
        baseline = _measure(call(fixed_config), FINAL_TRIALS)
        screened, differs, cannot_launch = [], 0, 0
        candidates = _candidates(reference, allowed, m)
        for index, config in enumerate(candidates):
            try:
                outputs = call(config)()
                torch.cuda.synchronize()
            except Exception as exc:  # a config can exceed LDS or registers
                cannot_launch += 1
                print(
                    f"M={m} [{index + 1}/{len(candidates)}] {_label(config)}: cannot launch: "
                    f"{type(exc).__name__}",
                    flush=True,
                )
                continue
            if not _same_bits(outputs, references):
                differs += 1
                print(
                    f"M={m} [{index + 1}/{len(candidates)}] {_label(config)}: bits differ, "
                    "excluded",
                    flush=True,
                )
                continue
            screen = _measure(call(config), SCREEN_TRIALS)
            screened.append((screen[0], config))
            print(
                f"M={m} [{index + 1}/{len(candidates)}] {_label(config)}: {_cell(screen)}",
                flush=True,
            )
        screened.sort(key=lambda item: item[0])
        finalists = [
            (_measure(call(config), FINAL_TRIALS), config) for _, config in screened[:FINALISTS]
        ]
        finalists.sort(key=lambda item: item[0][0])
    residual = None  # free before the next M
    torch.cuda.empty_cache()
    return {
        "m": m,
        "baseline": baseline,
        "finalists": finalists,
        "timed": len(screened),
        "differs": differs,
        "cannot_launch": cannot_launch,
    }


def to_json(results):
    return {
        f"M_LEQ_{r['m']}": r["finalists"][0][1]
        for r in sorted(results, key=lambda r: r["m"])
        if r["finalists"]
    }


def to_markdown(results, reference, allowed, header_lines):
    lines = list(header_lines) + [
        "",
        f"mHC pre forward device time, ms per call: n={N}, C={C}, bf16 residual, fp32 fn, "
        f"{SINKHORN_ITERS} Sinkhorn rounds. Median over {FINAL_TRIALS} trials of {ITERS} calls "
        "(min-max). Baseline: `FIXED_PRE_CONFIG`. Ours: fastest config whose three outputs "
        f"have the same bits as the reference `{_label(reference)}` at that M.",
        "",
        "| M | baseline: FIXED_PRE_CONFIG | ours: fastest with the reference's bits "
        "| ours / baseline (median) "
        "| ours: config | candidates timed / bits differ / cannot launch |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["m"]):
        counts = f"{r['timed']} / {r['differs']} / {r['cannot_launch']}"
        if not r["finalists"]:
            lines.append(
                f"| {r['m']} | {_cell(r['baseline'])} | no candidate kept "
                f"| n/a: no candidate kept | n/a | {counts} |"
            )
            continue
        best, config = r["finalists"][0]
        ratio = f"{best[0] / r['baseline'][0]:.2f}x"
        lines.append(
            f"| {r['m']} | {_cell(r['baseline'])} | {_cell(best)} | {ratio} "
            f"| `{_label(config)}` | {counts} |"
        )
    lines += [
        "",
        f"Fastest {FINALISTS} per M ({FINAL_TRIALS} trials):",
        "",
        "| M | config | ms (median, min-max) |",
        "|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["m"]):
        for result, config in r["finalists"]:
            lines.append(f"| {r['m']} | `{_label(config)}` | {_cell(result)} |")
    lines += ["", f"Allowed values: `{allowed}`"]
    return "\n".join(lines) + "\n"


def merge(parts):
    """Join JSON parts of one NUM_KSPLIT into one aiter-format config table."""
    table = {}
    for path in parts:
        with open(path) as f:
            for key, config in json.load(f).items():
                if key in table and table[key] != config:
                    print(f"{key}: {path} replaces the config of an earlier part")
                table[key] = config
    if not table:
        raise SystemExit("the parts contain no config")
    num_ksplit = {config["NUM_KSPLIT"] for config in table.values()}
    if len(num_ksplit) != 1:
        raise SystemExit(
            f"parts mix NUM_KSPLIT {sorted(num_ksplit)}; batch invariance needs one value"
        )
    entries = sorted(table.items(), key=lambda item: int(item[0].removeprefix("M_LEQ_")))
    merged = dict(entries)
    # M above the largest bucket uses "any": the largest bucket's config.
    merged["any"] = entries[-1][1]
    return merged


def _with_suffix(path, num_ksplit):
    if not path:
        return None
    stem, dot, ext = path.rpartition(".")
    return f"{stem}-ksplit{num_ksplit}.{ext}" if dot else f"{path}-ksplit{num_ksplit}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--num-ksplit",
        type=int,
        nargs="+",
        default=[1],
        help="reference NUM_KSPLIT values, each used for every M",
    )
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    for key, values in ALLOWED_VALUES.items():
        parser.add_argument(
            f"--{key.lower().replace('_', '-')}",
            dest=key,
            type=int,
            nargs="+",
            default=values,
            help=f"allowed {key} values",
        )
    parser.add_argument(
        "--output-json",
        help="write the aiter-format config table here "
        "(with a -ksplit<NUM_KSPLIT> suffix when tuning)",
    )
    parser.add_argument(
        "--output",
        help="also write the markdown table to this file " "(with a -ksplit<NUM_KSPLIT> suffix)",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="PART",
        help="join these JSON parts into --output-json; no GPU needed",
    )
    args = parser.parse_args()

    if args.merge:
        merged = merge(args.merge)
        text = json.dumps(merged, indent=4) + "\n"
        print(text)
        if args.output_json:
            with open(args.output_json, "w") as f:
                f.write(text)
        return

    import torch
    import triton

    from rl_engine.backends.rocm.mhc.aiter_mhc import FIXED_PRE_CONFIG

    allowed = {key: getattr(args, key) for key in ALLOWED_VALUES}
    gen = torch.Generator(device="cuda").manual_seed(0)
    fn = torch.randn(MIX, K, generator=gen, device="cuda") / K**0.5
    scale = 0.5 + torch.rand(3, generator=gen, device="cuda")
    base = 0.1 * torch.randn(MIX, generator=gen, device="cuda")
    for num_ksplit in args.num_ksplit:
        reference = dict(FIXED_PRE_CONFIG, NUM_KSPLIT=num_ksplit)
        header = [
            f"# mHC pre config tuning, NUM_KSPLIT={num_ksplit}",
            "",
            f"- Node: `{socket.gethostname()}`, GPU: {torch.cuda.get_device_name()}",
            f"- torch {torch.__version__}, triton {triton.__version__}",
            f"- M values: {args.m_values}",
        ]
        if num_ksplit != 1:
            header.append(
                "- This NUM_KSPLIT splits K, so its bits differ from FIXED_PRE_CONFIG's; "
                "a fixed NUM_KSPLIT for every M is batch invariant (see "
                "docs/operators/mhc.md)."
            )
        output = _with_suffix(args.output, num_ksplit)
        output_json = _with_suffix(args.output_json, num_ksplit)

        results = []
        for m in args.m_values:
            print(f"{time.strftime('%H:%M:%S')} NUM_KSPLIT={num_ksplit} M={m}: start", flush=True)
            results.append(tune_m(m, reference, allowed, (fn, scale, base), FIXED_PRE_CONFIG))
            table = to_markdown(results, reference, allowed, header)
            print(table, flush=True)
            if output:
                with open(output, "w") as f:
                    f.write(table)
            if output_json:
                with open(output_json, "w") as f:
                    f.write(json.dumps(to_json(results), indent=4) + "\n")


if __name__ == "__main__":
    main()
