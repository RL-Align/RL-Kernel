# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""mHC head device time: per-M config table vs aiter's own head config (DSV4 RFC task 8, #508).

- baseline: mhc_head_dsv4 with aiter's own head config;
- ours: AiterMHCHeadOp, i.e. mhc_head_dsv4 with the config the launch-config
  dispatch (rl_engine/backends/rocm/config_utils.py) picks for M;
- comparison: ours / baseline device time, per M; plus whether ours gives the
  same output bits as the baseline at that M.

Forward only: the backward is the same PyTorch recompute for both. Device time
from GPU events around a CUDA graph of calls (CPU launch overhead is larger than
the kernel at small M), median and min-max over several replays; the timing is
shared with benchmarks/tuning/tune_mhc_head_config.py. Shapes follow
DeepSeek-V4-Flash-0731: n=4 streams, C=4096, bf16 residual, FP32 hc_head_fn.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/operators/mhc/benchmark_mhc_head.py --output artifacts/benchmarks/mhc_head.md
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tuning.tune_mhc_head_config import (  # noqa: E402
    EPS,
    ITERS,
    TRIALS,
    C,
    N,
    bits_differ,
    head_inputs,
    time_ms,
)

M_VALUES = [1, 16, 17, 64, 100, 256, 1024, 2048, 3000, 4096, 5000, 8192, 16384, 32768, 65536]


def _cell(result):
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def _row(m, ours_op, mhc_head_dsv4, ours_config):
    residual, fn, scale, base = head_inputs(m)
    with torch.no_grad():

        def baseline_call():
            return mhc_head_dsv4(residual, fn, scale, base, rms_eps=EPS, hc_pre_eps=EPS)

        def ours_call():
            return ours_op(residual, fn, scale, base, rms_eps=EPS, hc_pre_eps=EPS)

        count = bits_differ(ours_call(), baseline_call())
        baseline, ours = time_ms(baseline_call), time_ms(ours_call)
    bits = "same" if count == 0 else f"differs: {count}/{m * C}"
    config = ours_config(m)
    ratio = f"{ours[0] / baseline[0]:.2f}x"
    print(f"M={m}: baseline {_cell(baseline)}, ours {_cell(ours)}, {ratio}, {bits}", flush=True)
    return m, _cell(baseline), _cell(ours), ratio, bits, config


def run(m_values):
    from aiter.ops.triton.fusions import mhc_head_dsv4

    from rl_engine.backends.rocm.config_utils import get_config
    from rl_engine.backends.rocm.mhc.aiter_mhc import (
        FIXED_HEAD_CONFIG,
        HEAD_BIT_PARAMS,
        AiterMHCHeadOp,
    )

    def ours_config(m):
        file_name = f"MHC_HEAD-n={N}-C={C}.json"
        return get_config(
            "mhc_head", file_name, m, FIXED_HEAD_CONFIG, HEAD_BIT_PARAMS, torch.device("cuda")
        )

    ours_op = AiterMHCHeadOp()
    rows = []
    for m in m_values:
        rows.append(_row(m, ours_op, mhc_head_dsv4, ours_config))
        torch.cuda.empty_cache()
    return rows


def to_markdown(rows, device):
    header = (
        f"mHC head forward device time, ms per call, on {device}: n={N}, C={C}, bf16 residual, "
        f"FP32 hc_head_fn. Median over {TRIALS} CUDA graph replays of {ITERS} calls (min-max).\n\n"
        "| M | baseline: aiter's head config | ours: per-M table "
        "| ours / baseline (median) | ours bits vs baseline | ours config |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "\n".join(f"| {m} | {b} | {o} | {r} | {s} | `{c}` |" for m, b, o, r, s, c in rows)
    return header + body + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--output", help="also write the markdown table to this file")
    args = parser.parse_args()

    table = to_markdown(run(args.m_values), torch.cuda.get_device_name(0))
    print(table)
    if args.output:
        with open(args.output, "w") as f:
            f.write(table)


if __name__ == "__main__":
    main()
