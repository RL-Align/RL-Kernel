# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark RL-Kernel's batch-invariant Triton mHC pre, post and head against aiter's on ROCm.

- baseline: aiter's ``mhc_pre_dsv4``, ``mhc_post_dsv4`` and ``mhc_head_dsv4``
  (aiter.ops.triton.fusions) with their default configs, written inline below;
- ours: ``TritonMHCPreOp``, ``TritonMHCPostOp`` and ``TritonMHCHeadOp``
  (rl_engine/backends/rocm/mhc/triton_mhc.py), the ops the registry hands out.

Two parts, forward only, DeepSeek-V4-Flash shapes (n = 4, C = 4096, bf16
residual, fp32 weights, 20 Sinkhorn rounds):

1. device time per M (benchmarks/common/device_time.py), ours / baseline;
2. error of baseline and ours against the PyTorch reference
   (rl_engine/ops/mhc/training.py, backend "rocm") at M = 1024.

Batch invariance, launch-config invariance and dynamic shape are checked by
tests/backends/rocm/.

Usage (ROCm GPU; aiter with the DeepSeek-V4 mHC ops for the baseline):
    python benchmarks/operators/mhc/benchmark_dsv4_mhc.py --output mhc.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.device_time import (  # noqa: E402
    environment,
    launch_error,
    measure,
    ratio_cell,
    time_cell,
)
from rl_engine.contracts.operators.mhc import (  # noqa: E402
    HC_POST_MULT_VALUE,
    HC_PRE_EPS,
    HC_SINKHORN_EPS,
    RMS_EPS,
    SINKHORN_REPEAT,
)

DEV = "cuda"
N, C = 4, 4096
K = N * C
PRE_ROWS = 2 * N + N * N
PERF_M = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
ERROR_M = 1024
OPS = ("pre", "post", "head")
OUTPUT_NAMES = {"pre": ["post_mix", "comb_mix", "layer_input"], "post": ["out"], "head": ["out"]}


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def pre_weights(seed=0):
    gen = _gen(seed)
    fn = torch.randn(PRE_ROWS, K, generator=gen, device=DEV) / K**0.5
    scale = 0.5 + torch.rand(3, generator=gen, device=DEV)
    base = 0.1 * torch.randn(PRE_ROWS, generator=gen, device=DEV)
    return fn, scale, base


def head_weights(seed=0):
    gen = _gen(seed + 1)
    fn = torch.randn(N, K, generator=gen, device=DEV) / K**0.5
    scale = 0.5 + torch.rand(1, generator=gen, device=DEV)
    base = 0.1 * torch.randn(N, generator=gen, device=DEV)
    return fn, scale, base


def residual(m, seed=0):
    return torch.randn(m, N, C, generator=_gen(seed), device=DEV).to(torch.bfloat16)


def post_inputs(m, seed=0):
    """y [M, C], residual [M, n, C], post_mix [M, n], comb_mix [M, n, n]."""
    gen = _gen(seed)
    y = torch.randn(m, C, generator=gen, device=DEV).to(torch.bfloat16)
    res = torch.randn(m, N, C, generator=gen, device=DEV).to(torch.bfloat16)
    post_mix = 2.0 * torch.sigmoid(torch.randn(m, N, generator=gen, device=DEV))
    comb_mix = torch.softmax(torch.randn(m, N, N, generator=gen, device=DEV), dim=-1)
    return y, res, post_mix, comb_mix


def make_inputs(op, m, seed=0):
    return post_inputs(m, seed) if op == "post" else residual(m, seed)


def ours(pre_w, head_w):
    """{op: callable(inputs) -> list of outputs}, inputs as made by make_inputs."""
    from rl_engine.backends.rocm.mhc.triton_mhc import (
        TritonMHCHeadOp,
        TritonMHCPostOp,
        TritonMHCPreOp,
    )

    pre, post, head = TritonMHCPreOp(), TritonMHCPostOp(), TritonMHCHeadOp()
    return {
        "pre": lambda r: list(pre(r, *pre_w)),
        "post": lambda inputs: [post(*inputs)],
        "head": lambda r: [head(r, *head_w)],
    }


def baseline(pre_w, head_w):
    """Stock aiter with its default configs: no ``config`` argument."""
    from aiter.ops.triton.fusions import mhc_head_dsv4, mhc_post_dsv4, mhc_pre_dsv4

    def pre(r):
        post_mix, comb_mix, layer_input = mhc_pre_dsv4(
            r.contiguous(),
            *pre_w,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
        )
        return [post_mix.reshape(-1, N), comb_mix, layer_input]

    def post(inputs):
        y, res, post_mix, comb_mix = inputs
        return [mhc_post_dsv4(y, res, post_mix.unsqueeze(-1), comb_mix)]

    def head(r):
        return [mhc_head_dsv4(r.contiguous(), *head_w, rms_eps=RMS_EPS, hc_pre_eps=HC_PRE_EPS)]

    return {"pre": pre, "post": post, "head": head}


def _baseline_or_reason(pre_w, head_w):
    """The baseline ops, or "cannot launch: <reason>" if aiter does not have them."""
    try:
        return baseline(pre_w, head_w)
    except Exception as exc:
        return launch_error(exc)


def section_time(m_values, pre_w, head_w):
    run = ours(pre_w, head_w)
    base_run = _baseline_or_reason(pre_w, head_w)
    lines = []
    for op in OPS:
        lines += [
            f"## Device time, mHC {op}",
            "",
            f"Forward of mHC {op}, n={N}, C={C}, bf16 residual, fp32 weights, "
            f"{SINKHORN_REPEAT} Sinkhorn rounds. ms per call from GPU events around a CUDA "
            "graph, median (min-max) over replays. Baseline: aiter's op with its default "
            "config. Ours: RL-Kernel's batch-invariant Triton op. Ratio: ours / baseline, "
            "below 1 means ours is faster.",
            "",
            "| M | baseline (ms) | ours (ms) | ours / baseline (ratio) |",
            "|---|---|---|---|",
        ]
        for m in m_values:
            inputs = make_inputs(op, m, seed=3)
            if isinstance(base_run, str):
                base = base_run
            else:
                base = measure(lambda inputs=inputs: base_run[op](inputs))
            result = measure(lambda inputs=inputs: run[op](inputs))
            lines.append(
                f"| {m} | {time_cell(base)} | {time_cell(result)} | {ratio_cell(result, base)} |"
            )
            print(lines[-1], flush=True)
            del inputs
            torch.cuda.empty_cache()
        lines.append("")
    return lines


def section_error(pre_w, head_w):
    from rl_engine.ops.mhc import training

    m = ERROR_M
    r = residual(m, seed=7)
    p_in = post_inputs(m, seed=8)
    inputs = {"pre": r, "post": p_in, "head": r}
    refs = {
        "pre": list(training.mhc_pre(r, *pre_w, backend="rocm")),
        "post": [training.mhc_post(*p_in, backend="rocm")],
        "head": [training.mhc_head(r, *head_w, backend="rocm")],
    }
    columns = [f"{op} {name}" for op in OPS for name in OUTPUT_NAMES[op]]
    lines = [
        "## Error against the PyTorch reference",
        "",
        f"Forward at M = {m}, random inputs. Reference: rl_engine/ops/mhc/training.py, backend "
        '"rocm" (fp32 projection; bf16 layer_input and outputs, as DeepSeek-V4). Cells: '
        "largest / mean absolute error.",
        "",
        "| variant | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    variants = {
        "baseline: aiter, default config": _baseline_or_reason(pre_w, head_w),
        "ours": ours(pre_w, head_w),
    }
    for name, run in variants.items():
        cells = []
        for op in OPS:
            if isinstance(run, str):
                cells += [run] * len(refs[op])
                continue
            try:
                for out, ref in zip(run[op](inputs[op]), refs[op]):
                    err = (out.float() - ref.float()).abs()
                    cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
            except Exception as exc:
                cells += [launch_error(exc)] * len(refs[op])
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="markdown report, rewritten after every part")
    parser.add_argument(
        "--m",
        type=lambda s: [int(v) for v in s.split(",")],
        default=PERF_M,
        help=f"comma-separated M values to time (default {','.join(map(str, PERF_M))})",
    )
    args = parser.parse_args()

    lines = ["# Batch-invariant Triton mHC on ROCm: device time and error", "", environment(), ""]

    def flush():
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n")

    with torch.no_grad():
        pre_w, head_w = pre_weights(), head_weights()
        lines += section_time(args.m, pre_w, head_w)
        flush()
        lines += section_error(pre_w, head_w) + [""]
    print("\n".join(lines))
    flush()


if __name__ == "__main__":
    main()
