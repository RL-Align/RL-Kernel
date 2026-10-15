#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Is Miles' routed-MoE backward reproducible? Measured on real layer-0 weights.

    TRITON_F32_DEFAULT=ieee python benchmarks/models/qwen3_next/qwen3_next_miles_moe_backward.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --miles-src <miles checkout> --output <dir>

Miles' own autograd functions (``miles/kernels/moe/fused_experts.py``: forward
through SGLang's Triton kernels, backward through Miles' Triton kernels in
``fused_experts_bwd.py``, which accumulate with ``tl.atomic_add``) are run on
the official layer-0 router and 512 experts, with the routes of an FP32 router
and the same seeded tokens and output gradient every time. The routed-MoE of
RL-Kernel (``_RoutedExperts`` + ``combine_routes``) runs next to it as the
control, on the same inputs and leaves.

For ``--repeats`` identical launches it records, per tensor (input gradient,
both expert-weight gradients, route-weight gradient and the output), how many
repeats equal the first bit for bit, the largest difference and the share of
elements that differ; then whether 8 probe rows keep their input gradient when
computed alone or inside batches of 8-1024 rows at the first, middle and last
position.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from benchmarks.models.qwen3_next import thirdparty_sources  # noqa: E402
from benchmarks.models.qwen3_next.moe_ep_candidates import (  # noqa: E402
    EXPERTS,
    HIDDEN,
    TOP_K,
    expert_weights,
)
from tools.validation.models.qwen3_next.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    provenance,
)

PROBE = 8
BATCHES = (8, 64, 256, 1024)
TOKENS = 1024
TENSORS = ("out", "dx", "dw_gate_up", "dw_down", "d_route_weights")


def routes(x, router):
    """FP32 router, softmax, top-10, renormalised; weights rounded to BF16 as Miles passes them.

    Miles' Qwen3-MoE path (``miles/backends/fsdp_utils/models/qwen3_moe.py``) casts the
    routing weights to the hidden-state dtype before ``fused_experts``; its kernels
    do not compile with FP32 weights. Returns BF16 weights and int32 ids.
    """
    probabilities = torch.softmax(x.float() @ router.float().t(), dim=-1)
    weights, ids = probabilities.topk(TOP_K, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    return weights.contiguous(), ids.int().contiguous()


def miles_run(x, w1, w2, weights, ids, dy):
    from miles.kernels.moe.fused_experts import fused_experts

    x = x.clone().requires_grad_(True)
    w1, w2 = w1.detach().requires_grad_(True), w2.detach().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    out = fused_experts(x, w1, w2, weights, ids)
    out.backward(dy)
    return {
        "out": out.detach(),
        "dx": x.grad,
        "dw_gate_up": w1.grad,
        "dw_down": w2.grad,
        "d_route_weights": weights.grad,
    }


def rl_kernel_run(x, w1, w2, weights, ids, dy):
    from rl_engine.models.qwen3_next.qwen3_next_forward import _RoutedExperts, combine_routes

    x = x.clone().requires_grad_(True)
    w1, w2 = w1.detach().requires_grad_(True), w2.detach().requires_grad_(True)
    weights = weights.float().requires_grad_(
        True
    )  # the same values Miles gets, in RL-Kernel's FP32
    out = combine_routes(_RoutedExperts.apply(x, w1, w2, ids.long()), weights)
    out.backward(dy)
    return {
        "out": out.detach(),
        "dx": x.grad,
        "dw_gate_up": w1.grad,
        "dw_down": w2.grad,
        "d_route_weights": weights.grad.to(torch.bfloat16),
    }


def _compare(first, other):
    """Bitwise equality, largest difference and the share of differing elements."""
    if torch.equal(first, other):
        return True, 0.0, 0.0
    difference = first.float() - other.float()
    return False, float(difference.abs().max()), float((difference != 0).float().mean())


def repeat_report(run, repeats):
    first = run()
    identical = {name: 0 for name in TENSORS}
    worst = {name: (0.0, 0.0) for name in TENSORS}
    for _ in range(repeats - 1):
        again = run()
        for name in TENSORS:
            same, largest, share = _compare(first[name], again[name])
            identical[name] += same
            worst[name] = (max(worst[name][0], largest), max(worst[name][1], share))
        del again
    report = {
        name: {
            "repeats_equal_to_first": identical[name] + 1,
            "repeats": repeats,
            "max_abs_diff_between_repeats": worst[name][0],
            "max_share_of_elements_differing": worst[name][1],
            "dtype": str(first[name].dtype).removeprefix("torch."),
        }
        for name in TENSORS
    }
    return first, report


def batch_grid(run_rows, x_all, dy_all):
    """Input gradient of 8 probe rows alone vs inside batches, first / middle / last."""
    probe, filler = x_all[:PROBE], x_all[TOKENS:]
    dprobe, dfiller = dy_all[:PROBE], dy_all[TOKENS:]
    alone = run_rows(probe, dprobe)["dx"]
    grid = {}
    for size in BATCHES:
        for position in sorted({0, (size - PROBE) // 2, size - PROBE}):
            rows = torch.cat((filler[:position], probe, filler[position : size - PROBE]))
            grads = torch.cat((dfiller[:position], dprobe, dfiller[position : size - PROBE]))
            got = run_rows(rows, grads)["dx"][position : position + PROBE]
            grid[f"{size}@{position}"] = bool(torch.equal(got, alone))
    return grid


def latency_ms(run, repeats=5):
    run()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def measure(args):
    third_party = [thirdparty_sources.load_miles(args.miles_src)]
    thirdparty_sources.install_sglang_aliases()
    from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import moe_hf_shapes

    generator = torch.Generator(device="cuda").manual_seed(4281)
    rows = 2 * TOKENS
    x_all = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    x_all = (x_all * 0.1).contiguous()
    dy_all = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    with ExitStack() as stack:
        weights = CheckpointWeights(args.checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
        router = weights["gate.weight"].cuda()
        w1, w2 = (t.cuda() for t in expert_weights(weights, range(EXPERTS)))
    x, dy = x_all[:TOKENS], dy_all[:TOKENS]
    result = {}
    kept = {}
    for name, runner in (("miles", miles_run), ("rl_kernel", rl_kernel_run)):

        def run_rows(rows_x, rows_dy, runner=runner):
            tw, ids = routes(rows_x, router)
            return runner(rows_x, w1, w2, tw, ids, rows_dy)

        tw, ids = routes(x, router)
        first, report = repeat_report(lambda: runner(x, w1, w2, tw, ids, dy), args.repeats)
        kept[name] = first
        result[name] = {
            "repeat": report,
            "dx_batch_grid": (grid := batch_grid(run_rows, x_all, dy_all)),
            "dx_batch_invariant": all(grid.values()),
            "latency_ms_forward_backward": latency_ms(lambda: runner(x, w1, w2, tw, ids, dy)),
        }
    for name, other in (("miles", "rl_kernel"), ("rl_kernel", "miles")):
        result[name]["max_abs_vs_other"] = {
            tensor: float((kept[name][tensor].float() - kept[other][tensor].float()).abs().max())
            for tensor in ("out", "dx", "d_route_weights")
        }
    report = {
        "kind": "qwen3_next_miles_moe_backward",
        "tokens": TOKENS,
        "repeats": args.repeats,
        "third_party": third_party,
        "environment": provenance()["environment"],
        "rl_kernel_commit": provenance()["rl_kernel_commit"],
        "tracked_tree_dirty": provenance()["tracked_tree_dirty"],
        "candidates": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, row in result.items():
        print(
            name, {k: (v["repeats_equal_to_first"], v["repeats"]) for k, v in row["repeat"].items()}
        )
        print(
            "   dx batch invariant:",
            row["dx_batch_invariant"],
            "latency ms:",
            row["latency_ms_forward_backward"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--miles-src", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
