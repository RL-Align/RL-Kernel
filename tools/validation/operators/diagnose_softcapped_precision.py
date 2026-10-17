# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare shifted and legacy ROW arithmetic with Native FP32 and independent FP64."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    _FORWARD_LAUNCHERS,
    SoftcappedLogprobStrategy,
    TritonSoftcappedSelectedLogprobOp,
    _ordered_tile_sum,
)
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)


@triton.jit
def _trace_row(
    x_ptr,
    ids_ptr,
    quotient_ptr,
    tanh_ptr,
    score_ptr,
    exp_ptr,
    tiles_ptr,
    stats_ptr,
    V: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFTED: tl.constexpr,
):
    """Instrument the existing ROW arithmetic; compare its output with production."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    row_max = tl.full((), 0.0, tl.float32)
    if SHIFTED:
        row_max = tl.full((), float("-inf"), tl.float32)
        for start in range(0, V, BLOCK):
            offsets = start + cols
            mask = offsets < V
            x = tl.load(x_ptr + row * V + offsets, mask=mask, other=0.0).to(tl.float32)
            s = CAP * libdevice.tanh(tl.div_rn(x, CAP))
            row_max = tl.maximum(row_max, tl.max(tl.where(mask, s, float("-inf")), axis=0))
        row_max = tl.where(row_max == 0.0, 0.0, row_max)
    total = tl.full((), 0.0, tl.float32)
    for start in range(0, V, BLOCK):
        offsets = start + cols
        mask = offsets < V
        x = tl.load(x_ptr + row * V + offsets, mask=mask, other=0.0).to(tl.float32)
        q = tl.div_rn(x, CAP)
        t = libdevice.tanh(q)
        s = CAP * t
        e = tl.where(mask, libdevice.exp(s - row_max), 0.0)
        if SHIFTED:
            partial = _ordered_tile_sum(e)
        else:
            # Historical unshifted path, not the current production tree.
            partial = tl.sum(e, axis=0)
        total = total + partial
        tl.store(quotient_ptr + row * V + offsets, q, mask=mask)
        tl.store(tanh_ptr + row * V + offsets, t, mask=mask)
        tl.store(score_ptr + row * V + offsets, s, mask=mask)
        tl.store(exp_ptr + row * V + offsets, e, mask=mask)
        tl.store(tiles_ptr + row * tl.cdiv(V, BLOCK) + start // BLOCK, partial)
    lse = tl.log(total)
    token = tl.load(ids_ptr + row).to(tl.int64)
    selected_x = tl.load(x_ptr + row * V + token).to(tl.float32)
    selected = CAP * libdevice.tanh(tl.div_rn(selected_x, CAP))
    selected_shifted = selected - row_max
    output = selected_shifted - lse
    tl.store(stats_ptr + row * 7, total)
    tl.store(stats_ptr + row * 7 + 1, lse)
    tl.store(stats_ptr + row * 7 + 2, selected)
    tl.store(stats_ptr + row * 7 + 3, output)
    tl.store(stats_ptr + row * 7 + 4, row_max)
    tl.store(stats_ptr + row * 7 + 5, selected_shifted)
    # Diagnostic counterfactual only; production contains no FP64 finishing code.
    output_f64_finish = (
        selected.to(tl.float64) - row_max.to(tl.float64) - libdevice.log(total.to(tl.float64))
    ).to(tl.float32)
    tl.store(stats_ptr + row * 7 + 6, output_f64_finish)


def same_bits(a, b):
    """Require identical shape, dtype and raw bytes, independent of strides."""
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8),
            b.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8),
        )
    )


def run_trace(out: Path) -> None:
    """Reproduce the fixed FP32 failure and save cumulative rounding diagnostics.

    Instrumentation is diagnostic only. Its output, maximum and shifted log-sum must match the
    production ROW kernel before its stages can explain a production failure.
    The JSON reports precision failures without turning this into a replacement
    for pytest; callers must run the FP32 acceptance and independent FP64
    mathematical test layers separately. Their numerical limits are unchanged.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA or ROCm GPU is required for the precision trace.")
    out.mkdir(parents=True, exist_ok=True)
    cap = 70.0
    generator = torch.Generator().manual_seed(438)
    cpu_x = torch.randn((17, 32767), generator=generator) * cap
    ids_cpu = torch.arange(17) * 1025
    x, ids = cpu_x.cuda(), ids_cpu.cuda()
    torch.save(
        {"logits": cpu_x, "token_ids": ids_cpu, "softcap": cap},
        out / "failure-input.pt",
    )
    scores64 = cap * torch.tanh(cpu_x.double() / cap)
    expected = torch.log_softmax(scores64, dim=-1).gather(1, ids_cpu[:, None]).squeeze(1)
    stages = {
        "fp64_math": expected,
        "fp64_rounded_once_to_fp32": expected.float().double(),
    }

    def selected(scores):
        return scores.gather(1, ids_cpu[:, None]).squeeze(1)

    def from_scores(scores):
        return torch.log_softmax(scores, dim=-1).gather(1, ids_cpu[:, None]).squeeze(1)

    q, t, s, e = [torch.empty_like(x) for _ in range(4)]
    tiles = torch.empty((17, triton.cdiv(x.shape[1], 1024)), device=x.device)
    stats = torch.empty((17, 7), device=x.device)
    with torch.cuda.device(x.device):
        _trace_row[(17,)](
            x,
            ids,
            q,
            t,
            s,
            e,
            tiles,
            stats,
            V=x.shape[1],
            CAP=cap,
            BLOCK=1024,
            SHIFTED=True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    q64, t64, s64, e64 = [v.cpu().double() for v in (q, t, s, e)]
    stats64 = stats.cpu().double()
    selected32 = stats64[:, 2]
    selected_shifted64 = selected32 - stats64[:, 4]
    rounded_selected_shifted64 = stats64[:, 5]
    stages.update(
        {
            "after_fp32_division_then_fp64": from_scores(cap * torch.tanh(q64)),
            "after_fp32_tanh_then_fp64": from_scores(cap * t64),
            "after_fp32_softcap_then_fp64": from_scores(s64),
            "after_fp32_shift_then_fp64": (
                rounded_selected_shifted64
                - torch.logsumexp((s - stats[:, 4, None]).cpu().double(), dim=-1)
            ),
            "after_fp32_exp_then_fp64_sum_log_subtract": rounded_selected_shifted64
            - e64.sum(-1).log(),
            "after_fp32_tile_reduction_then_fp64": rounded_selected_shifted64
            - tiles.cpu().double().sum(-1).log(),
            "after_fp32_row_sum_then_fp64": rounded_selected_shifted64 - stats64[:, 0].log(),
            "after_fp32_log_then_fp64_subtract": rounded_selected_shifted64 - stats64[:, 1],
            "shifted_fp32_output": stats64[:, 3],
            "shifted_fp64_finish_counterfactual": stats64[:, 6],
            "after_fp32_row_sum_with_fp64_shift_and_finish": selected_shifted64
            - stats64[:, 0].log(),
        }
    )
    native = NativeSoftcappedSelectedLogprobOp()
    stages["native_gpu"] = native(x, ids, softcap=cap).cpu().double()
    stages["native_cpu"] = native(cpu_x, ids_cpu, softcap=cap).double()
    nq = x / torch.tensor(cap, device=x.device, dtype=torch.float32)
    nt = torch.tanh(nq)
    ns = cap * nt
    nlse = torch.exp(ns).sum(-1).log()
    ns64, nlse64 = ns.cpu().double(), nlse.cpu().double()
    stages["native_gpu_after_softcap_then_fp64"] = from_scores(ns64)
    # Native now uses log_softmax; retain this old calculation only as a diagnostic.
    stages["legacy_native_gpu_before_final_fp32_subtract"] = selected(ns64) - nlse64
    stages["legacy_native_gpu_unshifted_output"] = (selected(ns.cpu()) - nlse.cpu()).double()

    checks = {
        "probe_division_matches_native_bits": same_bits(q, nq),
        "probe_tanh_matches_native_bits": same_bits(t, nt),
        "probe_scores_match_native_bits": same_bits(s, ns),
        "probe_selected_recompute_matches_vector_bits": same_bits(
            stats[:, 2], s.gather(1, ids[:, None]).squeeze(1)
        ),
    }
    for strategy in SoftcappedLogprobStrategy:
        output, row_max, lse = _FORWARD_LAUNCHERS[strategy](x, ids, cap)
        stages["production_" + strategy.value] = output.cpu().double()
        if strategy == SoftcappedLogprobStrategy.ROW:
            checks["probe_output_matches_production_row_bits"] = same_bits(output, stats[:, 3])
            checks["probe_lse_matches_production_row_bits"] = same_bits(lse, stats[:, 1])
            checks["probe_max_matches_production_row_bits"] = same_bits(row_max, stats[:, 4])
        op = TritonSoftcappedSelectedLogprobOp(forward_impl=strategy)
        training = op(x.detach().requires_grad_(True), ids, softcap=cap)
        with torch.inference_mode():
            inference = op(x, ids, softcap=cap)
        checks[strategy.value + "_training_inference_bits"] = same_bits(training, inference)

    # Reuse instrument buffers only after shifted stages/checks have been recorded.
    legacy_stats = torch.empty_like(stats)
    with torch.cuda.device(x.device):
        _trace_row[(17,)](
            x,
            ids,
            q,
            t,
            s,
            e,
            tiles,
            legacy_stats,
            V=x.shape[1],
            CAP=cap,
            BLOCK=1024,
            SHIFTED=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    stages["legacy_unshifted_fp32_output"] = legacy_stats[:, 3].cpu().double()
    stages["legacy_unshifted_fp64_finish"] = legacy_stats[:, 6].cpu().double()

    stage_results = {}
    for name, values in stages.items():
        error = values - expected
        native_error = values - stages["native_gpu"]
        stage_results[name] = {
            "values": values.tolist(),
            "signed_errors_vs_fp64": error.tolist(),
            "max_abs_error_vs_fp64": error.abs().max().item(),
            "rows_exceeding_1e_5": (error.abs() > 1e-5).nonzero().flatten().tolist(),
            "max_abs_error_vs_native_fp32": native_error.abs().max().item(),
            "rows_exceeding_native_1e_5": (
                (native_error.abs() > 1e-5).nonzero().flatten().tolist()
            ),
        }
    details = []
    for row in range(17):
        rounded = expected[row].float()
        details.append(
            {
                "row": row,
                "selected_x": cpu_x[row, ids_cpu[row]].item(),
                "fp64_logprob": expected[row].item(),
                "nearest_fp32": rounded.item(),
                "fp32_spacing": (
                    torch.nextafter(rounded, torch.full_like(rounded, float("inf"))) - rounded
                ).item(),
                "fp64_selected_softcap": selected(scores64)[row].item(),
                "triton_selected_softcap": selected32[row].item(),
                "native_gpu_selected_softcap": selected(ns64)[row].item(),
                "fp64_lse": torch.logsumexp(scores64[row], dim=0).item(),
                "triton_row_max": stats64[row, 4].item(),
                "triton_log_sum_exp_shifted": stats64[row, 1].item(),
                "triton_reconstructed_lse_fp64": (stats64[row, 4] + stats64[row, 1]).item(),
                "legacy_native_gpu_unshifted_lse": nlse64[row].item(),
            }
        )
    result = {
        "shape": list(x.shape),
        "dtype": "float32",
        "softcap": cap,
        "seed": 438,
        "notes": (
            "Diagnostic only. FP64 continuation separates cumulative rounding by stage. "
            "This diagnostic does not modify tolerances or production kernels. "
            "Native uses FP32 log_softmax. Legacy unshifted traces are "
            "counterfactual comparisons, not the current Op."
        ),
        "probe_checks": checks,
        "stages": stage_results,
        "row_details": details,
    }
    (out / "precision-trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)
    print("Row 13:", json.dumps(details[13], indent=2), flush=True)
    for name, values in stage_results.items():
        print(
            f"{name}: max_abs={values['max_abs_error_vs_fp64']:.9g}; "
            f"failing_rows={values['rows_exceeding_1e_5']}",
            flush=True,
        )
    assert checks[
        "probe_output_matches_production_row_bits"
    ], "Instrumented output differs from production; its stages cannot explain the failure."
    assert checks[
        "probe_lse_matches_production_row_bits"
    ], "Instrumented LSE differs from production."
    assert checks[
        "probe_max_matches_production_row_bits"
    ], "Instrumented maximum differs from production."


def main() -> None:
    """Write the fixed failure input and a per-stage numerical report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_trace(args.output_dir)


if __name__ == "__main__":
    main()
