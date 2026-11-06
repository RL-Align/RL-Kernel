# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare the HF-style eager scaled embedding with the Triton op on a CUDA or ROCm GPU.

Forward and backward are timed separately. Backward cases vary how often a token id
repeats, because the deterministic sorted-segment reducer walks each segment serially.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.cli import (  # noqa: E402
    command_output,
    measure,
    positive_int,
    run_and_save_report,
)
from benchmarks.common.profiler import GPUTargetInfo, PerformanceProfiler  # noqa: E402
from rl_engine.contracts.numerical import tolerance_contract_fingerprint  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward")
# unique: one row per segment. duplicate: ~64 rows per id. pad_flood: half the
# tokens share one id, the worst case for a serial per-segment loop.
ID_PATTERNS = ("unique", "duplicate", "pad_flood")
DEFAULT_TOKENS = (1, 2048, 32768)
DEFAULT_VOCAB = 262144
DEFAULT_HIDDEN = 5376
PAD_ID = 0


class HFScaledEmbedding:
    """``F.embedding(ids, w) * scale``, the ``Gemma4TextScaledWordEmbedding`` expression."""

    def __init__(self, scale: torch.Tensor) -> None:
        self.scale = scale

    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, weight) * self.scale


def make_token_ids(n_tokens, vocab, pattern, *, generator, device):
    if pattern == "unique":
        if n_tokens > vocab:
            raise ValueError("unique ids need n_tokens <= vocab")
        return torch.randperm(vocab, generator=generator, device=device)[:n_tokens]
    if pattern == "duplicate":
        distinct = max(1, n_tokens // 64)
        return torch.randint(0, distinct, (n_tokens,), generator=generator, device=device)
    if pattern == "pad_flood":
        ids = torch.randint(0, vocab, (n_tokens,), generator=generator, device=device)
        ids[::2] = PAD_ID
        return ids
    raise ValueError(f"unknown id pattern: {pattern}")


def _make_workload(fn, token_ids, weight, grad_out, mode):
    """Prepare inputs/graphs outside timing; discard results without accumulating .grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            fn(token_ids, weight)

        return forward
    if mode == "backward":
        leaf = weight.detach().requires_grad_(True)
        out = fn(token_ids, leaf)

        def backward():
            torch.autograd.grad(out, leaf, grad_outputs=grad_out, retain_graph=True)

        return backward
    raise ValueError(f"unknown mode: {mode}")


def check_forward_bitwise(baseline, candidate, token_ids, weight):
    """Forward must be bitwise HF; the backward contract differs from HF on purpose."""
    with torch.no_grad():
        expected = baseline(token_ids, weight)
        actual = candidate(token_ids, weight)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError("candidate forward shape or dtype differs from the HF expression")
    if not torch.equal(actual, expected):
        raise RuntimeError("candidate forward is not bitwise equal to the HF expression")


def backward_gap(baseline, candidate, token_ids, weight, grad_out):
    """Compare only the touched rows: every other row of both gradients is zero."""
    touched = token_ids.unique()

    def touched_grad(fn):
        leaf = weight.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(fn(token_ids, leaf), leaf, grad_outputs=grad_out)
        return grad.index_select(0, touched).float()

    gap = (touched_grad(candidate) - touched_grad(baseline)).abs()
    return {"max_abs_gap_vs_hf": gap.max().item(), "mean_abs_gap_vs_hf": gap.mean().item()}


def _environment(device, args, triton_version, blocks, target: GPUTargetInfo):
    props = torch.cuda.get_device_properties(device)
    is_rocm = target.backend == "rocm"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": command_output(
            ["git", "status", "--porcelain", "--untracked-files=no"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "backend": target.backend,
        "runtime": target.driver_version,
        "gpu": props.name,
        "architecture": target.architecture,
        "compute_capability": None if is_rocm else list(torch.cuda.get_device_capability(device)),
        "gpu_name_driver": command_output(
            ["rocm-smi", "--showproductname", "--showdriverversion"]
            if is_rocm
            else ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "device": str(device),
        "total_memory_gib": props.total_memory / 1024**3,
        "multiprocessors": props.multi_processor_count,
        "vocab": args.vocab,
        "hidden": args.hidden,
        "blocks": blocks,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "tolerance_contract_sha256": tolerance_contract_fingerprint(),
        "timer": (
            f"PerformanceProfiler {'HIP' if is_rocm else 'CUDA'} events; "
            "median and sample standard deviation"
        ),
        "baseline": "eager F.embedding(ids, w) * scale.to(w.dtype) on the same GPU",
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
        "backward": "Random upstream gradient in the weight dtype; prebuilt retained graph",
        "memory": "Extra peak allocated MiB per call; excludes inputs and any prebuilt graph",
    }


def _markdown(report):
    env = report["environment"]
    event_api = "HIP" if env["backend"] == "rocm" else "CUDA"
    lines = [
        "# Scaled token embedding benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Table: vocab {env['vocab']} x hidden {env['hidden']}; warmup {env['warmup']}; "
        f"measured repetitions {env['repeat']}.",
        "",
        f"Median {event_api} event latency for public wrappers, including allocation and "
        "autograd dispatch. Forward runs under no_grad and is checked bitwise against the HF "
        "expression first. Backward reuses a prebuilt graph; its gap to HF autograd is "
        "reported, not gated, because the contract sums fp32 products without an "
        "intermediate rounding.",
        "Speedup = HF eager / Triton; values below 1 mean Triton was slower.",
        "",
        "| Input | Tokens | Ids | Mode | HF ms | Triton ms | Speedup | "
        "HF extra MiB | Triton extra MiB |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        h, t = row["hf"], row["triton"]
        lines.append(
            f"| {row['dtype']} | {row['tokens']} | {row['ids']} | {row['mode']} | "
            f"{h['median_ms']:.6f} | {t['median_ms']:.6f} | {row['speedup']:.2f}x | "
            f"{h['peak_extra_mib']:.2f} | {t['peak_extra_mib']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--tokens", nargs="+", type=positive_int, default=list(DEFAULT_TOKENS))
    parser.add_argument("--vocab", type=positive_int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=positive_int, default=DEFAULT_HIDDEN)
    parser.add_argument("--patterns", nargs="+", choices=ID_PATTERNS, default=list(ID_PATTERNS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repeat", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/scaled-token-embedding"))
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    # CPU is rejected outright: its fallback timing must never be reported as GPU
    # performance. ROCm is accepted; AMD GPUs live under torch.cuda.
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    # Import only after the device check, so --help works without Triton/GPU.
    import triton

    from rl_engine.backends.shared.triton.embedding.embedding import _BACKWARD_BLOCK_H
    from rl_engine.backends.shared.triton.embedding.scaled_embedding import (
        TritonScaledTokenEmbeddingOp,
    )

    embed_scale = float(args.hidden) ** 0.5
    triton_op = TritonScaledTokenEmbeddingOp()
    blocks = {
        "forward_block_h": triton.next_power_of_2(args.hidden),
        "backward_block_h": _BACKWARD_BLOCK_H,
    }
    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        report = {
            "environment": _environment(
                device, args, triton.__version__, blocks, profiler.gpu_info
            ),
            "results": [],
        }
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            generator = torch.Generator(device=device).manual_seed(args.seed)
            weight = torch.randn(args.vocab, args.hidden, device=device, generator=generator)
            weight = weight.to(dtype)
            baseline = HFScaledEmbedding(torch.tensor(embed_scale, device=device).to(dtype))

            def candidate(token_ids, table):
                return triton_op(token_ids, table, embed_scale)

            for n_tokens in args.tokens:
                for pattern in args.patterns:
                    token_ids = make_token_ids(
                        n_tokens, args.vocab, pattern, generator=generator, device=device
                    )
                    grad_out = torch.randn(
                        n_tokens, args.hidden, device=device, generator=generator
                    ).to(dtype)
                    check_forward_bitwise(baseline, candidate, token_ids, weight)
                    for mode in args.modes:
                        if mode == "forward" and pattern != ID_PATTERNS[0]:
                            # Forward cost does not depend on repeats; time it once.
                            continue
                        measurements = {}
                        for name, fn in (("hf", baseline), ("triton", candidate)):
                            workload = _make_workload(fn, token_ids, weight, grad_out, mode)
                            measurements[name] = measure(profiler, workload)
                            del workload  # Release the retained graph before the next one.
                        row = {
                            "dtype": dtype_name,
                            "tokens": n_tokens,
                            "ids": pattern if mode == "backward" else "any",
                            "mode": mode,
                            **measurements,
                            "speedup": measurements["hf"]["median_ms"]
                            / measurements["triton"]["median_ms"],
                        }
                        if mode == "backward":
                            row["backward_gap"] = backward_gap(
                                baseline, candidate, token_ids, weight, grad_out
                            )
                        report["results"].append(row)
                        print(
                            f"{dtype_name} tokens={n_tokens} ids={row['ids']} {mode}: "
                            f"{row['speedup']:.2f}x",
                            flush=True,
                        )
                    del token_ids, grad_out
            del weight, baseline
            torch.cuda.empty_cache()
    return report


def main():
    run_and_save_report(build_arg_parser(), run_benchmark, _markdown)


if __name__ == "__main__":
    main()
