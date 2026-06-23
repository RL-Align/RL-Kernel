# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for the fused masking + pack-and-pad op (issue #42).

Two measurements per shape:

1. Pack latency: TritonPackOp vs a PyTorch boolean-index baseline
   (``x.reshape(-1, T)[mask]``), with max-abs drift between the two.
2. End-to-end peak VRAM: the motivation behind #42. Computing selected
   log-probs on the *dense* ``[B, S, V]`` tensor materializes full-sequence
   logits for masked-out tokens; packing first lets the log-prob run only on
   the ``[Total_Active, V]`` active rows. We report the peak GPU memory of
   ``dense logp`` vs ``pack -> logp`` to quantify the saving when the mask is
   sparse (long prompt, short response, padded batches).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.ops.pytorch.packing.pack import NativePackOp  # noqa: E402
from rl_engine.testing import make_synthetic_rl_kernel_batch  # noqa: E402

CSV_COLUMNS = [
    "timestamp",
    "case",
    "candidate",
    "device",
    "dtype",
    "num_prompts",
    "samples_per_prompt",
    "completion_len",
    "vocab_size",
    "mask_density",
    "valid_tokens",
    "baseline_ms",
    "candidate_ms",
    "speedup",
    "pack_drift",
    "dense_logp_mem_gb",
    "packed_logp_mem_gb",
    "mem_saving_pct",
    "status",
    "notes",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    case: str
    device: torch.device
    dtype: torch.dtype
    num_prompts: int
    samples_per_prompt: int
    completion_len: int
    vocab_size: int
    mask_density: float
    seed: int
    warmup: int
    repeat: int


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, device: torch.device, *, warmup: int = 3, repeat: int = 10) -> tuple[Any, float]:
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    _sync(device)

    elapsed: list[float] = []
    for _ in range(max(1, repeat)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        else:
            _sync(device)
            start_time = time.perf_counter()
            result = fn()
            _sync(device)
            elapsed.append((time.perf_counter() - start_time) * 1000.0)

    _sync(device)
    return result, statistics.median(elapsed)


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _baseline_pack(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: flatten and boolean-index the active rows."""
    return x.reshape(-1, x.shape[-1])[mask.reshape(-1)]


def _selected_logp(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log_softmax(logits)[ids] over the last dim; ids shape == logits.shape[:-1]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)


def _pack_row(config: BenchmarkConfig) -> dict[str, Any]:
    candidate_name = "TritonPackOp"

    batch = make_synthetic_rl_kernel_batch(
        num_prompts=config.num_prompts,
        samples_per_prompt=config.samples_per_prompt,
        prompt_len=0,
        completion_len=config.completion_len,
        vocab_size=config.vocab_size,
        valid_density=config.mask_density,
        dtype=config.dtype,
        device=config.device,
        seed=config.seed,
    )

    logit_shape = (batch.batch_size, batch.completion_len, config.vocab_size)
    logits = torch.randn(logit_shape, device=config.device, dtype=config.dtype)
    mask = batch.completion_mask
    ids = batch.token_ids

    native_pack = NativePackOp()

    status = "pass"
    notes = ""
    baseline_ms: float | str = ""
    candidate_ms: float | str = ""
    speedup: float | str = ""
    pack_drift: float | str = ""
    dense_logp_mem_gb: float | str = ""
    packed_logp_mem_gb: float | str = ""
    mem_saving_pct: float | str = ""

    # (1) pack latency: PyTorch boolean-index baseline vs Triton candidate.
    _reset_peak(config.device)
    base_packed, baseline_ms = _time_ms(
        lambda: _baseline_pack(logits, mask),
        config.device,
        warmup=config.warmup,
        repeat=config.repeat,
    )

    if config.device.type != "cuda":
        status = "blocked"
        notes = "candidate requires CUDA"
    else:
        try:
            from rl_engine.kernels.registry import kernel_registry

            candidate_op = kernel_registry.get_op("pack")
            if candidate_op.__class__.__name__ != candidate_name:
                raise RuntimeError(f"{candidate_name} backend is unavailable")

            (cand_packed, _), candidate_ms = _time_ms(
                lambda: candidate_op(logits, mask),
                config.device,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            speedup = baseline_ms / candidate_ms if candidate_ms else float("inf")
            pack_drift = (cand_packed.float() - base_packed.float()).abs().max().item()

            # (2) end-to-end peak VRAM: dense logp vs pack->logp.
            _reset_peak(config.device)
            _ = _selected_logp(logits, ids)
            _sync(config.device)
            dense_logp_mem_gb = _peak_memory_gb(config.device)

            _reset_peak(config.device)
            packed_logits, _ = candidate_op(logits, mask)
            packed_ids, _ = candidate_op(ids.unsqueeze(-1), mask)
            _ = _selected_logp(packed_logits, packed_ids.squeeze(-1))
            _sync(config.device)
            packed_logp_mem_gb = _peak_memory_gb(config.device)

            if dense_logp_mem_gb > 0:
                mem_saving_pct = 100.0 * (1.0 - packed_logp_mem_gb / dense_logp_mem_gb)
        except Exception as exc:
            status = "blocked"
            notes = f"candidate unavailable: {str(exc).splitlines()[0]}"

    metadata = batch.benchmark_metadata()
    timing_mode = "cuda_event_median_ms" if config.device.type == "cuda" else "wall_median_ms"
    timing_notes = f"warmup={config.warmup}; repeat={config.repeat}; {timing_mode}"
    notes = f"{notes}; {timing_notes}" if notes else timing_notes

    def _fmt(value: Any, spec: str) -> Any:
        return format(value, spec) if isinstance(value, float) else value

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case": config.case,
        "candidate": candidate_name,
        "device": str(config.device),
        "dtype": str(config.dtype),
        "num_prompts": config.num_prompts,
        "samples_per_prompt": config.samples_per_prompt,
        "completion_len": config.completion_len,
        "vocab_size": config.vocab_size,
        "mask_density": config.mask_density,
        "valid_tokens": metadata["valid_tokens"],
        "baseline_ms": f"{baseline_ms:.4f}" if isinstance(baseline_ms, float) else baseline_ms,
        "candidate_ms": _fmt(candidate_ms, ".4f"),
        "speedup": _fmt(speedup, ".2f"),
        "pack_drift": _fmt(pack_drift, ".3e"),
        "dense_logp_mem_gb": _fmt(dense_logp_mem_gb, ".6f"),
        "packed_logp_mem_gb": _fmt(packed_logp_mem_gb, ".6f"),
        "mem_saving_pct": _fmt(mem_saving_pct, ".2f"),
        "status": status,
        "notes": notes,
    }


def _write_rows(rows: list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fused pack-and-pad RL-Kernel benchmark runner")
    parser.add_argument("--case", default="pack", choices=["pack"])
    parser.add_argument("--smoke", action="store_true", help="Run a small local-development shape")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--g-sizes", default="8", help="Comma-separated samples-per-prompt values")
    parser.add_argument("--completion-lens", default="1024")
    parser.add_argument("--vocab-sizes", default="32768,131072")
    parser.add_argument(
        "--mask-densities",
        default="0.1,0.3,1.0",
        help="Active-token fraction; sparse masks show the largest VRAM saving",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if args.smoke:
        num_prompts = 1
        g_sizes = [2]
        completion_lens = [8]
        vocab_sizes = [128]
        mask_densities = [0.5, 1.0]
    else:
        num_prompts = args.num_prompts
        g_sizes = _parse_int_list(args.g_sizes)
        completion_lens = _parse_int_list(args.completion_lens)
        vocab_sizes = _parse_int_list(args.vocab_sizes)
        mask_densities = _parse_float_list(args.mask_densities)

    rows: list[dict[str, Any]] = []
    for samples_per_prompt in g_sizes:
        for completion_len in completion_lens:
            for vocab_size in vocab_sizes:
                for mask_density in mask_densities:
                    config = BenchmarkConfig(
                        case=args.case,
                        device=device,
                        dtype=dtype,
                        num_prompts=num_prompts,
                        samples_per_prompt=samples_per_prompt,
                        completion_len=completion_len,
                        vocab_size=vocab_size,
                        mask_density=mask_density,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    try:
                        rows.append(_pack_row(config))
                    except torch.cuda.OutOfMemoryError as exc:
                        rows.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "case": args.case,
                                "candidate": "TritonPackOp",
                                "device": str(device),
                                "dtype": str(dtype),
                                "num_prompts": num_prompts,
                                "samples_per_prompt": samples_per_prompt,
                                "completion_len": completion_len,
                                "vocab_size": vocab_size,
                                "mask_density": mask_density,
                                "valid_tokens": "",
                                "baseline_ms": "",
                                "candidate_ms": "",
                                "speedup": "",
                                "pack_drift": "",
                                "dense_logp_mem_gb": "",
                                "packed_logp_mem_gb": "",
                                "mem_saving_pct": "",
                                "status": "oom",
                                "notes": str(exc),
                            }
                        )

    _write_rows(rows, args.output)


if __name__ == "__main__":
    main()
