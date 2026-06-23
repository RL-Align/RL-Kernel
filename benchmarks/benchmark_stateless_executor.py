# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.executors.paged_kv_baseline import (  # noqa: E402
    PagedKVScoringBaseline,
    PagedKVScoringConfig,
)
from rl_engine.executors.stateless_executor import (  # noqa: E402
    StatelessForwardConfig,
    StatelessForwardExecutor,
    StatelessForwardInputs,
)

CSV_COLUMNS = [
    "timestamp",
    "candidate",
    "mode",
    "stage",
    "batch_size",
    "seq_len",
    "active_tokens",
    "device",
    "dtype",
    "elapsed_ms",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "paged_kv_baseline_mb",
    "kv_cache_output_mb",
    "savings_vs_local_reservation_baseline_mb",
    "status",
    "notes",
]

SCORING_MODES = {"reference", "reward", "both"}
DTYPE_CHOICES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class TinyReferenceModel(torch.nn.Module):
    """Small deterministic causal-LM-like module for executor smoke benchmarks."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)
        self.use_cache_calls: list[bool | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        del attention_mask
        self.use_cache_calls.append(use_cache)
        hidden = self.embedding(input_ids.long())
        return SimpleNamespace(
            logits=self.proj(hidden),
            past_key_values=_tiny_past_key_values(hidden, use_cache),
        )


class TinyRewardModel(torch.nn.Module):
    """Small deterministic scalar-reward module for executor smoke benchmarks."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.reward_head = torch.nn.Linear(hidden_dim, 1)
        self.use_cache_calls: list[bool | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.use_cache_calls.append(use_cache)
        hidden = self.embedding(input_ids.long())
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {
            "logits": self.reward_head(pooled),
            "past_key_values": _tiny_past_key_values(hidden, use_cache),
        }


class TinyBothModel(torch.nn.Module):
    """Tiny model with both LM logits and scalar rewards in one forward."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)
        self.reward_head = torch.nn.Linear(hidden_dim, 1)
        self.use_cache_calls: list[bool | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.use_cache_calls.append(use_cache)
        hidden = self.embedding(input_ids.long())
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {
            "logits": self.lm_head(hidden),
            "rewards": self.reward_head(pooled).squeeze(-1),
            "past_key_values": _tiny_past_key_values(hidden, use_cache),
        }


def _tiny_past_key_values(hidden: torch.Tensor, use_cache: bool | None):
    if use_cache is not True:
        return None
    batch_size, seq_len, hidden_dim = hidden.shape
    head_dim = max(1, min(hidden_dim, 8))
    key = torch.empty(
        batch_size,
        1,
        seq_len,
        head_dim,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    value = torch.empty_like(key)
    return ((key, value),)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment() -> str:
    parts = [f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        parts.append(f"cuda={torch.version.cuda}")
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    return ";".join(parts)


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_inputs(args: argparse.Namespace, device: torch.device) -> StatelessForwardInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
        generator=generator,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    completion_start = max(1, args.seq_len - args.completion_len)
    completion_mask[:, completion_start:] = True

    if args.valid_density < 1.0:
        keep = (
            torch.rand(
                args.batch_size,
                args.seq_len - completion_start,
                device=device,
                generator=generator,
            )
            < args.valid_density
        )
        keep[:, 0] = True
        completion_mask[:, completion_start:] &= keep
        prefix_positions = (
            torch.arange(args.seq_len, device=device).unsqueeze(0).lt(completion_start)
        )
        attention_mask &= completion_mask | prefix_positions

    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def _make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    mode = _scoring_mode(args)
    if mode == "reference":
        model: torch.nn.Module = TinyReferenceModel(args.vocab_size, args.hidden_dim)
    elif mode == "reward":
        model = TinyRewardModel(args.vocab_size, args.hidden_dim)
    else:
        model = TinyBothModel(args.vocab_size, args.hidden_dim)
    return model.to(device=device).eval()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "paged-kv-compare":
        return _run_paged_kv_only(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return _blocked_row(args, "CUDA is not available")

    torch.manual_seed(args.seed)
    inputs = _make_inputs(args, device)
    model = _make_model(args, device)
    scoring_mode = _scoring_mode(args)
    executor = StatelessForwardExecutor(
        model,
        StatelessForwardConfig(mode=scoring_mode, return_token_scores=scoring_mode != "reward"),
    )

    result = executor.score(inputs)
    peak_allocated = result.metrics.get("peak_allocated_mb", "")
    peak_reserved = result.metrics.get("peak_reserved_mb", "")

    row = {
        "timestamp": _timestamp(),
        "candidate": "issue-47-candidate-1",
        "mode": args.mode,
        "stage": "evaluation",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "active_tokens": result.metrics["active_completion_tokens"],
        "device": str(device),
        "dtype": str(inputs.input_ids.dtype).replace("torch.", ""),
        "elapsed_ms": f"{float(result.metrics['elapsed_ms']):.4f}",
        "peak_allocated_mb": f"{float(peak_allocated):.4f}" if peak_allocated != "" else "",
        "peak_reserved_mb": f"{float(peak_reserved):.4f}" if peak_reserved != "" else "",
        "paged_kv_baseline_mb": "",
        "kv_cache_output_mb": f"{float(result.metrics['kv_cache_output_mb']):.4f}",
        "savings_vs_local_reservation_baseline_mb": "",
        "status": "pass",
        "notes": (
            f"use_cache_passed={result.metrics['use_cache_passed']};"
            f"detached_outputs={result.metrics['detached_outputs']};"
            f"zero_kv_cache={result.metrics['zero_kv_cache']};"
            f"attention_backend={result.metrics['attention_backend']};"
            f"hidden_dim={args.hidden_dim};vocab_size={args.vocab_size};{_environment()}"
        ),
    }
    if args.compare_paged_kv:
        paged_result = _score_paged_kv(args, inputs, model)
        paged_metrics = paged_result.metrics
        row["paged_kv_baseline_mb"] = f"{float(paged_metrics['paged_kv_cache_reserved_mb']):.4f}"
        savings_mb = float(paged_metrics["total_kv_cache_mb"]) - float(
            result.metrics["kv_cache_output_mb"]
        )
        row["savings_vs_local_reservation_baseline_mb"] = f"{savings_mb:.4f}"
        row["notes"] += (
            f";paged_kv_elapsed_ms={float(paged_metrics['elapsed_ms']):.4f};"
            f"paged_kv_blocks={paged_metrics['paged_kv_blocks']};"
            f"paged_kv_required_blocks={paged_metrics['paged_kv_required_blocks']};"
            f"paged_kv_block_size={paged_metrics['paged_kv_block_size']};"
            f"paged_kv_dtype={paged_metrics['kv_cache_dtype']};"
            f"paged_kv_use_cache_passed={paged_metrics['use_cache_passed']};"
            f"paged_kv_model_cache_mb={float(paged_metrics['model_kv_cache_output_mb']):.4f};"
            f"paged_kv_total_cache_mb={float(paged_metrics['total_kv_cache_mb']):.4f}"
        )
        correctness = _paged_kv_correctness_notes(result, paged_result)
        if correctness:
            row["notes"] += f";{correctness}"
    return row


def _run_paged_kv_only(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return _blocked_row(args, "CUDA is not available")

    torch.manual_seed(args.seed)
    inputs = _make_inputs(args, device)
    model = _make_model(args, device)
    result = _score_paged_kv(args, inputs, model)
    return _paged_kv_row(args, inputs, result)


def _score_paged_kv(
    args: argparse.Namespace,
    inputs: StatelessForwardInputs,
    model: torch.nn.Module,
):
    baseline = PagedKVScoringBaseline(model, _paged_kv_config(args))
    return baseline.score(inputs)


def _paged_kv_config(args: argparse.Namespace) -> PagedKVScoringConfig:
    scoring_mode = _scoring_mode(args)
    return PagedKVScoringConfig(
        mode=scoring_mode,
        num_layers=args.paged_kv_layers,
        num_kv_heads=args.paged_kv_heads,
        head_dim=args.paged_kv_head_dim,
        block_size=args.paged_kv_block_size,
        kv_cache_dtype=DTYPE_CHOICES[args.paged_kv_dtype],
        kv_cache_blocks=args.paged_kv_blocks,
        return_token_scores=scoring_mode != "reward",
    )


def _paged_kv_row(
    args: argparse.Namespace,
    inputs: StatelessForwardInputs,
    result,
) -> dict[str, Any]:
    metrics = result.metrics
    peak_allocated = metrics.get("peak_allocated_mb", "")
    peak_reserved = metrics.get("peak_reserved_mb", "")
    return {
        "timestamp": _timestamp(),
        "candidate": "issue-47-candidate-1",
        "mode": args.mode,
        "stage": "evaluation",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "active_tokens": metrics["active_completion_tokens"],
        "device": str(inputs.input_ids.device),
        "dtype": str(inputs.input_ids.dtype).replace("torch.", ""),
        "elapsed_ms": f"{float(metrics['elapsed_ms']):.4f}",
        "peak_allocated_mb": f"{float(peak_allocated):.4f}" if peak_allocated != "" else "",
        "peak_reserved_mb": f"{float(peak_reserved):.4f}" if peak_reserved != "" else "",
        "paged_kv_baseline_mb": f"{float(metrics['paged_kv_cache_reserved_mb']):.4f}",
        "kv_cache_output_mb": f"{float(metrics['model_kv_cache_output_mb']):.4f}",
        "savings_vs_local_reservation_baseline_mb": "",
        "status": "pass",
        "notes": (
            f"baseline_kind={metrics['baseline_kind']};"
            f"scoring_mode={metrics['mode']};"
            f"use_cache_passed={metrics['use_cache_passed']};"
            f"paged_kv_layers={metrics['paged_kv_layers']};"
            f"paged_kv_heads={metrics['paged_kv_heads']};"
            f"paged_kv_head_dim={metrics['paged_kv_head_dim']};"
            f"paged_kv_block_size={metrics['paged_kv_block_size']};"
            f"paged_kv_blocks={metrics['paged_kv_blocks']};"
            f"paged_kv_required_blocks={metrics['paged_kv_required_blocks']};"
            f"paged_kv_dtype={metrics['kv_cache_dtype']};"
            f"model_kv_cache_output_present={metrics['model_kv_cache_output_present']};"
            f"total_kv_cache_mb={float(metrics['total_kv_cache_mb']):.4f};"
            f"hidden_dim={args.hidden_dim};vocab_size={args.vocab_size};{_environment()}"
        ),
    }


def _paged_kv_correctness_notes(stateless_result, paged_result) -> str:
    notes: list[str] = []
    if stateless_result.reference_logps is not None and paged_result.reference_logps is not None:
        reference_match = torch.allclose(
            stateless_result.reference_logps,
            paged_result.reference_logps,
        )
        notes.append(f"paged_kv_reference_allclose={bool(reference_match)}")
    if stateless_result.rewards is not None and paged_result.rewards is not None:
        reward_match = torch.allclose(
            stateless_result.rewards,
            paged_result.rewards,
        )
        notes.append(f"paged_kv_rewards_allclose={bool(reward_match)}")
    return ";".join(notes)


def _scoring_mode(args: argparse.Namespace) -> str:
    if args.mode in SCORING_MODES:
        return args.mode
    return "reference"


def _blocked_row(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(),
        "candidate": "issue-47-candidate-1",
        "mode": args.mode,
        "stage": "evaluation",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "device": args.device,
        "dtype": "int64",
        "status": "blocked",
        "notes": f"{reason};{_environment()}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stateless Reference/Reward executor benchmark")
    parser.add_argument(
        "--mode",
        choices=["reference", "reward", "both", "paged-kv-compare"],
        default="reference",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--compare-paged-kv",
        action="store_true",
        help="Run the local generation-style paged-KV reservation baseline too.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--completion-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--paged-kv-layers", type=int, default=4)
    parser.add_argument("--paged-kv-heads", type=int, default=8)
    parser.add_argument("--paged-kv-head-dim", type=int, default=32)
    parser.add_argument("--paged-kv-block-size", type=int, default=16)
    parser.add_argument("--paged-kv-blocks", type=int, default=None)
    parser.add_argument("--paged-kv-dtype", choices=sorted(DTYPE_CHOICES), default="float16")
    parser.add_argument("--valid-density", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "task-workspace/issues/issue_47/benchmark.csv",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.batch_size = min(args.batch_size, 2)
        args.seq_len = min(args.seq_len, 16)
        args.completion_len = min(args.completion_len, 8)
        args.vocab_size = min(args.vocab_size, 128)
        args.hidden_dim = min(args.hidden_dim, 32)
        args.paged_kv_layers = min(args.paged_kv_layers, 2)
        args.paged_kv_heads = min(args.paged_kv_heads, 2)
        args.paged_kv_head_dim = min(args.paged_kv_head_dim, 8)
        args.paged_kv_block_size = min(args.paged_kv_block_size, 16)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero")
    if args.seq_len < 2:
        raise ValueError("seq-len must be at least 2")
    if args.completion_len <= 0 or args.completion_len >= args.seq_len:
        raise ValueError("completion-len must be greater than zero and less than seq-len")
    if args.vocab_size <= 1:
        raise ValueError("vocab-size must be greater than one")
    if args.hidden_dim <= 0:
        raise ValueError("hidden-dim must be greater than zero")
    if args.paged_kv_layers <= 0:
        raise ValueError("paged-kv-layers must be greater than zero")
    if args.paged_kv_heads <= 0:
        raise ValueError("paged-kv-heads must be greater than zero")
    if args.paged_kv_head_dim <= 0:
        raise ValueError("paged-kv-head-dim must be greater than zero")
    if args.paged_kv_block_size <= 0:
        raise ValueError("paged-kv-block-size must be greater than zero")
    if args.paged_kv_blocks is not None and args.paged_kv_blocks <= 0:
        raise ValueError("paged-kv-blocks must be greater than zero")
    if not (0.0 < args.valid_density <= 1.0):
        raise ValueError("valid-density must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    row = _run(args)
    _append_row(args.output, row)
    _write_json(args.json_output, row)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
