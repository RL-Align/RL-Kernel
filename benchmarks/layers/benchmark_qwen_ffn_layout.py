# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark Qwen3 FFN deterministic layout and forward performance.

The baseline replays the deterministic single-GPU FFN layout contract from
commit 7207ebd with the preserved legacy CUDA bindings:

* forward materializes ``W.T.contiguous()`` for all three projections;
* weight-gradient GEMMs materialize ``dW.T.contiguous()`` for all three weights.

The optimized path replays the current TP=1 ``qwen3_ffn`` layout core, which
reads canonical ``[out, in]`` weights directly and writes canonical contiguous
weight gradients directly. A third, symmetric TP=1 path uses production
``torch.matmul`` on CUDA as a cuBLAS performance reference. A fourth,
forward-only path uses the batch-invariant persistent matmul vendored from
vLLM PR #53247. The script requires the two deterministic paths to match
bitwise and both numerical reference paths to agree with the optimized output
within explicit sanity bounds before timing.

Example:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/layers/benchmark_qwen_ffn_layout.py \
      --tokens 1,8,32,128 \
      --warmup 3 \
      --samples 20 \
      --training-samples 10 \
      --output-dir benchmarks/results/qwen_ffn_layout_h100

The output directory contains ``results.json``, ``results.csv``, ``report.md``,
and separate forward and forward+backward comparison SVGs. This is an operator-only
microbenchmark; it does not load a model, checkpoint, tokenizer, or dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

import torch
import triton
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rl_engine  # noqa: E402
from benchmarks.layers.vllm_batch_invariant_matmul import (  # noqa: E402
    VLLM_BATCH_INVARIANT_PR_URL,
    VLLM_BATCH_INVARIANT_SOURCE_SHA,
    VLLM_BATCH_INVARIANT_SOURCE_URL,
    matmul_config_metadata,
    matmul_persistent,
)
from rl_engine.backends.extension import _C, _EXT_AVAILABLE  # noqa: E402
from rl_engine.ops.ffn import qwen3_ffn  # noqa: E402

SCHEMA_VERSION = 4
LEGACY_COMMIT = "7207ebd"
LEGACY_LABEL = "Replayed legacy layout (7207ebd contract)"
OPTIMIZED_LABEL = "Optimized canonical layout"
CUBLAS_LABEL = "PyTorch CUDA BLAS / cuBLAS"
VLLM_LABEL = "vLLM batch-invariant persistent matmul"
CUBLAS_RELATIVE_L2_LIMIT = 0.02
CUBLAS_NORMALIZED_MAX_LIMIT = 0.05
INPUT_SCALE = 0.02
OPTIMIZED_COLOR = "#7c3aed"
CUBLAS_COLOR = "#0891b2"
VLLM_COLOR = "#d97706"
GRID_COLOR = "#d1d5db"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4b5563"


class _LegacyLayoutFFNFunction(torch.autograd.Function):
    """Replay the TP=1 deterministic FFN layout path from ``7207ebd``."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Tensor:
        input_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, input_shape[-1]).contiguous()
        gate = _C.det_gemm_fwd(hidden_2d, gate_weight.t().contiguous())
        up = _C.det_gemm_fwd(hidden_2d, up_weight.t().contiguous())
        activated = _C.swiglu_forward(gate, up)
        output = _C.det_gemm_fwd(activated, down_weight.t().contiguous())
        ctx.save_for_backward(
            hidden_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            hidden,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()

        grad_down_weight = _C.det_gemm_db(activated, grad_output).t().contiguous()
        grad_activated = _C.det_gemm_fwd(grad_output, down_weight)
        grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)
        grad_gate_weight = _C.det_gemm_db(hidden, grad_gate).t().contiguous()
        grad_up_weight = _C.det_gemm_db(hidden, grad_up).t().contiguous()
        grad_hidden_gate = _C.det_gemm_fwd(grad_gate, gate_weight)
        grad_hidden_up = _C.det_gemm_fwd(grad_up, up_weight)
        grad_hidden = grad_hidden_gate.add_(grad_hidden_up)
        return (
            grad_hidden.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
        )


def _legacy_qwen3_ffn(
    hidden: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    return _LegacyLayoutFFNFunction.apply(hidden, gate_weight, up_weight, down_weight)


class _OptimizedLayoutFFNFunction(torch.autograd.Function):
    """Run the current TP=1 layout core with timing scope symmetric to baseline."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Tensor:
        input_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, input_shape[-1]).contiguous()
        gate = _C.det_gemm_fwd_rhs_transposed(hidden_2d, gate_weight)
        up = _C.det_gemm_fwd_rhs_transposed(hidden_2d, up_weight)
        activated = _C.swiglu_forward(gate, up)
        output = _C.det_gemm_fwd_rhs_transposed(activated, down_weight)
        ctx.save_for_backward(
            hidden_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            hidden,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()

        grad_down_weight = _C.det_gemm_db_transposed(activated, grad_output)
        grad_activated = _C.det_gemm_fwd(grad_output, down_weight)
        grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)
        grad_gate_weight = _C.det_gemm_db_transposed(hidden, grad_gate)
        grad_up_weight = _C.det_gemm_db_transposed(hidden, grad_up)
        grad_hidden_gate = _C.det_gemm_fwd(grad_gate, gate_weight)
        grad_hidden_up = _C.det_gemm_fwd(grad_up, up_weight)
        grad_hidden = grad_hidden_gate.add_(grad_hidden_up)
        return (
            grad_hidden.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
        )


def _optimized_qwen3_ffn(
    hidden: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    return _OptimizedLayoutFFNFunction.apply(hidden, gate_weight, up_weight, down_weight)


class _ProductionMatmulFFNFunction(torch.autograd.Function):
    """Run the TP=1 production matmul core with the same wrapper scope."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Tensor:
        input_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, input_shape[-1]).contiguous()
        gate = torch.matmul(hidden_2d, gate_weight.t())
        up = torch.matmul(hidden_2d, up_weight.t())
        activated = _C.swiglu_forward(gate, up)
        output = torch.matmul(activated, down_weight.t())
        ctx.save_for_backward(
            hidden_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            hidden,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()

        grad_down_weight = torch.matmul(grad_output.t(), activated)
        grad_activated = torch.matmul(grad_output, down_weight)
        grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)
        grad_gate_weight = torch.matmul(grad_gate.t(), hidden)
        grad_up_weight = torch.matmul(grad_up.t(), hidden)
        grad_hidden_gate = torch.matmul(grad_gate, gate_weight)
        grad_hidden_up = torch.matmul(grad_up, up_weight)
        grad_hidden = grad_hidden_gate.add_(grad_hidden_up)
        return (
            grad_hidden.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
        )


def _production_matmul_qwen3_ffn(
    hidden: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    return _ProductionMatmulFFNFunction.apply(hidden, gate_weight, up_weight, down_weight)


def _public_production_matmul_qwen3_ffn(
    hidden: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    return qwen3_ffn(
        hidden,
        gate_weight,
        up_weight,
        down_weight,
        deterministic=False,
    )


def _vllm_batch_invariant_qwen3_ffn(
    hidden: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Run the forward-only vLLM batch-invariant persistent-matmul core."""

    input_shape = hidden.shape
    hidden_2d = hidden.reshape(-1, input_shape[-1]).contiguous()
    gate = matmul_persistent(hidden_2d, gate_weight.t())
    up = matmul_persistent(hidden_2d, up_weight.t())
    activated = _C.swiglu_forward(gate, up)
    output = matmul_persistent(activated, down_weight.t())
    return output.reshape(*input_shape[:-1], output.size(-1))


def _parse_tokens(value: str) -> tuple[int, ...]:
    try:
        tokens = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be comma-separated integers") from exc
    if not tokens:
        raise argparse.ArgumentTypeError("at least one token count is required")
    if any(item <= 0 for item in tokens):
        raise argparse.ArgumentTypeError("token counts must be positive")
    if len(set(tokens)) != len(tokens):
        raise argparse.ArgumentTypeError("token counts must not contain duplicates")
    return tokens


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one timing sample is required")
    return {
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    scale: float = INPUT_SCALE,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(device=device, dtype=torch.bfloat16)


def _training_step(
    operator: Callable[..., Tensor],
    inputs: list[Tensor],
    grad_output: Tensor,
) -> Tensor:
    for value in inputs:
        value.grad = None
    output = operator(*inputs)
    output.backward(grad_output)
    return output


def _balanced_permutation_cycle(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Order every execution permutation to keep short prefixes well balanced."""

    remaining = list(permutations(names))
    ordered: list[tuple[str, ...]] = []
    position_counts = {(name, position): 0 for name in names for position in range(len(names))}
    predecessor_counts = {(left, right): 0 for left in names for right in names if left != right}

    while remaining:

        def balance_score(order: tuple[str, ...]) -> tuple[Any, ...]:
            candidate_positions = position_counts.copy()
            candidate_predecessors = predecessor_counts.copy()
            for position, name in enumerate(order):
                candidate_positions[name, position] += 1
            for left, right in zip(order, order[1:], strict=False):
                candidate_predecessors[left, right] += 1
            position_values = tuple(candidate_positions.values())
            predecessor_values = tuple(candidate_predecessors.values())
            return (
                max(position_values) - min(position_values),
                sum(value * value for value in position_values),
                max(predecessor_values, default=0) - min(predecessor_values, default=0),
                sum(value * value for value in predecessor_values),
                order,
            )

        selected = min(remaining, key=balance_score)
        remaining.remove(selected)
        ordered.append(selected)
        for position, name in enumerate(selected):
            position_counts[name, position] += 1
        for left, right in zip(selected, selected[1:], strict=False):
            predecessor_counts[left, right] += 1
    return tuple(ordered)


def _interleaved_gpu_samples(
    functions: dict[str, Callable[[], Any]],
    *,
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, list[float]]:
    if len(functions) < 2:
        raise ValueError("interleaved timing requires at least two implementations")
    orders = _balanced_permutation_cycle(tuple(functions))
    for index in range(warmup):
        for name in orders[index % len(orders)]:
            functions[name]()
    torch.cuda.synchronize(device)

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in functions
    }
    for index in range(samples):
        for name in orders[index % len(orders)]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            events[name].append((start, end))
    torch.cuda.synchronize(device)
    return {
        name: [float(start.elapsed_time(end)) for start, end in implementation_events]
        for name, implementation_events in events.items()
    }


def _raw_bf16_mismatch(left: Tensor, right: Tensor) -> int:
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise RuntimeError("bitwise comparison expects bfloat16 tensors")
    left_bits = left.detach().contiguous().view(torch.int16)
    right_bits = right.detach().contiguous().view(torch.int16)
    return int(torch.count_nonzero(left_bits != right_bits).item())


def _numerical_error(reference: Tensor, candidate: Tensor) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    absolute_error = (reference_float - candidate_float).abs()
    reference_norm = torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
    reference_max = reference_float.abs().max().clamp_min(1e-12)
    bitwise_mismatches = _raw_bf16_mismatch(reference, candidate)
    return {
        "bitwise_mismatch_count": bitwise_mismatches,
        "bitwise_mismatch_fraction": bitwise_mismatches / reference.numel(),
        "max_abs": float(absolute_error.max().item()),
        "mean_abs": float(absolute_error.mean().item()),
        "relative_l2": float((torch.linalg.vector_norm(absolute_error) / reference_norm).item()),
        "max_abs_over_reference_max": float((absolute_error.max() / reference_max).item()),
        "candidate_finite": int(torch.isfinite(candidate_float).all().item()),
    }


def _require_numerical_agreement(
    reference: Tensor,
    candidate: Tensor,
    *,
    name: str,
) -> dict[str, float | int]:
    metrics = _numerical_error(reference, candidate)
    if (
        not metrics["candidate_finite"]
        or metrics["relative_l2"] > CUBLAS_RELATIVE_L2_LIMIT
        or metrics["max_abs_over_reference_max"] > CUBLAS_NORMALIZED_MAX_LIMIT
    ):
        raise RuntimeError(
            f"{name} exceeds the production matmul sanity bounds "
            f"(relative_l2<={CUBLAS_RELATIVE_L2_LIMIT}, "
            f"normalized_max<={CUBLAS_NORMALIZED_MAX_LIMIT}): {metrics}"
        )
    return metrics


def _tensor_layout(tensor: Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def _training_result(
    operator: Callable[..., Tensor],
    values: tuple[Tensor, Tensor, Tensor, Tensor],
    grad_output: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    inputs = [value.detach().clone().requires_grad_(True) for value in values]
    output = _training_step(operator, inputs, grad_output)
    gradients = [value.grad.detach().clone() for value in inputs]
    return output.detach().clone(), gradients


def _verify_bitwise(
    values: tuple[Tensor, Tensor, Tensor, Tensor],
    grad_output: Tensor,
) -> dict[str, Any]:
    vllm_inference = None
    with torch.no_grad():
        baseline_inference = _legacy_qwen3_ffn(*values)
        optimized_inference = _optimized_qwen3_ffn(*values)
        public_inference = qwen3_ffn(*values)
        matmul_inference = _production_matmul_qwen3_ffn(*values)
        public_matmul_inference = _public_production_matmul_qwen3_ffn(*values)
        if values[0].is_cuda:
            vllm_inference = _vllm_batch_invariant_qwen3_ffn(*values)
    baseline_training, baseline_gradients = _training_result(
        _legacy_qwen3_ffn,
        values,
        grad_output,
    )
    optimized_training, optimized_gradients = _training_result(
        _optimized_qwen3_ffn,
        values,
        grad_output,
    )
    public_training, public_gradients = _training_result(
        qwen3_ffn,
        values,
        grad_output,
    )
    matmul_training, matmul_gradients = _training_result(
        _production_matmul_qwen3_ffn,
        values,
        grad_output,
    )
    public_matmul_training, public_matmul_gradients = _training_result(
        _public_production_matmul_qwen3_ffn,
        values,
        grad_output,
    )

    names = ("dHidden", "dGateWeight", "dUpWeight", "dDownWeight")
    mismatches = {
        "inference_output": _raw_bf16_mismatch(
            baseline_inference,
            optimized_inference,
        ),
        "training_output": _raw_bf16_mismatch(
            baseline_training,
            optimized_training,
        ),
        "baseline_train_infer": _raw_bf16_mismatch(
            baseline_inference,
            baseline_training,
        ),
        "optimized_train_infer": _raw_bf16_mismatch(
            optimized_inference,
            optimized_training,
        ),
    }
    mismatches.update(
        {
            name: _raw_bf16_mismatch(baseline, optimized)
            for name, baseline, optimized in zip(
                names,
                baseline_gradients,
                optimized_gradients,
                strict=True,
            )
        }
    )
    public_parity_mismatches = {
        "inference_output": _raw_bf16_mismatch(
            optimized_inference,
            public_inference,
        ),
        "training_output": _raw_bf16_mismatch(
            optimized_training,
            public_training,
        ),
    }
    public_parity_mismatches.update(
        {
            name: _raw_bf16_mismatch(optimized, public)
            for name, optimized, public in zip(
                names,
                optimized_gradients,
                public_gradients,
                strict=True,
            )
        }
    )
    matmul_numerical_error = {
        "inference_output": _require_numerical_agreement(
            optimized_inference,
            matmul_inference,
            name="optimized versus production matmul inference output",
        ),
        "training_output": _require_numerical_agreement(
            optimized_training,
            matmul_training,
            name="optimized versus production matmul training output",
        ),
    }
    matmul_numerical_error.update(
        {
            name: _require_numerical_agreement(
                optimized,
                matmul,
                name=f"optimized versus production matmul {name}",
            )
            for name, optimized, matmul in zip(
                names,
                optimized_gradients,
                matmul_gradients,
                strict=True,
            )
        }
    )
    matmul_public_parity_error = {
        "inference_output": _require_numerical_agreement(
            matmul_inference,
            public_matmul_inference,
            name="timed matmul core versus public matmul inference output",
        ),
        "training_output": _require_numerical_agreement(
            matmul_training,
            public_matmul_training,
            name="timed matmul core versus public matmul training output",
        ),
    }
    matmul_public_parity_error.update(
        {
            name: _require_numerical_agreement(
                matmul,
                public_matmul,
                name=f"timed matmul core versus public matmul {name}",
            )
            for name, matmul, public_matmul in zip(
                names,
                matmul_gradients,
                public_matmul_gradients,
                strict=True,
            )
        }
    )
    vllm_numerical_error = None
    if vllm_inference is not None:
        vllm_numerical_error = _require_numerical_agreement(
            optimized_inference,
            vllm_inference,
            name="optimized versus vLLM batch-invariant inference output",
        )
    optimized_layouts = {
        name: _tensor_layout(gradient)
        for name, gradient in zip(names, optimized_gradients, strict=True)
    }
    public_layouts = {
        name: _tensor_layout(gradient)
        for name, gradient in zip(names, public_gradients, strict=True)
    }
    matmul_layouts = {
        name: _tensor_layout(gradient)
        for name, gradient in zip(names, matmul_gradients, strict=True)
    }
    weight_layouts = [optimized_layouts[name] for name in names[1:]]
    public_weight_layouts = [public_layouts[name] for name in names[1:]]
    matmul_weight_layouts = [matmul_layouts[name] for name in names[1:]]
    if any(mismatches.values()):
        raise RuntimeError(f"legacy and optimized FFN are not bitwise identical: {mismatches}")
    if any(public_parity_mismatches.values()):
        raise RuntimeError(
            "optimized timing core and public qwen3_ffn are not bitwise identical: "
            f"{public_parity_mismatches}"
        )
    if not all(layout["contiguous"] for layout in weight_layouts):
        raise RuntimeError(f"optimized weight gradients must be contiguous: {weight_layouts}")
    if not all(layout["contiguous"] for layout in public_weight_layouts):
        raise RuntimeError(
            f"public qwen3_ffn weight gradients must be contiguous: {public_weight_layouts}"
        )
    if not all(layout["contiguous"] for layout in matmul_weight_layouts):
        raise RuntimeError(
            f"production matmul weight gradients must be contiguous: {matmul_weight_layouts}"
        )
    return {
        "mismatch_count": mismatches,
        "public_parity_mismatch_count": public_parity_mismatches,
        "production_matmul_numerical_error": matmul_numerical_error,
        "production_matmul_public_parity_error": matmul_public_parity_error,
        "vllm_batch_invariant_numerical_error": vllm_numerical_error,
        "optimized_gradient_layouts": optimized_layouts,
        "public_gradient_layouts": public_layouts,
        "production_matmul_gradient_layouts": matmul_layouts,
    }


def _verify_vllm_batch_invariance(
    *,
    token_counts: tuple[int, ...],
    hidden_size: int,
    weights: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Require the identical first row to produce identical BF16 bits for every M."""

    tested_tokens = tuple(sorted({1, *token_counts}))
    shared_hidden = _randn(
        (max(tested_tokens), hidden_size),
        seed=seed + 30_000,
        device=device,
    )
    outputs: dict[int, Tensor] = {}
    with torch.no_grad():
        for tokens in tested_tokens:
            outputs[tokens] = _vllm_batch_invariant_qwen3_ffn(
                shared_hidden[:tokens],
                *weights,
            )
    reference = outputs[1][0]
    mismatches = {
        str(tokens): _raw_bf16_mismatch(reference, outputs[tokens][0]) for tokens in tested_tokens
    }
    if any(mismatches.values()):
        raise RuntimeError(
            "vLLM persistent-matmul FFN is not batch invariant for the shared first row: "
            f"{mismatches}"
        )
    return {
        "status": "pass",
        "contract": "raw BF16 first-row equality against M=1 with identical input rows",
        "reference_tokens": 1,
        "tested_tokens": list(tested_tokens),
        "first_row_bitwise_mismatch_count": mismatches,
    }


def _vllm_projection_configs(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, dict[str, Any]]:
    return {
        "gate": matmul_config_metadata(
            tokens,
            intermediate_size,
            hidden_size,
            torch.bfloat16,
        ),
        "up": matmul_config_metadata(
            tokens,
            intermediate_size,
            hidden_size,
            torch.bfloat16,
        ),
        "down": matmul_config_metadata(
            tokens,
            hidden_size,
            intermediate_size,
            torch.bfloat16,
        ),
    }


def _case_inputs(
    *,
    tokens: int,
    hidden_size: int,
    weights: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
    seed: int,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor]:
    hidden = _randn(
        (tokens, hidden_size),
        seed=seed + 10_000 + tokens,
        device=device,
    )
    grad_output = _randn(
        (tokens, hidden_size),
        seed=seed + 20_000 + tokens,
        device=device,
    )
    return (hidden, *weights), grad_output


def _benchmark_case(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    values: tuple[Tensor, Tensor, Tensor, Tensor],
    grad_output: Tensor,
    device: torch.device,
    warmup: int,
    samples: int,
    training_samples: int,
) -> list[dict[str, Any]]:

    def baseline_forward() -> Tensor:
        with torch.no_grad():
            return _legacy_qwen3_ffn(*values)

    def optimized_forward() -> Tensor:
        with torch.no_grad():
            return _optimized_qwen3_ffn(*values)

    def cublas_forward() -> Tensor:
        with torch.no_grad():
            return _production_matmul_qwen3_ffn(*values)

    def vllm_forward() -> Tensor:
        with torch.no_grad():
            return _vllm_batch_invariant_qwen3_ffn(*values)

    # Compile both Triton shape specializations before any timed launch.
    vllm_forward()
    torch.cuda.synchronize(device)
    forward_samples = _interleaved_gpu_samples(
        {
            "baseline": baseline_forward,
            "optimized": optimized_forward,
            "cublas": cublas_forward,
            "vllm": vllm_forward,
        },
        device=device,
        warmup=warmup,
        samples=samples,
    )

    baseline_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    optimized_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    cublas_inputs = [value.detach().clone().requires_grad_(True) for value in values]

    def baseline_training() -> Tensor:
        return _training_step(
            _legacy_qwen3_ffn,
            baseline_inputs,
            grad_output,
        )

    def optimized_training() -> Tensor:
        return _training_step(
            _optimized_qwen3_ffn,
            optimized_inputs,
            grad_output,
        )

    def cublas_training() -> Tensor:
        return _training_step(
            _production_matmul_qwen3_ffn,
            cublas_inputs,
            grad_output,
        )

    training_samples_by_implementation = _interleaved_gpu_samples(
        {
            "baseline": baseline_training,
            "optimized": optimized_training,
            "cublas": cublas_training,
        },
        device=device,
        warmup=warmup,
        samples=training_samples,
    )

    rows = []
    for direction, samples_by_implementation in (
        ("forward", forward_samples),
        ("forward_backward", training_samples_by_implementation),
    ):
        baseline_summary = _summary_ms(samples_by_implementation["baseline"])
        optimized_summary = _summary_ms(samples_by_implementation["optimized"])
        cublas_summary = _summary_ms(samples_by_implementation["cublas"])
        baseline_median = baseline_summary["median_ms"]
        optimized_median = optimized_summary["median_ms"]
        cublas_median = cublas_summary["median_ms"]
        row = {
            "tokens": tokens,
            "hidden": hidden_size,
            "intermediate": intermediate_size,
            "dtype": "bfloat16",
            "direction": direction,
            "baseline": baseline_summary,
            "optimized": optimized_summary,
            "cublas": cublas_summary,
            "speedup": baseline_median / optimized_median,
            "optimized_speedup_vs_legacy": baseline_median / optimized_median,
            "latency_reduction_percent": 100.0 * (1.0 - optimized_median / baseline_median),
            "cublas_speedup_vs_optimized": optimized_median / cublas_median,
            "optimized_overhead_vs_cublas_percent": 100.0
            * (optimized_median / cublas_median - 1.0),
        }
        if direction == "forward":
            vllm_summary = _summary_ms(samples_by_implementation["vllm"])
            vllm_median = vllm_summary["median_ms"]
            row.update(
                {
                    "vllm": vllm_summary,
                    "vllm_matmul_configs": _vllm_projection_configs(
                        tokens=tokens,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    ),
                    "vllm_speedup_vs_optimized": optimized_median / vllm_median,
                    "optimized_overhead_vs_vllm_percent": 100.0
                    * (optimized_median / vllm_median - 1.0),
                    "cublas_speedup_vs_vllm": vllm_median / cublas_median,
                    "vllm_overhead_vs_cublas_percent": 100.0 * (vllm_median / cublas_median - 1.0),
                }
            )
        rows.append(row)
    torch.cuda.empty_cache()
    return rows


def _profile_counts(function: Callable[[], Any], device: torch.device) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        function()
    torch.cuda.synchronize(device)
    counts = Counter(event.key for event in profiler.key_averages() for _ in range(event.count))
    cuda_gemm_kernels = sorted(
        name
        for name in counts
        if not name.startswith("aten::")
        and any(token in name.lower() for token in ("gemm", "nvjet", "cublas", "cutlass"))
    )
    return {
        "direct_copy_kernel": sum(
            count for name, count in counts.items() if "direct_copy_kernel" in name
        ),
        "det_gemm_sm90_kernel": sum(
            count for name, count in counts.items() if "det_gemm_sm90_kernel" in name
        ),
        "aten_copy": counts.get("aten::copy_", 0),
        "aten_mm": counts.get("aten::mm", 0),
        "cuda_gemm_kernels": cuda_gemm_kernels,
    }


def _profile_implementation(
    operator: Callable[..., Tensor],
    values: tuple[Tensor, Tensor, Tensor, Tensor],
    grad_output: Tensor,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    def forward() -> Tensor:
        with torch.no_grad():
            return operator(*values)

    forward()
    torch.cuda.synchronize(device)
    forward_counts = _profile_counts(forward, device)

    inputs = [value.detach().clone().requires_grad_(True) for value in values]

    def training() -> Tensor:
        return _training_step(operator, inputs, grad_output)

    training()
    torch.cuda.synchronize(device)
    training_counts = _profile_counts(training, device)
    return {"forward": forward_counts, "forward_backward": training_counts}


def _profile_layout_paths(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    weights: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    hidden = _randn(
        (tokens, hidden_size),
        seed=seed + 10_000 + tokens,
        device=device,
    )
    grad_output = _randn(
        (tokens, hidden_size),
        seed=seed + 20_000 + tokens,
        device=device,
    )
    values = (hidden, *weights)
    return {
        "tokens": tokens,
        "baseline": _profile_implementation(
            _legacy_qwen3_ffn,
            values,
            grad_output,
            device,
        ),
        "optimized": _profile_implementation(
            _optimized_qwen3_ffn,
            values,
            grad_output,
            device,
        ),
        "production_matmul": _profile_implementation(
            _production_matmul_qwen3_ffn,
            values,
            grad_output,
            device,
        ),
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _nvidia_driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return "unavailable"
    return completed.stdout.splitlines()[0].strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(device: torch.device, sm90_probe_launches: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    benchmark_path = Path(__file__).resolve()
    vllm_kernel_path = benchmark_path.with_name("vllm_batch_invariant_matmul.py")
    vllm_config_path = benchmark_path.with_name("vllm_batch_invariant_configs.py")
    package_path = Path(rl_engine.__file__).resolve()
    extension_path = Path(_C.__file__).resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_gib": properties.total_memory / (1024**3),
        "device": str(device),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nvidia_driver": _nvidia_driver_version(),
        "preferred_blas_library": torch.backends.cuda.preferred_blas_library().name,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "python": platform.python_version(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "vllm_kernel_sha256": _sha256_file(vllm_kernel_path),
        "vllm_config_sha256": _sha256_file(vllm_config_path),
        "extension_sha256": _sha256_file(extension_path),
        "rl_engine_path": str(package_path),
        "extension_path": str(extension_path),
        "sm90_probe_kernel_launches": sm90_probe_launches,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _validate_environment(device_index: int) -> tuple[torch.device, int]:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("the RL-Kernel CUDA extension is unavailable")
    package_path = Path(rl_engine.__file__).resolve()
    extension_path = Path(_C.__file__).resolve()
    if REPO_ROOT not in package_path.parents or REPO_ROOT not in extension_path.parents:
        raise RuntimeError(
            "benchmark must load rl_engine and its extension from this checkout; "
            f"got {package_path} and {extension_path}"
        )
    required = (
        "det_gemm_fwd",
        "det_gemm_db",
        "det_gemm_fwd_rhs_transposed",
        "det_gemm_db_transposed",
        "swiglu_forward",
        "swiglu_backward",
    )
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(f"CUDA extension is missing required symbols: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            f"device index {device_index} is outside {torch.cuda.device_count()} visible GPUs"
        )
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < (9, 0):
        raise RuntimeError(f"this layout benchmark requires SM90 or newer, got SM{capability}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.preferred_blas_library("cublas")
    probe_a = torch.zeros((128, 128), device=device, dtype=torch.bfloat16)
    probe_b = torch.zeros((128, 128), device=device, dtype=torch.bfloat16)
    _C.det_gemm_fwd(probe_a, probe_b)
    torch.cuda.synchronize(device)
    probe_counts = _profile_counts(lambda: _C.det_gemm_fwd(probe_a, probe_b), device)
    sm90_probe_launches = probe_counts["det_gemm_sm90_kernel"]
    if sm90_probe_launches != 1:
        raise RuntimeError(
            "det_gemm SM90 support is not compiled; rebuild with KERNEL_ALIGN_DET_GEMM_SM90=1"
        )
    return device, sm90_probe_launches


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    anchor: str = "middle",
    weight: int = 400,
    color: str = TEXT_COLOR,
    transform: str | None = None,
) -> str:
    transform_attribute = f' transform="{escape(transform)}"' if transform else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}"'
        f"{transform_attribute}>{escape(text)}</text>"
    )


def _log_ticks(low: float, high: float) -> list[float]:
    first_exponent = math.floor(math.log10(low))
    last_exponent = math.ceil(math.log10(high))
    ticks = []
    for exponent in range(first_exponent, last_exponent + 1):
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * (10**exponent)
            if low <= value <= high:
                ticks.append(value)
    return ticks


def _write_production_context_figure(payload: dict[str, Any], output_path: Path) -> None:
    methodology = payload["methodology"]
    rows = sorted(
        (row for row in payload["results"] if row["direction"] == "forward"),
        key=lambda row: row["tokens"],
    )
    if not rows:
        raise ValueError("production-context figure requires forward result rows")
    series = [
        ("optimized", "RL-Kernel optimized deterministic GEMM", OPTIMIZED_COLOR),
        ("vllm", f"{VLLM_LABEL} (PR #53247)", VLLM_COLOR),
        ("cublas", CUBLAS_LABEL, CUBLAS_COLOR),
    ]
    # Keep the writer usable with pre-vLLM payloads used by downstream tooling.
    series = [entry for entry in series if all(entry[0] in row for row in rows)]
    all_values = [row[key]["median_ms"] for row in rows for key, _, _ in series]
    low = min(all_values) * 0.72
    high = max(all_values) * 1.42
    if math.isclose(low, high):
        low *= 0.5
        high *= 2.0

    width, height = 1600, 820
    left, right = 135, 1535
    plot_top, plot_bottom = 165, 655
    plot_width = right - left
    plot_height = plot_bottom - plot_top

    def y_position(value: float) -> float:
        fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        return plot_bottom - fraction * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        'font-family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, '
        "'Segoe UI', sans-serif\">",
        "<title>Qwen3 FFN forward latency comparison</title>",
        "<desc>Equivalent BF16 FFN forward calls in a shared local harness compare "
        "optimized RL-Kernel, vLLM batch-invariant persistent matmul, and PyTorch "
        "CUDA BLAS.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _svg_text(width / 2, 42, "Qwen3 FFN forward latency", size=30, weight=500),
        _svg_text(
            width / 2,
            72,
            f"H={methodology['hidden']}, I={methodology['intermediate']}, "
            "BF16 · symmetric TP=1 cores · CUDA-event medians",
            size=17,
            color=MUTED_COLOR,
        ),
    ]
    legend_width = 440
    legend_left = (width - legend_width * len(series)) / 2
    for index, (_, label, color) in enumerate(series):
        x = legend_left + index * legend_width
        parts.append(f'<rect x="{x:.1f}" y="96" width="18" height="18" fill="{color}"/>')
        parts.append(_svg_text(x + 28, 111, label, anchor="start", size=15))
    parts.append(
        f'<rect x="{left}" y="{plot_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="{GRID_COLOR}"/>'
    )
    for tick in _log_ticks(low, high):
        y = y_position(tick)
        parts.append(
            f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(_svg_text(left - 12, y + 5, f"{tick:g}", anchor="end", size=14))
    group_width = plot_width / len(rows)
    bar_width = min(58.0, group_width * 0.2)
    offsets = [bar_width * (index - (len(series) - 1) / 2) * 1.18 for index in range(len(series))]
    for row_index, row in enumerate(rows):
        center = left + group_width * (row_index + 0.5)
        for offset, (key, _, color) in zip(offsets, series, strict=True):
            value = row[key]["median_ms"]
            x = center + offset - bar_width / 2
            y = y_position(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{plot_bottom - y:.1f}" fill="{color}"/>'
            )
            parts.append(_svg_text(x + bar_width / 2, y - 8, f"{value:.3f}", size=12))
        parts.append(_svg_text(center, plot_bottom + 30, f"M={row['tokens']}", size=15))
        if "vllm_speedup_vs_optimized" in row:
            parts.append(
                _svg_text(
                    center,
                    plot_bottom + 55,
                    f"RL / vLLM: {row['vllm_speedup_vs_optimized']:.2f}x",
                    size=13,
                    weight=500,
                )
            )
    parts.append(
        _svg_text(
            left - 78,
            (plot_top + plot_bottom) / 2,
            "Median forward latency (ms, log scale)",
            size=15,
            transform=f"rotate(-90 {left - 78:.1f} {(plot_top + plot_bottom) / 2:.1f})",
        )
    )
    selections = {
        metadata.get("selection")
        for row in rows
        for metadata in row.get("vllm_matmul_configs", {}).values()
    }
    config_note = (
        "Qwen3-8B projection shapes use the upstream default BF16 fallback configuration."
        if selections == {"default"}
        else "vLLM configuration selection is recorded per projection in results.json."
    )
    parts.append(
        _svg_text(
            width / 2,
            742,
            f"vLLM is forward-only. {config_note}",
            size=15,
            color=MUTED_COLOR,
        )
    )
    parts.append(
        _svg_text(
            width / 2,
            772,
            "Operator benchmark only; the vLLM series uses vendored persistent matmul "
            "code, not end-to-end vLLM serving.",
            size=15,
            color=MUTED_COLOR,
        )
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_forward_backward_context_figure(payload: dict[str, Any], output_path: Path) -> None:
    methodology = payload["methodology"]
    rows = sorted(
        (row for row in payload["results"] if row["direction"] == "forward_backward"),
        key=lambda row: row["tokens"],
    )
    if not rows:
        raise ValueError("forward+backward figure requires forward_backward result rows")
    series = [
        ("optimized", "RL-Kernel optimized deterministic GEMM", OPTIMIZED_COLOR),
        ("cublas", CUBLAS_LABEL, CUBLAS_COLOR),
    ]
    all_values = [row[key]["median_ms"] for row in rows for key, _, _ in series]
    low = min(all_values) * 0.72
    high = max(all_values) * 1.42
    if math.isclose(low, high):
        low *= 0.5
        high *= 2.0

    width, height = 1600, 820
    left, right = 135, 1535
    plot_top, plot_bottom = 165, 655
    plot_width = right - left
    plot_height = plot_bottom - plot_top

    def y_position(value: float) -> float:
        fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        return plot_bottom - fraction * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        'font-family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, '
        "'Segoe UI', sans-serif\">",
        "<title>Qwen3 FFN forward plus backward latency comparison</title>",
        "<desc>Equivalent BF16 FFN forward-plus-backward calls in a shared local harness "
        "compare optimized RL-Kernel and PyTorch CUDA BLAS. The vLLM path is omitted "
        "because it is forward-only.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _svg_text(
            width / 2,
            42,
            "Qwen3 FFN forward + backward latency",
            size=30,
            weight=500,
        ),
        _svg_text(
            width / 2,
            72,
            f"H={methodology['hidden']}, I={methodology['intermediate']}, "
            "BF16 · symmetric TP=1 cores · CUDA-event medians",
            size=17,
            color=MUTED_COLOR,
        ),
    ]
    legend_width = 520
    legend_left = (width - legend_width * len(series)) / 2
    for index, (_, label, color) in enumerate(series):
        x = legend_left + index * legend_width
        parts.append(f'<rect x="{x:.1f}" y="96" width="18" height="18" fill="{color}"/>')
        parts.append(_svg_text(x + 28, 111, label, anchor="start", size=15))
    parts.append(
        f'<rect x="{left}" y="{plot_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="{GRID_COLOR}"/>'
    )
    for tick in _log_ticks(low, high):
        y = y_position(tick)
        parts.append(
            f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(_svg_text(left - 12, y + 5, f"{tick:g}", anchor="end", size=14))
    group_width = plot_width / len(rows)
    bar_width = min(68.0, group_width * 0.24)
    offsets = (-bar_width * 0.62, bar_width * 0.62)
    for row_index, row in enumerate(rows):
        center = left + group_width * (row_index + 0.5)
        for offset, (key, _, color) in zip(offsets, series, strict=True):
            value = row[key]["median_ms"]
            x = center + offset - bar_width / 2
            y = y_position(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{plot_bottom - y:.1f}" fill="{color}"/>'
            )
            parts.append(_svg_text(x + bar_width / 2, y - 8, f"{value:.3f}", size=12))
        parts.append(_svg_text(center, plot_bottom + 30, f"M={row['tokens']}", size=15))
        parts.append(
            _svg_text(
                center,
                plot_bottom + 55,
                f"RL / cuBLAS: {row['cublas_speedup_vs_optimized']:.2f}x",
                size=13,
                weight=500,
            )
        )
    parts.append(
        _svg_text(
            left - 78,
            (plot_top + plot_bottom) / 2,
            "Median forward + backward latency (ms, log scale)",
            size=15,
            transform=f"rotate(-90 {left - 78:.1f} {(plot_top + plot_bottom) / 2:.1f})",
        )
    )
    parts.append(
        _svg_text(
            width / 2,
            742,
            "vLLM PR #53247 is forward-only; this benchmark has no vLLM backward path.",
            size=15,
            color=MUTED_COLOR,
        )
    )
    parts.append(
        _svg_text(
            width / 2,
            772,
            "Both series include complete three-GEMM FFN forward and backward calls.",
            size=15,
            color=MUTED_COLOR,
        )
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_csv(payload: dict[str, Any], output_path: Path) -> None:
    fieldnames = (
        "tokens",
        "hidden",
        "intermediate",
        "dtype",
        "direction",
        "baseline_median_ms",
        "baseline_p95_ms",
        "baseline_min_ms",
        "baseline_max_ms",
        "optimized_median_ms",
        "optimized_p95_ms",
        "optimized_min_ms",
        "optimized_max_ms",
        "cublas_median_ms",
        "cublas_p95_ms",
        "cublas_min_ms",
        "cublas_max_ms",
        "vllm_median_ms",
        "vllm_p95_ms",
        "vllm_min_ms",
        "vllm_max_ms",
        "optimized_speedup_vs_legacy",
        "latency_reduction_percent",
        "cublas_speedup_vs_optimized",
        "optimized_overhead_vs_cublas_percent",
        "vllm_speedup_vs_optimized",
        "optimized_overhead_vs_vllm_percent",
        "cublas_speedup_vs_vllm",
        "vllm_overhead_vs_cublas_percent",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["results"]:
            vllm = row.get("vllm", {})
            writer.writerow(
                {
                    "tokens": row["tokens"],
                    "hidden": row["hidden"],
                    "intermediate": row["intermediate"],
                    "dtype": row["dtype"],
                    "direction": row["direction"],
                    "baseline_median_ms": row["baseline"]["median_ms"],
                    "baseline_p95_ms": row["baseline"]["p95_ms"],
                    "baseline_min_ms": row["baseline"]["min_ms"],
                    "baseline_max_ms": row["baseline"]["max_ms"],
                    "optimized_median_ms": row["optimized"]["median_ms"],
                    "optimized_p95_ms": row["optimized"]["p95_ms"],
                    "optimized_min_ms": row["optimized"]["min_ms"],
                    "optimized_max_ms": row["optimized"]["max_ms"],
                    "cublas_median_ms": row["cublas"]["median_ms"],
                    "cublas_p95_ms": row["cublas"]["p95_ms"],
                    "cublas_min_ms": row["cublas"]["min_ms"],
                    "cublas_max_ms": row["cublas"]["max_ms"],
                    "vllm_median_ms": vllm.get("median_ms", ""),
                    "vllm_p95_ms": vllm.get("p95_ms", ""),
                    "vllm_min_ms": vllm.get("min_ms", ""),
                    "vllm_max_ms": vllm.get("max_ms", ""),
                    "optimized_speedup_vs_legacy": row["optimized_speedup_vs_legacy"],
                    "latency_reduction_percent": row["latency_reduction_percent"],
                    "cublas_speedup_vs_optimized": row["cublas_speedup_vs_optimized"],
                    "optimized_overhead_vs_cublas_percent": row[
                        "optimized_overhead_vs_cublas_percent"
                    ],
                    "vllm_speedup_vs_optimized": row.get("vllm_speedup_vs_optimized", ""),
                    "optimized_overhead_vs_vllm_percent": row.get(
                        "optimized_overhead_vs_vllm_percent", ""
                    ),
                    "cublas_speedup_vs_vllm": row.get("cublas_speedup_vs_vllm", ""),
                    "vllm_overhead_vs_cublas_percent": row.get(
                        "vllm_overhead_vs_cublas_percent", ""
                    ),
                }
            )


def _write_report(payload: dict[str, Any], output_path: Path) -> None:
    environment = payload["environment"]
    methodology = payload["methodology"]
    rows = payload["results"]
    correctness = payload["correctness"]
    has_vllm = any("vllm" in row for row in rows)
    lines = [
        "# Qwen3 FFN deterministic GEMM layout benchmark",
        "",
        "This is an operator-only single-GPU benchmark. It does not load or benchmark a model "
        "checkpoint, tokenizer, dataset, or serving engine.",
        "",
        "## Comparison contract",
        "",
        f"- **Baseline:** deterministic single-GPU layout contract from `{LEGACY_COMMIT}`. "
        "The benchmark replays the removed weight and weight-gradient transpose/copy "
        "materializations with the preserved legacy CUDA APIs.",
        "- **Candidate:** the current TP=1 `qwen3_ffn` layout core, which consumes canonical "
        "`[out, in]` weights and directly returns canonical contiguous weight gradients.",
        "- **Production reference:** symmetric TP=1 `torch.matmul` core matching "
        "`qwen3_ffn(deterministic=False)`. This environment prefers the cuBLAS backend; "
        "the exact CUDA BLAS algorithm is not a `torch.matmul` API guarantee.",
        f"- **Batch-invariant reference (forward only):** vLLM's persistent Triton "
        f"matmul, including the configuration selection from "
        f"[PR #53247]({VLLM_BATCH_INVARIANT_PR_URL}), pinned to merge commit "
        f"`{VLLM_BATCH_INVARIANT_SOURCE_SHA}`. The three-GEMM core uses the same weight "
        "transpose views and RL-Kernel SwiGLU operation as the other stripped TP=1 paths.",
        "- All timed paths have the same stripped TP=1 wrapper scope and use identical "
        "seeded BF16 tensors. The optimized deterministic core must match the public "
        "deterministic `qwen3_ffn` bitwise; cuBLAS and vLLM outputs must agree with the "
        "optimized output within the recorded numerical sanity bounds.",
        "- Every requested M passes correctness before any timing. Bitwise acceptance only "
        "applies to the two RL-Kernel deterministic paths. The vLLM path separately must "
        "produce a bitwise-identical first output row across batch sizes for an identical "
        "first input row.",
        "- This is an in-process layout-path comparison, not a separately loaded old "
        "binary. Keeping the GEMM arithmetic and build fixed isolates the removed layout "
        "materializations from cross-build and cross-run noise.",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in environment.items():
        if key in {"rl_engine_path", "extension_path"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(
        (
            "",
            "## Methodology",
            "",
            f"- Shape: H={methodology['hidden']}, I={methodology['intermediate']}, "
            f"M={methodology['tokens']}; dtype=BF16.",
            "- CUDA events measure complete FFN forward and complete forward+backward calls.",
            "- Forward samples follow a prefix-balanced 24-permutation cycle over legacy, "
            "optimized, CUDA BLAS, and vLLM. Forward+backward samples follow the balanced "
            "six-permutation cycle for the three training-capable paths.",
            f"- {methodology['warmup']} warmups per path, {methodology['samples']} forward "
            f"samples, and {methodology['training_samples']} forward+backward samples.",
            "- Tables and figures report median latency; JSON also contains p95, min, max, "
            "and every raw sample.",
            "",
            "Reproduction command:",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
            "## Performance",
            "",
            "| M | Direction | Replayed legacy (ms) | Optimized deterministic (ms) | "
            "CUDA BLAS (ms) | vLLM batch-invariant (ms) | Layout speedup | "
            "Optimized / cuBLAS | Optimized / vLLM |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in rows:
        direction = "Forward" if row["direction"] == "forward" else "Forward + backward"
        vllm_latency = f"{row['vllm']['median_ms']:.4f}" if "vllm" in row else "—"
        vllm_gap = f"{row['vllm_speedup_vs_optimized']:.2f}x" if "vllm" in row else "—"
        lines.append(
            f"| {row['tokens']} | {direction} | {row['baseline']['median_ms']:.4f} | "
            f"{row['optimized']['median_ms']:.4f} | {row['cublas']['median_ms']:.4f} | "
            f"{vllm_latency} | "
            f"{row['optimized_speedup_vs_legacy']:.2f}x | "
            f"{row['cublas_speedup_vs_optimized']:.2f}x | {vllm_gap} |"
        )
    lines.extend(
        (
            "",
            "![Forward production performance context](qwen_ffn_cublas_comparison.svg)",
            "",
            "![Forward + backward performance context]"
            "(qwen_ffn_forward_backward_comparison.svg)",
            "",
            "`Layout speedup = replayed / optimized`; `determinism overhead = optimized / "
            "CUDA BLAS`; `optimized / vLLM = optimized latency / vLLM latency`. vLLM is "
            "forward-only; both numerical references are outside the RL-Kernel bitwise "
            "acceptance contract.",
            "",
            "## Deterministic bitwise consistency",
            "",
            "All entries are raw BF16 bit mismatch counts. Timing is skipped if any value "
            "is non-zero.",
            "",
            "| M | Output | Training output | dHidden | dGateW | dUpW | dDownW |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for token_count in methodology["tokens"]:
        mismatch = correctness[str(token_count)]["mismatch_count"]
        lines.append(
            f"| {token_count} | {mismatch['inference_output']} | "
            f"{mismatch['training_output']} | {mismatch['dHidden']} | "
            f"{mismatch['dGateWeight']} | {mismatch['dUpWeight']} | "
            f"{mismatch['dDownWeight']} |"
        )
    lines.extend(
        (
            "",
            "For every M, legacy train/inference parity, optimized train/inference parity, "
            "and timed optimized-core/public-`qwen3_ffn` parity also have zero raw BF16 "
            "mismatches.",
            "",
            "## Production matmul numerical agreement",
            "",
            "CUDA BLAS uses a different reduction order and is not expected to match "
            "bitwise. The seeded benchmark requires relative L2 error <= "
            f"{CUBLAS_RELATIVE_L2_LIMIT:.1%} and normalized maximum error <= "
            f"{CUBLAS_NORMALIZED_MAX_LIMIT:.1%} for every output and gradient before "
            "timing.",
            "",
            "| M | Output rel. L2 | dHidden rel. L2 | Max dWeight rel. L2 | "
            "Max normalized error | Max bitwise mismatch |",
            "|---:|---:|---:|---:|---:|---:|",
        )
    )
    for token_count in methodology["tokens"]:
        errors = correctness[str(token_count)]["production_matmul_numerical_error"]
        weight_errors = [
            errors[name]["relative_l2"] for name in ("dGateWeight", "dUpWeight", "dDownWeight")
        ]
        max_normalized = max(metrics["max_abs_over_reference_max"] for metrics in errors.values())
        max_mismatch_fraction = max(
            metrics["bitwise_mismatch_fraction"] for metrics in errors.values()
        )
        lines.append(
            f"| {token_count} | {errors['training_output']['relative_l2']:.3%} | "
            f"{errors['dHidden']['relative_l2']:.3%} | {max(weight_errors):.3%} | "
            f"{max_normalized:.3%} | {max_mismatch_fraction:.3%} |"
        )
    if has_vllm:
        lines.extend(
            (
                "",
                "## vLLM batch-invariant forward checks",
                "",
                "The vendored vLLM path uses fixed-K persistent matmuls with FP32 "
                "accumulation. It is checked numerically against the optimized RL-Kernel "
                "output for each M; it is not expected to match RL-Kernel bitwise because "
                "the two kernels use different accumulation orders.",
                "",
                "| M | Relative L2 | Normalized max error | Bitwise mismatch fraction |",
                "|---:|---:|---:|---:|",
            )
        )
        for token_count in methodology["tokens"]:
            error = correctness[str(token_count)]["vllm_batch_invariant_numerical_error"]
            lines.append(
                f"| {token_count} | {error['relative_l2']:.3%} | "
                f"{error['max_abs_over_reference_max']:.3%} | "
                f"{error['bitwise_mismatch_fraction']:.3%} |"
            )
        batch_invariance = payload.get("vllm_batch_invariance")
        if batch_invariance:
            mismatch_text = ", ".join(
                f"M={tokens}: {count}"
                for tokens, count in batch_invariance["first_row_bitwise_mismatch_count"].items()
            )
            lines.extend(
                (
                    "",
                    "The identical-first-row batch-invariance gate passed with raw BF16 "
                    f"mismatch counts `{mismatch_text}` against M=1.",
                )
            )
        lines.extend(
            (
                "",
                "### vLLM matmul configuration selection",
                "",
                "Each projection records the exact configuration selected by the vendored "
                "upstream table. `default` means the upstream BF16 fallback, not a shape-"
                "tuned PR entry.",
                "",
                "| M | Projection | GEMM shape (M,N,K) | Selection | BM / BN / BK | "
                "Warps | Stages |",
                "|---:|---|---|---|---|---:|---:|",
            )
        )
        for row in rows:
            if row["direction"] != "forward":
                continue
            for projection, metadata in row["vllm_matmul_configs"].items():
                shape = metadata["shape"]
                config = metadata["config"]
                lines.append(
                    f"| {row['tokens']} | {projection} | "
                    f"({shape['M']},{shape['N']},{shape['K']}) | "
                    f"{metadata['selection']} | {config['BLOCK_SIZE_M']} / "
                    f"{config['BLOCK_SIZE_N']} / {config['BLOCK_SIZE_K']} | "
                    f"{config['num_warps']} | {config['num_stages']} |"
                )
        selections = {
            metadata["selection"]
            for row in rows
            if row["direction"] == "forward"
            for metadata in row["vllm_matmul_configs"].values()
        }
        if selections == {"default"}:
            lines.extend(
                (
                    "",
                    f"For H={methodology['hidden']} and I={methodology['intermediate']}, "
                    "none of the three projection shapes matches PR #53247's BF16 tuned "
                    "table. These measurements therefore exercise the vendored persistent "
                    "kernel with its upstream default configuration; they are not results "
                    "for a shape-tuned PR entry.",
                )
            )
        lines.extend(
            (
                "",
                f"Vendored kernel source: [{VLLM_BATCH_INVARIANT_SOURCE_SHA}]"
                f"({VLLM_BATCH_INVARIANT_SOURCE_URL}).",
            )
        )
    kernel_profile = payload.get("kernel_profile")
    if kernel_profile:
        lines.extend(
            (
                "",
                "## Layout-copy profile",
                "",
                f"One warmed call at M={kernel_profile['tokens']} was captured with "
                "`torch.profiler`.",
                "",
                "| Direction | Legacy direct copies | Optimized direct copies | "
                "Legacy GEMMs | Optimized GEMMs |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for direction, label in (
            ("forward", "Forward"),
            ("forward_backward", "Forward + backward"),
        ):
            baseline = kernel_profile["baseline"][direction]
            optimized = kernel_profile["optimized"][direction]
            lines.append(
                f"| {label} | {baseline['direct_copy_kernel']} | "
                f"{optimized['direct_copy_kernel']} | "
                f"{baseline['det_gemm_sm90_kernel']} | "
                f"{optimized['det_gemm_sm90_kernel']} |"
            )
        production_forward = kernel_profile["production_matmul"]["forward"]
        production_training = kernel_profile["production_matmul"]["forward_backward"]
        representative_kernels = sorted(
            set(production_forward["cuda_gemm_kernels"])
            | set(production_training["cuda_gemm_kernels"])
        )
        kernel_text = ", ".join(f"`{name}`" for name in representative_kernels)
        lines.extend(
            (
                "",
                "The production reference executes "
                f"{production_forward['aten_mm']} forward and "
                f"{production_training['aten_mm']} forward+backward `aten::mm` calls. "
                f"Representative profiled CUDA GEMM kernels: {kernel_text or 'unavailable'}.",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_parse_tokens, default="1,8,32,128")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--training-samples", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/qwen_ffn_layout_h100"),
    )
    parser.add_argument(
        "--skip-profiler",
        action="store_true",
        help=(
            "Skip layout-path copy/GEMM launch counting; the one-GEMM SM90 "
            "availability probe still runs."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.hidden <= 0 or args.intermediate <= 0:
        raise ValueError("hidden and intermediate sizes must be positive")
    if args.hidden % 128 or args.intermediate % 128:
        raise ValueError("hidden and intermediate sizes must be multiples of 128")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.samples <= 0 or args.training_samples <= 0:
        raise ValueError("sample counts must be positive")


def _portable_command(args: argparse.Namespace) -> str:
    command = [
        "python",
        "benchmarks/layers/benchmark_qwen_ffn_layout.py",
        "--tokens",
        ",".join(str(value) for value in args.tokens),
        "--hidden",
        str(args.hidden),
        "--intermediate",
        str(args.intermediate),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--training-samples",
        str(args.training_samples),
        "--device-index",
        str(args.device_index),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.skip_profiler:
        command.append("--skip-profiler")
    return shlex.join(command)


def _exact_invocation() -> str:
    command = shlex.join([sys.executable, *sys.argv])
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return command
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} {command}"


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_args(args)
    device, sm90_probe_launches = _validate_environment(args.device_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = (
        _randn(
            (args.intermediate, args.hidden),
            seed=args.seed + 1,
            device=device,
        ),
        _randn(
            (args.intermediate, args.hidden),
            seed=args.seed + 2,
            device=device,
        ),
        _randn(
            (args.hidden, args.intermediate),
            seed=args.seed + 3,
            device=device,
        ),
    )
    rows: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {}
    case_inputs: dict[int, tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor]] = {}
    vllm_batch_invariance = _verify_vllm_batch_invariance(
        token_counts=args.tokens,
        hidden_size=args.hidden,
        weights=weights,
        device=device,
        seed=args.seed,
    )
    for tokens in args.tokens:
        values, grad_output = _case_inputs(
            tokens=tokens,
            hidden_size=args.hidden,
            weights=weights,
            device=device,
            seed=args.seed,
        )
        case_inputs[tokens] = (values, grad_output)
        correctness[str(tokens)] = _verify_bitwise(values, grad_output)
        print(
            json.dumps(
                {
                    "phase": "correctness",
                    "tokens": tokens,
                    "deterministic_bitwise": "pass",
                    "production_matmul_numerical": "pass",
                    "vllm_numerical": "pass",
                }
            ),
            flush=True,
        )
    for tokens in args.tokens:
        values, grad_output = case_inputs[tokens]
        case_rows = _benchmark_case(
            tokens=tokens,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            values=values,
            grad_output=grad_output,
            device=device,
            warmup=args.warmup,
            samples=args.samples,
            training_samples=args.training_samples,
        )
        rows.extend(case_rows)
        print(
            json.dumps(
                {
                    "phase": "timing",
                    "tokens": tokens,
                    "forward_layout_speedup": case_rows[0]["optimized_speedup_vs_legacy"],
                    "forward_determinism_overhead": case_rows[0]["cublas_speedup_vs_optimized"],
                    "forward_vllm_speedup_vs_optimized": case_rows[0]["vllm_speedup_vs_optimized"],
                    "forward_backward_layout_speedup": case_rows[1]["optimized_speedup_vs_legacy"],
                    "forward_backward_determinism_overhead": case_rows[1][
                        "cublas_speedup_vs_optimized"
                    ],
                }
            ),
            flush=True,
        )
    kernel_profile = None
    if not args.skip_profiler:
        kernel_profile = _profile_layout_paths(
            tokens=max(args.tokens),
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            weights=weights,
            device=device,
            seed=args.seed,
        )
        for implementation in ("baseline", "optimized"):
            if kernel_profile[implementation]["forward_backward"]["det_gemm_sm90_kernel"] != 9:
                raise RuntimeError(
                    f"{implementation} profile did not execute nine SM90 GEMMs: "
                    f"{kernel_profile[implementation]}"
                )
        production_profile = kernel_profile["production_matmul"]
        if (
            production_profile["forward"]["aten_mm"] != 3
            or production_profile["forward_backward"]["aten_mm"] != 9
            or production_profile["forward_backward"]["det_gemm_sm90_kernel"] != 0
        ):
            raise RuntimeError(
                "production matmul profile did not execute 3/9 aten::mm calls: "
                f"{production_profile}"
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "comparison_contract": {
            "baseline": LEGACY_LABEL,
            "baseline_commit": LEGACY_COMMIT,
            "baseline_type": "in-process legacy-layout replay",
            "optimized": OPTIMIZED_LABEL,
            "production_reference": CUBLAS_LABEL,
            "vllm_batch_invariant_reference": VLLM_LABEL,
            "vllm_forward_only": True,
            "vllm_upstream_pr": VLLM_BATCH_INVARIANT_PR_URL,
            "vllm_upstream_source_sha": VLLM_BATCH_INVARIANT_SOURCE_SHA,
            "vllm_upstream_source_url": VLLM_BATCH_INVARIANT_SOURCE_URL,
            "scope": "symmetric stripped single-GPU TP=1 Qwen3 FFN cores",
        },
        "environment": _environment(device, sm90_probe_launches),
        "methodology": {
            "operator_only": True,
            "execution_mode": "PyTorch eager, no torch.compile",
            "timing": (
                "CUDA events; forward cycles through a prefix-balanced ordering of all "
                "24 four-way permutations, while forward+backward cycles through all six "
                "three-way permutations"
            ),
            "summary": "median, p95, min, max, and raw samples",
            "tokens": list(args.tokens),
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "dtype": "bfloat16",
            "input_scale": INPUT_SCALE,
            "seed": args.seed,
            "warmup": args.warmup,
            "samples": args.samples,
            "training_samples": args.training_samples,
            "forward_implementations": ["baseline", "optimized", "cublas", "vllm"],
            "forward_permutation_cycle_length": 24,
            "forward_backward_implementations": ["baseline", "optimized", "cublas"],
            "forward_backward_permutation_cycle_length": 6,
            "vllm_forward_only": True,
            "vllm_matmul_config_by_tokens": {
                str(tokens): _vllm_projection_configs(
                    tokens=tokens,
                    hidden_size=args.hidden,
                    intermediate_size=args.intermediate,
                )
                for tokens in args.tokens
            },
            "production_matmul_sanity_bounds": {
                "relative_l2": CUBLAS_RELATIVE_L2_LIMIT,
                "max_abs_over_reference_max": CUBLAS_NORMALIZED_MAX_LIMIT,
            },
        },
        "command": _portable_command(args),
        "exact_invocation": _exact_invocation(),
        "results": rows,
        "correctness": correctness,
        "vllm_batch_invariance": vllm_batch_invariance,
        "kernel_profile": kernel_profile,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(payload, args.output_dir / "report.md")
    _write_csv(payload, args.output_dir / "results.csv")
    _write_production_context_figure(
        payload,
        args.output_dir / "qwen_ffn_cublas_comparison.svg",
    )
    _write_forward_backward_context_figure(
        payload,
        args.output_dir / "qwen_ffn_forward_backward_comparison.svg",
    )
    for stale_figure in ("qwen_ffn_layout_latency.svg", "qwen_ffn_layout_copies.svg"):
        (args.output_dir / stale_figure).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "results": str(results_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
