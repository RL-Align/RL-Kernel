# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm benchmark for the deterministic distributed Triton Qwen3 FFN.

Distributed performance compares four paths at the same TP/CP/SP topology:
H100 official/deterministic and MI300X official/deterministic. Determinism
compares every Triton TP/CP/SP layout bitwise with Triton TP=1. A separate
single-GPU section retains the official Hugging Face Qwen3MLP TP=1 context. No
model checkpoint or serving engine is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import statistics
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import __version__ as transformers_version
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

import rl_engine.kernels.ops.triton.ffn.ffn as ffn_module
from rl_engine.kernels.ops.triton.ffn import (
    pack_qwen3_ffn_forward_weights,
    qwen3_ffn,
)

_DISTRIBUTED_CONFIGS: dict[int, tuple[tuple[str, int, int, bool], ...]] = {
    2: (("tp2", 2, 1, False), ("tp2_sp", 2, 1, True)),
    4: (
        ("tp4", 4, 1, False),
        ("tp2_cp2", 2, 2, False),
        ("tp2_cp2_sp", 2, 2, True),
    ),
    8: (
        ("tp8", 8, 1, False),
        ("tp4_cp2", 4, 2, False),
        ("tp4_cp2_sp", 4, 2, True),
    ),
}
_TP1_FORWARD_CACHE_BYTES = 3 * 4096 * 12288 * torch.bfloat16.itemsize
_COMMUNICATION_CONTRACT = {
    "forward": {"all_gather": 1, "reduce_scatter": 1, "total": 2},
    "train_fwd_bwd": {
        "all_gather": 7,
        "reduce_scatter_logical_lanes": 3,
        "logical_total": 10,
        "collective_invocations": 9,
    },
}
_PREVIOUS_DISTRIBUTED_TRITON_MS = {
    ("tp2", "forward"): 0.9039933793246746,
    ("tp2", "train_fwd_bwd"): 2.6996671222150326,
    ("tp2_sp", "forward"): 1.0561398230493069,
    ("tp2_sp", "train_fwd_bwd"): 2.9646214097738266,
    ("tp4", "forward"): 0.8358820341527462,
    ("tp4", "train_fwd_bwd"): 2.6998785324394703,
    ("tp2_cp2", "forward"): 0.9015901014208794,
    ("tp2_cp2", "train_fwd_bwd"): 3.5605919547379017,
    ("tp2_cp2_sp", "forward"): 1.1296039447188377,
    ("tp2_cp2_sp", "train_fwd_bwd"): 3.992859274148941,
    ("tp8", "forward"): 1.0859542526304722,
    ("tp8", "train_fwd_bwd"): 2.5620022788643837,
    ("tp4_cp2", "forward"): 1.0536308400332928,
    ("tp4_cp2", "train_fwd_bwd"): 3.259910736232996,
    ("tp4_cp2_sp", "forward"): 1.1927778832614422,
    ("tp4_cp2_sp", "train_fwd_bwd"): 3.7497887387871742,
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    denominator = torch.linalg.vector_norm(expected_float)
    numerator = torch.linalg.vector_norm(actual_float - expected_float)
    if denominator.item() == 0.0:
        return float(numerator.item())
    return float((numerator / denominator).item())


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - expected.detach().float()
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": _relative_l2(actual, expected),
        "exact_fraction": float((actual.detach() == expected.detach()).float().mean().item()),
    }


def _mismatches(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.detach() != right.detach()).sum().item())


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    scale: float = 0.02,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(device=device, dtype=dtype)


def _gpu_event_samples(
    function: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _official_qwen3_mlp(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> Qwen3MLP:
    """Build the upstream Transformers Qwen3 FFN with the benchmark weights."""
    config = Qwen3Config(
        hidden_size=gate_weight.size(1),
        intermediate_size=gate_weight.size(0),
        hidden_act="silu",
    )
    module = Qwen3MLP(config).to(
        device=gate_weight.device,
        dtype=gate_weight.dtype,
    )
    with torch.no_grad():
        module.gate_proj.weight.copy_(gate_weight)
        module.up_proj.weight.copy_(up_weight)
        module.down_proj.weight.copy_(down_weight)
    return module


def _official_inference(module: Qwen3MLP, hidden: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module(hidden)


def _training_step(
    function: Callable[..., torch.Tensor],
    inputs: list[torch.Tensor],
    grad_output: torch.Tensor,
) -> torch.Tensor:
    for value in inputs:
        value.grad = None
    output = function(*inputs)
    output.backward(grad_output)
    return output


class _NativeAllReduce(torch.autograd.Function):
    """Autograd-aware native ProcessGroup all-reduce used by the baseline."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        output = input.contiguous().clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _NativeCopyToTensorParallel(torch.autograd.Function):
    """Identity in forward and native all-reduce in backward."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, group=ctx.group)
        return grad_input, None


class _NativeAllGather(torch.autograd.Function):
    """Autograd-aware native first-dimension all-gather baseline."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, group: Any) -> torch.Tensor:
        world_size = dist.get_world_size(group=group)
        ctx.group = group
        ctx.input_shape = tuple(input.shape)
        input_flat = input.contiguous().view(-1)
        output_flat = torch.empty(
            world_size * input_flat.numel(),
            dtype=input.dtype,
            device=input.device,
        )
        dist.all_gather_into_tensor(output_flat, input_flat, group=group)
        return output_flat.view(world_size * input.size(0), *input.shape[1:])

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_output_flat = grad_output.contiguous().view(-1)
        grad_input_flat = torch.empty(
            math.prod(ctx.input_shape),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(
            grad_input_flat,
            grad_output_flat,
            group=ctx.group,
        )
        return grad_input_flat.view(ctx.input_shape), None


class _NativeReduceScatter(torch.autograd.Function):
    """Autograd-aware native first-dimension reduce-scatter baseline."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, group: Any) -> torch.Tensor:
        world_size = dist.get_world_size(group=group)
        if input.size(0) % world_size != 0:
            raise ValueError("native reduce-scatter requires divisible dimension 0")
        ctx.group = group
        ctx.input_shape = tuple(input.shape)
        output_shape = (input.size(0) // world_size, *input.shape[1:])
        output_flat = torch.empty(
            math.prod(output_shape),
            dtype=input.dtype,
            device=input.device,
        )
        dist.reduce_scatter_tensor(
            output_flat,
            input.contiguous().view(-1),
            group=group,
        )
        return output_flat.view(output_shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_output_flat = grad_output.contiguous().view(-1)
        grad_input_flat = torch.empty(
            math.prod(ctx.input_shape),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_gather_into_tensor(
            grad_input_flat,
            grad_output_flat,
            group=ctx.group,
        )
        return grad_input_flat.view(ctx.input_shape), None


def _official_distributed_ffn(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    tp_group: Any,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Upstream Qwen3 FFN math with native distributed collectives."""
    full_hidden = (
        _NativeAllGather.apply(hidden, tp_group)
        if sequence_parallel
        else _NativeCopyToTensorParallel.apply(hidden, tp_group)
    )
    activated = F.silu(F.linear(full_hidden, gate_weight)) * F.linear(full_hidden, up_weight)
    partial_output = F.linear(activated, down_weight)
    if sequence_parallel:
        return _NativeReduceScatter.apply(partial_output, tp_group)
    return _NativeAllReduce.apply(partial_output, tp_group)


def _official_distributed_training_step(
    inputs: list[torch.Tensor],
    grad_output: torch.Tensor,
    *,
    tp_group: Any,
    cp_group: Any,
    sequence_parallel: bool,
) -> torch.Tensor:
    for value in inputs:
        value.grad = None
    output = _official_distributed_ffn(
        *inputs,
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
    )
    output.backward(grad_output)
    if cp_group is not None:
        for weight in inputs[1:]:
            dist.all_reduce(weight.grad, group=cp_group)
    return output


def _single_gpu_benchmarks(
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    results: dict[str, list[dict[str, Any]]] = {"speed": [], "dtype_accuracy": []}
    gate_weight = _randn((12288, 4096), seed=3000, device=device)
    up_weight = _randn((12288, 4096), seed=3001, device=device)
    down_weight = _randn((4096, 12288), seed=3002, device=device)
    forward_weights = pack_qwen3_ffn_forward_weights(
        gate_weight,
        up_weight,
        down_weight,
    )
    official = _official_qwen3_mlp(gate_weight, up_weight, down_weight)
    for index, tokens in enumerate((1, 8, 32)):
        hidden = _randn((tokens, 4096), seed=3010 + index * 2, device=device)
        grad_output = _randn((tokens, 4096), seed=3011 + index * 2, device=device)
        weights = (gate_weight, up_weight, down_weight)
        official_timing = _summary_ms(
            _gpu_event_samples(
                lambda: _official_inference(official, hidden),
                warmup=warmup,
                samples=samples,
            )
        )
        triton_timing = _summary_ms(
            _gpu_event_samples(
                lambda: qwen3_ffn(
                    hidden,
                    *weights,
                    forward_weights=forward_weights,
                ),
                warmup=warmup,
                samples=samples,
            )
        )
        results["speed"].append(
            {
                "name": f"(M,H,I)=({tokens},4096,12288), forward",
                "direction": "forward",
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 12288,
                "dtype": "bfloat16",
                "weight_layout": "packed_forward_cache",
                "official_tp1": official_timing,
                "triton": triton_timing,
                "latency_ratio_vs_official_tp1": (
                    triton_timing["median_ms"] / official_timing["median_ms"]
                ),
            }
        )

        official_hidden = hidden.detach().clone().requires_grad_(True)
        triton_inputs = [
            value.detach().clone().requires_grad_(True) for value in (hidden, *weights)
        ]
        triton_forward_weights = pack_qwen3_ffn_forward_weights(*triton_inputs[1:])

        def triton_training_step() -> torch.Tensor:
            return _training_step(
                lambda *values: qwen3_ffn(
                    *values,
                    forward_weights=triton_forward_weights,
                ),
                triton_inputs,
                grad_output,
            )

        def official_training_step() -> torch.Tensor:
            official.zero_grad(set_to_none=True)
            official_hidden.grad = None
            output = official(official_hidden)
            output.backward(grad_output)
            return output

        official_train_timing = _summary_ms(
            _gpu_event_samples(
                official_training_step,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            )
        )
        triton_train_timing = _summary_ms(
            _gpu_event_samples(
                triton_training_step,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            )
        )
        results["speed"].append(
            {
                "name": f"(M,H,I)=({tokens},4096,12288), forward+backward",
                "direction": "train_fwd_bwd",
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 12288,
                "dtype": "bfloat16",
                "weight_layout": "packed_forward_cache",
                "official_tp1": official_train_timing,
                "triton": triton_train_timing,
                "latency_ratio_vs_official_tp1": (
                    triton_train_timing["median_ms"] / official_train_timing["median_ms"]
                ),
            }
        )
        del (
            hidden,
            grad_output,
            official_hidden,
            triton_inputs,
            triton_forward_weights,
        )
        torch.cuda.empty_cache()

    # This is intentionally separate from the determinism and speed results.
    del official, forward_weights, gate_weight, up_weight, down_weight
    torch.cuda.empty_cache()
    tokens = 8
    fp32_hidden = _randn((tokens, 4096), seed=3100, device=device, dtype=torch.float32)
    fp32_gate = _randn((12288, 4096), seed=3101, device=device, dtype=torch.float32)
    fp32_up = _randn((12288, 4096), seed=3102, device=device, dtype=torch.float32)
    fp32_down = _randn((4096, 12288), seed=3103, device=device, dtype=torch.float32)
    official_fp32 = _official_qwen3_mlp(fp32_gate, fp32_up, fp32_down)
    with torch.no_grad():
        fp32_output = official_fp32(fp32_hidden)
    del official_fp32
    torch.cuda.empty_cache()
    official_fp16 = _official_qwen3_mlp(fp32_gate.half(), fp32_up.half(), fp32_down.half())
    with torch.no_grad():
        fp16_output = official_fp16(fp32_hidden.half())
    results["dtype_accuracy"].append(
        {
            "name": "Official Qwen3MLP TP=1 FP16 vs FP32",
            "tokens": tokens,
            "hidden": 4096,
            "intermediate": 12288,
            "candidate_dtype": "float16",
            "reference_dtype": "float32",
            **_accuracy(fp16_output, fp32_output),
        }
    )
    return results


def _mesh_groups(
    world_size: int,
    tp_size: int,
    cp_size: int,
) -> tuple[list[Any], list[Any]]:
    if tp_size * cp_size != world_size:
        raise ValueError("TP size times CP size must equal world size")
    if tp_size == world_size and cp_size == 1:
        return [dist.group.WORLD], []
    tp_groups = []
    if tp_size > 1:
        for cp_rank in range(cp_size):
            ranks = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
            tp_groups.append(dist.new_group(ranks=ranks))
    cp_groups = []
    if cp_size > 1:
        for tp_rank in range(tp_size):
            ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
            cp_groups.append(dist.new_group(ranks=ranks))
    return tp_groups, cp_groups


def _shard_ranges(
    rank: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
    token_count: int,
    intermediate_size: int,
) -> tuple[int, int, int, int]:
    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    cp_tokens = token_count // cp_size
    local_tokens = cp_tokens // tp_size if sequence_parallel else cp_tokens
    token_start = cp_rank * cp_tokens
    if sequence_parallel:
        token_start += tp_rank * local_tokens
    token_end = token_start + local_tokens
    local_intermediate = intermediate_size // tp_size
    feature_start = tp_rank * local_intermediate
    feature_end = feature_start + local_intermediate
    return token_start, token_end, feature_start, feature_end


def _distributed_wall_samples(
    function: Callable[[], Any],
    *,
    group: Any,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    dist.barrier(group=group)
    return timings


def _slowest_rank_summary(local_timings: list[float], group: Any) -> dict[str, float]:
    world_size = dist.get_world_size(group=group)
    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_timings, group=group)
    slowest = [
        max(float(rank_values[index]) for rank_values in gathered if rank_values)
        for index in range(len(local_timings))
    ]
    return _summary_ms(slowest)


def _distributed_ffn_benchmark(
    rank: int,
    world_size: int,
    configs: tuple[tuple[str, int, int, bool], ...],
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda", rank)
    token_count = 32
    hidden_size = 4096
    intermediate_size = 12288
    hidden_full = _randn((token_count, hidden_size), seed=5000, device=device)
    gate_full = _randn((intermediate_size, hidden_size), seed=5001, device=device)
    up_full = _randn((intermediate_size, hidden_size), seed=5002, device=device)
    down_full = _randn((hidden_size, intermediate_size), seed=5003, device=device)
    grad_output_full = _randn((token_count, hidden_size), seed=5004, device=device)

    # Exactness reference: the same deterministic Triton implementation at TP=1.
    full_values = (hidden_full, gate_full, up_full, down_full)
    tp1_forward_weights = pack_qwen3_ffn_forward_weights(*full_values[1:])
    with torch.no_grad():
        tp1_forward = (
            qwen3_ffn(
                *full_values,
                forward_weights=tp1_forward_weights,
            )
            .detach()
            .clone()
        )
    tp1_inputs = [value.detach().clone().requires_grad_(True) for value in full_values]
    tp1_training_weights = pack_qwen3_ffn_forward_weights(*tp1_inputs[1:])
    tp1_train = (
        _training_step(
            lambda *values: qwen3_ffn(
                *values,
                forward_weights=tp1_training_weights,
            ),
            tp1_inputs,
            grad_output_full,
        )
        .detach()
        .clone()
    )
    tp1_grads = [value.grad.detach().clone() for value in tp1_inputs]
    del tp1_inputs, tp1_forward_weights, tp1_training_weights
    torch.cuda.empty_cache()

    meshes: dict[tuple[int, int], tuple[list[Any], list[Any]]] = {}
    results = []

    for name, tp_size, cp_size, sequence_parallel in configs:
        mesh_key = (tp_size, cp_size)
        if mesh_key not in meshes:
            meshes[mesh_key] = _mesh_groups(world_size, tp_size, cp_size)
        tp_groups, cp_groups = meshes[mesh_key]
        tp_rank = rank % tp_size
        cp_rank = rank // tp_size
        tp_group = tp_groups[cp_rank] if tp_size > 1 else None
        cp_group = cp_groups[tp_rank] if cp_size > 1 else None
        token_start, token_end, feature_start, feature_end = _shard_ranges(
            rank,
            tp_size=tp_size,
            cp_size=cp_size,
            sequence_parallel=sequence_parallel,
            token_count=token_count,
            intermediate_size=intermediate_size,
        )
        shard = (
            hidden_full[token_start:token_end].contiguous(),
            gate_full[feature_start:feature_end].contiguous(),
            up_full[feature_start:feature_end].contiguous(),
            down_full[:, feature_start:feature_end].contiguous(),
        )
        shard_forward_weights = pack_qwen3_ffn_forward_weights(*shard[1:])
        local_grad_output = grad_output_full[token_start:token_end].contiguous()

        def official_distributed_forward():
            with torch.no_grad():
                return _official_distributed_ffn(
                    *shard,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                )

        official_forward_summary = _slowest_rank_summary(
            _distributed_wall_samples(
                official_distributed_forward,
                group=dist.group.WORLD,
                warmup=warmup,
                samples=samples,
            ),
            dist.group.WORLD,
        )
        official_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        official_train_summary = _slowest_rank_summary(
            _distributed_wall_samples(
                lambda: _official_distributed_training_step(
                    official_inputs,
                    local_grad_output,
                    tp_group=tp_group,
                    cp_group=cp_group,
                    sequence_parallel=sequence_parallel,
                ),
                group=dist.group.WORLD,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            ),
            dist.group.WORLD,
        )

        def triton_forward():
            return qwen3_ffn(
                *shard,
                forward_weights=shard_forward_weights,
                tp_group=tp_group,
                cp_group=cp_group,
                sequence_parallel=sequence_parallel,
            )

        with torch.no_grad():
            triton_output = triton_forward()
            triton_repeat = triton_forward()
        triton_forward_summary = _slowest_rank_summary(
            _distributed_wall_samples(
                triton_forward,
                group=dist.group.WORLD,
                warmup=warmup,
                samples=samples,
            ),
            dist.group.WORLD,
        )

        triton_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        repeat_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        triton_forward_weights = pack_qwen3_ffn_forward_weights(*triton_inputs[1:])
        repeat_forward_weights = pack_qwen3_ffn_forward_weights(*repeat_inputs[1:])

        def triton_training_step(inputs, packed_weights):
            for value in inputs:
                value.grad = None
            output = qwen3_ffn(
                *inputs,
                forward_weights=packed_weights,
                tp_group=tp_group,
                cp_group=cp_group,
                sequence_parallel=sequence_parallel,
            )
            output.backward(local_grad_output)
            return output

        triton_train = (
            triton_training_step(
                triton_inputs,
                triton_forward_weights,
            )
            .detach()
            .clone()
        )
        triton_grads = [value.grad.detach().clone() for value in triton_inputs]
        repeat_train = (
            triton_training_step(
                repeat_inputs,
                repeat_forward_weights,
            )
            .detach()
            .clone()
        )
        repeat_grads = [value.grad.detach().clone() for value in repeat_inputs]
        triton_train_summary = _slowest_rank_summary(
            _distributed_wall_samples(
                lambda: triton_training_step(
                    triton_inputs,
                    triton_forward_weights,
                ),
                group=dist.group.WORLD,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            ),
            dist.group.WORLD,
        )

        expected_forward = tp1_forward[token_start:token_end]
        expected_train = tp1_train[token_start:token_end]
        expected_grads = (
            tp1_grads[0][token_start:token_end],
            tp1_grads[1][feature_start:feature_end],
            tp1_grads[2][feature_start:feature_end],
            tp1_grads[3][:, feature_start:feature_end],
        )
        local_exactness = {
            "tp1_forward_output": _mismatches(triton_output, expected_forward),
            "tp1_training_output": _mismatches(triton_train, expected_train),
            "tp1_hidden_gradient": _mismatches(triton_grads[0], expected_grads[0]),
            "tp1_weight_gradient": sum(
                _mismatches(actual, expected)
                for actual, expected in zip(triton_grads[1:], expected_grads[1:], strict=True)
            ),
            "repeat_forward": _mismatches(triton_output, triton_repeat),
            "train_infer_mismatch_count": _mismatches(triton_output, triton_train),
            "repeat_training": _mismatches(triton_train, repeat_train)
            + sum(
                _mismatches(actual, repeat)
                for actual, repeat in zip(triton_grads, repeat_grads, strict=True)
            ),
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_exactness)
        if rank == 0:
            valid = [value for value in gathered if value is not None]
            common = {
                "name": name,
                "world_size": world_size,
                "tp_size": tp_size,
                "cp_size": cp_size,
                "sequence_parallel": sequence_parallel,
                "tokens": token_count,
                "hidden": hidden_size,
                "intermediate": intermediate_size,
                "weight_layout": "packed_forward_cache",
            }
            results.extend(
                (
                    {
                        **common,
                        "direction": "forward",
                        "official_distributed": official_forward_summary,
                        "triton": triton_forward_summary,
                        "latency_ratio_triton_vs_official_distributed": (
                            triton_forward_summary["median_ms"]
                            / official_forward_summary["median_ms"]
                        ),
                        "tp1_mismatch": {
                            "forward_output": sum(value["tp1_forward_output"] for value in valid),
                        },
                        "repeat_mismatch_count": sum(value["repeat_forward"] for value in valid),
                        "train_infer_mismatch_count": sum(
                            value["train_infer_mismatch_count"] for value in valid
                        ),
                    },
                    {
                        **common,
                        "direction": "train_fwd_bwd",
                        "official_distributed": official_train_summary,
                        "triton": triton_train_summary,
                        "latency_ratio_triton_vs_official_distributed": (
                            triton_train_summary["median_ms"] / official_train_summary["median_ms"]
                        ),
                        "tp1_mismatch": {
                            "training_output": sum(value["tp1_training_output"] for value in valid),
                            "hidden_gradient": sum(value["tp1_hidden_gradient"] for value in valid),
                            "weight_gradient": sum(value["tp1_weight_gradient"] for value in valid),
                        },
                        "repeat_mismatch_count": sum(value["repeat_training"] for value in valid),
                        "train_infer_mismatch_count": sum(
                            value["train_infer_mismatch_count"] for value in valid
                        ),
                    },
                )
            )
        del (
            shard,
            shard_forward_weights,
            official_inputs,
            triton_inputs,
            triton_forward_weights,
            repeat_inputs,
            repeat_forward_weights,
        )
        torch.cuda.empty_cache()
        dist.barrier()

    for collective in list(ffn_module._COLLECTIVES.values()):
        collective.close()
    ffn_module._COLLECTIVES.clear()
    return results


def _distributed_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
    warmup: int,
    samples: int,
    training_samples: int,
) -> None:
    try:
        try:
            available_cpus = sorted(os.sched_getaffinity(0))
            numa_span = max(1, len(available_cpus) // 2)
            numa_index = 0 if rank < 4 else 1
            local_rank = rank % 4
            cpu_index = min(
                numa_index * numa_span + local_rank,
                len(available_cpus) - 1,
            )
            os.sched_setaffinity(0, {available_cpus[cpu_index]})
        except (AttributeError, OSError):
            pass
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=15),
        )
        distributed_ffn = _distributed_ffn_benchmark(
            rank,
            world_size,
            _DISTRIBUTED_CONFIGS[world_size],
            warmup=warmup,
            samples=samples,
            training_samples=training_samples,
        )
        if rank == 0:
            result_queue.put(
                {
                    "ok": True,
                    "world_size": world_size,
                    "distributed_ffn": distributed_ffn,
                }
            )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "rank": rank,
                "world_size": world_size,
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        for collective in list(ffn_module._COLLECTIVES.values()):
            collective.close()
        ffn_module._COLLECTIVES.clear()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_world(
    world_size: int,
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as temporary_directory:
        init_method = (Path(temporary_directory) / "rccl_init").as_uri()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_worker,
                args=(
                    rank,
                    world_size,
                    init_method,
                    result_queue,
                    warmup,
                    samples,
                    training_samples,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        result = None
        try:
            result = result_queue.get(timeout=1800)
            if not result["ok"]:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
        except queue.Empty as exc:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(f"timed out waiting for world_size={world_size} benchmark") from exc
        finally:
            for process in processes:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            result_queue.close()
            result_queue.join_thread()
    if result is None:
        raise RuntimeError(f"world_size={world_size} returned no result")
    if not result["ok"]:
        raise RuntimeError(result.get("traceback", str(result)))
    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(f"world_size={world_size} worker exited with {process.exitcode}")
    return result


def _topology_exactness_rows(
    distributed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in distributed_rows:
        entry = merged.setdefault(
            row["name"],
            {
                "name": row["name"],
                "world_size": row["world_size"],
                "tp_size": row["tp_size"],
                "cp_size": row["cp_size"],
                "sequence_parallel": row["sequence_parallel"],
                "forward_output": 0,
                "training_output": 0,
                "hidden_gradient": 0,
                "weight_gradient": 0,
                "repeat": 0,
                "train_infer": 0,
            },
        )
        for key, value in row["tp1_mismatch"].items():
            entry[key] += value
        entry["repeat"] += row["repeat_mismatch_count"]
        entry["train_infer"] += row["train_infer_mismatch_count"]
    return list(merged.values())


def _load_cuda_cpu_comparison(
    output_directory: Path,
) -> dict[str, Any] | None:
    comparison_path = output_directory / "cuda_cpu_comparison.json"
    if not comparison_path.exists():
        return None
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def _distributed_platform_comparison_rows(
    current_rows: list[dict[str, Any]],
    comparison_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Join H100 and MI300X distributed timings by topology and direction."""
    if comparison_payload is None:
        return []
    h100_lookup = {
        (row["name"], row["direction"]): row for row in comparison_payload["distributed"]
    }
    rows: list[dict[str, Any]] = []
    for current in current_rows:
        key = (current["name"], current["direction"])
        h100 = h100_lookup.get(key)
        if h100 is None:
            continue
        h100_official_ms = float(h100["official_h100_ms"])
        h100_deterministic_ms = float(h100["cuda_h100_ms"])
        mi300x_official_ms = float(current["official_distributed"]["median_ms"])
        mi300x_deterministic_ms = float(current["triton"]["median_ms"])
        rows.append(
            {
                "name": current["name"],
                "direction": current["direction"],
                "tp_size": current["tp_size"],
                "cp_size": current["cp_size"],
                "sequence_parallel": current["sequence_parallel"],
                "h100_official_distributed_ms": h100_official_ms,
                "h100_deterministic_cuda_ms": h100_deterministic_ms,
                "mi300x_official_distributed_ms": mi300x_official_ms,
                "mi300x_deterministic_triton_ms": mi300x_deterministic_ms,
                "h100_deterministic_over_official_ratio": (
                    h100_deterministic_ms / h100_official_ms
                ),
                "mi300x_deterministic_over_official_ratio": (
                    mi300x_deterministic_ms / mi300x_official_ms
                ),
                "deterministic_mi300x_over_h100_ratio": (
                    mi300x_deterministic_ms / h100_deterministic_ms
                ),
            }
        )
    return rows


def _previous_deterministic_comparison_rows(
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare the current MI300X run with the previous checked report data."""
    rows: list[dict[str, Any]] = []
    for current in current_rows:
        key = (current["name"], current["direction"])
        previous_ms = _PREVIOUS_DISTRIBUTED_TRITON_MS.get(key)
        if previous_ms is None:
            continue
        current_ms = float(current["triton"]["median_ms"])
        rows.append(
            {
                "name": current["name"],
                "direction": current["direction"],
                "previous_ms": previous_ms,
                "current_ms": current_ms,
                "latency_reduction_ratio": 1.0 - current_ms / previous_ms,
            }
        )
    return rows


def _write_report(
    payload: dict[str, Any],
    output_directory: Path,
    comparison_payload: dict[str, Any] | None = None,
) -> None:
    environment = payload["environment"]
    methodology = payload["methodology"]
    single_speed = payload["single_gpu"]["speed"]
    dtype_rows = payload["single_gpu"]["dtype_accuracy"]
    distributed_speed = payload["distributed_ffn"]
    platform_comparison = _distributed_platform_comparison_rows(
        distributed_speed, comparison_payload
    )
    previous_comparison = _previous_deterministic_comparison_rows(distributed_speed)
    previous_reductions = [row["latency_reduction_ratio"] for row in previous_comparison]
    exactness_rows = _topology_exactness_rows(distributed_speed)
    single_ratios = [row["latency_ratio_vs_official_tp1"] for row in single_speed]
    platform_ratios = [row["deterministic_mi300x_over_h100_ratio"] for row in platform_comparison]
    total_tp1_mismatch = sum(
        row[key]
        for row in exactness_rows
        for key in (
            "forward_output",
            "training_output",
            "hidden_gradient",
            "weight_gradient",
        )
    )
    total_repeat_mismatch = sum(row["repeat"] for row in exactness_rows)
    total_train_infer_mismatch = sum(row["train_infer"] for row in exactness_rows)
    dtype_row = dtype_rows[0]

    lines = [
        "# PR #325 ROCm deterministic Triton FFN report",
        "",
        "This is an operator-only MI300X report. It does not load or benchmark a "
        "model checkpoint.",
        "",
        "## Comparison contract",
        "",
        "1. **Determinism:** every Triton TP/CP/SP result is compared bitwise with "
        "the same deterministic Triton FFN at **TP=1**. The reported metric is "
        "element mismatch count; acceptance requires 0.",
        "2. **FP16/FP32:** one separate, simple output comparison runs official "
        "Hugging Face `Qwen3MLP` at TP=1 in FP16 and FP32. FP32 is the reference.",
        "3. **Speed:** single-GPU speed retains the official Qwen3MLP TP=1 "
        "context. Distributed speed compares four same-topology paths: H100 "
        "official/deterministic and MI300X official/deterministic.",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in environment.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        (
            "",
            "## Methodology",
            "",
            "- Operator shape: H=4096, I=12288; BF16 is used for all speed and "
            "determinism measurements.",
            "- Single-GPU shapes use M=1/8/32. Distributed cases use the same full "
            "logical M=32 input for TP2/4/8, TP+CP, and sequence parallelism.",
            "- The distributed comparison joins rows by topology and direction: "
            "H100 and MI300X use the same logical M=32 workload and TP/CP/SP layout. "
            "Official TP=1 latency is neither collected nor used in that distributed "
            "ratio.",
            "- MI300X official distributed uses upstream Qwen3 FFN math with native "
            "PyTorch BF16 GEMMs and native RCCL collectives over the same shards; "
            "the deterministic path uses the current Triton FFN and fixed-order "
            "transport.",
            "- Deterministic Triton timings use the explicit prepacked forward-weight "
            "cache. Packing happens once outside the timed region; canonical source "
            "weights remain the autograd and optimizer source of truth.",
            f"- The TP=1 cache adds {_TP1_FORWARD_CACHE_BYTES / 2**20:.0f} MiB; "
            "each TP rank holds that amount divided by TP size. Refresh cost is "
            "excluded because the benchmark measures the steady-state FFN call.",
            "- The distributed exactness baseline is the PR's deterministic Triton "
            "FFN at TP=1. Local outputs, dHidden, and sharded dWeights are compared "
            "against their exact TP=1 slices.",
            "- Communication contract for TP+CP+SP: forward uses 1 TP AllGather "
            "plus 1 TP ReduceScatter (2 calls). Forward+backward retains 7 "
            "AllGathers plus 3 logical ReduceScatter lanes; PR #357 merges the "
            "two independent backward gate/up lanes into one "
            "`reduce_scatter_many` call, for 9 collective invocations.",
            "- Implementation note: the current ROCm deterministic communication "
            "operator is adopted from PR #357. This changes the implementation "
            "under test, not the benchmark comparison contract.",
            f"- Single-GPU timing: {methodology['single_gpu_timing']}; distributed "
            f"timing: {methodology['distributed_timing']}.",
            f"- Distributed workers: {methodology['distributed_worker_cpu_affinity']} "
            "to reduce host-scheduler noise in synchronized wall-clock samples.",
            f"- {methodology['warmup']} warmups, {methodology['samples']} measured "
            f"forward samples, and {methodology['training_samples']} measured "
            "forward+backward samples.",
            "- `NCCL_IB_DISABLE=1` keeps the distributed run on intra-node XGMI. "
            "Median, p95, min, and max values are available in `results.json`.",
            "",
            "Reproduce from the repository root:",
            "",
            "```bash",
            "python benchmarks/benchmark_rocm_ffn.py \\",
            f"  --warmup {methodology['warmup']} \\",
            f"  --samples {methodology['samples']} \\",
            f"  --training-samples {methodology['training_samples']} \\",
            "  --output-dir benchmarks/results/pr325_rocm_mi300x",
            "```",
            "",
            "## Results summary",
            "",
            f"- TP=1 exactness baseline: **{total_tp1_mismatch} mismatched "
            "elements** across topology forward outputs, training outputs, "
            "dHidden, and dWeights.",
            f"- Repeat mismatch: **{total_repeat_mismatch}**; training/inference "
            f"forward mismatch: **{total_train_infer_mismatch}**.",
            f"- Single-GPU deterministic Triton packed-cache latency is "
            f"**{min(single_ratios):.2f}-{max(single_ratios):.2f}x** the official "
            "Qwen3MLP TP=1 latency across M=1/8/32 and forward/training.",
            (
                f"- MI300X deterministic Triton latency is **{min(platform_ratios):.2f}-"
                f"{max(platform_ratios):.2f}x** the H100 deterministic CUDA latency "
                "for the same distributed layouts. Both official distributed "
                "paths are reported alongside them."
                if platform_ratios
                else "- No matching H100 distributed rows were supplied."
            ),
            (
                f"- Versus the previous deterministic MI300X benchmark, PR #357 "
                f"improves **{sum(value > 0 for value in previous_reductions)}/"
                f"{len(previous_reductions)}** rows, with a mean latency reduction "
                f"of **{statistics.mean(previous_reductions) * 100:.1f}%**."
                if previous_reductions
                else "- No previous deterministic MI300X rows were available."
            ),
            f"- The separate official-Qwen3MLP FP16 versus FP32 observation has "
            f"relative-L2 error **{dtype_row['relative_l2']:.3e}** for "
            "(M,H,I)=(8,4096,12288).",
            "",
            "## Single-GPU FFN speed",
            "",
            "Performance only; no official-versus-Triton accuracy metric is " "reported here.",
            "",
            "| Shape / direction | Official Qwen3MLP TP=1 (ms) | Deterministic "
            "Triton, packed (ms) | Triton / official TP=1 |",
            "|---|---:|---:|---:|",
        )
    )
    for row in single_speed:
        lines.append(
            f"| {row['name']} | {row['official_tp1']['median_ms']:.4f} | "
            f"{row['triton']['median_ms']:.4f} | "
            f"{row['latency_ratio_vs_official_tp1']:.2f}x |"
        )

    lines.extend(("", "## Distributed FFN speed", ""))
    if platform_comparison:
        lines.extend(
            (
                "Every row compares the same distributed topology and direction. "
                "No TP=1 latency is used in this table.",
                "",
                "| Parallel layout | Direction | H100 official distributed (ms) | "
                "H100 deterministic CUDA (ms) | MI300X official distributed (ms) | "
                "MI300X deterministic Triton (ms) | H100 det / official | "
                "MI300X det / official |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in platform_comparison:
            lines.append(
                f"| {row['name']} | {row['direction']} | "
                f"{row['h100_official_distributed_ms']:.4f} | "
                f"{row['h100_deterministic_cuda_ms']:.4f} | "
                f"{row['mi300x_official_distributed_ms']:.4f} | "
                f"{row['mi300x_deterministic_triton_ms']:.4f} | "
                f"{row['h100_deterministic_over_official_ratio']:.2f}x | "
                f"{row['mi300x_deterministic_over_official_ratio']:.2f}x |"
            )
    else:
        lines.extend(
            (
                "No matching H100 distributed data was supplied; current MI300X "
                "latency is reported without a TP=1 ratio.",
                "",
                "| Parallel layout | Direction | MI300X official distributed (ms) | "
                "MI300X deterministic Triton (ms) | Deterministic / official |",
                "|---|---|---:|---:|---:|",
            )
        )
        for row in distributed_speed:
            lines.append(
                f"| {row['name']} | {row['direction']} | "
                f"{row['official_distributed']['median_ms']:.4f} | "
                f"{row['triton']['median_ms']:.4f} | "
                f"{row['latency_ratio_triton_vs_official_distributed']:.2f}x |"
            )

    if previous_comparison:
        lines.extend(
            (
                "",
                "### PR #357 latency change versus the previous benchmark",
                "",
                "This comparison changes only the deterministic ROCm communication "
                "implementation. It is not included as another series in the main "
                "four-path figure.",
                "",
                "| Parallel layout | Direction | Previous (ms) | Current (ms) | "
                "Latency reduction |",
                "|---|---|---:|---:|---:|",
            )
        )
        for row in previous_comparison:
            lines.append(
                f"| {row['name']} | {row['direction']} | "
                f"{row['previous_ms']:.4f} | {row['current_ms']:.4f} | "
                f"{row['latency_reduction_ratio'] * 100:.1f}% |"
            )

    if comparison_payload is not None:
        comparison_environment = comparison_payload["environment"]
        comparison_source = comparison_payload["source"]
        local_single = {(row["tokens"], row["direction"]): row for row in single_speed}
        lines.extend(
            (
                "",
                "## CUDA GPU and CPU performance context",
                "",
                f"The additional measurements come from [{comparison_source['report']}]"
                f"({comparison_source['pull_request']}) at CUDA commit "
                f"`{comparison_source['cuda_commit']}`. H100 Triton replays use "
                "this PR's code at "
                f"`{comparison_source['h100_triton_replay_commit']}`.",
                "",
                "The same-H100 CUDA/Triton ratio is the hardware-matched comparison. "
                "CPU and MI300X columns provide absolute-latency context only; they "
                "are not hardware-normalized speed claims.",
                "",
                "| Comparison environment | Value |",
                "|---|---|",
                f"| CUDA GPU | {comparison_environment['gpu']} "
                f"({comparison_environment['architecture']}) |",
                f"| CUDA / PyTorch | {comparison_environment['cuda']} / "
                f"{comparison_environment['torch']} |",
                f"| CPU | {comparison_environment['cpu']}, "
                f"{comparison_environment['cpu_threads']} intra-op threads |",
                f"| Transformers | {comparison_environment['transformers']} |",
                "",
                "### Single-GPU and CPU absolute latency",
                "",
                "| Shape / direction | CPU official (ms) | H100 official TP=1 "
                "(ms) | H100 Triton replay (ms) | H100 CUDA (ms) | CUDA / "
                "Triton H100 | MI300X official TP=1 (ms) | MI300X Triton (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in comparison_payload["single_gpu"]:
            local = local_single[(row["tokens"], row["direction"])]
            direction = "forward" if row["direction"] == "forward" else "forward+backward"
            lines.append(
                f"| M={row['tokens']}, {direction} | "
                f"{row['official_cpu_ms']:.4f} | "
                f"{row['official_h100_ms']:.4f} | "
                f"{row['triton_h100_ms']:.4f} | "
                f"{row['cuda_h100_ms']:.4f} | "
                f"{row['cuda_h100_ms'] / row['triton_h100_ms']:.2f}x | "
                f"{local['official_tp1']['median_ms']:.4f} | "
                f"{local['triton']['median_ms']:.4f} |"
            )
        lines.extend(
            (
                "",
                "Both H100 columns used in the main distributed table are the "
                "user-supplied distributed timings. They are joined directly with "
                "MI300X rows of the same topology and direction; TP=1 values are "
                "excluded from all four columns.",
            )
        )

    lines.extend(
        (
            "",
            "## Topology exactness versus Triton TP=1",
            "",
            "All columns are element mismatch counts. This table does not compare "
            "against the official FFN.",
            "",
            "| Parallel layout | Forward output | Training output | dHidden | "
            "dWeights | Repeat | Train/infer |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in exactness_rows:
        lines.append(
            f"| {row['name']} | {row['forward_output']} | "
            f"{row['training_output']} | {row['hidden_gradient']} | "
            f"{row['weight_gradient']} | {row['repeat']} | "
            f"{row['train_infer']} |"
        )

    lines.extend(
        (
            "",
            "## Simple FP16 versus FP32 observation",
            "",
            "This is an official `Qwen3MLP` TP=1 output comparison only; it is not "
            "used to judge deterministic Triton and is not included in speed "
            "ratios.",
            "",
            "| Shape | Candidate | Reference | Max abs | Mean abs | Relative L2 |",
            "|---|---|---|---:|---:|---:|",
            f"| (M,H,I)=({dtype_row['tokens']},{dtype_row['hidden']},"
            f"{dtype_row['intermediate']}) | FP16 | FP32 | "
            f"{dtype_row['max_abs']:.3e} | {dtype_row['mean_abs']:.3e} | "
            f"{dtype_row['relative_l2']:.3e} |",
            "",
            "## Deterministic communication overlap",
            "",
            "The current timing includes the fixed-order communication schedule "
            "and makes no overlap claim. Forward SP all-gather must finish before "
            "gate/up projection, and TP reduction consumes the down-projection "
            "output, so those edges are hard dependencies.",
            "",
            "In backward, the gate and up contributions to dHidden are independent "
            "until their final ordered addition. A future implementation can place "
            "the fixed-rank reduction of one contribution on a second stream while "
            "computing the other, but it must preserve rank order, reduction tree, "
            "wait points, and gate-then-up addition order. Any optimization is "
            "accepted only if every TP=1 mismatch column remains zero.",
            "",
            "## Figures",
            "",
            "![Single-GPU CUDA, packed Triton, and CPU latency]" "(single_gpu_overhead.png)",
            "",
            "![Topology mismatch versus Triton TP=1](collective_overhead.png)",
            "",
            "![Distributed H100 CUDA and MI300X packed Triton latency]"
            "(distributed_ffn_overhead.png)",
            "",
        )
    )
    (output_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figures(
    payload: dict[str, Any],
    output_directory: Path,
    comparison_payload: dict[str, Any] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 25,
            "axes.labelsize": 21,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
        }
    )

    single_rows = payload["single_gpu"]["speed"]
    if comparison_payload is None:
        single_labels = [
            f"M={row['tokens']}\n" f"{'FWD' if row['direction'] == 'forward' else 'FWD+BWD'}"
            for row in single_rows
        ]
        positions = np.arange(len(single_rows))
        width = 0.37
        figure, axis = plt.subplots(figsize=(17, 10))
        official_values = [row["official_tp1"]["median_ms"] for row in single_rows]
        triton_values = [row["triton"]["median_ms"] for row in single_rows]
        official_bars = axis.bar(
            positions - width / 2,
            official_values,
            width,
            label="Official Qwen3MLP TP=1",
            color="#2563eb",
        )
        triton_bars = axis.bar(
            positions + width / 2,
            triton_values,
            width,
            label="Deterministic Triton FFN (packed)",
            color="#7c3aed",
        )
        for bars, values in (
            (official_bars, official_values),
            (triton_bars, triton_values),
        ):
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=4,
                fontsize=13,
                rotation=90,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Token count M and measured direction\nH=4096, I=12288, BF16")
        axis.set_ylabel("Median latency (ms, log scale)")
        axis.set_title("MI300X single-GPU FFN speed: official TP=1 vs Triton")
        axis.set_xticks(positions, single_labels)
        axis.legend(loc="upper left")
        figure.tight_layout()
    else:
        local_lookup = {(row["tokens"], row["direction"]): row for row in single_rows}
        comparison_lookup = {
            (row["tokens"], row["direction"]): row for row in comparison_payload["single_gpu"]
        }
        figure, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
        series = (
            ("CPU official BF16", "official_cpu_ms", "#6b7280"),
            ("H100 official TP=1", "official_h100_ms", "#60a5fa"),
            ("H100 Triton replay", "triton_h100_ms", "#06b6d4"),
            ("H100 deterministic CUDA", "cuda_h100_ms", "#dc2626"),
            ("MI300X official TP=1", "official_mi300x_ms", "#86efac"),
            (
                "MI300X deterministic Triton (packed)",
                "triton_mi300x_ms",
                "#7c3aed",
            ),
        )
        width = 0.13
        for axis, direction, direction_label in (
            (axes[0], "forward", "Forward"),
            (axes[1], "train_fwd_bwd", "Forward + backward"),
        ):
            tokens = (1, 8, 32)
            positions = np.arange(len(tokens))
            combined_rows = []
            for token_count in tokens:
                external = comparison_lookup[(token_count, direction)]
                local = local_lookup[(token_count, direction)]
                combined_rows.append(
                    {
                        **external,
                        "official_mi300x_ms": local["official_tp1"]["median_ms"],
                        "triton_mi300x_ms": local["triton"]["median_ms"],
                    }
                )
            for series_index, (label, key, color) in enumerate(series):
                values = [row[key] for row in combined_rows]
                offset = (series_index - (len(series) - 1) / 2) * width
                bars = axis.bar(
                    positions + offset,
                    values,
                    width,
                    label=label,
                    color=color,
                )
                axis.bar_label(
                    bars,
                    labels=[f"{value:.2f}" for value in values],
                    padding=3,
                    fontsize=9,
                    rotation=90,
                )
            axis.set_yscale("log")
            axis.set_xlabel("Token count M\nH=4096, I=12288, BF16")
            axis.set_title(direction_label)
            axis.set_xticks(positions, [f"M={value}" for value in tokens])
        axes[0].set_ylabel("Median latency (ms, log scale)")
        axes[0].set_ylim(0.07, 140.0)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=3,
        )
        figure.suptitle(
            "Single-GPU / CPU FFN absolute latency across platforms",
            y=0.99,
        )
        figure.text(
            0.5,
            0.01,
            "Cross-hardware values are context only; H100 CUDA vs H100 Triton "
            "is the hardware-matched comparison.",
            ha="center",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    figure.savefig(output_directory / "single_gpu_overhead.png", dpi=180)
    plt.close(figure)

    exactness_rows = _topology_exactness_rows(payload["distributed_ffn"])
    mismatch_keys = (
        "forward_output",
        "training_output",
        "hidden_gradient",
        "weight_gradient",
    )
    mismatch_labels = (
        "Forward\noutput",
        "Training\noutput",
        "dHidden",
        "dWeights",
    )
    matrix = np.asarray(
        [[row[key] for key in mismatch_keys] for row in exactness_rows],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(1.0, matrix.max()),
    )
    for y_position in range(matrix.shape[0]):
        for x_position in range(matrix.shape[1]):
            axis.text(
                x_position,
                y_position,
                str(int(matrix[y_position, x_position])),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )
    axis.set_xticks(range(len(mismatch_labels)), mismatch_labels)
    axis.set_yticks(
        range(len(exactness_rows)),
        [
            f"TP={row['tp_size']}, CP={row['cp_size']}, "
            f"SP={'on' if row['sequence_parallel'] else 'off'}"
            for row in exactness_rows
        ],
    )
    axis.set_xlabel("Compared tensor category")
    axis.set_ylabel("Distributed parallel configuration")
    axis.set_title("Topology mismatch vs Triton TP=1")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colorbar.set_label("Mismatched elements")
    figure.tight_layout()
    figure.savefig(output_directory / "collective_overhead.png", dpi=180)
    plt.close(figure)

    rows = payload["distributed_ffn"]
    platform_comparison = _distributed_platform_comparison_rows(rows, comparison_payload)
    figure, axes = plt.subplots(1, 2, figsize=(26, 10), sharey=True)
    if platform_comparison:
        comparison_lookup = {(row["name"], row["direction"]): row for row in platform_comparison}
        distributed_series = (
            ("H100 official distributed", "h100_official_distributed_ms", "#60a5fa"),
            ("H100 deterministic CUDA", "h100_deterministic_cuda_ms", "#dc2626"),
            ("MI300X official distributed", "mi300x_official_distributed_ms", "#86efac"),
            (
                "MI300X deterministic Triton",
                "mi300x_deterministic_triton_ms",
                "#7c3aed",
            ),
        )
        width = 0.19
    else:
        distributed_series = (
            ("MI300X official distributed", "mi300x_official_distributed_ms", "#86efac"),
            (
                "MI300X deterministic Triton",
                "mi300x_deterministic_triton_ms",
                "#7c3aed",
            ),
        )
        width = 0.34
    for axis, direction, direction_label in (
        (axes[0], "forward", "Forward"),
        (axes[1], "train_fwd_bwd", "Forward + backward"),
    ):
        direction_rows = [row for row in rows if row["direction"] == direction]
        positions = np.arange(len(direction_rows))
        labels = [row["name"].upper().replace("_", "\n") for row in direction_rows]
        combined_rows = []
        for row in direction_rows:
            combined = {
                "mi300x_official_distributed_ms": row["official_distributed"]["median_ms"],
                "mi300x_deterministic_triton_ms": row["triton"]["median_ms"],
            }
            if platform_comparison:
                combined.update(comparison_lookup[(row["name"], direction)])
            combined_rows.append(combined)
        for series_index, (label, key, color) in enumerate(distributed_series):
            values = [row[key] for row in combined_rows]
            offset = (series_index - (len(distributed_series) - 1) / 2) * width
            bars = axis.bar(
                positions + offset,
                values,
                width,
                label=label,
                color=color,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=3,
                fontsize=10,
                rotation=90,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Parallel configuration\nM=32, H=4096, I=12288, BF16")
        axis.set_title(direction_label)
        axis.set_xticks(positions, labels)
    axes[0].set_ylabel("Median latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    if not platform_comparison:
        axes[0].legend(loc="upper left")
        figure.suptitle(
            "MI300X distributed FFN latency",
            y=0.99,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        axes[0].set_ylim(0.1, 30.0)
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=4,
        )
        figure.suptitle(
            "Distributed FFN latency: H100 CUDA vs MI300X Triton",
            y=0.99,
        )
        figure.text(
            0.5,
            0.01,
            "All four paths use the same M=32 workload, direction, and TP/CP/SP "
            "topology; no TP=1 latency is included.",
            ha="center",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    figure.savefig(output_directory / "distributed_ffn_overhead.png", dpi=180)
    plt.close(figure)


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "architecture": properties.gcnArchName,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "hip": torch.version.hip,
        "python": os.sys.version.split()[0],
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "single_gpu_speed_context": "Hugging Face Transformers Qwen3MLP, TP=1",
        "distributed_speed_comparison": "four same-topology H100/MI300X paths",
        "deterministic_compute": "ROCm-native Triton",
        "deterministic_transport": "fixed-tree HIP IPC with RCCL fallback on ROCm",
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", ""),
    }


def _validate_environment() -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this benchmark requires a ROCm PyTorch build")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("this benchmark requires eight visible ROCm GPUs")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("PyTorch RCCL/ProcessGroupNCCL support is unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pr325_rocm_mi300x"),
    )
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        choices=(2, 4, 8),
        default=(2, 4, 8),
        help="distributed world sizes to benchmark (default: 2 4 8)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--training-samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_environment()
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "environment": _environment(),
        "methodology": {
            "single_gpu_timing": "GPU events, median and p95",
            "distributed_timing": "synchronized wall clock, slowest rank/sample",
            "distributed_worker_cpu_affinity": "one NUMA-local CPU per GPU rank",
            "warmup": args.warmup,
            "samples": args.samples,
            "training_samples": args.training_samples,
            "operator_only": True,
            "triton_weight_layout": "packed_forward_cache_outside_timed_region",
            "tp1_forward_cache_bytes": _TP1_FORWARD_CACHE_BYTES,
        },
        "communication_contract": _COMMUNICATION_CONTRACT,
        "single_gpu": _single_gpu_benchmarks(
            warmup=args.warmup,
            samples=args.samples,
            training_samples=args.training_samples,
        ),
        "distributed_ffn": [],
    }
    torch.cuda.empty_cache()
    for world_size in args.world_sizes:
        result = _run_distributed_world(
            world_size,
            warmup=args.warmup,
            samples=args.samples,
            training_samples=args.training_samples,
        )
        payload["distributed_ffn"].extend(result["distributed_ffn"])
    comparison_payload = _load_cuda_cpu_comparison(args.output_dir)
    platform_comparison = _distributed_platform_comparison_rows(
        payload["distributed_ffn"], comparison_payload
    )
    if platform_comparison:
        payload["distributed_platform_comparison"] = {
            "source": "cuda_cpu_comparison.json",
            "contract": "same M=32 workload, direction, and TP/CP/SP topology",
            "rows": platform_comparison,
        }
    previous_comparison = _previous_deterministic_comparison_rows(payload["distributed_ffn"])
    if previous_comparison:
        payload["previous_deterministic_comparison"] = {
            "source": "previous checked MI300X benchmark before PR #357",
            "rows": previous_comparison,
        }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(payload, args.output_dir, comparison_payload)
    _write_figures(payload, args.output_dir, comparison_payload)
    print(json.dumps({"output_dir": str(args.output_dir), "status": "ok"}))


if __name__ == "__main__":
    main()
