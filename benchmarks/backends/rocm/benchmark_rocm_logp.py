# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""ROCm benchmark for the WS2 vocab-parallel logprob path (PR #328).

Operator-only: seeded logits, no checkpoint, tokenizer, or model server.

Single GPU (TP=1), Qwen3 vocabulary ``V=151936`` (64 tiles of 2374 columns):

- ``native``: ``torch.logsumexp`` + ``gather`` on FP32 logits (plain PyTorch,
  not batch-invariant by contract).
- ``ws1-pytorch`` / ``ws1-triton``: existing single-shard batch-invariant ops.
- ``ws2-reference``: ``pytorch-vocab-parallel-logp-ws2`` (PyTorch tile loop).
- ``ws2-rocm``: ``rocm-vocab-parallel-logp-ws2`` (HIP tile-stats kernel).

Plus a component table for the HIP ``deterministic_logp_tile_stats`` kernel
against the PyTorch ``_local_tile_stats`` loop it replaces.

Distributed (one process per GPU, RCCL via ProcessGroupNCCL):

- ``native``: Megatron-style vocab-parallel logprob (all-reduce MAX, all-reduce
  SUM of exp, all-reduce SUM of the owned target logit).
- ``ws2-reference`` and ``ws2-rocm``: the contract-aware WS2 operator with the
  fixed tile-order merge; CP ranks shard tokens and never join the merge.

Every path reports latency (GPU events on a single GPU; synchronized wall
clock and slowest rank per sample when distributed), peak device memory,
FP64 accuracy, repeat bitwise stability, and batch invariance.

Usage:
    python benchmarks/backends/rocm/benchmark_rocm_logp.py \
        --warmup 5 --samples 20 --training-samples 10 \
        --output-dir benchmarks/results/pr328_rocm_mi300x
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.backends.cuda.logprob.vocab_parallel_logp import (
    CudaVocabParallelLogprobOp,
    native_tile_stats_available,
)
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.backends.rocm.logprob.vocab_parallel_logp import RocmVocabParallelLogprobOp
from rl_engine.backends.shared.triton.logprob.vocab_parallel_logp import (
    TritonVocabParallelLogprobOp,
)
from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.reference.logprob.vocab_parallel_logp import (
    VocabParallelLogprobOp,
    _local_tile_stats,
)

REAL_VOCAB = 151936  # Qwen3 tokenizer/lm_head width; 151936 = 64 * 2374, so no padding.
NUM_TILES = 64
IGNORE_INDEX = -100
LOGIT_SCALE = 2.0
SINGLE_TOKENS = (1, 8, 32, 128, 512, 2048)
SINGLE_DTYPES = ("bf16", "fp32")
DISTRIBUTED_TOKENS = (256, 2048)
TOPOLOGIES = (
    ("tp2", 2, 1),
    ("tp4", 4, 1),
    ("tp8", 8, 1),
    ("tp2_cp2", 2, 2),
    ("tp4_cp2", 4, 2),
    ("tp2_cp4", 2, 4),
)
DISTRIBUTED_PATHS = ("native", "ws2-reference", "ws2-triton", "ws2-cuda", "ws2-rocm")
WS2_KERNEL_PATHS = ("ws2-triton", "ws2-cuda", "ws2-rocm")
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_SPAWN_TIMEOUT_S = 1800


# --------------------------------------------------------------------------- helpers


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
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
    actual_float = actual.detach().double()
    expected_float = expected.detach().double()
    denominator = torch.linalg.vector_norm(expected_float)
    if denominator.item() == 0.0:
        return float(torch.linalg.vector_norm(actual_float - expected_float).item())
    return float((torch.linalg.vector_norm(actual_float - expected_float) / denominator).item())


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().double() - expected.detach().double()
    return {
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "relative_l2": _relative_l2(actual, expected),
    }


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().contiguous().view(torch.int32)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _mismatch_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum().item())


def _gpu_event_samples(function: Callable[[], Any], *, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _host_wall_samples(function: Callable[[], Any], *, warmup: int, samples: int) -> list[float]:
    """Wall-clock timing for host execution, where CUDA events do not apply."""
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _timed_samples(
    function: Callable[[], Any], *, warmup: int, samples: int, device: torch.device
) -> list[float]:
    if device.type == "cuda":
        return _gpu_event_samples(function, warmup=warmup, samples=samples)
    return _host_wall_samples(function, warmup=warmup, samples=samples)


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _device_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()


def _rss_mib() -> float:
    with open("/proc/self/statm", "r", encoding="ascii") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _host_peak_rss_mib(function: Callable[[], Any]) -> float:
    """Peak resident-set increase during one host call, sampled from /proc.

    This is the closest host analogue of ``torch.cuda.max_memory_allocated``,
    but it is an RSS high-water delta rather than an allocator statistic: it
    includes caching-allocator reuse and page-level granularity, so it is an
    approximation and not directly comparable to the device figures.  A call
    served entirely from already-resident pages can legitimately report ~0.
    """
    gc.collect()
    baseline = _rss_mib()
    peak = baseline
    stop = threading.Event()

    def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _rss_mib())
            stop.wait(0.001)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        function()
    finally:
        stop.set()
        thread.join()
    return float(max(peak, _rss_mib()) - baseline)


def _peak_memory_mib(function: Callable[[], Any], device: torch.device) -> float:
    """Peak memory used by one call, above what was live before it."""
    if device.type != "cuda":
        return _host_peak_rss_mib(function)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    function()
    torch.cuda.synchronize()
    return float((torch.cuda.max_memory_allocated() - baseline) / (1024.0 * 1024.0))


def _seeded_logits(
    num_tokens: int, vocab: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identical FP32 logits, targets, and active mask on every rank.

    Generated on-device from a seeded CUDA generator so multi-GB inputs are not
    materialized on the host; the same seed yields identical values on every
    MI300X rank.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(num_tokens, vocab, generator=generator, device=device) * LOGIT_SCALE
    targets = torch.randint(0, vocab, (num_tokens,), generator=generator, device=device)
    active = (torch.arange(num_tokens, device=device) % 7) != 5
    return logits, targets, active


def _fp64_oracle(
    logits_fp32: torch.Tensor, targets: torch.Tensor, active: torch.Tensor, real_vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    z = logits_fp32[:, :real_vocab].double()
    lse = torch.logsumexp(z, dim=-1)
    safe = torch.where(active, targets, torch.zeros_like(targets))
    selected = z.gather(1, safe.unsqueeze(1)).squeeze(1)
    logp = torch.where(active, selected - lse, torch.zeros_like(lse))
    return logp, lse


def _contract(
    *,
    num_tokens: int,
    active: tuple[bool, ...],
    tp_rank: int,
    tp_world_size: int,
    bounds: tuple[tuple[int, int], ...],
    real_vocab: int,
    padded_vocab: int,
    dtype: str,
    cp_rank: int = 0,
    cp_world_size: int = 1,
) -> LogprobContract:
    return LogprobContract(
        role="train",
        dtype=dtype,
        mask=MaskSpec(num_tokens=num_tokens, active_mask=active),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            vocab_shard_bounds=bounds,
            real_vocab_size=real_vocab,
            padded_vocab_size=padded_vocab,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        ),
        reduction=ReductionSpec(),
    )


# --------------------------------------------------------------------------- single GPU


def _native_logp(
    logits: torch.Tensor, targets: torch.Tensor, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    z = logits.float()
    lse = torch.logsumexp(z, dim=-1)
    safe = torch.where(active, targets, torch.zeros_like(targets))
    selected = z.gather(1, safe.unsqueeze(1)).squeeze(1)
    logp = torch.where(active, selected - lse, torch.zeros_like(lse))
    return logp, lse


def _single_gpu_paths(
    device: torch.device,
) -> dict[str, tuple[Callable[..., Any], Callable[..., Any]]]:
    ws1_pytorch = NativeBatchInvariantLogpOp()
    ws2_reference = VocabParallelLogprobOp()

    def ws1_pytorch_fn(logits, targets, active, contract):
        ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
        return ws1_pytorch.forward_with_lse(logits, ignore_targets, IGNORE_INDEX, validate=False)

    def ws1_pytorch_train(logits, targets, active, contract):
        ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
        return ws1_pytorch.apply(logits, ignore_targets, IGNORE_INDEX, validate=False)

    def ws2_reference_fn(logits, targets, active, contract):
        return ws2_reference.apply(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
        )

    # path -> (forward returning (logp, lse), training forward returning logp with autograd)
    paths: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
        "native": (
            lambda logits, targets, active, contract: _native_logp(logits, targets, active),
            lambda logits, targets, active, contract: _native_logp(logits, targets, active)[0],
        ),
        "ws1-pytorch": (ws1_pytorch_fn, ws1_pytorch_train),
    }
    if device.type != "cuda":
        # Triton and the native extensions have no host backend; the reference
        # tile loop and the plain-PyTorch baseline are the whole CPU story.
        paths["ws2-reference"] = (ws2_reference_fn, lambda *a: ws2_reference_fn(*a)[0])
        return paths
    try:
        from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        ws1_triton = TritonBatchInvariantLogpOp()
        probe = torch.randn(2, 256, device=device, dtype=torch.bfloat16)
        ws1_triton.forward_with_lse(probe, torch.zeros(2, device=device, dtype=torch.long))
        torch.cuda.synchronize()

        def ws1_triton_fn(logits, targets, active, contract):
            ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
            return ws1_triton.forward_with_lse(logits, ignore_targets, IGNORE_INDEX)

        def ws1_triton_train(logits, targets, active, contract):
            ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
            return ws1_triton.apply(logits, ignore_targets, IGNORE_INDEX)

        paths["ws1-triton"] = (ws1_triton_fn, ws1_triton_train)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ws1-triton unavailable: {exc}")
    paths["ws2-reference"] = (
        ws2_reference_fn,
        lambda *a: ws2_reference_fn(*a)[0],
    )
    ws2_triton = TritonVocabParallelLogprobOp()

    def ws2_triton_fn(logits, targets, active, contract):
        return ws2_triton.apply(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
        )

    paths["ws2-triton"] = (ws2_triton_fn, lambda *a: ws2_triton_fn(*a)[0])

    if native_tile_stats_available():
        ws2_cuda = CudaVocabParallelLogprobOp()

        def ws2_cuda_fn(logits, targets, active, contract):
            return ws2_cuda.apply(
                logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
            )

        paths["ws2-cuda"] = (ws2_cuda_fn, lambda *a: ws2_cuda_fn(*a)[0])

    if torch.version.hip is not None:
        ws2_rocm = RocmVocabParallelLogprobOp()

        def ws2_rocm_fn(logits, targets, active, contract):
            return ws2_rocm.apply(
                logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
            )

        paths["ws2-rocm"] = (ws2_rocm_fn, lambda *a: ws2_rocm_fn(*a)[0])
    return paths


def _single_gpu_benchmarks(
    *,
    warmup: int,
    samples: int,
    training_samples: int,
    tokens: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    paths = _single_gpu_paths(device)
    bounds = ((0, REAL_VOCAB),)
    cases: list[dict[str, Any]] = []
    validate_overhead: list[dict[str, Any]] = []
    # The validate=True overhead is measured on the fastest native backend the
    # platform actually has: HIP on ROCm, the CUDA tile-stats kernel on CUDA,
    # and the reference tile loop on the host.
    if torch.version.hip is not None:
        validate_op: Any = RocmVocabParallelLogprobOp()
        validate_op_path = "ws2-rocm"
    elif device.type == "cuda" and native_tile_stats_available():
        validate_op = CudaVocabParallelLogprobOp()
        validate_op_path = "ws2-cuda"
    else:
        validate_op = VocabParallelLogprobOp()
        validate_op_path = "ws2-reference"

    for dtype_name in SINGLE_DTYPES:
        dtype = _DTYPES[dtype_name]
        for num_tokens in tokens:
            logits_fp32, targets, active = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=2026 + num_tokens, device=device
            )
            logits = logits_fp32.to(dtype).contiguous()
            oracle_logp, oracle_lse = _fp64_oracle(logits.float(), targets, active, REAL_VOCAB)
            logits_fp32 = None
            contract = _contract(
                num_tokens=num_tokens,
                active=tuple(bool(flag) for flag in active.tolist()),
                tp_rank=0,
                tp_world_size=1,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype=dtype_name,
            )
            row = min(3, num_tokens - 1)
            row_contract = _contract(
                num_tokens=1,
                active=(bool(active[row].item()),),
                tp_rank=0,
                tp_world_size=1,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype=dtype_name,
            )
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for path_name, (path, train_path) in paths.items():

                def forward():
                    return path(logits, targets, active, contract)

                def train_step():
                    leaf = logits.detach().clone().requires_grad_(True)
                    logp = train_path(leaf, targets, active, contract)
                    (logp * active).sum().backward()
                    return leaf.grad

                try:
                    first_logp, first_lse = forward()
                    second_logp, second_lse = forward()
                    row_logp, row_lse = path(
                        logits[row : row + 1].contiguous(),
                        targets[row : row + 1],
                        active[row : row + 1],
                        row_contract,
                    )
                    train_step()
                    _device_synchronize(device)
                except Exception as exc:
                    print(f"{path_name} failed for {dtype_name} M={num_tokens}: {exc}")
                    continue
                outputs[path_name] = (first_logp.detach(), first_lse.detach())
                forward_times = _timed_samples(
                    forward, warmup=warmup, samples=samples, device=device
                )
                train_times = _timed_samples(
                    train_step,
                    warmup=max(1, warmup // 2),
                    samples=training_samples,
                    device=device,
                )
                grad = train_step()
                _device_synchronize(device)
                active_idx = active.nonzero().squeeze(1)
                cases.append(
                    {
                        "dtype": dtype_name,
                        "tokens": num_tokens,
                        "path": path_name,
                        "forward": _summary_ms(forward_times),
                        "train_fwd_bwd": _summary_ms(train_times),
                        "forward_peak_mib": _peak_memory_mib(forward, device),
                        "train_peak_mib": _peak_memory_mib(train_step, device),
                        "logp_vs_fp64": _accuracy(first_logp[active_idx], oracle_logp[active_idx]),
                        "lse_vs_fp64": _accuracy(first_lse, oracle_lse),
                        "repeat_bitwise": _bitwise_equal(first_logp, second_logp)
                        and _bitwise_equal(first_lse, second_lse),
                        "batch_invariant": _bitwise_equal(row_logp[0], first_logp[row])
                        and _bitwise_equal(row_lse[0], first_lse[row]),
                        "grad_finite": bool(torch.isfinite(grad).all().item()),
                    }
                )
                print(
                    f"single {dtype_name} M={num_tokens:5d} {path_name:14s} "
                    f"fwd={cases[-1]['forward']['median_ms']:.4f}ms "
                    f"train={cases[-1]['train_fwd_bwd']['median_ms']:.4f}ms "
                    f"peak={cases[-1]['train_peak_mib']:.1f}MiB",
                    flush=True,
                )
            if "ws2-reference" in outputs:
                ref_logp, ref_lse = outputs["ws2-reference"]
                for kernel_path in WS2_KERNEL_PATHS:
                    if kernel_path not in outputs:
                        continue
                    k_logp, k_lse = outputs[kernel_path]
                    for case in cases:
                        if (
                            case["dtype"] == dtype_name
                            and case["tokens"] == num_tokens
                            and case["path"] == kernel_path
                        ):
                            case["mismatch_vs_reference"] = _mismatch_count(
                                k_logp, ref_logp
                            ) + _mismatch_count(k_lse, ref_lse)
                            case["rel_l2_vs_reference"] = max(
                                _relative_l2(k_logp, ref_logp), _relative_l2(k_lse, ref_lse)
                            )
            # validate=True production entry point overhead (host-side checks + .item() sync)
            if dtype_name == "bf16":

                def validated():
                    return validate_op.apply(
                        logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=True
                    )

                def unvalidated():
                    return validate_op.apply(
                        logits,
                        targets,
                        contract=contract,
                        num_vocab_tiles=NUM_TILES,
                        validate=False,
                    )

                validate_overhead.append(
                    {
                        "tokens": num_tokens,
                        "path": validate_op_path,
                        "validate_true": _summary_ms(
                            _timed_samples(validated, warmup=warmup, samples=samples, device=device)
                        ),
                        "validate_false": _summary_ms(
                            _timed_samples(
                                unvalidated, warmup=warmup, samples=samples, device=device
                            )
                        ),
                    }
                )
            logits = oracle_logp = oracle_lse = outputs = None
            _device_empty_cache(device)

    return {
        "cases": cases,
        "validate_overhead": validate_overhead,
        "paths": list(paths),
        "device": device.type,
    }


def _tile_stats_component(
    *, warmup: int, samples: int, tokens: tuple[int, ...], device: torch.device
) -> list[dict[str, Any]]:
    """Native tile-stats kernel versus the PyTorch tile loop it replaces.

    ``hip_deterministic_logp_tile_stats`` on ROCm, ``deterministic_logp_tile_stats``
    (``csrc/cuda/logprob/deterministic_logp_kernel.cu``) on CUDA.  The two kernels share the
    same fixed per-tile reduction contract, so the row means the same thing on
    either platform; the ``hip_*`` result keys are kept for schema stability.
    """
    if device.type != "cuda":
        return []
    rows: list[dict[str, Any]] = []
    tile = REAL_VOCAB // NUM_TILES
    for dtype_name in SINGLE_DTYPES:
        dtype = _DTYPES[dtype_name]
        for num_tokens in tokens:
            logits_fp32, _, _ = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=99 + num_tokens, device=device
            )
            logits = logits_fp32.to(dtype).contiguous()
            z32 = logits.float()
            logits_fp32 = None

            def pytorch_loop():
                return _local_tile_stats(z32, tile)

            tile_stats = getattr(
                _C, "hip_deterministic_logp_tile_stats", _C.deterministic_logp_tile_stats
            )
            kernel_symbol = (
                "hip_deterministic_logp_tile_stats"
                if hasattr(_C, "hip_deterministic_logp_tile_stats")
                else "deterministic_logp_tile_stats"
            )

            def hip_kernel_fp32():
                return tile_stats(z32, 0, REAL_VOCAB, NUM_TILES)

            def hip_kernel_input_dtype():
                return tile_stats(logits, 0, REAL_VOCAB, NUM_TILES)

            ref_m, ref_s = pytorch_loop()
            hip_m, hip_s = hip_kernel_fp32()
            hip_m2, hip_s2 = hip_kernel_fp32()
            rows.append(
                {
                    "dtype": dtype_name,
                    "tokens": num_tokens,
                    "kernel_symbol": kernel_symbol,
                    "pytorch_loop": _summary_ms(
                        _timed_samples(pytorch_loop, warmup=warmup, samples=samples, device=device)
                    ),
                    "hip_fp32_input": _summary_ms(
                        _timed_samples(
                            hip_kernel_fp32, warmup=warmup, samples=samples, device=device
                        )
                    ),
                    "hip_native_dtype_input": _summary_ms(
                        _timed_samples(
                            hip_kernel_input_dtype, warmup=warmup, samples=samples, device=device
                        )
                    ),
                    "pytorch_loop_peak_mib": _peak_memory_mib(pytorch_loop, device),
                    "hip_peak_mib": _peak_memory_mib(hip_kernel_fp32, device),
                    "max_bitwise": _bitwise_equal(hip_m, ref_m),
                    "sumexp_rel_l2": _relative_l2(hip_s, ref_s),
                    "sumexp_max_rel": float(
                        ((hip_s - ref_s).abs() / ref_s.abs().clamp_min(1e-30)).max().item()
                    ),
                    "repeat_bitwise": _bitwise_equal(hip_m, hip_m2)
                    and _bitwise_equal(hip_s, hip_s2),
                }
            )
            print(
                f"tile-stats {dtype_name} M={num_tokens:5d} "
                f"loop={rows[-1]['pytorch_loop']['median_ms']:.4f}ms "
                f"hip={rows[-1]['hip_fp32_input']['median_ms']:.4f}ms",
                flush=True,
            )
            logits = z32 = None
            _device_empty_cache(device)
    return rows


# --------------------------------------------------------------------------- distributed


class _NativeVocabParallelLogp(torch.autograd.Function):
    """Megatron-style vocab-parallel logprob with RCCL all-reduce, no fixed order."""

    @staticmethod
    def forward(ctx, local_logits, targets, active, vocab_start, group):
        z = local_logits.float()
        local_vocab = z.shape[1]
        global_max = z.max(dim=-1).values
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        sum_exp = (z - global_max.unsqueeze(1)).exp().sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)
        lse = global_max + sum_exp.log()
        owned = active & (targets >= vocab_start) & (targets < vocab_start + local_vocab)
        local_index = torch.where(owned, targets - vocab_start, torch.zeros_like(targets))
        selected = z.gather(1, local_index.unsqueeze(1)).squeeze(1) * owned
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
        logp = torch.where(active, selected - lse, torch.zeros_like(lse))
        ctx.save_for_backward(z, lse, local_index, owned, active)
        ctx.input_dtype = local_logits.dtype
        ctx.set_materialize_grads(False)
        return logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        z, lse, local_index, owned, active = ctx.saved_tensors
        probabilities = (z - lse.unsqueeze(1)).exp()
        scale = torch.zeros_like(lse)
        if grad_logp is not None:
            scale = scale - grad_logp * active
        if grad_lse is not None:
            scale = scale + grad_lse
        grad = probabilities * scale.unsqueeze(1)
        if grad_logp is not None:
            grad.scatter_add_(
                1, local_index.unsqueeze(1), (grad_logp * owned).unsqueeze(1).to(grad.dtype)
            )
        return grad.to(ctx.input_dtype), None, None, None, None


def _tp_bounds(tp_world_size: int) -> tuple[tuple[int, int], ...]:
    tile = REAL_VOCAB // NUM_TILES
    per_rank = NUM_TILES // tp_world_size
    return tuple(
        (rank * per_rank * tile, (rank + 1) * per_rank * tile) for rank in range(tp_world_size)
    )


def _token_bounds(num_tokens: int, cp_world_size: int) -> tuple[tuple[int, int], ...]:
    quotient, remainder = divmod(num_tokens, cp_world_size)
    bounds, cursor = [], 0
    for cp_rank in range(cp_world_size):
        count = quotient + int(cp_rank < remainder)
        bounds.append((cursor, cursor + count))
        cursor += count
    return tuple(bounds)


def _distributed_wall_samples(
    function: Callable[[], Any], *, warmup: int, samples: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    dist.barrier()
    return timings


def _slowest_rank_summary(local_timings: list[float]) -> dict[str, float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_timings)
    slowest = [
        max(float(rank_timings[index]) for rank_timings in gathered if rank_timings is not None)
        for index in range(len(local_timings))
    ]
    return _summary_ms(slowest)


def _all_max(value: float) -> float:
    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, float(value))
    return max(float(item) for item in gathered if item is not None)


def _all_all(flag: bool) -> bool:
    gathered: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, bool(flag))
    return all(bool(item) for item in gathered)


def _all_sum(value: int) -> int:
    gathered: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, int(value))
    return sum(int(item) for item in gathered if item is not None)


def _tp_replicated(
    logp: torch.Tensor, lse: torch.Tensor, tp_group: Any, tp_world_size: int
) -> bool:
    if tp_world_size == 1:
        return True
    payload = torch.stack([_bits(logp), _bits(lse)])
    gathered = [torch.empty_like(payload) for _ in range(tp_world_size)]
    dist.all_gather(gathered, payload, group=tp_group)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def _distributed_worker(
    rank: int,
    world_size: int,
    init_method: str,
    topology: tuple[str, int, int],
    tokens_list: tuple[int, ...],
    config: dict[str, Any],
    result_queue: Any,
) -> None:
    name, tp_world_size, cp_world_size = topology
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
        cp_rank, tp_rank = divmod(rank, tp_world_size)
        tp_group = None
        for group_cp_rank in range(cp_world_size):
            ranks = list(range(group_cp_rank * tp_world_size, (group_cp_rank + 1) * tp_world_size))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group
        bounds = _tp_bounds(tp_world_size)
        vocab_start, vocab_end = bounds[tp_rank]
        ops: dict[str, Any] = {
            "ws2-reference": VocabParallelLogprobOp(),
            "ws2-triton": TritonVocabParallelLogprobOp(),
        }
        if native_tile_stats_available():
            ops["ws2-cuda"] = CudaVocabParallelLogprobOp()
        if torch.version.hip is not None:
            ops["ws2-rocm"] = RocmVocabParallelLogprobOp()
        results: list[dict[str, Any]] = []

        for num_tokens in tokens_list:
            full_logits, full_targets, full_active = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=4100 + num_tokens, device=device
            )
            token_start, token_end = _token_bounds(num_tokens, cp_world_size)[cp_rank]
            local_tokens = token_end - token_start
            targets = full_targets[token_start:token_end].contiguous()
            active = full_active[token_start:token_end].contiguous()
            local_fp32 = full_logits[token_start:token_end]
            shard = local_fp32[:, vocab_start:vocab_end].to(torch.bfloat16).contiguous()
            # The FP64 oracle sees the BF16-rounded logits every path actually consumes.
            oracle_logp, oracle_lse = _fp64_oracle(
                local_fp32.to(torch.bfloat16).float(), targets, active, REAL_VOCAB
            )
            full_logits = local_fp32 = None
            torch.cuda.empty_cache()
            contract = _contract(
                num_tokens=local_tokens,
                active=tuple(bool(flag) for flag in active.tolist()),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype="bf16",
                cp_rank=cp_rank,
                cp_world_size=cp_world_size,
            )
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for path_name in DISTRIBUTED_PATHS:
                if path_name != "native" and path_name not in ops:
                    continue
                if path_name == "native":

                    def forward(x=shard):
                        return _NativeVocabParallelLogp.apply(
                            x, targets, active, vocab_start, tp_group
                        )

                else:
                    op = ops[path_name]

                    def forward(x=shard, op=op):
                        return op.apply(
                            x,
                            targets,
                            contract=contract,
                            tp_group=tp_group,
                            num_vocab_tiles=NUM_TILES,
                            validate=False,
                        )

                def train_step():
                    leaf = shard.detach().clone().requires_grad_(True)
                    logp, _ = forward(leaf)
                    (logp * active).sum().backward()
                    return leaf.grad

                first_logp, first_lse = forward()
                second_logp, second_lse = forward()
                torch.cuda.synchronize()
                outputs[path_name] = (first_logp.detach(), first_lse.detach())
                forward_times = _distributed_wall_samples(
                    forward, warmup=config["warmup"], samples=config["samples"]
                )
                train_times = _distributed_wall_samples(
                    train_step,
                    warmup=max(1, config["warmup"] // 2),
                    samples=config["training_samples"],
                )
                grad = train_step()
                torch.cuda.synchronize()
                active_idx = active.nonzero().squeeze(1)
                logp_acc = _accuracy(first_logp[active_idx], oracle_logp[active_idx])
                lse_acc = _accuracy(first_lse, oracle_lse)
                entry = {
                    "topology": name,
                    "tp": tp_world_size,
                    "cp": cp_world_size,
                    "tokens": num_tokens,
                    "tokens_per_cp_rank": local_tokens,
                    "local_vocab": vocab_end - vocab_start,
                    "path": path_name,
                    "forward": _slowest_rank_summary(forward_times),
                    "train_fwd_bwd": _slowest_rank_summary(train_times),
                    "forward_peak_mib": _all_max(_peak_memory_mib(forward, device)),
                    "train_peak_mib": _all_max(_peak_memory_mib(train_step, device)),
                    "logp_vs_fp64_max_abs": _all_max(logp_acc["max_abs"]),
                    "logp_vs_fp64_rel_l2": _all_max(logp_acc["relative_l2"]),
                    "lse_vs_fp64_max_abs": _all_max(lse_acc["max_abs"]),
                    "tp_replicated": _all_all(
                        _tp_replicated(first_logp, first_lse, tp_group, tp_world_size)
                    ),
                    "repeat_bitwise": _all_all(
                        _bitwise_equal(first_logp, second_logp)
                        and _bitwise_equal(first_lse, second_lse)
                    ),
                    "grad_finite": _all_all(bool(torch.isfinite(grad).all().item())),
                }
                results.append(entry)
                if rank == 0:
                    print(
                        f"dist {name} M={num_tokens:5d} {path_name:14s} "
                        f"fwd={entry['forward']['median_ms']:.4f}ms "
                        f"train={entry['train_fwd_bwd']['median_ms']:.4f}ms "
                        f"peak={entry['train_peak_mib']:.1f}MiB",
                        flush=True,
                    )
            ref_logp, ref_lse = outputs["ws2-reference"]
            for other_path in ("native",) + WS2_KERNEL_PATHS:
                if other_path not in outputs:
                    continue
                o_logp, o_lse = outputs[other_path]
                mismatch = _mismatch_count(o_logp, ref_logp) + _mismatch_count(o_lse, ref_lse)
                rel = max(_relative_l2(o_logp, ref_logp), _relative_l2(o_lse, ref_lse))
                # Count once per TP group (outputs are replicated inside the group).
                mismatch_total = _all_sum(mismatch if tp_rank == 0 else 0)
                rel_max = _all_max(rel)
                for entry in results:
                    if entry["tokens"] == num_tokens and entry["path"] == other_path:
                        entry["mismatch_vs_reference"] = mismatch_total
                        entry["rel_l2_vs_reference"] = rel_max
            shard = oracle_logp = oracle_lse = outputs = None
            torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            result_queue.put({"ok": True, "results": results})
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_world(
    topology: tuple[str, int, int], tokens_list: tuple[int, ...], config: dict[str, Any]
) -> list[dict[str, Any]]:
    name, tp_world_size, cp_world_size = topology
    world_size = tp_world_size * cp_world_size
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "rccl_init").as_uri()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_worker,
                args=(rank, world_size, init_method, topology, tokens_list, config, result_queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        try:
            payload = result_queue.get(timeout=_SPAWN_TIMEOUT_S)
        finally:
            for process in processes:
                process.join(timeout=120)
                if process.is_alive():
                    process.terminate()
    if not payload.get("ok"):
        raise RuntimeError(f"{name} rank {payload.get('rank')} failed:\n{payload.get('traceback')}")
    return payload["results"]


# --------------------------------------------------------------------------- report


PATH_DESCRIPTIONS = {
    "native": (
        "`torch.logsumexp` + `gather` on FP32 logits (plain PyTorch, not batch-invariant "
        "by contract)"
    ),
    "ws1-pytorch": (
        "`pytorch-batch-invariant-logp-ws1`, the single-shard batch-invariant PyTorch op"
    ),
    "ws1-triton": (
        "`triton-batch-invariant-logp-ws1`, the single-shard Triton online-softmax op; it has "
        "no vocab-parallel (TP) path, so it appears only in the single-GPU results"
    ),
    "ws2-reference": (
        "`pytorch-vocab-parallel-logp-ws2`, the WS2 vocab-parallel reference operator: a PyTorch "
        "tile loop for the per-tile FP32 `(max, sumexp)` partials, all-gather of the partials, "
        "fixed global tile-order merge, and a PyTorch autograd backward"
    ),
    "ws2-triton": (
        "`triton-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two "
        "Triton kernels (tile statistics read from the stored shard, fused backward); one "
        "source for CUDA and ROCm"
    ),
    "ws2-cuda": (
        "`cuda-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two "
        "CUDA kernels from `csrc/cuda/logprob/deterministic_logp_kernel.cu`: "
        "`deterministic_logp_tile_stats` reads the stored BF16/FP16/FP32 shard directly "
        "(16-byte vector loads, no FP32 copy) and `deterministic_logp_backward` produces the "
        "gradient in one fused pass"
    ),
    "ws2-rocm": (
        "`rocm-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two HIP "
        "kernels: `hip_deterministic_logp_tile_stats` reads the stored BF16/FP16/FP32 shard "
        "directly (8-element vector loads, no FP32 copy) and `hip_deterministic_logp_backward` "
        "produces the gradient in one fused pass"
    ),
}
DISTRIBUTED_DESCRIPTIONS = {
    "native": (
        "`native` is a Megatron-style vocab-parallel logprob using RCCL all-reduce (MAX, SUM of "
        "exp, SUM of the owned target logit) through ProcessGroupNCCL"
    ),
    "ws2-reference": (
        "the WS2 operators all-gather per-tile `(max, sumexp)` partials over RCCL and merge them "
        "in fixed global tile order; CP ranks shard tokens and never enter the merge"
    ),
    "ws2-triton": "",
    "ws2-cuda": "",
    "ws2-rocm": "",
}


class ReportStyle:
    """Which measured paths a report shows, how it names them, and what it compares against."""

    def __init__(
        self,
        *,
        paths: tuple[str, ...],
        names: dict[str, str],
        baseline: str,
        table_tokens: tuple[int, ...] | None,
        show_tuning_baseline: bool,
        command: str,
    ) -> None:
        if baseline not in paths:
            raise ValueError(
                f"report baseline {baseline!r} is not among the reported paths {paths}"
            )
        self.paths = paths
        self.names = names
        self.baseline = baseline
        self.table_tokens = table_tokens
        self.show_tuning_baseline = show_tuning_baseline
        self.command = command

    def name(self, path: str) -> str:
        return self.names.get(path, path)

    def keep_tokens(self, tokens: int) -> bool:
        return self.table_tokens is None or tokens in self.table_tokens


def _fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.2f}×"


def _median_ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None, key: str
) -> str:
    if numerator is None or denominator is None:
        return "n/a"
    return _fmt_ratio(numerator[key]["median_ms"], denominator[key]["median_ms"])


def _field_ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None, key: str
) -> str:
    if numerator is None or denominator is None:
        return "n/a"
    return _fmt_ratio(float(numerator[key]), float(denominator[key]))


def _range_text(values: list[float], fmt: str = "{:.1f}") -> str:
    if not values:
        return "n/a"
    low, high = fmt.format(min(values)), fmt.format(max(values))
    if low == high:
        return low
    return f"{low}-{high}"


def _lookup(cases: list[dict[str, Any]], **match: Any) -> dict[str, Any] | None:
    for case in cases:
        if all(case.get(key) == value for key, value in match.items()):
            return case
    return None


def _yes(flag: Any) -> str:
    return "yes" if flag else "no"


def _default_compare_label(payload: dict[str, Any]) -> str:
    env = payload.get("environment", {})
    if env.get("device") == "cpu":
        return "cpu"
    gpu = str(env.get("gpu", "device"))
    for token in ("MI300X", "MI250X", "MI325X", "H100", "H200", "A100", "B200"):
        if token.lower() in gpu.lower():
            return token.lower()
    return gpu.split()[0].lower() if gpu else "device"


def _report_platforms(
    payload: dict[str, Any], comparisons: list[str], label: str | None
) -> list[tuple[str, dict[str, Any]]]:
    """This run first, then every ``--compare-with`` platform, in the given order.

    Each entry feeds the same tables, so one report can carry several devices.
    """

    platforms: list[tuple[str, dict[str, Any]]] = [
        (label or _default_compare_label(payload), payload)
    ]
    for spec in comparisons:
        if "=" not in spec:
            raise ValueError(f"--compare-with must be LABEL=path/to/results.json, got {spec!r}")
        other_label, _, other_path = spec.partition("=")
        platforms.append(
            (other_label.strip(), json.loads(Path(other_path).read_text(encoding="utf-8")))
        )
    return platforms


def _write_report(
    payload: dict[str, Any],
    output_directory: Path,
    style: ReportStyle,
    platforms: list[tuple[str, dict[str, Any]]] | None = None,
) -> None:
    env = payload["environment"]
    cfg = payload["config"]
    if platforms is None:
        platforms = _report_platforms(payload, [], None)
    multi = len(platforms) > 1
    plat_head = "Platform | " if multi else ""
    plat_div = "---|" if multi else ""

    def plat_cell(label: str) -> str:
        return f"{label} | " if multi else ""

    def tagged(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return [dict(row, platform=label) for row in rows]

    single = [
        case
        for label, pl in platforms
        for case in tagged(pl["single_gpu"]["cases"], label)
        if case["path"] in style.paths
    ]
    component = [
        row for label, pl in platforms for row in tagged(pl.get("tile_stats_component", []), label)
    ]
    distributed = [
        row
        for label, pl in platforms
        for row in tagged(pl.get("distributed", []), label)
        if row["path"] in style.paths
    ]
    base = style.baseline
    base_name = style.name(base)
    others = [path for path in style.paths if path != base]
    lines: list[str] = []
    add = lines.append

    device_type = env.get("device", "cuda")
    is_host = device_type == "cpu"
    platform_label = (
        "host (CPU)" if is_host else ("ROCm" if env.get("hip") not in (None, "None") else "CUDA")
    )
    if multi:
        add("# PR #328 vocab-parallel logprob performance analysis")
    else:
        add(f"# PR #328 {platform_label} vocab-parallel logprob performance analysis")
    add("")
    add("> Operator-only benchmark. No model checkpoint or serving engine was used.")
    if multi:
        add("")
        add(
            "> Every platform below ran this same harness, so the seeded logits, the "
            f"`V={REAL_VOCAB}` / {NUM_TILES}-tile split, and the FP64 oracle are identical and "
            "only the device and backend differ. Backends are not available everywhere: "
            "`ws2-rocm` is ROCm-only, `ws2-cuda` is CUDA-only, `ws2-triton` compiles from one "
            "source on both, and the PyTorch paths are the only ones that also run on the host. "
            "A missing row means the backend cannot exist on that platform, not that it failed."
        )
    add("")
    add("## Environment")
    add("")
    if multi:
        keys = sorted({key for _, pl in platforms for key in pl["environment"]})
        add("| Item | " + " | ".join(label for label, _ in platforms) + " |")
        add("|---|" + "---|" * len(platforms))
        for key in keys:
            values = " | ".join(str(pl["environment"].get(key, "n/a")) for _, pl in platforms)
            add(f"| {key} | {values} |")
    else:
        add("| Item | Value |")
        add("|---|---|")
        for key in sorted(env):
            add(f"| {key} | {env[key]} |")
    add("")
    add("## Methodology")
    add("")
    add(
        f"- Qwen3 vocabulary `V={REAL_VOCAB}` split into {NUM_TILES} tiles of "
        f"{REAL_VOCAB // NUM_TILES} columns; seeded logits (`randn * {LOGIT_SCALE}`), "
        "random targets, every seventh token inactive."
    )
    add("- Measured paths:")
    for path in style.paths:
        add(f"  - `{style.name(path)}`: {PATH_DESCRIPTIONS[path]}.")
    dist_notes = [
        DISTRIBUTED_DESCRIPTIONS.get(p, "") for p in style.paths if DISTRIBUTED_DESCRIPTIONS.get(p)
    ]
    if dist_notes and distributed:
        collective = "/".join(
            sorted(
                {
                    "RCCL" if pl["environment"].get("hip") not in (None, "None") else "NCCL"
                    for _, pl in platforms
                    if pl.get("distributed")
                }
            )
        )
        add(
            "- Distributed: one process per GPU; "
            + "; ".join(note.replace("RCCL", collective) for note in dist_notes)
            + "."
        )
    add(
        "- Forward returns the selected-token logprob and the vocabulary LSE; forward+backward "
        "computes `grad_logits` for `sum(active * logp)`. The WS2 operators run with "
        "`validate=False`; the `validate=True` production entry point is measured separately."
    )
    if is_host:
        add(
            "- Timing: `time.perf_counter` wall clock, median and p95. Peak memory is the "
            "per-call increase in resident set size sampled from `/proc/self/statm`, which is "
            "an allocator-inclusive high-water approximation rather than the exact tensor "
            "bytes reported by `torch.cuda.max_memory_allocated` on device runs; treat the "
            "host memory column as indicative only."
        )
    else:
        add(
            "- Single-GPU timing: GPU events, median and p95. Distributed timing: synchronized "
            "wall clock, slowest rank per sample. Peak memory is the per-call increase in "
            "`torch.cuda.max_memory_allocated` (distributed: max over ranks)."
        )
    add(
        "- Accuracy is against an FP64 `logsumexp` of the same (BF16-rounded) logits. Repeat = "
        "two identical calls are bitwise equal; batch-invariant = a row computed alone is "
        "bitwise equal to the same row inside the batch; TP-replicated = every TP rank holds "
        "identical bits."
    )
    add(
        f"- {cfg['warmup']} warmups, {cfg['samples']} measured forward samples, "
        f"{cfg['training_samples']} measured forward+backward samples. Raw medians, p95, "
        "minimum, maximum, and every measured path are in `results.json`."
    )
    if style.table_tokens is not None:
        add(
            "- Tables show "
            + ", ".join(f"{t}" for t in style.table_tokens)
            + " tokens; the figures cover the full token sweep."
        )
    add("")
    add("Reproduce this report from the repository root:")
    add("")
    add("```bash")
    add(style.command)
    add("```")
    add("")

    # ---- key findings
    all_single, all_distributed, all_component = single, distributed, component
    add("## Key findings")
    add("")
    for _plat, _payload in platforms:
        prefix = f"{_plat} " if multi else ""
        single = [c for c in all_single if c["platform"] == _plat]
        distributed = [d for d in all_distributed if d["platform"] == _plat]
        component = [r for r in all_component if r["platform"] == _plat]
        _env = _payload["environment"]
        single_label = "Host" if _env.get("device") == "cpu" else "Single GPU"
        _tile_symbol = (
            "hip_deterministic_logp_tile_stats"
            if _env.get("hip") not in (None, "None")
            else "deterministic_logp_tile_stats"
        )
        if component:
            _tile_symbol = component[0].get("kernel_symbol", _tile_symbol)
        for path in others:
            name = style.name(path)
            speed_fwd, speed_train, mem = [], [], []
            for dtype_name in SINGLE_DTYPES:
                for case in single:
                    if case["dtype"] != dtype_name or case["path"] != path:
                        continue
                    if not style.keep_tokens(case["tokens"]):
                        continue
                    ref = _lookup(single, dtype=dtype_name, tokens=case["tokens"], path=base)
                    if ref is None:
                        continue
                    speed_fwd.append(ref["forward"]["median_ms"] / case["forward"]["median_ms"])
                    speed_train.append(
                        ref["train_fwd_bwd"]["median_ms"] / case["train_fwd_bwd"]["median_ms"]
                    )
                    mem.append(case["train_peak_mib"] / max(ref["train_peak_mib"], 1e-6))
            if speed_fwd:
                add(
                    f"- {prefix}{single_label}: `{name}` is "
                    f"{_range_text(speed_fwd, '{:.2f}')}x faster than "
                    f"`{base_name}` in forward and {_range_text(speed_train, '{:.2f}')}x in "
                    f"forward+backward, with {_range_text(mem, '{:.2f}')}x its peak memory."
                )
            d_fwd, d_train, d_mem, d_abs = [], [], [], []
            for d in distributed:
                if d["path"] != path or not style.keep_tokens(d["tokens"]):
                    continue
                ref = _lookup(distributed, topology=d["topology"], tokens=d["tokens"], path=base)
                if ref is None:
                    continue
                d_fwd.append(ref["forward"]["median_ms"] / d["forward"]["median_ms"])
                d_train.append(ref["train_fwd_bwd"]["median_ms"] / d["train_fwd_bwd"]["median_ms"])
                d_mem.append(d["train_peak_mib"] / max(ref["train_peak_mib"], 1e-6))
                d_abs.append(d["forward"]["median_ms"])
            if d_fwd:
                add(
                    f"- {prefix}Distributed: `{name}` is "
                    f"{_range_text(d_fwd, '{:.2f}')}x faster than "
                    f"`{base_name}` in forward and {_range_text(d_train, '{:.2f}')}x in "
                    f"forward+backward across {len({d['topology'] for d in distributed})} TP/CP "
                    f"topologies, at {_range_text(d_mem, '{:.2f}')}x the per-rank peak memory "
                    f"(absolute forward {_range_text(d_abs, '{:.3f}')} ms)."
                )
        for kernel_path in WS2_KERNEL_PATHS:
            if kernel_path not in style.paths or "ws1-triton" not in style.paths:
                continue
            ratios_fwd, ratios_train = [], []
            for dtype_name in SINGLE_DTYPES:
                for case in single:
                    if case["path"] != kernel_path or case["dtype"] != dtype_name:
                        continue
                    if not style.keep_tokens(case["tokens"]):
                        continue
                    tri = _lookup(
                        single, dtype=dtype_name, tokens=case["tokens"], path="ws1-triton"
                    )
                    if tri is None:
                        continue
                    ratios_fwd.append(case["forward"]["median_ms"] / tri["forward"]["median_ms"])
                    ratios_train.append(
                        case["train_fwd_bwd"]["median_ms"] / tri["train_fwd_bwd"]["median_ms"]
                    )
            if ratios_fwd:
                add(
                    f"- {prefix}`{style.name(kernel_path)}` runs at "
                    f"{_range_text(ratios_fwd, '{:.2f}')}x the "
                    f"latency of `{style.name('ws1-triton')}` in forward and "
                    f"{_range_text(ratios_train, '{:.2f}')}x in forward+backward with the same "
                    "peak memory, while carrying the vocab-parallel contract (tile partials, "
                    "all-gather, fixed tile-order merge, vocab-domain LSE export) that the "
                    "single-shard Triton op does not provide; the gap is the operator's fixed "
                    "Python/launch floor, not the kernels."
                )
        if component:
            comp_speed = [
                r["pytorch_loop"]["median_ms"] / r["hip_fp32_input"]["median_ms"]
                for r in component
                if style.keep_tokens(r["tokens"])
            ]
            comp_mem = [
                r["pytorch_loop_peak_mib"] / max(r["hip_peak_mib"], 1e-6)
                for r in component
                if style.keep_tokens(r["tokens"])
            ]
            add(
                f"- {prefix}The `{_tile_symbol}` kernel alone is "
                f"{_range_text(comp_speed, '{:.1f}')}x faster than the PyTorch tile loop and "
                f"allocates {_range_text(comp_mem, '{:.0f}')}x less transient memory (it writes "
                f"only the `[tokens, {NUM_TILES}]` FP32 partials)."
            )
        for kernel_path in WS2_KERNEL_PATHS:
            if kernel_path not in style.paths or "ws2-reference" not in style.paths:
                continue
            mism = [
                c.get("mismatch_vs_reference")
                for c in single
                if c["path"] == kernel_path and c.get("mismatch_vs_reference") is not None
            ]
            relr = [c.get("rel_l2_vs_reference", 0.0) for c in single if c["path"] == kernel_path]
            if not mism:
                continue
            add(
                f"- {prefix}`{style.name(kernel_path)}` vs "
                f"`{style.name('ws2-reference')}`: tile maxima are "
                "bitwise equal; sumexp partials differ only by FP32 summation order, so final "
                "outputs differ in "
                f"{_range_text([float(m) for m in mism], '{:.0f}')} elements per case with "
                f"relative-L2 {_range_text(relr, '{:.1e}')}. Both paths are equally close to FP64."
            )
        ws2 = [c for c in single if c["path"].startswith("ws2")]
        if ws2:
            add(
                f"- {prefix}Repeat bitwise: "
                f"{_yes(all(c['repeat_bitwise'] for c in ws2))}; batch-invariant: "
                f"{_yes(all(c['batch_invariant'] for c in ws2))}; all gradients finite: "
                f"{_yes(all(c['grad_finite'] for c in ws2))}."
            )
        ws2_dist = [d for d in distributed if d["path"].startswith("ws2")]
        if ws2_dist:
            add(
                f"- {prefix}Distributed: TP-replicated and repeat bitwise on every topology: "
                f"{_yes(all(d['tp_replicated'] and d['repeat_bitwise'] for d in ws2_dist))}."
            )
    add("")

    single, distributed, component = all_single, all_distributed, all_component

    # ---- single GPU tables
    for dtype_name in SINGLE_DTYPES:
        rows = [c for c in single if c["dtype"] == dtype_name and style.keep_tokens(c["tokens"])]
        if not rows:
            continue
        add(f"## Single-GPU logprob ({dtype_name.upper()} logits, V={REAL_VOCAB})")
        add("")
        add("### Forward")
        add("")
        add(
            f"| Tokens | {plat_head}Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | "
            "Peak MiB | logp max-abs vs FP64 | LSE max-abs vs FP64 | Repeat | Batch-inv |"
        )
        add("|---:|" + plat_div + "---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
        for tokens in sorted({c["tokens"] for c in rows}):
            for label, _ in platforms:
                # Speedups are always against that platform's own baseline.
                ref = _lookup(single, dtype=dtype_name, tokens=tokens, path=base, platform=label)
                for path in style.paths:
                    case = _lookup(rows, tokens=tokens, path=path, platform=label)
                    if case is None:
                        continue
                    add(
                        f"| {tokens} | {plat_cell(label)}{style.name(path)} | "
                        f"{case['forward']['median_ms']:.4f} | "
                        f"{case['forward']['p95_ms']:.4f} | "
                        f"{_median_ratio(ref, case, 'forward')} | "
                        f"{case['forward_peak_mib']:.1f} | "
                        f"{case['logp_vs_fp64']['max_abs']:.3e} | "
                        f"{case['lse_vs_fp64']['max_abs']:.3e} | "
                        f"{_yes(case['repeat_bitwise'])} | "
                        f"{_yes(case['batch_invariant'])} |"
                    )
        add("")
        add("### Forward+backward")
        add("")
        add(
            f"| Tokens | {plat_head}Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | "
            f"Peak MiB | Memory vs {base_name} | Grad finite |"
        )
        add("|---:|" + plat_div + "---|---:|---:|---:|---:|---:|:---:|")
        for tokens in sorted({c["tokens"] for c in rows}):
            for label, _ in platforms:
                ref = _lookup(single, dtype=dtype_name, tokens=tokens, path=base, platform=label)
                for path in style.paths:
                    case = _lookup(rows, tokens=tokens, path=path, platform=label)
                    if case is None:
                        continue
                    add(
                        f"| {tokens} | {plat_cell(label)}{style.name(path)} | "
                        f"{case['train_fwd_bwd']['median_ms']:.4f} | "
                        f"{case['train_fwd_bwd']['p95_ms']:.4f} | "
                        f"{_median_ratio(ref, case, 'train_fwd_bwd')} | "
                        f"{case['train_peak_mib']:.1f} | "
                        f"{_field_ratio(case, ref, 'train_peak_mib')} | "
                        f"{_yes(case['grad_finite'])} |"
                    )
        add("")
        kernel_paths = [p for p in WS2_KERNEL_PATHS if p in style.paths]
        if kernel_paths and "ws2-reference" in style.paths:
            add(f"### Numerics versus `{style.name('ws2-reference')}`")
            add("")
            add(f"| Tokens | {plat_head}Path | Mismatched elements (logp+LSE) | Relative L2 |")
            add("|---:|" + plat_div + "---|---:|---:|")
            for tokens in sorted({c["tokens"] for c in rows}):
                for label, _ in platforms:
                    for kernel_path in kernel_paths:
                        case = _lookup(rows, tokens=tokens, path=kernel_path, platform=label)
                        if case is None:
                            continue
                        add(
                            f"| {tokens} | {plat_cell(label)}{style.name(kernel_path)} | "
                            f"{case.get('mismatch_vs_reference', 'n/a')} | "
                            f"{case.get('rel_l2_vs_reference', float('nan')):.3e} |"
                        )
            add("")

    overhead_rows: list[tuple[str, str, dict[str, Any]]] = []
    for label, pl in platforms:
        for row in pl["single_gpu"].get("validate_overhead", []):
            if not style.keep_tokens(row["tokens"]):
                continue
            row_path = row.get("path", "ws2-rocm")
            if row_path in style.paths:
                overhead_rows.append((label, row_path, row))
    if overhead_rows:
        measured = sorted({style.name(path) for _, path, _ in overhead_rows})
        add(f"### `validate=True` production entry point ({', '.join(measured)}, BF16)")
        add("")
        add(f"| Tokens | {plat_head}Path | validate=False (ms) | validate=True (ms) | Overhead |")
        add("|---:|" + plat_div + "---|---:|---:|---:|")
        for label, row_path, row in overhead_rows:
            overhead_ratio = _fmt_ratio(
                row["validate_true"]["median_ms"], row["validate_false"]["median_ms"]
            )
            add(
                f"| {row['tokens']} | {plat_cell(label)}{style.name(row_path)} | "
                f"{row['validate_false']['median_ms']:.4f} | "
                f"{row['validate_true']['median_ms']:.4f} | {overhead_ratio} |"
            )
        add("")
        add(
            "`validate=True` adds host-side target-range checks and a non-finite LSE check that "
            "synchronizes the stream; the cost is a fixed per-call overhead."
        )
        add("")

    # ---- tile-stats component
    plat_order = {label: index for index, (label, _) in enumerate(platforms)}
    comp_rows = [r for r in component if style.keep_tokens(r["tokens"])]
    comp_rows.sort(key=lambda r: (r["dtype"], r["tokens"], plat_order.get(r["platform"], 99)))
    if comp_rows:
        add("## Tile-stats kernel")
        add("")

        def _symbol(row: dict[str, Any]) -> str:
            # results.json written before kernel_symbol existed: infer from the platform.
            hip_build = dict(platforms)[row["platform"]]["environment"].get("hip") not in (
                None,
                "None",
            )
            default = (
                "hip_deterministic_logp_tile_stats"
                if hip_build
                else "deterministic_logp_tile_stats"
            )
            return row.get("kernel_symbol", default)

        symbols = sorted({_symbol(row) for row in comp_rows})
        add(
            f"{', '.join(f'`{s}`' for s in symbols)} computes the per-row, per-tile FP32 "
            "`(max, sumexp)` partials that the operator all-gathers and merges; the PyTorch tile "
            f"loop is what `{style.name('ws2-reference')}` uses for the same step. Tile maxima are "
            "bitwise equal; sums differ only by FP32 summation order."
        )
        add("")
        add(
            f"| Logits dtype | Tokens | {plat_head}Kernel | PyTorch tile loop (ms) | "
            "Kernel on FP32 (ms) | Kernel on stored dtype (ms) | Speedup | Loop peak MiB | "
            "Kernel peak MiB | Max bitwise | sumexp max rel | Repeat |"
        )
        add("|---|---:|" + plat_div + "---|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|")
        for row in comp_rows:
            kernel_speedup = _fmt_ratio(
                row["pytorch_loop"]["median_ms"], row["hip_fp32_input"]["median_ms"]
            )
            add(
                f"| {row['dtype']} | {row['tokens']} | {plat_cell(row['platform'])}"
                f"`{_symbol(row)}` | {row['pytorch_loop']['median_ms']:.4f} | "
                f"{row['hip_fp32_input']['median_ms']:.4f} | "
                f"{row['hip_native_dtype_input']['median_ms']:.4f} | "
                f"{kernel_speedup} | "
                f"{row['pytorch_loop_peak_mib']:.1f} | {row['hip_peak_mib']:.1f} | "
                f"{_yes(row['max_bitwise'])} | {row['sumexp_max_rel']:.2e} | "
                f"{_yes(row['repeat_bitwise'])} |"
            )
        add("")

    # ---- distributed
    topo_order = {name: index for index, (name, _, _) in enumerate(TOPOLOGIES)}
    path_order = {path: index for index, path in enumerate(style.paths)}
    dist_rows = [d for d in distributed if style.keep_tokens(d["tokens"])]
    dist_rows.sort(
        key=lambda d: (
            topo_order.get(d["topology"], 99),
            d["tokens"],
            plat_order.get(d["platform"], 99),
            path_order.get(d["path"], 99),
        )
    )
    if dist_rows:
        collectives = sorted(
            {
                "RCCL" if pl["environment"].get("hip") not in (None, "None") else "NCCL"
                for label, pl in platforms
                if pl.get("distributed")
            }
        )
        add(f"## Distributed vocab-parallel logprob (BF16, {'/'.join(collectives)})")
        add("")
        absent = [
            style.name(path) for path in style.paths if path not in {d["path"] for d in distributed}
        ]
        if absent:
            add(
                "Only the vocab-parallel operators take part here. "
                + ", ".join(f"`{name}`" for name in absent)
                + " is a single-shard op that consumes the full `[tokens, V]` logits on one GPU; "
                "it has no TP implementation (no vocab shard input, TP group, or partial merge), "
                "so there is no comparable distributed row for it."
            )
            add("")
        add("### Forward")
        add("")
        add(
            f"| Topology | Tokens | {plat_head}Path | Median (ms) | p95 (ms) | "
            f"Speedup vs {base_name} | Peak MiB/rank | logp max-abs vs FP64 | TP-replicated | "
            "Repeat |"
        )
        add("|---|---:|" + plat_div + "---|---:|---:|---:|---:|---:|:---:|:---:|")
        for d in dist_rows:
            ref = _lookup(
                distributed,
                topology=d["topology"],
                tokens=d["tokens"],
                path=base,
                platform=d["platform"],
            )
            add(
                f"| {d['topology']} | {d['tokens']} | {plat_cell(d['platform'])}"
                f"{style.name(d['path'])} | "
                f"{d['forward']['median_ms']:.4f} | {d['forward']['p95_ms']:.4f} | "
                f"{_median_ratio(ref, d, 'forward')} | {d['forward_peak_mib']:.1f} | "
                f"{d['logp_vs_fp64_max_abs']:.3e} | {_yes(d['tp_replicated'])} | "
                f"{_yes(d['repeat_bitwise'])} |"
            )
        add("")
        add("### Forward+backward")
        add("")
        add(
            f"| Topology | Tokens | {plat_head}Path | Median (ms) | p95 (ms) | "
            f"Speedup vs {base_name} | Peak MiB/rank | Memory vs {base_name} | Grad finite |"
        )
        add("|---|---:|" + plat_div + "---|---:|---:|---:|---:|---:|:---:|")
        for d in dist_rows:
            ref = _lookup(
                distributed,
                topology=d["topology"],
                tokens=d["tokens"],
                path=base,
                platform=d["platform"],
            )
            add(
                f"| {d['topology']} | {d['tokens']} | {plat_cell(d['platform'])}"
                f"{style.name(d['path'])} | "
                f"{d['train_fwd_bwd']['median_ms']:.4f} | {d['train_fwd_bwd']['p95_ms']:.4f} | "
                f"{_median_ratio(ref, d, 'train_fwd_bwd')} | {d['train_peak_mib']:.1f} | "
                f"{_field_ratio(d, ref, 'train_peak_mib')} | {_yes(d['grad_finite'])} |"
            )
        add("")
        kernel_paths = [p for p in WS2_KERNEL_PATHS if p in style.paths]
        if kernel_paths and "ws2-reference" in style.paths:
            add(f"### Numerics versus `{style.name('ws2-reference')}` (distributed)")
            add("")
            add(
                f"| Topology | Tokens | {plat_head}Path | Mismatched elements (logp+LSE) | "
                "Relative L2 |"
            )
            add("|---|---:|" + plat_div + "---|---:|---:|")
            for d in dist_rows:
                if d["path"] not in kernel_paths:
                    continue
                add(
                    f"| {d['topology']} | {d['tokens']} | {plat_cell(d['platform'])}"
                    f"{style.name(d['path'])} | "
                    f"{d.get('mismatch_vs_reference', 'n/a')} | "
                    f"{d.get('rel_l2_vs_reference', float('nan')):.3e} |"
                )
            add("")

    # ---- optional tuning history
    baseline = payload.get("baseline")
    if baseline and style.show_tuning_baseline and "ws2-rocm" in style.paths:
        name = style.name("ws2-rocm")
        add("## ROCm tuning: before versus after")
        add("")
        add(
            f"Baseline commit `{baseline.get('git_commit')}` ran the PR's first ROCm backend: the "
            "HIP tile kernel on an FP32 copy of the shard, with the shared PyTorch autograd "
            f"backward. `{name}` rows only."
        )
        add("")
        add(
            "| dtype | Tokens | Fwd before (ms) | Fwd after (ms) | Speedup | "
            "Fwd+bwd before (ms) | Fwd+bwd after (ms) | Speedup | Peak before (MiB) | "
            "Peak after (MiB) | Memory ratio |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for before in baseline["single_gpu"]:
            if not style.keep_tokens(before["tokens"]):
                continue
            after = _lookup(single, dtype=before["dtype"], tokens=before["tokens"], path="ws2-rocm")
            if after is None:
                continue
            add(
                f"| {before['dtype']} | {before['tokens']} | "
                f"{before['forward']['median_ms']:.4f} | "
                f"{after['forward']['median_ms']:.4f} | "
                f"{_median_ratio(before, after, 'forward')} | "
                f"{before['train_fwd_bwd']['median_ms']:.4f} | "
                f"{after['train_fwd_bwd']['median_ms']:.4f} | "
                f"{_median_ratio(before, after, 'train_fwd_bwd')} | "
                f"{before['train_peak_mib']:.1f} | {after['train_peak_mib']:.1f} | "
                f"{_field_ratio(after, before, 'train_peak_mib')} |"
            )
        add("")

    add("## Figures")
    add("")
    if multi:
        add(
            "One line per backend and device across the full token sweep. The grid puts "
            "latency and peak memory, forward and forward+backward, on one page."
        )
        add("")
    add("![Single-device latency and memory grid](single_gpu_grid.png)")
    add("")
    if multi:
        add(
            "The host and reference paths span three orders of magnitude, which flattens the "
            "kernel backends against each other. The second grid drops them and re-scales to "
            "the kernel backends alone, where the differences between Triton and the two "
            "vendor kernels are legible."
        )
        add("")
        add("![Kernel backends only](single_gpu_grid_kernels.png)")
        add("")
    add("![Single-device latency](single_gpu_latency.png)")
    add("")
    add("![Single-device peak memory](single_gpu_memory.png)")
    add("")
    if distributed:
        add("![Distributed latency](distributed_logp_latency.png)")
        add("")
    (output_directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _platform_kind(payload: dict[str, Any]) -> str:
    env = payload.get("environment", {})
    if env.get("device") == "cpu" or str(env.get("gpu", "")).startswith("n/a"):
        return "cpu"
    return "rocm" if env.get("hip") not in (None, "None") else "cuda"


# Fixed colours so a series looks the same in every figure of the set.
_SERIES_STYLE = {
    "cpu": {"color": "#7f7f7f", "marker": "v", "linestyle": ":"},
    "native-torch": {"color": "#000000", "marker": "o", "linestyle": "-"},
    "triton-cuda": {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
    "triton-rocm": {"color": "#d62728", "marker": "^", "linestyle": "--"},
    "cuda": {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    "hip": {"color": "#ff7f0e", "marker": "P", "linestyle": "-"},
}
_SERIES_ORDER = ("cpu", "native-torch", "triton-cuda", "triton-rocm", "cuda", "hip")


def _figure_series(
    platforms: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, str, str]]:
    """``(display label, platform label, path)`` for one line per backend/device.

    The PyTorch reference is drawn once from the primary accelerator as
    ``native-torch``; the host run contributes the ``cpu`` line.  Each
    accelerator adds its Triton line and its vendor-kernel line, so a ROCm +
    CUDA + host set yields the six series in ``_SERIES_ORDER``.
    """

    kinds = {label: _platform_kind(pl) for label, pl in platforms}
    measured = {
        label: {case["path"] for case in pl["single_gpu"]["cases"]} for label, pl in platforms
    }
    series: list[tuple[str, str, str]] = []

    for label, kind in kinds.items():
        if kind == "cpu" and "ws2-reference" in measured[label]:
            series.append(("cpu", label, "ws2-reference"))
            break

    # The reference line is drawn once, from the primary accelerator. On a
    # host-only run the "cpu" entry above already is that line, so skip it.
    primary = next((label for label, kind in kinds.items() if kind != "cpu"), None)
    if primary is not None and "ws2-reference" in measured.get(primary, set()):
        series.append(("native-torch", primary, "ws2-reference"))

    for label, kind in kinds.items():
        if kind == "cpu":
            continue
        if "ws2-triton" in measured[label]:
            series.append((f"triton-{kind}", label, "ws2-triton"))
        vendor = "ws2-rocm" if kind == "rocm" else "ws2-cuda"
        if vendor in measured[label]:
            series.append(("hip" if kind == "rocm" else "cuda", label, vendor))

    rank = {name: index for index, name in enumerate(_SERIES_ORDER)}
    series.sort(key=lambda item: rank.get(item[0], len(rank)))
    return series


def _series_value(case: dict[str, Any] | None, key: str) -> float:
    if case is None:
        return float("nan")
    value = case[key]
    return float(value["median_ms"] if isinstance(value, dict) else value)


def _write_figures(
    payload: dict[str, Any],
    output_directory: Path,
    style: ReportStyle,
    platforms: list[tuple[str, dict[str, Any]]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if platforms is None:
        platforms = _report_platforms(payload, [], None)
    series = _figure_series(platforms)
    if not series:
        return
    by_label = dict(platforms)

    single: dict[str, list[dict[str, Any]]] = {
        label: [c for c in pl["single_gpu"]["cases"] if c["dtype"] == "bf16"]
        for label, pl in platforms
    }
    tokens = sorted({c["tokens"] for cases in single.values() for c in cases})

    panels = (
        ("forward", "Forward latency", "median ms", True),
        ("train_fwd_bwd", "Forward+backward latency", "median ms", True),
        ("forward_peak_mib", "Forward peak memory", "peak MiB above live", False),
        ("train_peak_mib", "Forward+backward peak memory", "peak MiB above live", False),
    )

    def draw(axis, key, title, ylabel, log_y, chosen_series=None) -> None:
        for index, (name, platform_label, path) in enumerate(chosen_series or series):
            ys = [
                _series_value(_lookup(single[platform_label], tokens=t, path=path), key)
                for t in tokens
            ]
            # Series routinely coincide (cuda tracks the reference's memory; triton
            # and hip share it). Draw later ones thinner so the overlap stays legible.
            axis.plot(
                tokens,
                ys,
                label=name,
                linewidth=3.2 - 0.4 * index,
                markersize=7 - 0.5 * index,
                zorder=3 + index,
                **_SERIES_STYLE.get(name, {}),
            )
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        else:
            # symlog keeps the host run's legitimate ~0 MiB readings on the axis;
            # memory is never negative, so clip the mirrored half away.
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylim(bottom=0)
        axis.set_xlabel("tokens")
        axis.set_ylabel(ylabel)
        axis.set_title(f"BF16: {title}", fontsize=11)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    # Latency and memory keep their own files; the grid puts all four panels together.
    for filename, chosen in (
        ("single_gpu_latency.png", panels[:2]),
        ("single_gpu_memory.png", panels[2:]),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, (key, title, ylabel, log_y) in zip(axes, chosen):
            draw(axis, key, title, ylabel, log_y)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)

    def grid(filename: str, chosen_series, subtitle: str) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        for axis, (key, title, ylabel, log_y) in zip(axes.flat, panels):
            draw(axis, key, title, ylabel, log_y, chosen_series)
        figure.suptitle(
            f"Single-device vocab-parallel logprob, BF16, V={REAL_VOCAB} — {subtitle} ("
            + ", ".join(name for name, _, _ in chosen_series)
            + ")",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)

    grid("single_gpu_grid.png", series, "all paths")

    # The host and reference lines span three orders of magnitude, which flattens
    # the kernel backends against each other. Re-draw them on their own scale.
    kernel_series = [item for item in series if item[0] not in ("cpu", "native-torch")]
    if len(kernel_series) >= 2 and len(kernel_series) != len(series):
        grid("single_gpu_grid_kernels.png", kernel_series, "kernel backends only")

    # ---- distributed: same series minus the host run
    dist_series = [item for item in series if item[0] != "cpu"]
    distributed = {
        label: pl.get("distributed", []) or [] for label, pl in platforms if label in by_label
    }
    dist_series = [item for item in dist_series if distributed.get(item[1])]
    if not dist_series:
        return

    cells: list[tuple[str, int]] = []
    for _, platform_label, _ in dist_series:
        for row in distributed[platform_label]:
            key = (row["topology"], row["tokens"])
            if key not in cells:
                cells.append(key)
    topo_rank = {name: index for index, (name, _, _) in enumerate(TOPOLOGIES)}
    cells.sort(key=lambda c: (topo_rank.get(c[0], 99), c[1]))
    labels = [f"{topology}\nM={tokens_}" for topology, tokens_ in cells]

    figure, axes = plt.subplots(1, 2, figsize=(max(12, 1.15 * len(labels)), 5.0))
    xs = list(range(len(labels)))
    width = 0.8 / max(len(dist_series), 1)
    for axis, key, direction in zip(
        axes, ("forward", "train_fwd_bwd"), ("Forward", "Forward+backward")
    ):
        for index, (name, platform_label, path) in enumerate(dist_series):
            values = [
                _series_value(
                    _lookup(distributed[platform_label], path=path, topology=topology, tokens=t),
                    key,
                )
                for topology, t in cells
            ]
            offset = (index - (len(dist_series) - 1) / 2) * width
            axis.bar(
                [x + offset for x in xs],
                values,
                width,
                label=name,
                color=_SERIES_STYLE.get(name, {}).get("color"),
                zorder=3,
            )
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("slowest-rank median ms")
        axis.set_title(f"Distributed vocab-parallel logprob, BF16: {direction}")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "distributed_logp_latency.png", dpi=180)
    plt.close(figure)


def _extension_symbols() -> str:
    if not _EXT_AVAILABLE or _C is None:
        return "none (pure-Python fallback)"
    candidates = (
        "deterministic_logp_tile_stats",
        "hip_deterministic_logp_tile_stats",
        "hip_deterministic_logp_backward",
    )
    present = [name for name in candidates if hasattr(_C, name)]
    return ", ".join(present) if present else "none"


def _environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "device": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "python": os.sys.version.split()[0],
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "extension_symbols": _extension_symbols(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        environment.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "architecture": (
                    getattr(properties, "gcnArchName", "unknown")
                    if torch.version.hip is not None
                    else f"sm_{properties.major}{properties.minor}"
                ),
                "native_collective": (
                    "torch.distributed ProcessGroupNCCL"
                    + (" (RCCL on ROCm)" if torch.version.hip is not None else " (NCCL)")
                ),
            }
        )
    else:
        environment.update(
            {
                "gpu": "n/a (host execution)",
                "gpu_count": 0,
                "architecture": platform.processor() or platform.machine(),
                "cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "native_collective": "n/a (single-process host run)",
            }
        )
    return environment


def _validate_environment(require_distributed: bool, device: torch.device) -> None:
    if device.type == "cpu":
        if require_distributed:
            raise RuntimeError(
                "--device cpu cannot run the distributed section; add --skip-distributed"
            )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/ROCm GPU is visible")
    if torch.version.hip is not None:
        if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "hip_deterministic_logp_backward"):
            raise RuntimeError(
                "rl_engine._C with hip_deterministic_logp_* is unavailable; build with "
                "PYTORCH_ROCM_ARCH=gfx942 RL_KERNEL_REQUIRE_EXT=1 "
                "python setup.py build_ext --inplace"
            )
    elif not native_tile_stats_available():
        # Not fatal: ws2-triton and the reference still run, only ws2-cuda drops out.
        print(
            "warning: rl_engine._C with deterministic_logp_tile_stats is unavailable; "
            "the ws2-cuda path will be skipped. Build with "
            "TORCH_CUDA_ARCH_LIST=9.0 RL_KERNEL_REQUIRE_EXT=1 "
            "python setup.py build_ext --inplace"
        )
    if require_distributed and (not dist.is_available() or not dist.is_nccl_available()):
        raise RuntimeError("PyTorch NCCL/ProcessGroupNCCL support is unavailable")


ALL_PATHS = (
    "native",
    "ws1-pytorch",
    "ws1-triton",
    "ws2-reference",
    "ws2-triton",
    "ws2-cuda",
    "ws2-rocm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/rocm_logp"))
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="device for the single-device section; 'cpu' implies --skip-distributed",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--training-samples", type=int, default=5)
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="earlier results.json; adds a before/after tuning section for the ROCm backend",
    )
    parser.add_argument(
        "--render-from",
        type=Path,
        default=None,
        help="skip measurement and render report/figures from this results.json",
    )
    parser.add_argument(
        "--report-paths",
        type=str,
        default=",".join(ALL_PATHS),
        help="comma-separated measured paths to show, in order (default: all)",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default="",
        help="display names, e.g. 'ws2-reference=native,ws2-rocm=strict-hip'",
    )
    parser.add_argument(
        "--report-baseline",
        type=str,
        default=None,
        help="path used for speedup/memory ratio columns (default: first reported path)",
    )
    parser.add_argument(
        "--table-tokens",
        type=str,
        default="",
        help="comma-separated token counts to show in tables (default: all measured)",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=",".join(name for name, _, _ in TOPOLOGIES),
        help="comma-separated subset of " + ",".join(name for name, _, _ in TOPOLOGIES),
    )
    parser.add_argument(
        "--compare-with",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "results.json from another platform's run of this harness; adds a cross-platform "
            "comparison section to the report. Repeat for each platform, e.g. "
            "--compare-with h100=benchmarks/results/pr328_cuda_h100/results.json"
        ),
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default=None,
        help="label for this run inside the comparison section (default: derived from the GPU)",
    )
    parser.add_argument("--tokens", type=str, default=",".join(str(t) for t in SINGLE_TOKENS))
    parser.add_argument(
        "--distributed-tokens", type=str, default=",".join(str(t) for t in DISTRIBUTED_TOKENS)
    )
    return parser.parse_args()


def _report_style(
    args: argparse.Namespace, output_directory: Path, config: dict[str, Any]
) -> ReportStyle:
    paths = tuple(p.strip() for p in args.report_paths.split(",") if p.strip())
    unknown = [p for p in paths if p not in ALL_PATHS]
    if unknown:
        raise ValueError(f"unknown report paths {unknown}; choose from {ALL_PATHS}")
    names: dict[str, str] = {}
    for item in (piece.strip() for piece in args.rename.split(",") if piece.strip()):
        key, _, value = item.partition("=")
        names[key.strip()] = value.strip()
    table_tokens = tuple(int(t) for t in args.table_tokens.split(",") if t.strip()) or None
    # Always show the measurement command: rerunning it with the same report flags
    # regenerates this report; --render-from only re-renders an existing results.json.
    command_parts = [
        "python benchmarks/backends/rocm/benchmark_rocm_logp.py \\",
        f"  --warmup {config['warmup']} \\",
        f"  --samples {config['samples']} \\",
        f"  --training-samples {config['training_samples']} \\",
    ]
    if paths != ALL_PATHS:
        command_parts.append(f"  --report-paths {','.join(paths)} \\")
    if names:
        command_parts.append("  --rename " + ",".join(f"{k}={v}" for k, v in names.items()) + " \\")
    if args.report_baseline:
        command_parts.append(f"  --report-baseline {args.report_baseline} \\")
    if table_tokens:
        command_parts.append(f"  --table-tokens {','.join(str(t) for t in table_tokens)} \\")
    for spec in getattr(args, "compare_with", []) or []:
        command_parts.append(f"  --compare-with {spec} \\")
    if getattr(args, "compare_label", None):
        command_parts.append(f"  --compare-label {args.compare_label} \\")
    command_parts.append(f"  --output-dir {output_directory.as_posix()}")
    return ReportStyle(
        paths=paths,
        names=names,
        baseline=args.report_baseline or paths[0],
        table_tokens=table_tokens,
        show_tuning_baseline=args.baseline is not None,
        command="\n".join(command_parts),
    )


def main() -> None:
    args = parse_args()
    output_directory: Path = args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.render_from is not None:
        payload = json.loads(args.render_from.read_text(encoding="utf-8"))
    else:
        device = torch.device("cuda", 0) if args.device == "cuda" else torch.device("cpu")
        if device.type == "cpu":
            # Triton, the native kernels, and NCCL are all device-only.
            args.skip_distributed = True
        _validate_environment(require_distributed=not args.skip_distributed, device=device)
        config = {
            "warmup": args.warmup,
            "samples": args.samples,
            "training_samples": args.training_samples,
        }
        tokens = tuple(int(t) for t in args.tokens.split(",") if t)
        distributed_tokens = tuple(int(t) for t in args.distributed_tokens.split(",") if t)
        selected = {name.strip() for name in args.topologies.split(",") if name.strip()}
        payload = {
            "environment": _environment(device),
            "config": config,
            "single_gpu": {"cases": [], "validate_overhead": [], "paths": []},
            "tile_stats_component": [],
            "distributed": [],
        }
        if not args.skip_single:
            payload["single_gpu"] = _single_gpu_benchmarks(
                warmup=args.warmup,
                samples=args.samples,
                training_samples=args.training_samples,
                tokens=tokens,
                device=device,
            )
            payload["tile_stats_component"] = _tile_stats_component(
                warmup=args.warmup, samples=args.samples, tokens=tokens, device=device
            )
        if not args.skip_distributed:
            device_count = torch.cuda.device_count()
            for topology in TOPOLOGIES:
                name, tp, cp = topology
                if name not in selected:
                    continue
                if tp * cp > device_count:
                    print(f"skipping {name}: needs {tp * cp} GPUs, {device_count} visible")
                    continue
                payload["distributed"].extend(
                    _run_distributed_world(topology, distributed_tokens, config)
                )
    style = _report_style(args, output_directory, payload["config"])
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        payload["baseline"] = {
            "git_commit": baseline.get("environment", {}).get("git_commit"),
            "single_gpu": [c for c in baseline["single_gpu"]["cases"] if c["path"] == "ws2-rocm"],
            "tile_stats_component": baseline.get("tile_stats_component", []),
            "distributed": [d for d in baseline.get("distributed", []) if d["path"] == "ws2-rocm"],
        }
    elif args.render_from is not None:
        payload.pop("baseline", None)
    (output_directory / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    platforms = _report_platforms(payload, args.compare_with, args.compare_label)
    _write_report(payload, output_directory, style, platforms)
    _write_figures(payload, output_directory, style, platforms)
    print(json.dumps({"output_dir": str(output_directory), "status": "ok"}))


if __name__ == "__main__":
    main()
