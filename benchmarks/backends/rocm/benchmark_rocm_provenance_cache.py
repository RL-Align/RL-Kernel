# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark cached ROCm Attention provenance metadata lookups.

The baseline restores the pre-cache behavior by resolving the GPU description
and immutable Split-KV execution plan for every AITER KV-group launch. The
candidate uses the production core caches, but clears only its one-entry plan
cache at the start of every simulated model forward. This models a new decode
length while retaining the process-stable device description.

Both arms run 36 distinct layer inputs backed by vLLM's ROCm 5-D KV layout,
``[blocks, 2, block_size, kv_heads, head_dim]``. Page gathers, caller-output
writes, AITER launch count/order, and tensor arithmetic are otherwise
identical. This is an isolated strict-runtime timing, not an end-to-end VIME
result.

Example on one MI300X::

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_provenance_cache.py \
      --cached-lengths 32,127,512,2048 --blocks 12 --iterations 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore
from rl_engine.backends.rocm.attention.strict_runtime import StrictRocmAttentionRuntime
from rl_engine.contracts.operators.attention import SplitKVExecutionPlan

_BATCH = 1
_QUERY_HEADS = 8
_KV_HEADS = 2
_HEAD_DIM = 128
_PAGE_SIZE = 16
_LAYERS = 36


class _LegacyProvenanceCore(StrictRocmAiterCKAttentionCore):
    """Restore uncached host metadata construction for the baseline."""

    def _device_description(self, device: torch.device) -> tuple[str, str]:
        properties = torch.cuda.get_device_properties(device)
        return properties.name, getattr(properties, "gcnArchName", "unknown")

    def _resolve_split_kv_plan(self, total_kv_tokens: int) -> SplitKVExecutionPlan:
        return self.split_kv.resolve(total_kv_tokens, backend=self.backend_id)


class _NewDecodeCandidateCore(StrictRocmAiterCKAttentionCore):
    """Use production caches while modeling one new sequence length per forward."""

    def begin_simulated_forward(self) -> None:
        # Decode normally advances to a new cached length. Clear only the
        # one-entry plan cache so the first KV group resolves that new length;
        # the process-stable device description deliberately remains warm.
        self._split_kv_plan_cache = None


class _CountingLegacyCore(_LegacyProvenanceCore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device_helper_calls = 0
        self.device_property_lookups = 0
        self.plan_helper_calls = 0
        self.split_plan_resolutions = 0

    def _device_description(self, device: torch.device) -> tuple[str, str]:
        self.device_helper_calls += 1
        self.device_property_lookups += 1
        return super()._device_description(device)

    def _resolve_split_kv_plan(self, total_kv_tokens: int) -> SplitKVExecutionPlan:
        self.plan_helper_calls += 1
        self.split_plan_resolutions += 1
        return super()._resolve_split_kv_plan(total_kv_tokens)


class _CountingCandidateCore(_NewDecodeCandidateCore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device_helper_calls = 0
        self.device_property_lookups = 0
        self.plan_helper_calls = 0
        self.split_plan_resolutions = 0

    def _device_description(self, device: torch.device) -> tuple[str, str]:
        self.device_helper_calls += 1
        cached = self._device_description_cache
        if cached is None or cached[0] != device:
            self.device_property_lookups += 1
        return super()._device_description(device)

    def _resolve_split_kv_plan(self, total_kv_tokens: int) -> SplitKVExecutionPlan:
        self.plan_helper_calls += 1
        key = (self.split_kv, total_kv_tokens, self.backend_id)
        cached = self._split_kv_plan_cache
        if cached is None or cached[0] != key:
            self.split_plan_resolutions += 1
        return super()._resolve_split_kv_plan(total_kv_tokens)


def _digest(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _raw_equal(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        torch.equal(
            left_tensor.detach().contiguous().view(torch.uint8),
            right_tensor.detach().contiguous().view(torch.uint8),
        )
        for left_tensor, right_tensor in zip(left, right, strict=True)
    )


def _elapsed_ms(function: Callable[[], object], iterations: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1_000.0 / iterations
    return start.elapsed_time(end) / iterations, wall_ms


def _counter_snapshot(core) -> dict[str, int]:
    return {
        "device_helper_calls": core.device_helper_calls,
        "device_property_lookups": core.device_property_lookups,
        "plan_helper_calls": core.plan_helper_calls,
        "split_plan_resolutions": core.split_plan_resolutions,
    }


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before[name] for name in after}


def _assert_nested_provenance_is_disjoint(left: dict, right: dict) -> None:
    if left is right:
        raise AssertionError("separate Attention results shared their provenance dictionary")
    left_core = left["core"]
    right_core = right["core"]
    if left_core is right_core:
        raise AssertionError("separate Attention results shared core provenance")
    left_split = left_core["split_kv"]
    right_split = right_core["split_kv"]
    if left_split is right_split:
        raise AssertionError("cached Split-KV plans leaked a mutable provenance dictionary")
    left_boundaries = left_split["actual_split_boundaries"]
    right_boundaries = right_split["actual_split_boundaries"]
    if left_boundaries is right_boundaries:
        raise AssertionError("cached Split-KV plans leaked a mutable boundaries list")
    if any(
        left_boundary is right_boundary
        for left_boundary, right_boundary in zip(
            left_boundaries,
            right_boundaries,
            strict=True,
        )
    ):
        raise AssertionError("cached Split-KV plans leaked a mutable boundary row")


def _benchmark_length(
    *,
    cached_length: int,
    dtype: torch.dtype,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict[str, object]:
    pages_per_row = (cached_length + _PAGE_SIZE - 1) // _PAGE_SIZE
    generator = torch.Generator(device="cpu").manual_seed(391_000 + cached_length)
    layer_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(_LAYERS):
        query = torch.randn(
            _BATCH,
            _QUERY_HEADS,
            1,
            _HEAD_DIM,
            dtype=dtype,
            generator=generator,
        ).cuda()
        # vLLM ROCm 5-D cache: [blocks, K/V, block, KV heads, head dim].
        packed_cache = torch.randn(
            pages_per_row,
            2,
            _PAGE_SIZE,
            _KV_HEADS,
            _HEAD_DIM,
            dtype=dtype,
            generator=generator,
        ).cuda()
        key_cache, value_cache = packed_cache.unbind(1)
        layer_inputs.append((query, key_cache, value_cache))

    layer_cache_storages = {
        key_cache.untyped_storage().data_ptr() for _query, key_cache, _value_cache in layer_inputs
    }
    if len(layer_cache_storages) != _LAYERS:
        raise AssertionError("simulated decoder layers must own distinct KV cache storage")

    page_table = (
        torch.arange(pages_per_row, device="cuda", dtype=torch.int32)
        .reshape(_BATCH, pages_per_row)
        .flip(1)
        .contiguous()
    )
    seqused_k = torch.full((_BATCH,), cached_length, device="cuda", dtype=torch.int32)
    cached_lengths = (cached_length,)

    prototype = StrictRocmAiterCKAttentionCore()
    core_kwargs = {
        "_mha_fwd": prototype._mha_fwd,
        "_mha_bwd": prototype._mha_bwd,
        "_source_sha256": prototype.source_sha256,
    }
    baseline_core = _LegacyProvenanceCore(**core_kwargs)
    candidate_core = _NewDecodeCandidateCore(**core_kwargs)
    baseline_runtime = StrictRocmAttentionRuntime(core=baseline_core)
    candidate_runtime = StrictRocmAttentionRuntime(core=candidate_core)
    baseline_outputs = [torch.empty_like(query) for query, _key, _value in layer_inputs]
    candidate_outputs = [torch.empty_like(query) for query, _key, _value in layer_inputs]

    @torch.inference_mode()
    def run_layers(runtime, outputs, *, capture_lse):
        epoch = runtime.new_page_bounds_epoch()
        last_result = None
        captured_lse = []
        launches = 0
        for (query, key_cache, value_cache), output in zip(
            layer_inputs,
            outputs,
            strict=True,
        ):
            last_result = runtime.forward_paged_with_lse(
                query,
                key_cache,
                value_cache,
                page_table=page_table,
                seqused_k=seqused_k,
                max_seqlen_k=pages_per_row * _PAGE_SIZE,
                scale=_HEAD_DIM**-0.5,
                out=output,
                cached_lengths=cached_lengths,
                page_bounds_epoch=epoch,
            )
            if capture_lse:
                captured_lse.append(last_result.lse)
                launches += int(last_result.provenance["core_launch_count"])
        if last_result is None:
            raise AssertionError("the simulated decoder must contain at least one layer")
        return last_result, tuple(captured_lse), launches

    def baseline():
        return run_layers(baseline_runtime, baseline_outputs, capture_lse=False)[0]

    def candidate():
        candidate_core.begin_simulated_forward()
        return run_layers(candidate_runtime, candidate_outputs, capture_lse=False)[0]

    counting_baseline_core = _CountingLegacyCore(**core_kwargs)
    counting_candidate_core = _CountingCandidateCore(**core_kwargs)
    counting_baseline_runtime = StrictRocmAttentionRuntime(core=counting_baseline_core)
    counting_candidate_runtime = StrictRocmAttentionRuntime(core=counting_candidate_core)
    counting_baseline_outputs = [torch.empty_like(query) for query, _key, _value in layer_inputs]
    counting_candidate_outputs = [torch.empty_like(query) for query, _key, _value in layer_inputs]

    def counted_baseline_forward():
        before = _counter_snapshot(counting_baseline_core)
        result = run_layers(
            counting_baseline_runtime,
            counting_baseline_outputs,
            capture_lse=True,
        )
        return result, _counter_delta(_counter_snapshot(counting_baseline_core), before)

    def counted_candidate_forward():
        counting_candidate_core.begin_simulated_forward()
        before = _counter_snapshot(counting_candidate_core)
        result = run_layers(
            counting_candidate_runtime,
            counting_candidate_outputs,
            capture_lse=True,
        )
        return result, _counter_delta(_counter_snapshot(counting_candidate_core), before)

    (baseline_counted_first, baseline_counts_first) = counted_baseline_forward()
    (baseline_counted_next, baseline_counts_next) = counted_baseline_forward()
    (candidate_counted_first, candidate_counts_first) = counted_candidate_forward()
    (candidate_counted_next, candidate_counts_next) = counted_candidate_forward()
    expected_launches = _LAYERS * _BATCH * _KV_HEADS
    for result, _captured_lse, launches in (
        baseline_counted_first,
        baseline_counted_next,
        candidate_counted_first,
        candidate_counted_next,
    ):
        if launches != expected_launches:
            raise AssertionError("AITER launch count changed while caching host provenance")
        if result.provenance["core_launch_count"] != _BATCH * _KV_HEADS:
            raise AssertionError("per-layer AITER launch count changed")

    expected_helper_calls = expected_launches
    expected_counts = {
        "baseline_first_forward": {
            "device_helper_calls": expected_helper_calls,
            "device_property_lookups": expected_helper_calls,
            "plan_helper_calls": expected_helper_calls,
            "split_plan_resolutions": expected_helper_calls,
        },
        "baseline_next_forward": {
            "device_helper_calls": expected_helper_calls,
            "device_property_lookups": expected_helper_calls,
            "plan_helper_calls": expected_helper_calls,
            "split_plan_resolutions": expected_helper_calls,
        },
        "candidate_first_forward": {
            "device_helper_calls": expected_helper_calls,
            "device_property_lookups": 1,
            "plan_helper_calls": expected_helper_calls,
            "split_plan_resolutions": 1,
        },
        "candidate_next_forward": {
            "device_helper_calls": expected_helper_calls,
            "device_property_lookups": 0,
            "plan_helper_calls": expected_helper_calls,
            "split_plan_resolutions": 1,
        },
    }
    observed_counts = {
        "baseline_first_forward": baseline_counts_first,
        "baseline_next_forward": baseline_counts_next,
        "candidate_first_forward": candidate_counts_first,
        "candidate_next_forward": candidate_counts_next,
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            f"provenance cache lookup counts changed: {observed_counts} != {expected_counts}"
        )

    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_result, baseline_lses, baseline_launches = run_layers(
        baseline_runtime,
        baseline_outputs,
        capture_lse=True,
    )
    candidate_core.begin_simulated_forward()
    candidate_result, candidate_lses, candidate_launches = run_layers(
        candidate_runtime,
        candidate_outputs,
        capture_lse=True,
    )
    candidate_output_snapshot = tuple(output.clone() for output in candidate_outputs)
    candidate_lse_snapshot = tuple(lse.clone() for lse in candidate_lses)
    candidate_core.begin_simulated_forward()
    replay_result, replay_lses, replay_launches = run_layers(
        candidate_runtime,
        candidate_outputs,
        capture_lse=True,
    )
    torch.cuda.synchronize()

    output_equal = _raw_equal(baseline_outputs, candidate_output_snapshot) and _raw_equal(
        candidate_output_snapshot,
        candidate_outputs,
    )
    lse_equal = _raw_equal(baseline_lses, candidate_lse_snapshot) and _raw_equal(
        candidate_lse_snapshot,
        replay_lses,
    )
    provenance_equal = (
        baseline_result.provenance == candidate_result.provenance == replay_result.provenance
    )
    if not output_equal or not lse_equal:
        raise AssertionError("provenance caching changed output or LSE bytes")
    if not provenance_equal:
        raise AssertionError("provenance caching changed the reported strict contract")
    _assert_nested_provenance_is_disjoint(
        baseline_result.provenance,
        candidate_result.provenance,
    )
    _assert_nested_provenance_is_disjoint(
        candidate_result.provenance,
        replay_result.provenance,
    )
    if {baseline_launches, candidate_launches, replay_launches} != {expected_launches}:
        raise AssertionError("AITER launch count changed in the exactness replay")

    functions = {"uncached_provenance": baseline, "cached_provenance": candidate}
    gpu_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    for block in range(blocks):
        order = (
            ("uncached_provenance", "cached_provenance")
            if block % 2 == 0
            else ("cached_provenance", "uncached_provenance")
        )
        for name in order:
            gpu_ms, wall_ms = _elapsed_ms(functions[name], iterations)
            gpu_samples[name].append(gpu_ms)
            wall_samples[name].append(wall_ms)

    baseline_ms = statistics.median(gpu_samples["uncached_provenance"])
    candidate_ms = statistics.median(gpu_samples["cached_provenance"])
    baseline_wall_ms = statistics.median(wall_samples["uncached_provenance"])
    candidate_wall_ms = statistics.median(wall_samples["cached_provenance"])
    return {
        "cached_length": cached_length,
        "dtype": str(dtype).removeprefix("torch."),
        "layout": "vllm_rocm_[blocks,2,block,kv_heads,head_dim]",
        "batch": _BATCH,
        "query_heads": _QUERY_HEADS,
        "kv_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "page_size": _PAGE_SIZE,
        "distinct_layers_per_forward": _LAYERS,
        "distinct_layer_kv_storages": True,
        "candidate_cache_model": {
            "split_plan": "cleared_once_per_simulated_forward",
            "device_description": "retained_across_simulated_forwards",
        },
        "host_metadata_counts": observed_counts,
        "aiter_core_launches_per_forward": expected_launches,
        "median_ms": {
            "gpu": {
                "uncached_provenance": baseline_ms,
                "cached_provenance": candidate_ms,
            },
            "wall": {
                "uncached_provenance": baseline_wall_ms,
                "cached_provenance": candidate_wall_ms,
            },
        },
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "wall_latency_reduction_percent": (
            100.0 * (baseline_wall_ms - candidate_wall_ms) / baseline_wall_ms
        ),
        "saved_ms_per_forward": baseline_ms - candidate_ms,
        "raw_output_equal": output_equal,
        "raw_lse_equal": lse_equal,
        "full_provenance_equal": provenance_equal,
        "nested_provenance_disjoint": True,
        "output_sha256_36_layers": _digest(candidate_output_snapshot),
        "lse_sha256_36_layers": _digest(candidate_lse_snapshot),
        "samples_ms": {"gpu": gpu_samples, "wall": wall_samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-lengths", default="32,127,512,2048")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    if args.warmup < 0 or args.blocks <= 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative; blocks and iterations must be positive")
    cached_lengths = tuple(int(value.strip()) for value in args.cached_lengths.split(","))
    if not cached_lengths or any(length <= 0 for length in cached_lengths):
        raise ValueError("--cached-lengths must contain positive integers")
    dtype = getattr(torch, args.dtype)
    results = [
        _benchmark_length(
            cached_length=cached_length,
            dtype=dtype,
            warmup=args.warmup,
            blocks=args.blocks,
            iterations=args.iterations,
        )
        for cached_length in cached_lengths
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "benchmark_scope": "strict_runtime_distinct_vllm_rocm_decoder_layers",
                "blocks": args.blocks,
                "iterations_per_block": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
