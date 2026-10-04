# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 strict ROCm Attention performance and bitwise-parity benchmark.

Operator-only. No model checkpoint or serving engine is loaded; the shapes are
Qwen3-8B's attention shapes (``Hq=32``, ``Hkv=8``, ``D=128``).

Measurement matrix and presentation follow PR #325 (`benchmark_rocm_ffn.py`) and
PR #328 (`benchmark_rocm_logp.py`): the timing/accuracy helpers, the spawned
distributed world, and the figure style are taken from those scripts so the three
reports can be read side by side.

Paths measured:

- ``sdpa``            PyTorch ``scaled_dot_product_attention``. Speed baseline only,
                      exactly as PR #325 uses upstream ``Qwen3MLP`` at TP=1: no
                      accuracy claim is mixed into the speed comparison.
- ``strict-aiter``    ``StrictRocmAiterCKAttentionCore`` — the ROCm production core
                      (AITER CK dense MHA, non-split API).
- ``reference-native``   ``_C.deterministic_attention_forward/backward`` — the materializing
                      FP32 reference core, hipified from the shared ``.cu``.
- ``triton-bitwise``  ``TritonDeterministicAttentionOp`` — the Triton port whose
                      contract is bit-identity with ``reference-native``.

The headline column is ``triton-bitwise`` versus ``reference-native``: acceptance is
0 mismatched elements on out, lse, dQ, dK and dV.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import statistics
import tempfile
import threading
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

QWEN3_8B_Q_HEADS = 32
QWEN3_8B_KV_HEADS = 8
QWEN3_8B_HEAD_DIM = 128

DEFAULT_SEQ_LENS = (512, 1024, 2048, 4096)
DEFAULT_TP_DEGREES = (2, 4, 8)
# (label, tp_world_size, cp_world_size, replicas) -- world_size = tp * cp * replicas.
# Replicas run independent CP groups side by side, which is how PR #319 exercised
# 8 ranks at TP=2/CP=2.
DISTRIBUTED_TOPOLOGIES = (
    ("tp1_cp2", 1, 2, 1),
    ("tp2_cp2", 2, 2, 1),
    ("tp1_cp4", 1, 4, 1),
    ("tp2_cp2_x2", 2, 2, 2),
    ("tp2_cp4", 2, 4, 1),
    ("tp1_cp8", 1, 8, 1),
)


# ---------------------------------------------------------------------------
# Measurement helpers (PR #328 benchmark_rocm_logp.py)
# ---------------------------------------------------------------------------


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


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def _mismatch_count(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape:
        return -1
    return int((a != b).sum().item())


def _gpu_event_samples(
    function: Callable[[], Any], *, warmup: int, samples: int, deadline: float = 0.0
) -> list[float]:
    for _ in range(warmup):
        function()
        if deadline and time.perf_counter() > deadline:
            break
    torch.cuda.synchronize()
    events = []
    for _ in range(samples):
        if deadline and events and time.perf_counter() > deadline:
            break
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _host_wall_samples(
    function: Callable[[], Any], *, warmup: int, samples: int, deadline: float = 0.0
) -> list[float]:
    """Wall-clock timing for host execution, where CUDA events do not apply."""
    for _ in range(warmup):
        function()
        if deadline and time.perf_counter() > deadline:
            break
    timings = []
    for _ in range(samples):
        if deadline and timings and time.perf_counter() > deadline:
            break
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _timed_samples(
    function: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
    device: torch.device,
    deadline: float = 0.0,
) -> list[float]:
    """Sample, stopping early once ``deadline`` (a perf_counter value) passes.

    The pre-flight projection can under-estimate badly: at S=4096 the materialized
    score matrix leaves cache and the compute model stops holding. So the budget is
    also enforced here as a hard wall clock, not only as an estimate. At least one
    sample is always taken, so a row is never empty.
    """
    if device.type == "cuda":
        return _gpu_event_samples(function, warmup=warmup, samples=samples, deadline=deadline)
    return _host_wall_samples(function, warmup=warmup, samples=samples, deadline=deadline)


def _rss_mib() -> float:
    with open("/proc/self/statm", "r", encoding="ascii") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _host_peak_rss_mib(function: Callable[[], Any]) -> float:
    """Peak resident-set increase during one host call, sampled from /proc.

    The closest host analogue of ``torch.cuda.max_memory_allocated``, but an RSS
    high-water delta rather than an allocator statistic: it includes caching-allocator
    reuse and page granularity, so it is an approximation and not directly comparable
    to the device figures. A call served from already-resident pages can report ~0.
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
    _empty_cache(device)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    function()
    torch.cuda.synchronize()
    return float((torch.cuda.max_memory_allocated() - baseline) / (1024.0 * 1024.0))


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()


# ---------------------------------------------------------------------------
# Attention paths
# ---------------------------------------------------------------------------


def _seeded_qkv(
    batch: int,
    q_heads: int,
    kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, q_heads, seq_len, head_dim, device=device, dtype=dtype, generator=generator
    )
    k = torch.randn(
        batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype, generator=generator
    )
    v = torch.randn(
        batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype, generator=generator
    )
    return q, k, v


def _positions(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0).expand(batch, -1)


def _fp64_oracle(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (out, lse) in FP64 from the BF16/FP16-rounded inputs."""
    q64 = q.double()
    k64 = k.double()
    v64 = v.double()
    group = q.size(1) // k.size(1)
    k64 = k64.repeat_interleave(group, dim=1)
    v64 = v64.repeat_interleave(group, dim=1)
    scores = torch.matmul(q64, k64.transpose(-1, -2)) * scale
    if causal:
        sq, skv = q.size(2), k.size(2)
        offset = skv - sq
        mask = torch.arange(skv, device=q.device)[None, :] > (
            torch.arange(sq, device=q.device)[:, None] + offset
        )
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v64), lse


def _sdpa_forward(q, k, v, *, causal, scale):
    group = q.size(1) // k.size(1)
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(group, dim=1),
        v.repeat_interleave(group, dim=1),
        is_causal=causal,
        scale=scale,
    )


class _Paths:
    """Lazily constructed attention paths, so a missing backend skips one row."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.errors: dict[str, str] = {}
        # The PyTorch reference is the only non-SDPA path that also runs on the host.
        self.native = self._try("pytorch-native", self._make_native)
        self.is_rocm = torch.version.hip is not None
        if device.type == "cuda":
            if self.is_rocm:
                self.strict = self._try("strict-aiter", self._make_strict)
                self.strict_fa4 = None
                self.errors["strict-fa4"] = "CUDA-only path; this run is ROCm"
            else:
                self.strict = None
                self.errors["strict-aiter"] = "ROCm-only path; this run is CUDA"
                self.strict_fa4 = self._try("strict-fa4", self._make_strict_fa4)
            self.reference = self._try("reference-native", self._make_reference)
            self.triton = self._try("triton-bitwise", self._make_triton)
        else:
            self.strict = self.strict_fa4 = self.reference = self.triton = None
            for name in ("strict-aiter", "strict-fa4", "reference-native", "triton-bitwise"):
                self.errors[name] = "GPU-only path; not available on the host"

    def _try(self, name: str, factory: Callable[[], Any]) -> Any:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - a missing backend is a reported row
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _make_native():
        from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp

        return NativeAttentionOp()

    @staticmethod
    def _make_strict():
        from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

        return StrictRocmAiterCKAttentionCore()

    @staticmethod
    def _make_strict_fa4():
        from rl_engine.kernels.ops.cuda.attention.flash_attn import StrictFlashAttention4Core

        return StrictFlashAttention4Core()

    @staticmethod
    def _make_reference():
        from rl_engine.kernels.ops.cuda.attention.deterministic_attn import DeterministicAttentionOp

        return DeterministicAttentionOp()

    def _make_triton(self):
        from rl_engine.kernels.ops.triton.attention.deterministic_attn import (
            BITWISE_LIBM_PARITY,
            TritonDeterministicAttentionOp,
        )

        # The bitwise expf/logf sequence is only ported for HIP, so on CUDA the op
        # refuses by default. Measure it anyway, but the report must not call it bitwise.
        return TritonDeterministicAttentionOp(require_bitwise_libm=BITWISE_LIBM_PARITY)

    def runner(self, name: str, q, k, v, *, causal: bool, scale: float, positions):
        """Return ``() -> (out, lse|None)`` for one path, or None when unavailable."""
        if name == "sdpa":
            return lambda: (_sdpa_forward(q, k, v, causal=causal, scale=scale), None)
        if name == "pytorch-native" and self.native is not None:
            # NativeAttentionOp is the repo's ground-truth reference; it returns out only.
            return lambda: (
                self.native.forward(q, k, v, causal=causal, scale=scale),
                None,
            )
        if name == "strict-aiter" and self.strict is not None:

            def run_strict():
                result = self.strict.forward_with_lse(
                    q,
                    k,
                    v,
                    causal=causal,
                    scale=scale,
                    query_position_ids=positions if causal else None,
                    key_position_ids=positions if causal else None,
                )
                return result.out, result.lse

            return run_strict
        if name == "strict-fa4" and self.strict_fa4 is not None:

            def run_fa4():
                result = self.strict_fa4.forward_with_lse(
                    q,
                    k,
                    v,
                    causal=causal,
                    scale=scale,
                    query_position_ids=positions if causal else None,
                    key_position_ids=positions if causal else None,
                )
                return result.out, result.lse

            return run_fa4
        if name == "reference-native" and self.reference is not None:
            return lambda: self.reference.forward_with_lse(q, k, v, causal=causal, scale=scale)
        if name == "triton-bitwise" and self.triton is not None:
            return lambda: self.triton.forward_with_lse(q, k, v, causal=causal, scale=scale)
        return None


PATH_NAMES = (
    "sdpa",
    "pytorch-native",
    "strict-aiter",
    "strict-fa4",
    "reference-native",
    "triton-bitwise",
)


def _single_gpu_benchmarks(
    *,
    paths: _Paths,
    seq_lens: tuple[int, ...],
    dtypes: tuple[torch.dtype, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    warmup: int,
    samples: int,
    training_samples: int,
    device: torch.device,
    budget_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    scale = 1.0 / math.sqrt(head_dim)

    for dtype in dtypes:
        dtype_name = str(dtype).replace("torch.", "").replace("float", "fp").replace("bfp", "bf")
        for seq_len in seq_lens:
            q, k, v = _seeded_qkv(
                batch, q_heads, kv_heads, seq_len, head_dim, dtype, device, seed=1234
            )
            positions = _positions(batch, seq_len, device)
            oracle_out, oracle_lse = _fp64_oracle(q, k, v, causal=True, scale=scale)

            row: dict[str, Any] = {
                "dtype": dtype_name,
                "seq_len": seq_len,
                "batch": batch,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "paths": {},
            }
            captured: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}

            for name in PATH_NAMES:
                runner = paths.runner(name, q, k, v, causal=True, scale=scale, positions=positions)
                if runner is None:
                    continue

                # One untimed call both captures the outputs and prices the path. A cell
                # whose sampling would blow the budget is skipped and says so, rather
                # than silently costing an hour.
                probe_start = time.perf_counter()
                out, lse = runner()
                _device_sync(device)
                probe_seconds = time.perf_counter() - probe_start
                captured[name] = (
                    out.detach().clone(),
                    None if lse is None else lse.detach().clone(),
                )

                forward_calls = warmup + samples + 2
                training_calls = max(1, warmup // 2) + training_samples + 1
                # Backward is empirically 2-4x the forward on these paths; 3x is the
                # midpoint and only decides whether to run, never a reported number.
                estimated = probe_seconds * (forward_calls + 3 * training_calls)
                if budget_seconds > 0 and estimated > budget_seconds:
                    row["paths"][name] = {
                        "skipped": (
                            f"one call took {probe_seconds:.1f}s; sampling would need about "
                            f"{estimated / 60:.0f} min, over the "
                            f"{budget_seconds / 60:.0f} min per-path budget"
                        ),
                        "probe_seconds": probe_seconds,
                        "estimated_seconds": estimated,
                        "out_vs_fp64": _accuracy(out.double(), oracle_out),
                    }
                    del out, lse
                    _empty_cache(device)
                    continue

                deadline = time.perf_counter() + budget_seconds if budget_seconds else 0.0
                forward_ms = _timed_samples(
                    runner, warmup=warmup, samples=samples, device=device, deadline=deadline
                )
                forward = _summary_ms(forward_ms)
                forward_peak = _peak_memory_mib(runner, device)

                entry: dict[str, Any] = {
                    "forward": forward,
                    "forward_samples": len(forward_ms),
                    "forward_truncated": len(forward_ms) < samples,
                    "forward_peak_mib": forward_peak,
                    "out_vs_fp64": _accuracy(out.double(), oracle_out),
                }
                if lse is not None:
                    entry["lse_vs_fp64"] = _accuracy(lse.double(), oracle_lse)

                # Repeat determinism: two identical calls must be bitwise equal.
                repeat_out, repeat_lse = runner()
                entry["repeat_bitwise"] = _bitwise_equal(out, repeat_out) and (
                    lse is None or _bitwise_equal(lse, repeat_lse)
                )

                training = _training_runner(
                    paths, name, q, k, v, causal=True, scale=scale, positions=positions
                )
                if training is not None:
                    train_deadline = time.perf_counter() + budget_seconds if budget_seconds else 0.0
                    train_ms = _timed_samples(
                        training,
                        warmup=max(1, warmup // 2),
                        samples=training_samples,
                        device=device,
                        deadline=train_deadline,
                    )
                    entry["train_fwd_bwd"] = _summary_ms(train_ms)
                    entry["train_samples"] = len(train_ms)
                    entry["train_truncated"] = len(train_ms) < training_samples
                    entry["train_peak_mib"] = _peak_memory_mib(training, device)

                row["paths"][name] = entry
                del out, lse, repeat_out, repeat_lse
                _empty_cache(device)

            # Headline: Triton must be bit-identical to the native reference core.
            if "triton-bitwise" in captured and "reference-native" in captured:
                t_out, t_lse = captured["triton-bitwise"]
                r_out, r_lse = captured["reference-native"]
                row["triton_vs_reference"] = {
                    "out_mismatched": _mismatch_count(t_out, r_out),
                    "lse_mismatched": _mismatch_count(t_lse, r_lse),
                    "out_relative_l2": _relative_l2(t_out, r_out),
                    "bitwise": _bitwise_equal(t_out, r_out) and _bitwise_equal(t_lse, r_lse),
                }
            # The production core is a different vendor kernel; report the gap, do
            # not claim parity with it.
            production = "strict-aiter" if "strict-aiter" in captured else "strict-fa4"
            if production in captured and "reference-native" in captured:
                s_out, s_lse = captured[production]
                r_out, r_lse = captured["reference-native"]
                row["strict_vs_reference"] = {
                    "production_path": production,
                    "out": _accuracy(s_out, r_out),
                    "lse": _accuracy(s_lse, r_lse),
                    "out_mismatched": _mismatch_count(s_out, r_out),
                }

            cases.append(row)
            del q, k, v, oracle_out, oracle_lse, captured
            _empty_cache(device)
    return cases


def _training_runner(paths, name, q, k, v, *, causal, scale, positions):
    """Return ``() -> None`` running one forward+backward, or None."""
    if name == "sdpa":

        def train_sdpa() -> None:
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            out = _sdpa_forward(qr, kr, vr, causal=causal, scale=scale)
            out.sum().backward()

        return train_sdpa

    if name == "pytorch-native":
        if paths.native is None:
            return None

        def train_native() -> None:
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            out = paths.native.forward(qr, kr, vr, causal=causal, scale=scale)
            out.sum().backward()

        return train_native

    op = {
        "strict-aiter": paths.strict,
        "strict-fa4": paths.strict_fa4,
        "reference-native": paths.reference,
        "triton-bitwise": paths.triton,
    }.get(name)
    if op is None:
        return None

    def train_op() -> None:
        qr = q.detach().requires_grad_(True)
        kr = k.detach().requires_grad_(True)
        vr = v.detach().requires_grad_(True)
        kwargs: dict[str, Any] = {"causal": causal, "scale": scale}
        if name in ("strict-aiter", "strict-fa4") and causal:
            kwargs["query_position_ids"] = positions
            kwargs["key_position_ids"] = positions
        result = op.forward_with_lse(qr, kr, vr, **kwargs)
        out = result.out if hasattr(result, "out") else result[0]
        out.sum().backward()

    return train_op


def _backward_parity(
    *,
    paths: _Paths,
    seq_lens: tuple[int, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """dQ/dK/dV bitwise parity, Triton port versus the native reference core."""
    if paths.reference is None or paths.triton is None:
        return []
    rows = []
    scale = 1.0 / math.sqrt(head_dim)
    for seq_len in seq_lens:
        q, k, v = _seeded_qkv(
            batch, q_heads, kv_heads, seq_len, head_dim, torch.bfloat16, device, seed=99
        )
        grad_out = torch.randn(
            batch,
            q_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(100),
        )
        grads = {}
        for name, op in (("reference-native", paths.reference), ("triton-bitwise", paths.triton)):
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            out, _lse = op.forward_with_lse(qr, kr, vr, causal=True, scale=scale)
            out.backward(grad_out)
            grads[name] = (qr.grad.clone(), kr.grad.clone(), vr.grad.clone())
        reference = grads["reference-native"]
        triton_grads = grads["triton-bitwise"]
        rows.append(
            {
                "seq_len": seq_len,
                "dq_mismatched": _mismatch_count(triton_grads[0], reference[0]),
                "dk_mismatched": _mismatch_count(triton_grads[1], reference[1]),
                "dv_mismatched": _mismatch_count(triton_grads[2], reference[2]),
                "bitwise": all(_bitwise_equal(t, r) for t, r in zip(triton_grads, reference)),
            }
        )
        del q, k, v, grad_out, grads
        _empty_cache(device)
    return rows


def _batch_composition(
    *,
    paths: _Paths,
    seq_lens: tuple[int, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """A row computed alone must be bitwise equal to the same row inside a batch.

    The strict ROCm core refuses ``B > 1`` outright (``_validate_inputs``: "executes
    one logical batch row at a time"), so for that path the property is structural
    rather than measured, and the row records that instead of a comparison.
    """
    rows = []
    scale = 1.0 / math.sqrt(head_dim)
    for seq_len in seq_lens:
        q, k, v = _seeded_qkv(
            4, q_heads, kv_heads, seq_len, head_dim, torch.bfloat16, device, seed=7
        )
        positions_batch = _positions(4, seq_len, device)
        positions_one = _positions(1, seq_len, device)
        row: dict[str, Any] = {"seq_len": seq_len, "paths": {}}
        for name in PATH_NAMES:
            if name == "strict-aiter":
                if paths.strict is not None:
                    row["paths"][name] = {
                        "batch_gt1_rejected": True,
                        "out_bitwise": True,
                        "out_mismatched": 0,
                        "note": "core executes one logical batch row per launch",
                    }
                continue
            batched = paths.runner(
                name, q, k, v, causal=True, scale=scale, positions=positions_batch
            )
            single = paths.runner(
                name,
                q[2:3].contiguous(),
                k[2:3].contiguous(),
                v[2:3].contiguous(),
                causal=True,
                scale=scale,
                positions=positions_one,
            )
            if batched is None or single is None:
                continue
            batch_out, batch_lse = batched()
            single_out, single_lse = single()
            row["paths"][name] = {
                "batch_gt1_rejected": False,
                "out_bitwise": _bitwise_equal(single_out[0], batch_out[2].contiguous()),
                "out_mismatched": _mismatch_count(single_out[0], batch_out[2].contiguous()),
                "out_max_abs": _accuracy(single_out[0], batch_out[2])["max_abs"],
                "lse_bitwise": (
                    None
                    if batch_lse is None
                    else _bitwise_equal(single_lse[0], batch_lse[2].contiguous())
                ),
            }
            del batch_out, batch_lse, single_out, single_lse
            _empty_cache(device)
        rows.append(row)
        del q, k, v
        _empty_cache(device)
    return rows


def _tp_head_sensitivity(
    *,
    paths: _Paths,
    seq_lens: tuple[int, ...],
    tp_degrees: tuple[int, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Is a head shard under TP=N bitwise equal to the same slice of an unsharded run?

    TP performs no cross-rank reduction in attention, so any nonzero value here
    means the kernel's arithmetic depends on how many heads shared the launch.
    Measured both on the raw production core and through the per-KV-group launch
    schedule that the Vime provider uses.
    """
    rows: list[dict[str, Any]] = []
    scale = 1.0 / math.sqrt(head_dim)
    for seq_len in seq_lens:
        q, k, v = _seeded_qkv(
            1, q_heads, kv_heads, seq_len, head_dim, torch.bfloat16, device, seed=21
        )
        positions = _positions(1, seq_len, device)

        for schedule in ("raw_launch", "one_kv_group_per_launch"):
            full = _tp_schedule_forward(paths, q, k, v, scale, positions, schedule)
            if full is None:
                continue
            full_out, full_lse = full
            for tp in tp_degrees:
                if q_heads % tp or kv_heads % tp:
                    continue
                local_q = q_heads // tp
                local_kv = kv_heads // tp
                shard = _tp_schedule_forward(
                    paths,
                    q[:, :local_q],
                    k[:, :local_kv],
                    v[:, :local_kv],
                    scale,
                    positions,
                    schedule,
                )
                shard_out, shard_lse = shard
                rows.append(
                    {
                        "seq_len": seq_len,
                        "schedule": schedule,
                        "tp": tp,
                        "local_q_heads": local_q,
                        "local_kv_heads": local_kv,
                        "out_max_abs": _accuracy(shard_out, full_out[:, :local_q])["max_abs"],
                        "lse_max_abs": _accuracy(shard_lse, full_lse[:, :local_q])["max_abs"],
                        "invariant": _bitwise_equal(shard_out, full_out[:, :local_q].contiguous())
                        and _bitwise_equal(shard_lse, full_lse[:, :local_q].contiguous()),
                    }
                )
                del shard_out, shard_lse
                _empty_cache(device)
            del full_out, full_lse
            _empty_cache(device)
        del q, k, v
        _empty_cache(device)
    return rows


def _tp_schedule_cost(
    *,
    paths: _Paths,
    seq_lens: tuple[int, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int,
    samples: int,
) -> list[dict[str, Any]]:
    """What the TP-degree invariance costs.

    ``raw_launch`` is one launch for all heads and is NOT the production schedule;
    ``one_kv_group_per_launch`` is what the provider runs (``Hkv`` launches per row)
    and is what makes the result independent of the TP degree.
    """
    if paths.strict is None and paths.strict_fa4 is None:
        return []
    rows: list[dict[str, Any]] = []
    scale = 1.0 / math.sqrt(head_dim)
    for seq_len in seq_lens:
        q, k, v = _seeded_qkv(
            1, q_heads, kv_heads, seq_len, head_dim, torch.bfloat16, device, seed=21
        )
        positions = _positions(1, seq_len, device)
        entry: dict[str, Any] = {"seq_len": seq_len, "launches": kv_heads}
        for schedule in ("raw_launch", "one_kv_group_per_launch"):

            # Bind the tensors as defaults: they are deleted at the end of each
            # iteration, so a late-binding closure would reference a dead name.
            def run(chosen=schedule, q=q, k=k, v=v, positions=positions):
                return _tp_schedule_forward(paths, q, k, v, scale, positions, chosen)

            entry[schedule] = _summary_ms(
                _timed_samples(run, warmup=warmup, samples=samples, device=device)
            )
            entry[f"{schedule}_peak_mib"] = _peak_memory_mib(run, device)

        def run_sdpa(q=q, k=k, v=v):
            return _sdpa_forward(q, k, v, causal=True, scale=scale)

        entry["sdpa"] = _summary_ms(
            _timed_samples(run_sdpa, warmup=warmup, samples=samples, device=device)
        )
        rows.append(entry)
        del q, k, v
        _empty_cache(device)
    return rows


def _tp_schedule_forward(paths, q, k, v, scale, positions, schedule):
    """Run the strict core either in one launch or one launch per KV group."""
    core = paths.strict if paths.strict is not None else paths.strict_fa4
    if core is None:
        return None
    if schedule == "raw_launch":
        result = core.forward_with_lse(
            q,
            k,
            v,
            causal=True,
            scale=scale,
            query_position_ids=positions,
            key_position_ids=positions,
        )
        return result.out.contiguous(), result.lse.contiguous()

    group = q.size(1) // k.size(1)
    outs, lses = [], []
    for kv_index in range(k.size(1)):
        lo, hi = kv_index * group, (kv_index + 1) * group
        result = core.forward_with_lse(
            q[:, lo:hi],
            k[:, kv_index : kv_index + 1],
            v[:, kv_index : kv_index + 1],
            causal=True,
            scale=scale,
            query_position_ids=positions,
            key_position_ids=positions,
        )
        outs.append(result.out)
        lses.append(result.lse)
    return torch.cat(outs, dim=1).contiguous(), torch.cat(lses, dim=1).contiguous()


# ---------------------------------------------------------------------------
# Distributed CP (spawned world; harness shape from PR #325)
# ---------------------------------------------------------------------------


def _distributed_cp_worker(
    rank: int,
    world_size: int,
    topology: tuple[str, int, int, int],
    init_method: str,
    result_queue: Any,
    warmup: int,
    samples: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=20),
        )
        payload = _distributed_cp_case(
            rank,
            world_size,
            topology,
            warmup=warmup,
            samples=samples,
            seq_len=seq_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        if rank == 0:
            result_queue.put({"ok": True, "topology": topology[0], "payload": payload})
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "rank": rank,
                "topology": topology[0],
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _distributed_cp_case(
    rank: int,
    world_size: int,
    topology: tuple[str, int, int],
    *,
    warmup: int,
    samples: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """One CP topology running the real AG/RS schedule over the platform's transport.

    Schedule (same one ``scripts/ws2_p2p_nccl_attention_reference_check.py`` accepts):
    all-gather Q/K/V and the position ids over the CP group, run the strict core
    once on the full sequence, then reduce-scatter the ``(out, lse)`` result back
    to this rank's query range. Bitwise acceptance is against a CP=1 run of the
    same core on the same full-sequence inputs.
    """
    from rl_engine.kernels.ops.cuda.attention.cp_comm import (
        AttentionCPBlockMetadata,
        AttentionCPCommunicationPlan,
        AttentionParallelSpec,
        CUDAAGRSAttentionCPCommunication,
        RCCLAGRSAttentionCPCommunication,
    )

    # ROCm runs AITER over the RCCL transport; CUDA runs FA4 over the CUDA IPC transport.
    is_rocm = torch.version.hip is not None
    if is_rocm:
        from rl_engine.kernels.ops.rocm.attention.flash_attn import (
            StrictRocmAiterCKAttentionCore as _StrictCore,
        )

        _Transport = RCCLAGRSAttentionCPCommunication
        transport_id = "rccl_ag_rs"
    else:
        from rl_engine.kernels.ops.cuda.attention.flash_attn import (
            StrictFlashAttention4Core as _StrictCore,
        )

        _Transport = CUDAAGRSAttentionCPCommunication
        transport_id = "cuda_ag_rs"

    label, tp_world, cp_world, replicas = topology
    device = torch.device("cuda", rank)
    scale = 1.0 / math.sqrt(head_dim)
    chunk_size = seq_len // (cp_world * 2)

    # Ranks that share a TP index form one CP group. Every rank must call
    # new_group for every group, in the same order.
    group_index = rank // cp_world
    tp_index = group_index % tp_world
    replica_index = group_index // tp_world
    cp_rank = rank % cp_world
    cp_group = None
    for slice_index in range(world_size // cp_world):
        ranks = list(range(slice_index * cp_world, (slice_index + 1) * cp_world))
        group = dist.new_group(ranks=ranks)
        if slice_index == group_index:
            cp_group = group

    local_q_heads = q_heads // tp_world
    local_kv_heads = kv_heads // tp_world

    generator = torch.Generator(device="cpu").manual_seed(2357 + tp_index + 100 * replica_index)
    q = torch.randn(
        1, local_q_heads, seq_len, head_dim, generator=generator, dtype=torch.bfloat16
    ).to(device)
    k = torch.randn(
        1, local_kv_heads, seq_len, head_dim, generator=generator, dtype=torch.bfloat16
    ).to(device)
    v = torch.randn(
        1, local_kv_heads, seq_len, head_dim, generator=generator, dtype=torch.bfloat16
    ).to(device)
    positions = _positions(1, seq_len, device).to(torch.int32)

    span = seq_len // cp_world
    owner_ranges = tuple((i * span, (i + 1) * span) for i in range(cp_world))
    blocks: list[AttentionCPBlockMetadata] = []
    for owner, (owner_start, owner_end) in enumerate(owner_ranges):
        for start in range(owner_start, owner_end, chunk_size):
            blocks.append(
                AttentionCPBlockMetadata(
                    global_block_index=len(blocks),
                    kv_block_start=start,
                    kv_block_end=min(start + chunk_size, owner_end),
                    owner_cp_rank=owner,
                    owner_tp_rank=tp_index,
                )
            )
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=tp_world,
            tp_rank=tp_index,
            cp_world_size=cp_world,
            cp_rank=cp_rank,
        ),
        backend=transport_id,
        status="implemented",
        expected_blocks=tuple(blocks),
        expected_kv_token_range=(0, seq_len),
        query_token_ranges=owner_ranges,
    )

    core = _StrictCore()
    communication = _Transport(process_group=cp_group)

    query_start, query_end = owner_ranges[cp_rank]
    q_local = q[:, :, query_start:query_end, :].contiguous()
    k_local = k[:, :, query_start:query_end, :].contiguous()
    v_local = v[:, :, query_start:query_end, :].contiguous()
    positions_local = positions[:, query_start:query_end].contiguous()

    def cp_forward():
        q_full = communication.all_gather_query(q_local, plan)
        k_full, v_full = communication.all_gather_kv(k_local, v_local, plan)
        query_positions, key_positions = communication.all_gather_position_ids(
            positions_local, positions_local, plan
        )
        result = core.forward_with_lse(
            q_full,
            k_full,
            v_full,
            causal=True,
            scale=scale,
            query_position_ids=query_positions,
            key_position_ids=key_positions,
        )
        return communication.reduce_scatter_strict_result(result.out, result.lse, plan)

    shard = cp_forward()
    forward_ms = _summary_ms(
        _gpu_event_samples(lambda: cp_forward(), warmup=warmup, samples=samples)
    )
    peak = _peak_memory_mib(lambda: cp_forward(), device)

    # CP=1 acceptance: the same core on the same full-sequence inputs, then take
    # this rank's query range out of it.
    def cp1_forward():
        return core.forward_with_lse(
            q,
            k,
            v,
            causal=True,
            scale=scale,
            query_position_ids=positions,
            key_position_ids=positions,
        )

    single = cp1_forward()
    cp1_ms = _summary_ms(_gpu_event_samples(cp1_forward, warmup=warmup, samples=samples))
    expected_out = single.out[:, :, query_start:query_end, :].contiguous()
    expected_lse = single.lse[:, :, query_start:query_end].contiguous()

    out_bitwise = _bitwise_equal(shard.out.contiguous(), expected_out)
    lse_bitwise = _bitwise_equal(shard.lse.contiguous(), expected_lse)
    repeat = cp_forward()
    repeat_bitwise = _bitwise_equal(shard.out.contiguous(), repeat.out.contiguous())

    flags = torch.tensor(
        [
            1.0 if out_bitwise else 0.0,
            1.0 if lse_bitwise else 0.0,
            1.0 if repeat_bitwise else 0.0,
        ],
        device=device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    mismatches = torch.tensor(
        [
            float(_mismatch_count(shard.out.contiguous(), expected_out)),
            float(_mismatch_count(shard.lse.contiguous(), expected_lse)),
        ],
        device=device,
    )
    dist.all_reduce(mismatches, op=dist.ReduceOp.SUM)

    return {
        "topology": label,
        "world_size": world_size,
        "tp_world_size": tp_world,
        "cp_world_size": cp_world,
        "replicas": replicas,
        "seq_len": seq_len,
        "local_q_heads": local_q_heads,
        "local_kv_heads": local_kv_heads,
        "transport": transport_id,
        "forward": forward_ms,
        "cp1_baseline": cp1_ms,
        "peak_mib_per_rank": peak,
        "out_bitwise_vs_cp1": bool(flags[0].item() == 1.0),
        "lse_bitwise_vs_cp1": bool(flags[1].item() == 1.0),
        "repeat_bitwise": bool(flags[2].item() == 1.0),
        "out_mismatched_all_ranks": int(mismatches[0].item()),
        "lse_mismatched_all_ranks": int(mismatches[1].item()),
    }


def _run_distributed_topology(
    topology: tuple[str, int, int, int],
    *,
    warmup: int,
    samples: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    _label, tp_world, cp_world, replicas = topology
    world_size = tp_world * cp_world * replicas
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as temporary_directory:
        init_method = (Path(temporary_directory) / "rccl_init").as_uri()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_cp_worker,
                args=(
                    rank,
                    world_size,
                    topology,
                    init_method,
                    result_queue,
                    warmup,
                    samples,
                    seq_len,
                    q_heads,
                    kv_heads,
                    head_dim,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        result = None
        try:
            result = result_queue.get(timeout=1800)
        except queue.Empty as exc:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(f"timed out waiting for {topology[0]}") from exc
        finally:
            for process in processes:
                process.join(timeout=90)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            result_queue.close()
            result_queue.join_thread()
    if result is None or not result["ok"]:
        raise RuntimeError((result or {}).get("traceback", f"{topology[0]} returned no result"))
    return result["payload"]


# ---------------------------------------------------------------------------
# Environment, report and figures
# ---------------------------------------------------------------------------


def _default_platform_label(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(0).lower()
    if "mi300" in name:
        return "mi300x"
    if "h100" in name:
        return "h100"
    return name.replace(" ", "-")[:24]


def _load_comparisons(specs: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Parse ``--compare-with LABEL=PATH`` into (label, payload) pairs."""
    loaded: list[tuple[str, dict[str, Any]]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--compare-with expects LABEL=PATH, got {spec!r}")
        label, _, path = spec.partition("=")
        loaded.append((label, json.loads(Path(path).read_text())))
    return loaded


def _environment(device: torch.device | None = None) -> dict[str, Any]:
    on_gpu = device is None or device.type == "cuda"
    properties = (
        torch.cuda.get_device_properties(0) if on_gpu and torch.cuda.is_available() else None
    )
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        symbols = sorted(name for name in dir(_C) if "attention" in name) if _EXT_AVAILABLE else []
    except Exception:  # noqa: BLE001
        symbols = []
    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # noqa: BLE001
        triton_version = "unavailable"
    # Device facts must not leak into a host run's column: a CPU row reporting
    # gpu_count=8 and an RCCL collective would misdescribe what was measured.
    return {
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpu": properties.name if properties else "n/a (host execution)",
        "architecture": getattr(properties, "gcnArchName", "unknown") if properties else "n/a",
        "gpu_count": (torch.cuda.device_count() if on_gpu and torch.cuda.is_available() else 0),
        "hip": torch.version.hip if on_gpu else None,
        "cuda": torch.version.cuda if on_gpu else None,
        "torch": torch.__version__,
        "triton": triton_version if on_gpu else "n/a (host execution)",
        "python": platform.python_version(),
        "extension_attention_symbols": symbols if on_gpu else [],
        "native_collective": (
            "torch.distributed ProcessGroupNCCL (RCCL on ROCm)"
            if on_gpu
            else "n/a (single-process host run)"
        ),
    }


def _case_for(payload: dict[str, Any], dtype: str, seq_len: int) -> dict[str, Any] | None:
    for case in payload.get("single_gpu", {}).get("cases", []):
        if case["dtype"] == dtype and case["seq_len"] == seq_len:
            return case
    return None


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _write_report(
    payload: dict[str, Any],
    output_directory: Path,
    comparisons: list[tuple[str, dict[str, Any]]] | None = None,
) -> None:
    platforms: list[tuple[str, dict[str, Any]]] = [
        (payload.get("platform_label", "this run"), payload)
    ] + list(comparisons or [])
    configuration = payload["configuration"]
    lines: list[str] = []
    add = lines.append

    add("# WS2 strict ROCm Attention — bitwise parity and performance")
    add("")
    add("> Operator-only benchmark. No model checkpoint or serving engine was used;")
    add("> the shapes are Qwen3-8B's attention shapes.")
    add("")
    add("## Environment")
    add("")
    keys = sorted({k for _, pl in platforms for k in pl["environment"]})
    add("| Item | " + " | ".join(label for label, _ in platforms) + " |")
    add("|---|" + "---|" * len(platforms))
    for key in keys:
        cells = []
        for _, pl in platforms:
            value = pl["environment"].get(key, "n/a")
            if isinstance(value, list):
                value = ", ".join(value) or "none"
            cells.append(str(value))
        add(f"| {key} | " + " | ".join(cells) + " |")
    add("")
    if len(platforms) > 1:
        add(
            "A missing row below means the backend cannot exist on that platform, not that it "
            "failed: `strict-aiter` is ROCm-only, `reference-native` and `triton-bitwise` need a "
            "GPU, and only `sdpa` and `pytorch-native` also run on the host."
        )
        add("")

    add("## Methodology")
    add("")
    add(
        f"- Operator shape: `Hq={configuration['q_heads']}`, `Hkv={configuration['kv_heads']}`, "
        f"`D={configuration['head_dim']}`, `B={configuration['batch']}`, causal; sequence sweep "
        + ", ".join(str(s) for s in configuration["seq_lens"])
        + "."
    )
    add("- Measured paths:")
    add(
        "  - `sdpa`: `torch.nn.functional.scaled_dot_product_attention`. **Speed baseline only** — "
        "as in PR #325, no accuracy comparison is mixed into the speed table."
    )
    add(
        "  - `strict-aiter`: `StrictRocmAiterCKAttentionCore` called **once for all heads**. "
        "This is the core, not the production schedule: the Vime provider launches it once "
        "per (batch row, KV group). See the per-KV-group schedule table for that cost."
    )
    add(
        "  - `reference-native`: `_C.deterministic_attention_forward/backward`, the materializing "
        "FP32 reference core hipified from the shared `.cu`."
    )
    add(
        "  - `triton-bitwise`: `TritonDeterministicAttentionOp`, whose contract is bit-identity "
        "with `reference-native`."
    )
    add(
        "- Timing: CUDA events, median and p95. Peak memory is the per-call increase in "
        "`torch.cuda.max_memory_allocated` above what was live before the call."
    )
    add(
        "- Accuracy is against an FP64 oracle over the same BF16/FP16-rounded inputs. "
        "Repeat = two identical calls are bitwise equal; batch-invariant = a row computed "
        "alone is bitwise equal to the same row inside a batch."
    )
    add(
        f"- {configuration['warmup']} warmups, {configuration['samples']} measured forward "
        f"samples, {configuration['training_samples']} measured forward+backward samples. "
        "Raw medians, p95, min and max are in `results.json`."
    )
    add("")
    add("Reproduce from the repository root:")
    add("")
    add("```bash")
    add("python benchmarks/benchmark_ws2_rocm_attention.py \\")
    add(f"  --seq-lens {','.join(str(s) for s in configuration['seq_lens'])} \\")
    add(f"  --dtypes {','.join(configuration['dtypes'])} \\")
    add(f"  --warmup {configuration['warmup']} --samples {configuration['samples']} \\")
    add(f"  --training-samples {configuration['training_samples']} \\")
    add("  --output-dir benchmarks/results/ws2_rocm_mi300x")
    add("```")
    add("")

    if payload.get("unavailable_paths"):
        add("### Unavailable paths")
        add("")
        for name, reason in payload["unavailable_paths"].items():
            add(f"- `{name}`: {reason}")
        add("")

    # ---- headline
    add("## Bitwise parity: Triton port vs the native reference core")
    add("")
    add("Acceptance is 0 mismatched elements. This is the contract the Triton core exists to hold.")
    add("")
    add("| dtype | S | out mismatched | lse mismatched | dQ | dK | dV | bitwise |")
    add("|---|---:|---:|---:|---:|---:|---:|:---:|")
    backward = {row["seq_len"]: row for row in payload.get("backward_parity", [])}
    for case in payload["single_gpu"]["cases"]:
        parity = case.get("triton_vs_reference")
        if not parity:
            continue
        grads = backward.get(case["seq_len"], {}) if case["dtype"] == "bf16" else {}
        add(
            f"| {case['dtype']} | {case['seq_len']} | {parity['out_mismatched']} | "
            f"{parity['lse_mismatched']} | {_fmt(grads.get('dq_mismatched'))} | "
            f"{_fmt(grads.get('dk_mismatched'))} | {_fmt(grads.get('dv_mismatched'))} | "
            f"{_fmt(parity['bitwise'])} |"
        )
    add("")
    add("`dQ/dK/dV` are measured on the BF16 sweep only; `n/a` marks the FP16 rows.")
    add("")

    skipped_rows = [
        (label, case["dtype"], case["seq_len"], name, entry["skipped"])
        for label, pl in platforms
        for case in pl.get("single_gpu", {}).get("cases", [])
        for name, entry in case["paths"].items()
        if isinstance(entry, dict) and "skipped" in entry
    ]
    if skipped_rows:
        add("## Skipped cells")
        add("")
        add(
            "A cell whose single call implies more sampling time than the per-path budget "
            "is not measured. The observed single-call cost is reported instead, which is "
            "the useful part: it is what made the cell unaffordable."
        )
        add("")
        add("| Platform | dtype | S | Path | Why |")
        add("|---|---|---:|---|---|")
        for label, dtype, seq_len, name, why in skipped_rows:
            add(f"| {label} | {dtype} | {seq_len} | {name} | {why} |")
        add("")

    # ---- single GPU speed
    multi = len(platforms) > 1
    platform_column = "Platform | " if multi else ""
    platform_rule = "---|" if multi else ""
    for dtype in configuration["dtypes"]:
        seq_lens = sorted(
            {
                c["seq_len"]
                for _, pl in platforms
                for c in pl.get("single_gpu", {}).get("cases", [])
                if c["dtype"] == dtype
            }
        )
        if not seq_lens:
            continue
        add(f"## Single-device Attention ({dtype})")
        add("")
        add("### Forward")
        add("")
        add(
            f"| S | {platform_column}Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB | "
            "out max-abs vs FP64 | lse max-abs vs FP64 | Repeat |"
        )
        add(f"|---:|{platform_rule}---|---:|---:|---:|---:|---:|---:|:---:|")
        for seq_len in seq_lens:
            for label, pl in platforms:
                case = _case_for(pl, dtype, seq_len)
                if case is None:
                    continue
                baseline = case["paths"].get("sdpa", {}).get("forward", {}).get("median_ms")
                for name in PATH_NAMES:
                    entry = case["paths"].get(name)
                    if not entry:
                        continue
                    if "skipped" in entry:
                        prefix = f"{label} | " if multi else ""
                        add(
                            f"| {seq_len} | {prefix}{name} | skipped | — | — | — | "
                            f"{entry['out_vs_fp64']['max_abs']:.3e} | — | — |"
                        )
                        continue
                    median = entry["forward"]["median_ms"]
                    ratio = f"{median / baseline:.2f}x" if baseline else "n/a"
                    prefix = f"{label} | " if multi else ""
                    add(
                        f"| {seq_len} | {prefix}{name} | {median:.4f} | "
                        f"{entry['forward']['p95_ms']:.4f} | {ratio} | "
                        f"{entry['forward_peak_mib']:.1f} | "
                        f"{entry['out_vs_fp64']['max_abs']:.3e} | "
                        f"{_fmt(entry.get('lse_vs_fp64', {}).get('max_abs'), '.3e')} | "
                        f"{_fmt(entry['repeat_bitwise'])} |"
                    )
        add("")
        add("### Forward+backward")
        add("")
        add(f"| S | {platform_column}Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB |")
        add(f"|---:|{platform_rule}---|---:|---:|---:|---:|")
        for seq_len in seq_lens:
            for label, pl in platforms:
                case = _case_for(pl, dtype, seq_len)
                if case is None:
                    continue
                baseline = case["paths"].get("sdpa", {}).get("train_fwd_bwd", {}).get("median_ms")
                for name in PATH_NAMES:
                    entry = case["paths"].get(name)
                    if not entry or "train_fwd_bwd" not in entry:
                        continue
                    median = entry["train_fwd_bwd"]["median_ms"]
                    ratio = f"{median / baseline:.2f}x" if baseline else "n/a"
                    prefix = f"{label} | " if multi else ""
                    add(
                        f"| {seq_len} | {prefix}{name} | {median:.4f} | "
                        f"{entry['train_fwd_bwd']['p95_ms']:.4f} | {ratio} | "
                        f"{entry['train_peak_mib']:.1f} |"
                    )
        add("")
        if multi:
            add(
                "Host peak memory is an RSS high-water delta sampled from `/proc`, not an "
                "allocator statistic, so the `cpu` rows approximate and are not directly "
                "comparable to the device figures."
            )
            add("")

    # ---- production core vs reference
    add("## Production core versus the reference core")
    add("")
    add(
        "These are two different kernels, so this is a tolerance comparison, not a parity "
        "claim. It is here to size the gap, not to assert equality."
    )
    add("")
    add("| dtype | S | out max-abs | out relative-L2 | lse max-abs |")
    add("|---|---:|---:|---:|---:|")
    for case in payload["single_gpu"]["cases"]:
        gap = case.get("strict_vs_reference")
        if not gap:
            continue
        add(
            f"| {case['dtype']} | {case['seq_len']} | {gap['out']['max_abs']:.3e} | "
            f"{gap['out']['relative_l2']:.3e} | {gap['lse']['max_abs']:.3e} |"
        )
    add("")

    # ---- batch composition
    add("## Batch-composition invariance")
    add("")
    add(
        "A row computed alone must be bitwise equal to the same row inside a batch. "
        "The strict ROCm core rejects `B > 1` outright, so for that path the property is "
        "structural rather than measured."
    )
    add("")
    add("| S | Path | Bitwise | Mismatched | Note |")
    add("|---:|---|:---:|---:|---|")
    for row in payload.get("batch_composition", []):
        for name, entry in row["paths"].items():
            note = entry.get("note", "measured")
            add(
                f"| {row['seq_len']} | {name} | {_fmt(entry['out_bitwise'])} | "
                f"{entry['out_mismatched']} | {note} |"
            )
    add("")

    # ---- TP degree
    add("## TP-degree invariance of the strict ROCm core")
    add("")
    add(
        "A head shard computed under TP=N versus the same slice of an unsharded run. TP performs "
        "no cross-rank reduction in attention, so any nonzero value means the kernel's result "
        "depends on how many heads shared the launch. `raw_launch` is one launch for all heads; "
        "`one_kv_group_per_launch` is the schedule the Vime provider actually uses."
    )
    add("")
    add("| S | Schedule | TP | Local Hq | Local Hkv | out max-abs | lse max-abs | Invariant |")
    add("|---:|---|---:|---:|---:|---:|---:|:---:|")
    for row in payload.get("tp_head_sensitivity", []):
        add(
            f"| {row['seq_len']} | {row['schedule']} | {row['tp']} | {row['local_q_heads']} | "
            f"{row['local_kv_heads']} | {row['out_max_abs']:.6e} | {row['lse_max_abs']:.6e} | "
            f"{_fmt(row['invariant'])} |"
        )
    add("")

    # ---- schedule cost
    schedule = payload.get("tp_schedule_cost") or []
    if schedule:
        add("## Cost of the per-KV-group launch schedule")
        add("")
        add(
            "§ TP-degree invariance is bought by launching the core once per "
            "`(batch row, KV group)` instead of once for all heads. This table is that "
            "bill. `raw_launch` is one launch for all heads and is **not** the production "
            "schedule; `per_kv_group` is what the Vime provider actually runs "
            "(`Hkv` launches per row)."
        )
        add("")
        add(
            "| S | Launches | sdpa (ms) | raw_launch (ms) | per_kv_group (ms) | "
            "vs raw | vs sdpa |"
        )
        add("|---:|---:|---:|---:|---:|---:|---:|")
        for row in schedule:
            raw = row["raw_launch"]["median_ms"]
            group = row["one_kv_group_per_launch"]["median_ms"]
            sdpa = row["sdpa"]["median_ms"]
            add(
                f"| {row['seq_len']} | {row['launches']} | {sdpa:.4f} | {raw:.4f} | "
                f"{group:.4f} | {group / raw:.2f}x | {group / sdpa:.2f}x |"
            )
        add("")

    # ---- distributed
    distributed = payload.get("distributed") or []
    if distributed:
        add("## Distributed CP (RCCL AG/RS transport)")
        add("")
        add(
            "Schedule: all-gather Q/K/V and the position ids over the CP group, run the strict "
            "core once on the full sequence, reduce-scatter `(out, lse)` back to this rank's "
            "query range. Acceptance is bitwise against a CP=1 run of the same core."
        )
        add("")
        add(
            "| Topology | World | TP | CP | Replicas | S | Median (ms) | p95 (ms) | "
            "Peak MiB/rank | out bitwise | lse bitwise | Repeat |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|")
        for row in distributed:
            if "error" in row:
                add(
                    f"| {row['topology']} | — | — | — | — | — | — | — | — | "
                    "error | error | error |"
                )
                continue
            add(
                f"| {row['topology']} | {row['world_size']} | {row['tp_world_size']} | "
                f"{row['cp_world_size']} | {row.get('replicas', 1)} | {row['seq_len']} | "
                f"{row['forward']['median_ms']:.4f} | "
                f"{row['forward']['p95_ms']:.4f} | {row['peak_mib_per_rank']:.1f} | "
                f"{_fmt(row['out_bitwise_vs_cp1'])} | {_fmt(row['lse_bitwise_vs_cp1'])} | "
                f"{_fmt(row['repeat_bitwise'])} |"
            )
        add("")
        errors = [row for row in distributed if "error" in row]
        for row in errors:
            add(f"- `{row['topology']}` failed: {row['error']}")
        if errors:
            add("")

    add("## Figures")
    add("")
    add(
        "`reference-native` and `triton-bitwise` allocate exactly the same buffers, so their "
        "memory curves coincide and the later-drawn series hides the earlier one."
    )
    add("")
    add("![Single-device latency and memory grid](single_gpu_grid.png)")
    add("")
    add("![Single-device latency](single_gpu_latency.png)")
    add("")
    add("![Single-device peak memory](single_gpu_memory.png)")
    add("")
    add("![Bitwise exactness matrix](exactness_matrix.png)")
    add("")
    add("![TP-degree invariance](tp_degree_invariance.png)")
    add("")
    if distributed:
        add("![Distributed CP latency](distributed_cp_latency.png)")
        add("")

    (output_directory / "report.md").write_text("\n".join(lines) + "\n")


def _write_figures(payload: dict[str, Any], output_directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8})

    style = {
        "sdpa": {"marker": "o", "color": "#888888", "linestyle": "--"},
        "strict-aiter": {"marker": "s", "color": "#d62728"},
        "reference-native": {"marker": "^", "color": "#1f77b4"},
        "triton-bitwise": {"marker": "D", "color": "#2ca02c"},
    }

    cases = [c for c in payload["single_gpu"]["cases"] if c["dtype"] == "bf16"]
    if not cases:
        return
    seq_lens = sorted({c["seq_len"] for c in cases})
    present = [n for n in PATH_NAMES if any(n in c["paths"] for c in cases)]

    panels = (
        ("forward", "median_ms", "Forward latency", "median ms", True),
        ("train_fwd_bwd", "median_ms", "Forward+backward latency", "median ms", True),
        ("forward_peak_mib", None, "Forward peak memory", "peak MiB above live", True),
        ("train_peak_mib", None, "Forward+backward peak memory", "peak MiB above live", True),
    )

    def value(case, name, key, sub):
        entry = case["paths"].get(name)
        if entry is None or key not in entry:
            return float("nan")
        return entry[key][sub] if sub else entry[key]

    def draw(axis, key, sub, title, ylabel, log_y):
        for index, name in enumerate(present):
            ys = [
                value(next(c for c in cases if c["seq_len"] == s), name, key, sub) for s in seq_lens
            ]
            axis.plot(
                seq_lens,
                ys,
                label=name,
                linewidth=3.0 - 0.35 * index,
                markersize=7 - 0.5 * index,
                zorder=3 + index,
                **style.get(name, {}),
            )
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("sequence length")
        axis.set_ylabel(ylabel)
        axis.set_title(f"BF16: {title}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    for filename, chosen in (
        ("single_gpu_latency.png", panels[:2]),
        ("single_gpu_memory.png", panels[2:]),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, (key, sub, title, ylabel, log_y) in zip(axes, chosen):
            draw(axis, key, sub, title, ylabel, log_y)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (key, sub, title, ylabel, log_y) in zip(axes.flat, panels):
        draw(axis, key, sub, title, ylabel, log_y)
    figure.suptitle(
        "Single-device strict Attention, BF16, Qwen3-8B shape "
        f"(Hq={QWEN3_8B_Q_HEADS}, Hkv={QWEN3_8B_KV_HEADS}, D={QWEN3_8B_HEAD_DIM})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_directory / "single_gpu_grid.png", dpi=180)
    plt.close(figure)

    # TP head-count sensitivity: raw launch versus one KV group per launch.
    tp_rows = payload.get("tp_head_sensitivity") or []
    if tp_rows:
        figure, axis = plt.subplots(figsize=(11, 5))
        for schedule, marker in (("raw_launch", "o"), ("one_kv_group_per_launch", "s")):
            rows = [r for r in tp_rows if r["schedule"] == schedule]
            if not rows:
                continue
            labels = [f"S={r['seq_len']}\nTP={r['tp']}" for r in rows]
            axis.plot(
                range(len(rows)),
                [max(r["out_max_abs"], 1e-12) for r in rows],
                marker=marker,
                label=schedule,
                linewidth=2.4,
            )
        axis.set_xticks(range(len([r for r in tp_rows if r["schedule"] == "raw_launch"])))
        axis.set_xticklabels(
            [f"S={r['seq_len']}\nTP={r['tp']}" for r in tp_rows if r["schedule"] == "raw_launch"],
            fontsize=8,
        )
        axis.set_yscale("log")
        axis.set_ylabel("out max-abs vs unsharded slice (1e-12 == bitwise)")
        axis.set_title("TP-degree invariance of the strict ROCm core, BF16")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_directory / "tp_degree_invariance.png", dpi=180)
        plt.close(figure)

    distributed = [r for r in (payload.get("distributed") or []) if "error" not in r]
    if distributed:
        figure, axis = plt.subplots(figsize=(max(9, 1.7 * len(distributed)), 5.0))
        labels = [f"{r['topology']}\nS={r['seq_len']}" for r in distributed]
        xs = list(range(len(distributed)))
        width = 0.38
        baseline = [r.get("cp1_baseline", {}).get("median_ms", float("nan")) for r in distributed]
        measured = [r["forward"]["median_ms"] for r in distributed]
        bars_a = axis.bar(
            [x - width / 2 for x in xs], baseline, width, label="CP=1 baseline", color="#888888"
        )
        bars_b = axis.bar(
            [x + width / 2 for x in xs], measured, width, label="CP AG/RS", color="#2ca02c"
        )
        for bars in (bars_a, bars_b):
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=9)
        axis.set_ylabel("median ms")
        axis.set_title(
            "Strict ROCm Attention: CP=1 baseline vs RCCL AG/RS CP transport, BF16 S=4096"
        )
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_directory / "distributed_cp_latency.png", dpi=180)
        plt.close(figure)

    # ---- exactness matrix, in the shape PR #325 uses for its topology mismatch heatmap
    single_rows, single_labels = [], []
    for case in payload["single_gpu"]["cases"]:
        parity = case.get("triton_vs_reference")
        if not parity:
            continue
        grads = next(
            (r for r in payload.get("backward_parity", []) if r["seq_len"] == case["seq_len"]),
            {},
        )
        use_grads = case["dtype"] == "bf16" and grads
        # Gradients are only measured on the BF16 sweep; NaN renders as "not measured"
        # rather than a zero that would read as "measured and equal".
        missing = float("nan")
        single_rows.append(
            [
                parity["out_mismatched"],
                parity["lse_mismatched"],
                grads.get("dq_mismatched", missing) if use_grads else missing,
                grads.get("dk_mismatched", missing) if use_grads else missing,
                grads.get("dv_mismatched", missing) if use_grads else missing,
            ]
        )
        single_labels.append(f"{case['dtype']}, S={case['seq_len']}")

    dist_rows = [
        [r["out_mismatched_all_ranks"], r["lse_mismatched_all_ranks"]] for r in distributed
    ]
    dist_labels = [
        f"{r['topology']} (TP={r['tp_world_size']}, CP={r['cp_world_size']})" for r in distributed
    ]

    if single_rows or dist_rows:
        panels = []
        if single_rows:
            panels.append(
                (
                    single_rows,
                    single_labels,
                    ["out", "lse", "dQ", "dK", "dV"],
                    "Triton core vs native reference core",
                )
            )
        if dist_rows:
            panels.append((dist_rows, dist_labels, ["out", "lse"], "CP topology vs CP=1"))
        figure, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.6), squeeze=False)
        for axis, (rows, row_labels, col_labels, title) in zip(axes.flat, panels):
            matrix = [[float(value) for value in row] for row in rows]
            colormap = plt.get_cmap("RdYlGn_r").copy()
            colormap.set_bad("#d9d9d9")
            image = axis.imshow(matrix, aspect="auto", cmap=colormap, vmin=0, vmax=1.0)
            axis.grid(False)
            for y in range(len(matrix)):
                for x in range(len(matrix[y])):
                    cell = matrix[y][x]
                    label = "n/m" if cell != cell else str(int(cell))
                    axis.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=15,
                        fontweight="bold",
                        color="#222222",
                    )
            axis.set_xticks(range(len(col_labels)), col_labels)
            axis.set_yticks(range(len(row_labels)), row_labels)
            axis.set_xlabel("Compared tensor")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03).set_label(
                "Mismatched elements"
            )
        figure.suptitle(
            "Bitwise exactness — every measured cell must be 0 (n/m = not measured)",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(output_directory / "exactness_matrix.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/ws2_rocm_mi300x")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--training-samples", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=QWEN3_8B_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=QWEN3_8B_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=QWEN3_8B_HEAD_DIM)
    parser.add_argument(
        "--seq-lens",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=DEFAULT_SEQ_LENS,
    )
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument(
        "--path-budget-seconds",
        type=float,
        default=300.0,
        help=(
            "Skip a (path, case) whose measured single call implies more than this many "
            "seconds of sampling. The row records the observed cost instead, which is "
            "itself the finding. 0 disables the budget."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cpu. On cpu only the sdpa and pytorch-native paths exist.",
    )
    parser.add_argument(
        "--platform-label",
        default=None,
        help="Name for this run in merged reports (default: mi300x / h100 / cpu, auto-detected).",
    )
    parser.add_argument(
        "--compare-with",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Merge another platform's results.json into the report. Repeatable.",
    )
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--distributed-only",
        action="store_true",
        help="Re-run only the distributed topologies and merge into an existing results.json.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-render report.md and the figures from an existing results.json.",
    )
    arguments = parser.parse_args()

    if arguments.report_only:
        payload = json.loads((arguments.output_dir / "results.json").read_text())
        _write_report(payload, arguments.output_dir, _load_comparisons(arguments.compare_with))
        if not arguments.skip_figures:
            _write_figures(payload, arguments.output_dir)
        print(json.dumps({"output_dir": str(arguments.output_dir)}, indent=2))
        return

    if arguments.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = arguments.device
    if device_type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA/ROCm device is visible")

    device = torch.device(device_type, 0) if device_type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtypes = tuple(dtype_map[name] for name in arguments.dtypes.split(","))

    paths = _Paths(device)
    payload: dict[str, Any] = {
        "platform_label": arguments.platform_label or _default_platform_label(device),
        "environment": _environment(device),
        "configuration": {
            "batch": arguments.batch,
            "q_heads": arguments.q_heads,
            "kv_heads": arguments.kv_heads,
            "head_dim": arguments.head_dim,
            "seq_lens": list(arguments.seq_lens),
            "dtypes": arguments.dtypes.split(","),
            "warmup": arguments.warmup,
            "samples": arguments.samples,
            "training_samples": arguments.training_samples,
        },
        "unavailable_paths": paths.errors,
    }

    existing: dict[str, Any] = {}
    if arguments.distributed_only:
        existing = json.loads((arguments.output_dir / "results.json").read_text())
        payload = dict(existing)
        payload["environment"] = _environment()

    if not arguments.distributed_only:
        payload["single_gpu"] = {
            "cases": _single_gpu_benchmarks(
                paths=paths,
                seq_lens=arguments.seq_lens,
                dtypes=dtypes,
                q_heads=arguments.q_heads,
                kv_heads=arguments.kv_heads,
                head_dim=arguments.head_dim,
                batch=arguments.batch,
                warmup=arguments.warmup,
                samples=arguments.samples,
                training_samples=arguments.training_samples,
                device=device,
            )
        }
        payload["backward_parity"] = _backward_parity(
            paths=paths,
            seq_lens=arguments.seq_lens,
            q_heads=arguments.q_heads,
            kv_heads=arguments.kv_heads,
            head_dim=arguments.head_dim,
            batch=arguments.batch,
            device=device,
        )
        payload["batch_composition"] = _batch_composition(
            paths=paths,
            seq_lens=arguments.seq_lens,
            q_heads=arguments.q_heads,
            kv_heads=arguments.kv_heads,
            head_dim=arguments.head_dim,
            device=device,
        )
        payload["tp_head_sensitivity"] = (
            []
            if device.type != "cuda"
            else _tp_head_sensitivity(
                paths=paths,
                seq_lens=arguments.seq_lens,
                tp_degrees=DEFAULT_TP_DEGREES,
                q_heads=arguments.q_heads,
                kv_heads=arguments.kv_heads,
                head_dim=arguments.head_dim,
                device=device,
            )
        )

    distributed: list[dict[str, Any]] = []
    if not arguments.skip_distributed and device.type == "cuda":
        available = torch.cuda.device_count()
        for topology in DISTRIBUTED_TOPOLOGIES:
            if topology[1] * topology[2] * topology[3] > available:
                continue
            try:
                distributed.append(
                    _run_distributed_topology(
                        topology,
                        warmup=arguments.warmup,
                        samples=arguments.samples,
                        seq_len=arguments.seq_lens[-1] if arguments.seq_lens else 2048,
                        q_heads=arguments.q_heads,
                        kv_heads=arguments.kv_heads,
                        head_dim=arguments.head_dim,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a failed topology is a reported row
                distributed.append(
                    {"topology": topology[0], "error": f"{type(exc).__name__}: {exc}"}
                )
    payload["distributed"] = distributed

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(payload, arguments.output_dir, _load_comparisons(arguments.compare_with))
    if not arguments.skip_figures:
        _write_figures(payload, arguments.output_dir)
    print(json.dumps({"output_dir": str(arguments.output_dir)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    main()
