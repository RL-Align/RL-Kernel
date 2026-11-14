# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Device time of a call from GPU events around a CUDA graph, and table cells for it.

WARMUP plain calls first (they also compile), then a CUDA graph of ITERS calls
is captured and replayed TRIALS times between two GPU events, so CPU launch
overhead is not counted. The host synchronizes only to read the events.
"""

from __future__ import annotations

import socket
import statistics

import torch

WARMUP, ITERS, TRIALS = 10, 20, 5


def time_ms(call):
    """(median, min, max) device time per call, in ms, over TRIALS graph replays."""
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ITERS):
            call()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(TRIALS):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / ITERS)
    return statistics.median(samples), min(samples), max(samples)


def measure(call):
    """``time_ms(call)``, or "cannot launch: <reason>" if the call raises."""
    try:
        return time_ms(call)
    except Exception as exc:  # e.g. a kernel that needs more LDS than the GPU has
        return launch_error(exc)


def launch_error(exc):
    """"cannot launch: <reason>" on one line, safe in a markdown table cell."""
    lines = str(exc).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    return f"cannot launch: {reason}"[:200].replace("|", "/")


def time_cell(result):
    """"median (min-max)" in ms, or the reason the call could not run."""
    if isinstance(result, str):
        return result
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def ratio_cell(ours, baseline):
    """Median of ours divided by median of the baseline."""
    if isinstance(ours, str) or isinstance(baseline, str):
        return "n/a"
    return f"{ours[0] / baseline[0]:.2f}"


def environment():
    """Host, GPU and the torch, Triton and aiter versions, for a table label."""
    import triton

    try:
        import aiter

        aiter_version = getattr(aiter, "__version__", "unknown version")
    except Exception as exc:
        aiter_version = f"not available ({type(exc).__name__})"
    return (
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}, Triton {triton.__version__}, aiter {aiter_version}."
    )
