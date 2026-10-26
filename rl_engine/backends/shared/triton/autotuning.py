# SPDX-License-Identifier: Apache-2.0
"""Triton launch adapter with isolated validation buffers and pinned autograd policy."""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import statistics
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from rl_engine.platforms.autotuner import PersistentAutotuner, TuneConfig, autotune_enabled


@dataclass
class TuneState:
    """Per-call tuning policy, plus the launch records collected for provenance."""

    enabled: bool
    core: bool = False
    records: list[dict] = field(default_factory=list)
    phase: str = "forward"


_STATE: ContextVar[TuneState | None] = ContextVar("rl_kernel_autotune", default=None)
_TUNER = PersistentAutotuner()
_DEVICES: dict[tuple[int, int], dict] = {}


@contextmanager
def tuning_scope(state: TuneState):
    """Apply ``state`` to the Triton launches made inside this context."""
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def pin_autotune(function_class):
    """Pin the forward policy on the autograd context, including core opt-in."""
    forward = function_class.forward
    backward = function_class.backward

    @wraps(forward)
    def pinned_forward(ctx, *args, **kwargs):
        """Run forward under the current policy and keep it for backward."""
        ctx.autotune_state = _STATE.get()
        if ctx.autotune_state is not None:
            return forward(ctx, *args, **kwargs)
        ctx.autotune_state = TuneState(autotune_enabled())
        with tuning_scope(ctx.autotune_state):
            return forward(ctx, *args, **kwargs)

    @wraps(backward)
    def pinned_backward(ctx, *args, **kwargs):
        """Run backward under the policy captured in forward."""
        with tuning_scope(replace(ctx.autotune_state, phase="backward")):
            return backward(ctx, *args, **kwargs)

    function_class.forward = staticmethod(pinned_forward)
    function_class.backward = staticmethod(pinned_backward)
    return function_class


def _device_identity(device: torch.device, capture: bool) -> dict | None:
    """Return cached device, driver and toolkit identity; None if uncached during capture."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (os.getpid(), index)
    if key in _DEVICES:
        return _DEVICES[key]
    if capture:
        return None
    props = torch.cuda.get_device_properties(index)
    driver_path = Path(
        "/sys/module/amdgpu/version" if torch.version.hip else "/proc/driver/nvidia/version"
    )
    try:
        driver = driver_path.read_text().strip()
    except OSError:
        driver = None
        if torch.version.hip:
            try:
                driver = (
                    platform.release()
                    + ":"
                    + Path("/sys/module/amdgpu/srcversion").read_text().strip()
                )
            except OSError:
                driver = None  # Without driver metadata, use only process-local selections.
    identity = {
        "index": index,
        "host": platform.node(),
        "name": props.name,
        "architecture": str(getattr(props, "gcnArchName", (props.major, props.minor))),
        "uuid": str(getattr(props, "uuid", "")),
        "pci_bus_id": str(getattr(props, "pci_bus_id", "")),
        "pci_domain_id": str(getattr(props, "pci_domain_id", "")),
        "pci_device_id": str(getattr(props, "pci_device_id", "")),
        "driver": driver,
        "torch": str(torch.__version__),
        "triton": version("triton"),
        "runtime": torch.version.hip or torch.version.cuda,
    }
    _DEVICES[key] = identity
    return identity


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors have identical bytes."""
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def _profile_candidates(candidates, args, outputs, inout, run):
    """Compare validated kernels using interleaved graph replays on the caller's stream.

    Multiple captured launches remove Python dispatch gaps from GPU timing.
    Read/write buffers are reset inside the graph; their identical reset cost
    is included for every candidate. Pure outputs need no reset between launches.
    """
    from triton.runtime.errors import OutOfResources

    reference, scratch = list(args), list(args)
    for index in outputs:
        reference[index] = args[index].clone()
        scratch[index] = args[index].clone()
    run(TuneConfig(), reference)
    valid = []
    timings = dict.fromkeys(candidates)
    # Finish all JIT compilation before timing any candidate.
    for config in candidates:
        for index in outputs:
            scratch[index].copy_(args[index])
        try:
            run(config, scratch)
        except OutOfResources:
            if config == TuneConfig():
                raise
            continue
        if all(_bitwise_equal(reference[index], scratch[index]) for index in outputs):
            valid.append(config)

    repeats = 32
    graphs = {}
    caller_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(caller_stream)
    for config in valid:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(repeats):
                for index in inout:
                    scratch[index].copy_(args[index])
                run(config, scratch)
        graphs[config] = graph
    caller_stream.wait_stream(capture_stream)
    # Replays (including timing events) run on the caller's current stream.
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {config: [] for config in valid}
    for iteration in range(9):
        order = valid if iteration % 2 == 0 else reversed(valid)
        for config in order:
            start.record()
            graphs[config].replay()
            end.record()
            end.synchronize()
            samples[config].append(start.elapsed_time(end) / repeats)
    for config in valid:
        timings[config] = statistics.median(samples[config])
    return timings


def launch_candidates(kernel) -> tuple[TuneConfig, ...]:
    """Single source of launch candidates for selection and numerical qualification."""
    candidates = [TuneConfig()]
    # Changing reduction warp layouts can reorder arithmetic on later inputs.
    if kernel.tune_warps:
        candidates.extend(
            TuneConfig((("num_warps", warps), ("num_stages", stages)))
            for warps in (4, 8)
            for stages in (1, 2)
        )
    else:
        candidates.extend(TuneConfig((("num_stages", stages),)) for stages in (1, 2))
    if kernel.flat_block is not None:
        candidates.extend(
            TuneConfig((("BLOCK", block), ("num_warps", 4))) for block in (256, 512, 2048)
        )
    return tuple(candidates)


class TunableKernel:
    """Wrap a JIT launch; outputs lists every mutated positional tensor.

    ``flat_block`` permits changing BLOCK and recomputes the one-dimensional
    grid from the specified positional element count. Reduction layouts stay fixed.
    ``inout`` marks outputs whose original contents must be restored before each
    benchmark iteration. Core kernels require an explicit performance-backend scope.
    """

    def __init__(
        self,
        kernel,
        *,
        outputs: tuple[int, ...],
        core: bool = False,
        flat_block: int | None = None,
        tune_warps: bool = False,
        inout: tuple[int, ...] = (),
    ):
        """Record the launch contract and hash kernel and tuner sources as the revision."""
        self._launches: dict[tuple, tuple[TuneConfig, Any, dict[str, Any]]] = {}
        self.kernel = kernel
        self.outputs = outputs
        if not set(inout).issubset(outputs):
            raise ValueError("Read/write buffers must also be declared as outputs")
        self.inout = inout
        self.core = core
        self.flat_block = flat_block
        self.tune_warps = tune_warps or flat_block is not None
        self.name = f"{kernel.fn.__module__}.{kernel.fn.__name__}"
        # Include helpers and host launch constants, not just the JIT body.
        kernel_module = inspect.getmodule(kernel.fn)
        tuner_module = inspect.getmodule(TuneState)
        if kernel_module is None or tuner_module is None:
            raise RuntimeError(f"Cannot resolve the source module of {self.name}")
        source = inspect.getsource(kernel_module) + inspect.getsource(tuner_module)
        self.revision = hashlib.sha256(source.encode()).hexdigest()

    def __getitem__(self, grid):
        """Return a launcher for ``grid`` that uses the cached or newly selected config."""

        def launch(*args, **kwargs):
            """Launch with the selected configuration, or the fixed one when tuning is off."""
            state = _STATE.get()
            enabled = state.enabled if state is not None else autotune_enabled()
            if (
                not enabled
                or (self.core and (state is None or not state.core))
                # An output passed by keyword cannot be cloned for benchmarking: launch it fixed.
                or len(args) <= max(self.outputs)
                or args[self.outputs[0]].device.type != "cuda"
            ):
                return self.kernel[grid](*args, **kwargs)
            device = args[self.outputs[0]].device
            device_scope = (
                nullcontext()
                if device.index == torch.cuda.current_device()
                else torch.cuda.device(device)
            )
            with device_scope:
                capture = torch.cuda.is_current_stream_capturing()
                phase = state.phase if state is not None else "forward"
                local_key = (
                    _TUNER,
                    os.getpid(),
                    phase,
                    grid,
                    tuple(
                        (
                            (a.shape, a.stride(), a.dtype, a.device)
                            if isinstance(a, torch.Tensor)
                            else a
                        )
                        for a in args
                    ),
                    tuple(sorted(kwargs.items())),
                )
                cached = self._launches.get(local_key)
                if cached is not None:
                    config, runner, options = cached
                    if state is not None and not capture:
                        state.records.append(
                            {
                                "kernel": self.name,
                                "phase": phase,
                                "config": config.to_dict(),
                                "source": "memory",
                            }
                        )
                    return runner(*args, **options)
                identity = _device_identity(device, capture)
                if identity is None:
                    return self.kernel[grid](*args, **kwargs)
                candidates = launch_candidates(self)
                signature = {
                    "kernel": self.name,
                    "phase": state.phase if state is not None else "forward",
                    "revision": self.revision,
                    "device": identity,
                    "args": [
                        (
                            {
                                "shape": list(a.shape),
                                "stride": list(a.stride()),
                                "dtype": str(a.dtype),
                                "device": str(a.device),
                            }
                            if isinstance(a, torch.Tensor)
                            else a
                        )
                        for a in args
                    ],
                    "kwargs": kwargs,
                    "grid": list(grid) if not callable(grid) else "config_grid",
                }

                def run(config, buffers):
                    """Launch ``config`` on private output buffers."""
                    options = {**kwargs, **config.to_dict()}
                    selected_grid = grid
                    if self.flat_block is not None and "BLOCK" in config.to_dict():
                        count = args[self.flat_block]
                        selected_grid = ((count + options["BLOCK"] - 1) // options["BLOCK"],)
                    return self.kernel[selected_grid](*buffers, **options)

                timings = None

                def evaluate(config):
                    """Time ``config``, or return None if its output bytes differ."""
                    nonlocal timings
                    if timings is None:
                        timings = _profile_candidates(
                            candidates, args, self.outputs, self.inout, run
                        )
                    return timings[config]

                selected = _TUNER.select(
                    signature,
                    candidates,
                    evaluate,
                    persistent=bool(
                        identity["driver"] and (identity["uuid"] or identity["pci_bus_id"])
                    ),
                    capture=capture,
                )
                if not capture:
                    options = {**kwargs, **selected.config.to_dict()}
                    selected_grid = grid
                    if self.flat_block is not None and "BLOCK" in selected.config.to_dict():
                        count = args[self.flat_block]
                        selected_grid = ((count + options["BLOCK"] - 1) // options["BLOCK"],)
                    self._launches[local_key] = (
                        selected.config,
                        self.kernel[selected_grid],
                        options,
                    )
                if state is not None and not capture:
                    state.records.append(
                        {
                            "kernel": self.name,
                            "phase": phase,
                            "config": selected.config.to_dict(),
                            "source": selected.source,
                        }
                    )
                return run(selected.config, args)

        return launch
