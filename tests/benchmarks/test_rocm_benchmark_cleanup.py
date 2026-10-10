# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for benchmark callbacks and buffer lifetimes, not GPU kernels."""

from __future__ import annotations

import ast
import hashlib
import math
import statistics
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]


def _load_definitions(filename, **scope):
    # Execute the real orchestration with isolated CPU backends; importing these
    # scripts normally requires Triton, Transformers, and a GPU installation.
    path = _ROOT / "benchmarks/backends/rocm" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or isinstance(node, ast.ClassDef)
        and node.name in {"LeafConfig", "LeafCase"}
    ]
    scope.update(
        __name__=__name__,
        dataclass=dataclass,
        asdict=asdict,
        hashlib=hashlib,
        math=math,
        statistics=statistics,
    )
    exec(compile(tree, str(path), "exec"), scope)
    return scope


class _CPUTorch:
    def __init__(self, empty_cache=lambda: None):
        self.cuda = SimpleNamespace(
            set_device=lambda device: None,
            synchronize=lambda: None,
            empty_cache=empty_cache,
        )
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()))

    def device(self, *args):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.mark.parametrize("transpose_output", [False, True])
def test_leaf_callbacks_release_candidate_buffers(transpose_output):
    scope = _load_definitions("benchmark_rocm_det_gemm_leaf.py", torch=_CPUTorch())
    baseline = scope["LeafConfig"](64, 64, 4)
    candidate = scope["LeafConfig"](32, 64, 4)
    case = scope["LeafCase"]("small", 2, 3, 4, transpose_output=transpose_output)
    plan = SimpleNamespace(
        host=SimpleNamespace(node_count=1, leaf_nodes=[0], reduction_levels=[]),
        leaf_nodes=torch.tensor([0]),
    )
    buffers = []
    callbacks = []

    def launch_leaf(case, config, a, b, workspace, plan):
        workspace[0].copy_(a @ b)

    def launch_tree(case, config, a, b, workspace, output, plan):
        launch_leaf(case, config, a, b, workspace, plan)
        output.copy_(workspace[0].T if case.transpose_output else workspace[0])

    def measure(callback, *, warmup, samples):
        # Retaining callbacks exposes accidental strong references after cleanup.
        callbacks.append(callback)
        cells = dict(zip(callback.__code__.co_freevars, callback.__closure__))
        buffers.append(weakref.ref(cells["workspace"].cell_contents))
        if "output" in cells:
            buffers.append(weakref.ref(cells["output"].cell_contents))
        for _ in range(warmup + samples):
            callback()
        return {"median_ms": 1.0}

    scope.update(
        _BASELINE=baseline,
        _inputs=lambda case, device: (torch.ones(2, 3), torch.ones(3, 4)),
        _device_tree_plan=lambda k, device: plan,
        _launch_leaf=launch_leaf,
        _launch_tree=launch_tree,
        _measure=measure,
    )
    result = scope["_run_case"](
        case, [baseline, candidate], device=torch.device("cpu"), warmup=1, samples=2
    )
    assert len(callbacks) == 4
    assert len(result["results"]) == 2
    assert all(row["leaf_raw_bytes_equal"] for row in result["results"])
    assert all(row["root_raw_bytes_equal"] for row in result["results"])
    assert all(ref() is None for ref in buffers)


class _PackedWeights:
    def __init__(self, weights):
        self.weights = weights


def _cpu_ffn(hidden, gate, up, down, *, forward_weights=None, **kwargs):
    if forward_weights is not None:
        assert all(a is b for a, b in zip(forward_weights.weights, (gate, up, down)))
    return F.linear(F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down)


class _CPUMLP(torch.nn.Module):
    def __init__(self, *weights):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(value.clone()) for value in weights]
        )

    def forward(self, hidden):
        return _cpu_ffn(hidden, *self.weights)


@pytest.mark.parametrize("distributed", [False, True])
def test_ffn_callbacks_release_packed_weights_between_cases(distributed):
    packed_refs = []
    live_at_cleanup = []
    callbacks = []

    def empty_cache():
        live_at_cleanup.append(sum(ref() is not None for ref in packed_refs))

    scope = _load_definitions("benchmark_rocm_ffn.py", torch=_CPUTorch(empty_cache))

    def randn(shape, *, seed, device, dtype=torch.float32):
        # Preserve token counts while reducing model dimensions for CPU tests.
        shape = tuple({4096: 4, 12288: 8}.get(size, size) for size in shape)
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)

    def pack(*weights):
        packed = _PackedWeights(weights)
        packed_refs.append(weakref.ref(packed))
        return packed

    def samples(callback, *, warmup, samples, **kwargs):
        callbacks.append(callback)
        for _ in range(warmup + samples):
            callback()
        return [1.0] * samples

    def gather(output, value, **kwargs):
        output[0] = value

    scope.update(
        _randn=randn,
        _official_qwen3_mlp=_CPUMLP,
        _official_distributed_ffn=_cpu_ffn,
        pack_qwen3_ffn_forward_weights=pack,
        qwen3_ffn=_cpu_ffn,
        _gpu_event_samples=samples,
        _distributed_wall_samples=samples,
        dist=SimpleNamespace(
            group=SimpleNamespace(WORLD=object()),
            get_world_size=lambda **kwargs: 1,
            all_gather_object=gather,
            barrier=lambda: None,
        ),
        ffn_module=SimpleNamespace(_COLLECTIVES={}),
    )
    if distributed:
        # Two single-rank cases exercise loop cleanup without claiming to test
        # real inter-rank communication or GPU arithmetic.
        rows = scope["_distributed_ffn_benchmark"](
            0,
            1,
            (("first", 1, 1, False), ("second", 1, 1, False)),
            warmup=1,
            samples=2,
            training_samples=2,
        )
        assert len(rows) == 4
        assert len(callbacks) == 8
        assert live_at_cleanup == [0, 0, 0]
        for row in rows:
            assert all(value == 0 for value in row["tp1_mismatch"].values())
            assert row["repeat_mismatch_count"] == 0
            assert row["train_infer_mismatch_count"] == 0
    else:
        result = scope["_single_gpu_benchmarks"](warmup=1, samples=2, training_samples=2)
        assert len(result["speed"]) == 6
        assert [row["tokens"] for row in result["speed"]] == [1, 1, 8, 8, 32, 32]
        assert len(result["dtype_accuracy"]) == 1
        assert len(callbacks) == 12
        # The shared inference cache survives the three token-count cases only.
        assert live_at_cleanup == [1, 1, 1, 0, 0]
    assert all(ref() is None for ref in packed_refs)
