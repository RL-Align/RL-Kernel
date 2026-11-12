# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks/distributed/benchmark_rocm_collectives.py"
_SPEC = importlib.util.spec_from_file_location(
    "rlkernel_rocm_collective_benchmark", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_rocm_collective_benchmark_parser() -> None:
    args = benchmark.parse_args(
        [
            "--size-bytes",
            "1024",
            "4096",
            "--dtype",
            "fp32",
            "--operations",
            "all_reduce",
            "reduce_scatter",
            "--warmup",
            "2",
            "--iterations",
            "3",
            "--samples",
            "4",
        ]
    )

    benchmark._validate_args(args)
    assert args.size_bytes == [1024, 4096]
    assert args.dtype == "fp32"
    assert args.operations == ["all_reduce", "reduce_scatter"]
    assert (args.warmup, args.iterations, args.samples) == (2, 3, 4)


@pytest.mark.parametrize(
    "argv",
    (
        ["--size-bytes", "0"],
        ["--warmup", "-1"],
        ["--iterations", "0"],
        ["--samples", "0"],
    ),
)
def test_rocm_collective_benchmark_rejects_invalid_counts(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        benchmark._validate_args(benchmark.parse_args(argv))


def test_rocm_collective_benchmark_aligns_reduce_scatter_input() -> None:
    tensor, actual_bytes = benchmark._make_inputs(
        size_bytes=35,
        dtype=torch.float32,
        world_size=4,
        rank=2,
        device=torch.device("cpu"),
    )

    assert tensor.is_contiguous()
    assert tensor.numel() % 4 == 0
    assert actual_bytes == tensor.numel() * tensor.element_size()
