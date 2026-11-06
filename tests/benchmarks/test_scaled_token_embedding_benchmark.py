# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for the benchmark's input patterns, timing boundaries and forward gate."""

import pytest
import torch

from benchmarks.operators.embedding.benchmark_scaled_token_embedding import (
    PAD_ID,
    HFScaledEmbedding,
    _make_workload,
    backward_gap,
    build_arg_parser,
    check_forward_bitwise,
    make_token_ids,
    run_benchmark,
)


def _generator():
    return torch.Generator().manual_seed(0)


def test_unique_ids_never_repeat():
    ids = make_token_ids(500, 1000, "unique", generator=_generator(), device="cpu")
    assert ids.shape == (500,) and ids.unique().numel() == 500
    with pytest.raises(ValueError, match="n_tokens <= vocab"):
        make_token_ids(1001, 1000, "unique", generator=_generator(), device="cpu")


def test_duplicate_ids_form_long_segments():
    ids = make_token_ids(4096, 262144, "duplicate", generator=_generator(), device="cpu")
    assert ids.unique().numel() <= 64


def test_pad_flood_puts_half_of_the_tokens_on_the_pad_id():
    ids = make_token_ids(4096, 262144, "pad_flood", generator=_generator(), device="cpu")
    assert int((ids == PAD_ID).sum()) >= 2048


@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_workload_reuses_inputs_without_accumulating_gradients(mode):
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    token_ids = torch.tensor([1, 1, 3])
    grad_out = torch.ones(3, 3)
    baseline = HFScaledEmbedding(torch.tensor(2.0))
    calls = []
    tables = []

    def observed(ids, table):
        calls.append(torch.is_grad_enabled())
        tables.append(table)
        return baseline(ids, table)

    fn = _make_workload(observed, token_ids, weight, grad_out, mode)
    assert fn() is None and fn() is None
    assert calls == ([False, False] if mode == "forward" else [True])
    assert weight.grad is None and not weight.requires_grad
    # The backward workload differentiates its own leaf, so check the table it was given.
    assert all(table.grad is None for table in tables)


def test_backward_gap_averages_over_touched_rows_only():
    weight = torch.zeros(4, 1)
    token_ids = torch.tensor([1, 1, 3])
    grad_out = torch.ones(3, 1)

    gap = backward_gap(
        HFScaledEmbedding(torch.tensor(2.0)),
        HFScaledEmbedding(torch.tensor(3.0)),
        token_ids,
        weight,
        grad_out,
    )

    assert gap == {"max_abs_gap_vs_hf": 2.0, "mean_abs_gap_vs_hf": 1.5}


def test_forward_gate_rejects_an_unrounded_scale():
    weight = torch.randn(64, 32).to(torch.bfloat16)
    weight[0, 0] = 1.25
    token_ids = torch.arange(64)
    baseline = HFScaledEmbedding(torch.tensor(5376**0.5).to(torch.bfloat16))

    def unrounded(ids, table):
        return (torch.nn.functional.embedding(ids, table).float() * 5376**0.5).to(table.dtype)

    check_forward_bitwise(baseline, baseline, token_ids, weight)
    with pytest.raises(RuntimeError, match="not bitwise equal"):
        check_forward_bitwise(baseline, unrounded, token_ids, weight)


def test_forward_is_timed_when_unique_ids_are_not_selected():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    args = build_arg_parser().parse_args(
        "--patterns duplicate pad_flood --modes forward --dtypes bf16 --tokens 64 "
        "--vocab 1024 --hidden 64 --warmup 1 --repeat 2".split()
    )

    results = run_benchmark(args)["results"]

    assert [(row["mode"], row["ids"]) for row in results] == [("forward", "any")]


def test_benchmark_rejects_cpu_instead_of_timing_fallback():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA or AMD ROCm GPU"):
        run_benchmark(args)
