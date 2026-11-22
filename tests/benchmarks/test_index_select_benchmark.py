# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for the shared selection benchmark's shapes and accuracy gates."""

from functools import partial

import pytest
import torch

from benchmarks.operators.embedding import benchmark_index_select as benchmark
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp


@pytest.mark.parametrize(
    "shape,dim,dtype",
    [
        ((2, 17, 32), 1, torch.float32),
        ((17, 32), 0, torch.float16),
        ((17, 4, 8), 0, torch.bfloat16),
        ((2, 17), -1, torch.float32),
    ],
)
def test_inputs_and_independent_checks(shape, dim, dtype):
    """The same accuracy and bit checks apply to H3, BAGEL and a trailing axis."""
    x, index, grad = benchmark.make_inputs(shape, dim, dtype, "cpu", 680)
    assert x.shape == shape
    assert grad.shape == x.index_select(dim, index).shape
    assert index.numel() == shape[dim] // 2
    assert index.unique().numel() == index.numel()
    candidate = partial(NativeIndexSelectOp(), dim=dim, unique_indices=True)
    expected_y, expected_dx = benchmark.reference_outputs(x, index, grad, dim=dim)
    tolerance = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=x.dtype
    )
    checks = benchmark.check_result(
        candidate, x, index, grad, (expected_y, expected_dx), tolerance, check_gradient_bits=True
    )
    assert not benchmark.checks_failed(checks)
    # Opposite signs of zero pass the tolerance check but differ by one bit.
    wrong_dx = expected_dx.masked_fill(expected_dx == 0, -0.0)
    checks = benchmark.check_result(
        candidate, x, index, grad, (expected_y, wrong_dx), tolerance, check_gradient_bits=True
    )
    assert checks["gradient_accuracy_failures"] == 0
    assert checks["gradient_bit_mismatches"] > 0
    assert benchmark.checks_failed(checks)


def test_bit_check_rejects_broadcasting_and_dtype_changes():
    """An extra singleton axis or a different 16-bit type can have identical bytes."""
    expected = torch.zeros(3, dtype=torch.float16)
    for wrong in (expected.unsqueeze(0), expected.to(torch.bfloat16)):
        with pytest.raises(ValueError, match="expected shape"):
            benchmark.bit_mismatches(wrong, expected)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--dim", "2", "--shapes", "7x9"],
        ["--dim", "0", "--shapes", "0x9"],
        ["--dim", "0", "--duplicates", "0.25"],
        ["--dim", "0", "--samples", "0"],
    ],
)
def test_cli_rejects_invalid_cases(arguments):
    with pytest.raises(SystemExit):
        benchmark.parse_args(arguments)


def test_dry_run_without_cuda(monkeypatch, capsys):
    import json

    monkeypatch.setattr(
        "sys.argv", ["benchmark_index_select", "--dim", "0", "--shapes", "17x4x8", "--dry-run"]
    )
    benchmark.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == [
        {"shape": [17, 4, 8], "dim": 0, "unique_indices": True, "duplicate_fraction": 0.0}
    ]
    assert plan["measurement_count"] == 9
