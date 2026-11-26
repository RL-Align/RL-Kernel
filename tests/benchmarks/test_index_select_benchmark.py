# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for the shared selection benchmark's shapes and accuracy gates."""

from functools import partial

import pytest
import torch

from benchmarks.operators.embedding import benchmark_index_select as benchmark
from benchmarks.operators.embedding.report_index_select import panel_rows
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
        "sys.argv",
        ["benchmark_index_select", "--dim", "0", "--shapes", "17x4x8", "--dry-run"],
    )
    benchmark.main()
    plan = json.loads(capsys.readouterr().out)
    cases = plan["cases"]
    assert all(case["shape"] == [17, 4, 8] and case["dim"] == 0 for case in cases)
    assert {(c["x_contiguous"], c["grad_y_contiguous"]): c["phases"] for c in cases} == {
        (True, True): ["forward", "backward", "forward_backward"],
        (True, False): ["backward", "forward_backward"],
        (False, True): ["forward", "forward_backward"],
        (False, False): ["forward_backward"],
    }
    assert plan["measurement_count"] == 24  # (2 forward + 2 backward + 4 combined) × 3 backends.


@pytest.mark.parametrize("shape", [(7, 5), (2,), (1,)])
def test_layout_cases_preserve_values_and_match_actual_strides(shape):
    """Check the controlled comparison, including gradients with only one element."""
    args = shape, 0, torch.float32, "cpu", 420
    dense_x, dense_index, dense_grad = benchmark.make_inputs(*args)
    cases = list(benchmark.layout_cases(shape, 0))
    assert len(cases) == (2 if dense_x.numel() > 1 else 1) * (2 if dense_grad.numel() > 1 else 1)
    for case in cases:
        x, index, grad = benchmark.make_inputs(
            *args,
            x_contiguous=case["x_contiguous"],
            grad_y_contiguous=case["grad_y_contiguous"],
        )
        assert x.is_contiguous() == case["x_contiguous"]
        assert grad.is_contiguous() == case["grad_y_contiguous"]
        assert torch.equal(x, dense_x) and torch.equal(grad, dense_grad)
        assert index.stride() == (1,) and torch.equal(index, dense_index)


def test_report_groups_only_timed_phases_and_converts_units():
    """Isolated phases use one layout label; combined time and peak use both."""
    results = []
    for case in benchmark.layout_cases((3, 7), 1):
        timings = {
            phase: {"median_wall_ms": 0.1, "median_event_ms": 0.08} for phase in case["phases"]
        }
        timings["incremental_peak_bytes"] = 2**20
        results.append({**case, "implementation": "triton", "timings": timings})
    assert panel_rows(results, "forward") == {
        ("C", "—"): {"triton": (100, 80)},
        ("S", "—"): {"triton": (100, 80)},
    }
    assert set(panel_rows(results, "backward")) == {("—", "C"), ("—", "S")}
    for phase, values in (
        ("forward_backward", (100, 80)),
        ("incremental_peak_bytes", (1,)),
    ):
        assert panel_rows(results, phase) == {
            (x, grad): {"triton": values} for x in ("C", "S") for grad in ("C", "S")
        }
