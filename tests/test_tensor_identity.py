# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for the raw-bit identity used by Qwen3-Next acceptance checks."""

import pytest
import torch

from rl_engine.kernels.gtest.forward_invariance import _compare_logical_tensors
from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal, tensor_bitwise_equal


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_signed_zero_is_not_identical(dtype):
    positive = torch.tensor([0.0], dtype=dtype)
    negative = torch.tensor([-0.0], dtype=dtype)
    assert not tensor_bitwise_equal(positive, negative)
    with pytest.raises(AssertionError, match="raw-bit"):
        assert_tensor_bitwise_equal(positive, negative)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_identical_nonfinite_values_are_rejected(value):
    tensor = torch.tensor([value])
    assert not tensor_bitwise_equal(tensor, tensor)


def test_same_numbers_in_different_dtypes_are_rejected():
    assert not tensor_bitwise_equal(torch.ones(2), torch.ones(2, dtype=torch.bfloat16))


def test_adjacent_bfloat16_values_are_rejected():
    value = torch.ones(2, dtype=torch.bfloat16)
    changed = (value.view(torch.int16) + 1).view(torch.bfloat16)
    assert not tensor_bitwise_equal(value, changed)


@pytest.mark.parametrize(
    "value", [torch.tensor(1.0), torch.empty(0), torch.arange(12).reshape(3, 4).T]
)
def test_logical_identity_does_not_require_matching_strides(value):
    assert_tensor_bitwise_equal(value, value.contiguous().clone())


@pytest.mark.parametrize("judgment", ["forward_invariance", "gradient_invariance"])
@pytest.mark.parametrize("fault", ["signed_zero", "dtype", "nan"])
def test_gtest_invariance_uses_original_bits(judgment, fault):
    canonical = torch.zeros(1, dtype=torch.bfloat16)
    changed = torch.tensor([-0.0], dtype=torch.bfloat16)
    if fault == "dtype":
        changed = canonical.float()
    elif fault == "nan":
        canonical = torch.tensor([float("nan")], dtype=torch.bfloat16)
        changed = canonical.clone()
    result = _compare_logical_tensors(
        canonical,
        changed,
        judgment=judgment,
        contract=load_contract(),
        op_class="reduction",
        dtype=torch.bfloat16,
        backend_profile="cuda_bf16",
    )
    assert result.passed is False


def test_accuracy_keeps_its_tolerance_semantics():
    result = _compare_logical_tensors(
        torch.ones(1),
        torch.tensor([1.0078125], dtype=torch.bfloat16),
        judgment="forward_accuracy",
        contract=load_contract(),
        op_class="reduction",
        dtype=torch.bfloat16,
        backend_profile="cuda_bf16",
    )
    assert result.passed is True
