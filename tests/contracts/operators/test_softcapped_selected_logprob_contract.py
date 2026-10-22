# SPDX-License-Identifier: Apache-2.0
"""Shared configuration without coupling independent accuracy and ordered references."""

import inspect

import pytest
import torch

from rl_engine.contracts.operators.softcapped_selected_logprob import (
    REDUCTION_CONFIG,
    SUPPORTED_DTYPES,
    TRITON_SUPPORTED_DEVICES,
    SoftcappedLogprobReductionConfig,
)
from rl_engine.reference.logprob import softcapped_selected_logprob as reference


def test_shared_defaults_preserve_the_existing_operator_contract():
    """Keep the established API and 1024-element arithmetic after moving metadata."""
    assert SUPPORTED_DTYPES == (torch.float16, torch.bfloat16, torch.float32)
    assert TRITON_SUPPORTED_DEVICES == ("cuda", "hip", "xpu", "musa")
    assert (
        REDUCTION_CONFIG.tile_size,
        REDUCTION_CONFIG.group_size,
        REDUCTION_CONFIG.section_count,
    ) == (1024, 8, 4)
    assert reference.SUPPORTED_DTYPES is SUPPORTED_DTYPES
    assert reference.REDUCTION_CONFIG is REDUCTION_CONFIG
    assert (
        inspect.signature(reference.NativeSoftcappedSelectedLogprobOp.forward)
        .parameters["softcap"]
        .default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(reference.softcapped_selected_logprob).parameters["softcap"].default
        is inspect.Parameter.empty
    )
    for ordered_function in (reference.ordered_exp_sum, reference.ordered_forward):
        assert (
            inspect.signature(ordered_function).parameters["tile_size"].default
            == REDUCTION_CONFIG.tile_size
        )


@pytest.mark.parametrize("tile_size", (0, -1, 3, 24, 1023))
def test_reduction_config_rejects_invalid_width_at_construction(tile_size):
    """Bad configurations fail before importing a backend or launching any GPU work."""
    with pytest.raises(ValueError, match="tile_size must be a positive power of two"):
        SoftcappedLogprobReductionConfig(tile_size=tile_size)


@pytest.mark.parametrize("tile_size", tuple(1 << exponent for exponent in range(14)))
def test_reduction_config_accepts_all_qualified_reference_widths(tile_size):
    """Small widths are valid: the implementation uses one or two groups when needed."""
    config = SoftcappedLogprobReductionConfig(tile_size=tile_size)
    assert config.tile_size == tile_size


def test_native_reference_does_not_use_the_ordered_reduction(monkeypatch):
    """Shared metadata must not turn the accuracy baseline into the ordered oracle."""

    def forbidden(*args, **kwargs):
        raise AssertionError("native accuracy reference called the fixed-order reference")

    monkeypatch.setattr(reference, "ordered_reduce", forbidden)
    monkeypatch.setattr(reference, "ordered_exp_sum", forbidden)
    monkeypatch.setattr(reference, "ordered_forward", forbidden)
    logits = torch.tensor([[1.5, -2.0, 0.5]], requires_grad=True)
    ids = torch.tensor([2])
    actual = reference.NativeSoftcappedSelectedLogprobOp()(logits, ids, softcap=30.0)
    divisor = torch.tensor(30.0)
    expected = torch.log_softmax(30.0 * torch.tanh(logits / divisor), dim=-1)[:, 2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), logits)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
