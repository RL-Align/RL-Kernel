# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Finite tensor identity in the original storage dtype, including signed zero."""

import torch


def tensor_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare logical values by bytes without widening or accepting NaN/Inf.

    Strides need not match. A cross-device comparison copies expected bytes to
    the actual tensor's device; it never converts either tensor's dtype.
    """
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    for value in (actual, expected):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            return False
    actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
    if actual_bytes.device != expected_bytes.device:
        expected_bytes = expected_bytes.to(actual_bytes.device)
    return bool(torch.equal(actual_bytes, expected_bytes))


def assert_tensor_bitwise_equal(
    actual: torch.Tensor, expected: torch.Tensor, *, name: str = "tensor"
) -> None:
    """Require finite, same-dtype, same-shape raw-bit identity."""
    if not tensor_bitwise_equal(actual, expected):
        raise AssertionError(
            f"{name}: finite raw-bit identity failed; "
            f"actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(expected.shape)}/{expected.dtype}"
        )
