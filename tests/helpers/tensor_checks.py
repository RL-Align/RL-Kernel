# SPDX-License-Identifier: Apache-2.0
"""Tensor assertions shared by operator and backend tests."""

import torch


def assert_bitwise(actual, expected, stage="tensor"):
    """Compare bytes after copying both tensors into an explicit dense layout.

    contiguous() can preserve a non-unit final stride on empty or singleton
    dimensions. A dtype view requires that stride to be one even in those
    cases, so force a contiguous-format clone before viewing the raw bytes.
    Flatten afterward so scalar reduction outputs also support a byte view.
    """
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, stage
    actual_bytes = (
        actual.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
    )
    expected_bytes = (
        expected.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
    )
    assert torch.equal(actual_bytes, expected_bytes), f"Bitwise mismatch: {stage}"
