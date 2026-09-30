# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exact tensor comparison for byte-level kernel contracts."""

import torch


def tensor_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare logical tensor contents, including floating-point signed zeros."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False

    actual_bytes = actual.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(actual_bytes, expected_bytes)
