# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native input validation for the H3 deterministic linear kernels."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.cuda.h3.det_linear import det_linear_available

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not det_linear_available(), reason="rl_engine._C lacks h3_det_linear_*"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_input_rejects_zero_rows(dtype):
    """Reject empty gradient batches in native input-gradient kernels for both dtypes."""

    grad = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    weight = torch.empty(16, 8, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="grad must have at least one row"):
        _C.h3_det_linear_backward_input(grad, weight, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_backward_weight_rejects_zero_rows(dtype, with_bias):
    """Reject empty gradient batches before parameter-gradient or bias-gradient reduction."""

    grad = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    x = torch.empty(0, 8, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="grad must have at least one row"):
        _C.h3_det_linear_backward_weight(grad, x, dtype, with_bias)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k_in,n_out", [(0, 16), (8, 0), (0, 0)])
def test_backward_input_handles_empty_columns(dtype, out_dtype, k_in, n_out):
    """Match the input VJP of empty linear dimensions in the requested output dtype."""

    grad = torch.arange(3 * n_out, device="cuda", dtype=torch.float32).reshape(3, n_out)
    weight = torch.empty(n_out, k_in, device="cuda", dtype=dtype)
    expected = (grad @ weight.float()).to(out_dtype)
    actual = _C.h3_det_linear_backward_input(grad, weight, out_dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k_in,n_out", [(0, 16), (8, 0), (0, 0)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_backward_weight_handles_empty_columns(dtype, w_dtype, k_in, n_out, with_bias):
    """Return empty weight gradients while preserving any nonempty bias reduction."""

    grad = torch.arange(3 * n_out, device="cuda", dtype=torch.float32).reshape(3, n_out)
    x = torch.arange(3 * k_in, device="cuda", dtype=dtype).reshape(3, k_in)
    expected_weight = (grad.T @ x.float()).to(w_dtype)
    actual = _C.h3_det_linear_backward_weight(grad, x, w_dtype, with_bias)
    assert len(actual) == (2 if with_bias else 1)
    torch.testing.assert_close(actual[0], expected_weight, atol=0, rtol=0)
    if with_bias:
        torch.testing.assert_close(actual[1], grad.sum(dim=0).to(w_dtype), atol=0, rtol=0)
