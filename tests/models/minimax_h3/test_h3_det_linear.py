# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native input validation for the H3 deterministic linear kernels."""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import (
    det_linear_available,
    det_linear_device_supported,
    det_linear_forward,
)
from rl_engine.backends.extension import _C

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


def test_device_support_follows_the_device_that_is_asked_about(monkeypatch):
    """The capability is read for the given device, not for the current one."""

    seen = []

    def capability(device=None):
        seen.append(device)
        return (7, 5) if device == torch.device("cuda:0") else (9, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert not det_linear_device_supported(torch.device("cuda:0"))
    assert det_linear_device_supported(torch.device("cuda:1"))
    assert not det_linear_device_supported(torch.device("cpu"))
    assert seen == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_no_cuda_means_no_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not det_linear_device_supported()


def test_bf16_forward_is_refused_on_a_pre_sm80_device_with_a_clear_error(monkeypatch):
    """The wrapper checks the tensors' device, so no opaque launch error reaches the caller."""

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    x = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="SM80"):
        det_linear_forward(x, w, None)
    # FP32 does not use mma.sync and is not refused.
    out = det_linear_forward(x.float(), w.float(), None)[0]
    assert out.shape == (1, 8)
