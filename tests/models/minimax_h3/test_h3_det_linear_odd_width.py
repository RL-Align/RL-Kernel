# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Output widths that are not a multiple of the column tile of the deterministic linear.

Each warp of the FP32 forward kernel owns kCols = 2 adjacent output columns, and each warp of the
BF16 forward (an ``mma.sync`` kernel) owns a 16-column tile (kMmaCols), so the last tile of an odd
or short N has columns that are not loaded and must not be consumed. The results for such widths
must match an FP64 golden, and must be bitwise the leading columns of the same product with zero
columns padded on (the tail does not influence the real columns). The BF16 forward needs SM80, so
its cases are skipped on older devices.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import (
    det_linear_available,
    det_linear_backward_input,
    det_linear_backward_weight,
    det_linear_device_supported,
    det_linear_forward,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not det_linear_available(), reason="rl_engine._C lacks h3_det_linear_*"),
]

DTYPES = [torch.float32, torch.bfloat16]
WIDTHS = [1, 2, 3, 5, 7, 9, 33, 65]  # odd widths, and tails past a 2- or 4-column tile
K_IN = 64
ROWS = [1, 3, 5]


def _skip_unsupported(dtype):
    if dtype == torch.bfloat16 and not det_linear_device_supported():
        pytest.skip("the BF16 forward needs SM80 or newer")


def _inputs(dtype, rows, n_out, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + n_out)
    x = torch.randn(rows, K_IN, generator=g).to(dtype).cuda()
    weight = (torch.randn(n_out, K_IN, generator=g) / K_IN**0.5).to(dtype).cuda()
    bias = (torch.randn(n_out, generator=g) * 0.1).to(dtype).cuda()
    return x, weight, bias


def _pad(weight, bias, extra):
    zeros = torch.zeros(extra, weight.shape[1], dtype=weight.dtype, device=weight.device)
    return torch.cat([weight, zeros]), torch.cat([bias, bias.new_zeros(extra)])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_out", WIDTHS)
@pytest.mark.parametrize("rows", ROWS)
def test_forward_matches_golden_and_ignores_the_tail(dtype, n_out, rows):
    _skip_unsupported(dtype)
    x, weight, bias = _inputs(dtype, rows, n_out)
    out = det_linear_forward(x, weight, bias)[0]
    assert out.shape == (rows, n_out) and torch.isfinite(out.float()).all()
    golden = x.double() @ weight.double().T + bias.double()
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.double(), golden, atol=tol, rtol=tol)
    padded = det_linear_forward(x, *_pad(weight, bias, 3))[0]
    assert torch.equal(out, padded[:, :n_out])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_out", WIDTHS)
def test_backward_matches_golden_and_ignores_the_tail(dtype, n_out):
    rows = 3
    x, weight, _ = _inputs(dtype, rows, n_out)
    g = torch.Generator(device="cpu").manual_seed(1)
    grad = torch.randn(rows, n_out, generator=g).cuda()
    tol = 1e-5 if dtype == torch.float32 else 1e-2

    d_input = det_linear_backward_input(grad, weight, torch.float32)
    torch.testing.assert_close(
        d_input.double(), grad.double() @ weight.double(), atol=tol, rtol=tol
    )
    d_weight, d_bias = det_linear_backward_weight(grad, x, dtype, with_bias=True)
    torch.testing.assert_close(d_weight.double(), grad.double().T @ x.double(), atol=tol, rtol=tol)
    torch.testing.assert_close(d_bias.double(), grad.double().sum(0), atol=tol, rtol=tol)

    zero = torch.zeros(rows, 3, device="cuda")
    padded_weight, _ = _pad(weight, weight[:, 0], 3)
    padded_grad = torch.cat([grad, zero], dim=1)
    assert torch.equal(
        d_input, det_linear_backward_input(padded_grad, padded_weight, torch.float32)
    )
    d_weight_p, d_bias_p = det_linear_backward_weight(padded_grad, x, dtype, with_bias=True)
    assert torch.equal(d_weight, d_weight_p[:n_out]) and torch.equal(d_bias, d_bias_p[:n_out])


@pytest.mark.parametrize(
    "dtype, k_in",
    [(torch.bfloat16, k) for k in (8, 24, 40, 72, 264, 520)]
    + [(torch.float32, k) for k in (4, 12, 20, 36, 132, 1044)],
)
@pytest.mark.parametrize("n_out", [1, 3, 33])
def test_k_that_is_not_a_multiple_of_the_lane_stride_matches_the_golden(dtype, k_in, n_out):
    """Row lengths that are a vector multiple but not a multiple of the 32-lane stride."""

    _skip_unsupported(dtype)

    g = torch.Generator(device="cpu").manual_seed(k_in + n_out)
    x = torch.randn(3, k_in, generator=g).to(dtype).cuda()
    weight = (torch.randn(n_out, k_in, generator=g) / k_in**0.5).to(dtype).cuda()
    bias = (torch.randn(n_out, generator=g) * 0.1).to(dtype).cuda()
    out = det_linear_forward(x, weight, bias)[0]
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    golden = x.double() @ weight.double().T + bias.double()
    torch.testing.assert_close(out.double(), golden, atol=tol, rtol=tol)
