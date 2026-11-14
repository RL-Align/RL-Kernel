# SPDX-License-Identifier: Apache-2.0
"""Independent analytical normalization VJP check; not worker parity."""

import torch

from rl_engine.backends.rocm.norm.qnorm_fp8 import rmsnorm_vjp


def test_norm_vjp_against_independent_autograd():
    torch.manual_seed(137)
    x = torch.randn(3, 2, 1024, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1024, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    expected = torch.autograd.grad(
        (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight), (x, weight), dy
    )
    actual = rmsnorm_vjp(x.detach(), weight.detach(), dy, 1e-6)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-14, atol=2e-14)
