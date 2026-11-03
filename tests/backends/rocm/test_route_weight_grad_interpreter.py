# SPDX-License-Identifier: Apache-2.0
"""Run in a separate process with TRITON_INTERPRET=1. This is CPU interpretation."""

import os

import pytest
import torch

from rl_engine.reference.moe.route_weight_grad import route_weight_grad_reference

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_INTERPRET") != "1", reason="requires TRITON_INTERPRET=1"
)


@pytest.mark.parametrize("width", [1, 255, 256, 257, 769, 2049])
def test_actual_kernel_arithmetic_in_interpreter(width):
    pytest.importorskip("triton")
    from rl_engine.backends.rocm.moe.route_weight_grad_kernels import _finish, _partials

    rng = torch.Generator().manual_seed(491)
    dh, s, u = [torch.randn(3, width * 2, generator=rng)[:, ::2] for _ in range(3)]
    u *= 20
    if width > 3:
        dh[0, :4] = torch.tensor([1e20, 1, -1e20, 1])
        s[0, :4] = u[0, :4] = 1
    chunks = (width + 255) // 256
    partials, dp = torch.empty(3, chunks), torch.empty(3)
    _partials[(3, chunks)](
        dh, s, u, partials, width, chunks, *dh.stride(), *s.stride(), *u.stride()
    )
    _finish[(3,)](partials, dp, chunks)
    expected = route_weight_grad_reference(dh, s, u)
    assert torch.equal(dp.view(torch.uint8), expected.view(torch.uint8))
