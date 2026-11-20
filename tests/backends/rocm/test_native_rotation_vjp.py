# SPDX-License-Identifier: Apache-2.0
"""Independent dense-Hadamard oracle for the native rotation and its VJP."""

import math

import pytest
import torch

from rl_engine.backends.rocm.attention.rotation import native_hadamard128

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_tail_rotation_and_input_gradient_against_fp64_matrix():
    gen = torch.Generator().manual_seed(337)
    x = (torch.randn((17, 128), generator=gen) / 8).bfloat16().cuda().requires_grad_(True)
    dy = (torch.randn((17, 128), generator=gen) / 8).bfloat16().cuda()
    matrix = torch.ones((1, 1), dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    matrix = matrix / math.sqrt(128)
    out = native_hadamard128(x)
    out.backward(dy)
    expected = x.detach().cpu().double() @ matrix.T
    expected_grad = dy.cpu().double() @ matrix
    # BF16 result accuracy only; all model logprob acceptance still uses bytes.
    torch.testing.assert_close(out.cpu().double(), expected, atol=0.004, rtol=0)
    torch.testing.assert_close(x.grad.cpu().double(), expected_grad, atol=0.004, rtol=0)
    assert bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
