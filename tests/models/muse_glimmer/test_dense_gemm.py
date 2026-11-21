# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #501 single-GPU qualification of the BF16 projection operators."""

import math

import pytest
import torch

from rl_engine.backends.cuda.gemm import deterministic_gemm
from rl_engine.backends.shared.triton.gemm import deterministic_gemm_triton


def _assert_bits(actual, expected, label):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    different = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    count = different.sum().item()
    if count:
        index = different.flatten().int().argmax().item()
        pytest.fail(f"{label}: {count}/{actual.numel()} BF16 words differ, first flat index {index}")


@pytest.mark.cuda_only
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="BF16 GEMM qualification requires an NVIDIA SM80+ GPU",
)
def test_kv_projection_parity():
    generator = torch.Generator(device="cuda").manual_seed(501)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(96, 6656, **options)
    weight = torch.randn(256, 6656, **options) / math.sqrt(6656)
    dy = torch.randn(96, 256, **options)

    def evaluate(gemm, inputs, upstream):
        a = inputs.detach().requires_grad_()
        w = weight.detach().requires_grad_()
        y = gemm(a, w.T)
        return (y.detach(), *torch.autograd.grad(y, (a, w), upstream))

    row = slice(17, 18)
    for inputs, upstream in ((x, dy), (x[row], dy[row])):
        cuda = evaluate(deterministic_gemm, inputs, upstream)
        triton = evaluate(deterministic_gemm_triton, inputs, upstream)
        for name, left, right in zip(("Y", "dX", "dW"), cuda, triton, strict=True):
            _assert_bits(left, right, f"M={len(inputs)} {name}")
        if len(inputs) == len(x):
            batch_y, batch_dx = cuda[0][row], cuda[1][row]
        else:
            _assert_bits(cuda[0], batch_y, "single vs mixed Y")
            _assert_bits(cuda[1], batch_dx, "single vs mixed dX")
