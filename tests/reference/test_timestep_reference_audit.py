# SPDX-License-Identifier: Apache-2.0
"""Finite differences independently check the reference's handwritten VJP."""

import pytest
import torch

from rl_engine.reference.embedding.timestep_embed_mlp import _ReferenceLinear


@pytest.mark.parametrize("batch", [1, 3, 33])
def test_reference_linear_vjp_finite_differences(batch):
    generator = torch.Generator().manual_seed(20261007)
    # Double is used only for numerical differentiation of this isolated
    # formula. Production gold and its official tolerance remain FP32.
    values = tuple(
        torch.randn(shape, dtype=torch.float64, generator=generator).requires_grad_()
        for shape in ((batch, 5), (7, 5), (7,))
    )
    assert torch.autograd.gradcheck(_ReferenceLinear.apply, values)


@pytest.mark.parametrize("batch", [0, 2])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_reference_linear_selective_vjp(batch, index):
    generator = torch.Generator().manual_seed(386)
    values = [torch.randn(shape, generator=generator) for shape in ((batch, 5), (7, 5), (7,))]
    values[index].requires_grad_()
    actual = _ReferenceLinear.apply(*values)
    expected = torch.nn.functional.linear(*values)
    upstream = torch.randn(actual.shape, generator=generator)
    (a,) = torch.autograd.grad(actual, values[index], upstream)
    (b,) = torch.autograd.grad(expected, values[index], upstream)
    torch.testing.assert_close(a, b)
