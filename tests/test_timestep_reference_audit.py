# SPDX-License-Identifier: Apache-2.0
"""Finite differences independently check the reference's handwritten VJP."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.timestep_embed_mlp import _ReferenceLinear


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
