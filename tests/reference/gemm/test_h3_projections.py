# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent H3 projection reference arithmetic."""

import torch

from rl_engine.reference.gemm.h3_projections import reference_projection


def test_fp32_reference_keeps_quantized_inputs_and_gradients():
    x = torch.tensor([[0.25, -0.5], [1.0, 0.125]], dtype=torch.bfloat16)
    w = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16)
    dy = torch.tensor([[0.5], [-0.25]], dtype=torch.bfloat16)
    y, dx, dw = reference_projection(x, w, dy)
    assert y.dtype == dx.dtype == dw.dtype == torch.float32
    torch.testing.assert_close(y.double(), x.double() @ w.double().t(), atol=0, rtol=0)
    torch.testing.assert_close(dx.double(), dy.double() @ w.double(), atol=0, rtol=0)
    torch.testing.assert_close(dw.double(), dy.double().t() @ x.double(), atol=0, rtol=0)
