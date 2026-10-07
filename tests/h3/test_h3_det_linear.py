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
    grad = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    weight = torch.empty(16, 8, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="grad must have at least one row"):
        _C.h3_det_linear_backward_input(grad, weight, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_backward_weight_rejects_zero_rows(dtype, with_bias):
    grad = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    x = torch.empty(0, 8, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="grad must have at least one row"):
        _C.h3_det_linear_backward_weight(grad, x, dtype, with_bias)
