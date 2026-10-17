# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.dsv4.attention.cuda_runtime import ensure_native_kernels
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import _det_gemm_linear
from rl_engine.kernels.ops import base


@pytest.fixture(scope="module")
def strict_det_gemm():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    ensure_native_kernels()
    from rl_engine.kernels.ops.cuda.matmul.det_gemm import det_gemm_backend

    if det_gemm_backend() == "sm90":
        marker = getattr(base._C, "det_gemm_sm90_compiled", None)
        if not callable(marker) or not marker():
            pytest.skip("strict SM90 DetGemm kernel is not compiled")
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("strict SM90 DetGemm requires Hopper")
    return _det_gemm_linear()
