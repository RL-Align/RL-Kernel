# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L2: indexed AdaLN forward/backward accuracy and byte comparisons on CUDA."""

import pytest

from tests.helpers.adaln_indexed import check_small_case, require_cuda


@pytest.mark.cuda_only
def test_cuda_indexed_small_fp32_direct_backend():
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    check_small_case(CudaAdaLNModulationOp(), "cuda")
