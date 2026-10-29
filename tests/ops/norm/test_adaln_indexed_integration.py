# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Prepared L2 integration; backend equality supplements independent oracles."""

import os

import pytest
import torch

from tests.helpers.adaln_indexed import (
    assert_bytes,
    check_geometry_bytes,
    check_small_case,
    require_cuda,
    require_triton_gpu,
    small_case,
)

pytestmark = pytest.mark.cuda_only


def test_cuda_triton_indexed_small_bytes_and_independent_oracles():
    require_cuda()
    require_triton_gpu()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    results = []
    for op in (CudaAdaLNModulationOp(), TritonAdaLNModulationOp()):
        # Each backend must pass fixed bytes AND independent FP32 accuracy first.
        check_small_case(op, "cuda")
        x, m, index, dy, dg = small_case("cuda")
        y, gate = op(x, m, modulate_index=index, eps=7.0)
        dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
        results.append((y, gate, dx, dm))
    for cuda, triton in zip(*results, strict=True):
        assert_bytes(cuda, triton)


@pytest.fixture(params=["cuda", "triton"])
def direct(request):
    if request.param == "cuda":
        require_cuda()
        from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

        return CudaAdaLNModulationOp()
    require_triton_gpu()
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    return TritonAdaLNModulationOp()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexed_direct_integration_fixed_bytes_and_accuracy(direct, dtype):
    check_geometry_bytes(direct, "cuda", dtype, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.parametrize("route", ["segmented", "dispersed"])
def test_indexed_direct_target_h3072(direct, dtype, seq, route):
    if os.environ.get("RLK_ADALN_REAL_SHAPES") != "1":
        pytest.skip("user-run workload: >=8 GiB free GPU / >=16 GiB available host")
    check_geometry_bytes(direct, "cuda", dtype, 3072, seq=seq, route=route)
