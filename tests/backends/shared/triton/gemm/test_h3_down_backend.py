# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU launch binding checks; these do not compile or execute device code."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.shared.triton.gemm import det_gemm
from rl_engine.backends.shared.triton.gemm.h3_down_fp32_backend import FixedFP32Backend


def test_h3_adapter_rejects_inner_dimension_mismatch_before_launch():
    backend = object.__new__(FixedFP32Backend)
    a = torch.empty((2, 3), dtype=torch.bfloat16, device="meta")
    b = torch.empty((4, 5), dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="GEMM inner dimensions differ: 3 != 4"):
        backend.gemm(a, b)


@pytest.mark.parametrize("role", ["forward", "dx", "dw"])
def test_h3_launch_binds_current_shared_kernel_signature(role):
    if not det_gemm._TRITON_AVAILABLE:
        pytest.skip("Triton is required to inspect the real kernel signature")
    backend = FixedFP32Backend()
    signature = inspect.signature(backend.kernel.fn)
    launches = []

    class BindingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                options = {
                    key: kwargs.pop(key) for key in ("num_warps", "num_stages", "enable_fp_fusion")
                }
                # Bind the adapter's actual call to the real JIT function, not a
                # copied signature that could drift with the implementation.
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                launches.append((grid, bound.arguments, options))
                return SimpleNamespace(
                    hash="h3_cpu_binding_fixture",
                    asm={"cubin": b"synthetic test binary"},
                    metadata=SimpleNamespace(num_warps=4),
                )

            return launch

    backend.kernel = BindingKernel()
    x = torch.empty((129, 14336), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((5376, 14336), dtype=torch.bfloat16, device="meta")
    grad = torch.empty((129, 5376), dtype=torch.bfloat16, device="meta")
    if role == "forward":
        output = backend.det_gemm_fwd_rhs_transposed(x, weight)
        dimensions, strides = (129, 5376, 14336), (14336, 1, 1, 14336, 5376, 1)
    elif role == "dx":
        output = backend.det_gemm_fwd(grad, weight)
        dimensions, strides = (129, 14336, 5376), (5376, 1, 14336, 1, 14336, 1)
    else:
        output = backend.det_gemm_db_transposed(x, grad)
        dimensions, strides = (5376, 14336, 129), (1, 5376, 14336, 1, 14336, 1)
    assert output.shape == dimensions[:2]
    assert output.dtype == torch.bfloat16
    assert len(launches) == 1
    grid, arguments, options = launches[0]
    assert grid == tuple((value + 63) // 64 for value in dimensions[:2])
    assert arguments["c_ptr"] is output
    assert arguments["bias_ptr"] is None
    assert arguments["HAS_BIAS"] is False
    assert tuple(arguments[key] for key in ("M", "N", "K")) == dimensions
    assert (
        tuple(
            arguments[key]
            for key in (
                "stride_am",
                "stride_ak",
                "stride_bk",
                "stride_bn",
                "stride_cm",
                "stride_cn",
            )
        )
        == strides
    )
    assert tuple(arguments[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")) == (64, 64, 32)
    assert arguments["PROMOTE_INPUTS"] is False
    assert arguments["CHUNK_K"] == 0
    assert arguments["EVEN_K"] is arguments["EVEN_N"] is False
    assert options == {"num_warps": 4, "num_stages": 2, "enable_fp_fusion": False}
