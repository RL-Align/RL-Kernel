# SPDX-License-Identifier: Apache-2.0
"""Check reduction bits separately from elementary-math/reference accuracy."""

import importlib

import pytest
import torch

from rl_engine.reference.norm.fused_add_rmsnorm import ordered_backward, ordered_row_sum
from tests.helpers.tensor_checks import assert_bitwise


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required")
    return importlib.import_module("rl_engine.backends.shared.triton.norm.fused_add_rmsnorm")


@pytest.mark.parametrize("width", [1, 4, 7, 16, 33, 129, 2688, 4097])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_production_row_reductions_match_fixed_tree_bits(kernels, width, num_warps):
    """Real production helpers see the same FP32 operands as the eager tree.

    Positive squares and signed dot products exercise different rounding risks.
    Compare sum bits directly; existing FP64 tests qualify rsqrt accuracy.
    """
    from tests.helpers.fused_rmsnorm_path_kernels import row_correction_probe, row_statistics_probe

    generator = torch.Generator().manual_seed(463)
    updated = torch.randn((3, width), generator=generator).cuda()
    updated[0].fill_(1)
    updated[0, 0] = 4096
    normalized = torch.randn((3, width), generator=generator).cuda()
    grad_normalized = torch.randn((3, width), generator=generator).cuda()
    normalized[0].fill_(1)
    grad_normalized[0].fill_(0)
    if width >= 4:
        grad_normalized[0, :4] = torch.tensor([2**24, 1, -(2**24), 1], device="cuda")
    actual_sum = torch.empty(3, device="cuda")
    actual_inverse = torch.empty_like(actual_sum)
    row_statistics_probe[(3,)](
        updated,
        actual_sum,
        actual_inverse,
        N_COLS=width,
        BLOCK_SIZE=1 << (width - 1).bit_length(),
        EPS=1e-5,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    expected_sum = ordered_row_sum(updated * updated).flatten()
    assert_bitwise(actual_sum, expected_sum, "forward square reduction")
    divisor = torch.tensor(width, dtype=torch.float32, device="cuda")
    # The public production launcher must use the same row arithmetic.
    _, saved_updated, saved_inverse = kernels._launch_fused_add_rmsnorm_fwd(
        updated, torch.zeros_like(updated), torch.ones(width, device="cuda"), eps=1e-5
    )
    assert_bitwise(saved_updated, updated)
    assert_bitwise(saved_inverse, actual_inverse, "saved inverse RMS")

    actual_correction = torch.empty_like(actual_sum)
    row_correction_probe[(3,)](
        grad_normalized,
        normalized,
        actual_sum,
        actual_correction,
        N_COLS=width,
        BLOCK_SIZE=1 << (width - 1).bit_length(),
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    expected_sum = ordered_row_sum(grad_normalized * normalized).flatten()
    assert_bitwise(actual_sum, expected_sum, "backward correction reduction")
    assert_bitwise(actual_correction, expected_sum / divisor, "backward correction division")


@pytest.mark.parametrize("width", [1, 7, 33, 129, 2688])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("branch", ["both", "y_only", "residual_only"])
def test_every_gradient_matches_ordered_backward_from_same_saved_state(
    kernels, width, dtype, branch
):
    """Feed actual saved FP32 statistics to the hand-written VJP, without autograd.

    This isolates backward from rsqrt implementation differences. Existing
    Native/autograd and independent FP64 tests still check end-to-end accuracy.
    """
    generator = torch.Generator().manual_seed(464)
    x = torch.randn((33, width), generator=generator).to(device="cuda", dtype=dtype)
    residual = torch.randn((33, width), generator=generator).to(device="cuda", dtype=dtype)
    weight = torch.randn(width, generator=generator).to(device="cuda", dtype=dtype)
    gy = torch.randn(x.shape, generator=generator).cuda() if branch != "residual_only" else None
    gr = torch.randn(x.shape, generator=generator).cuda() if branch != "y_only" else None
    _, u, inverse = kernels._launch_fused_add_rmsnorm_fwd(x, residual, weight, eps=1e-5)
    assert_bitwise(u, x.float() + residual.float())
    expected = ordered_backward(u, inverse, weight, gy, gr, x_dtype=dtype, residual_dtype=dtype)
    for strategy in kernels.RMSNormWeightGradStrategy:
        actual = kernels._launch_fused_add_rmsnorm_bwd(
            u,
            inverse,
            weight,
            gy,
            gr,
            x_dtype=dtype,
            residual_dtype=dtype,
            plan=kernels._default_weight_grad_plan(strategy),
        )
        for name, value in zip(("grad_x", "grad_residual", "grad_weight"), actual, strict=True):
            assert_bitwise(value, getattr(expected, name), f"{strategy.value} {name}")
