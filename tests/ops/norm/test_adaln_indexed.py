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


@pytest.mark.cuda_only
@pytest.mark.parametrize("route", ["all0", "all1", "alternating"])
def test_cuda_indexed_route_identity_literal(route):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_route_identity

    check_route_identity(CudaAdaLNModulationOp(), "cuda", route)


@pytest.mark.cuda_only
@pytest.mark.parametrize("explicit_zero", [False, True])
def test_cuda_indexed_absent_vs_zero_dy_nonfinite_literal(explicit_zero):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_absent_vs_zero_dy

    check_absent_vs_zero_dy(CudaAdaLNModulationOp(), "cuda", explicit_zero)


@pytest.mark.cuda_only
@pytest.mark.parametrize("order", ["lost_unit", "retained_unit"])
def test_cuda_indexed_selected_token_cancellation_literal(order):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_cancellation

    check_cancellation(CudaAdaLNModulationOp(), "cuda", order)


@pytest.mark.cuda_only
def test_cuda_indexed_dscale_product_rounds_before_add_literal():
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_product_rounding

    check_product_rounding(CudaAdaLNModulationOp(), "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("bank", [0, 1])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_cuda_indexed_nonfinite_selected_bank_isolation_literal(bank, nonfinite):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_nonfinite_isolation

    check_nonfinite_isolation(CudaAdaLNModulationOp(), "cuda", bank, nonfinite)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "frozen,outputs",
    [
        ("none", "y"),
        ("none", "gate"),
        ("none", "both"),
        ("x", "y"),
        ("x", "gate"),
        ("x", "both"),
        ("modulation", "y"),
        ("modulation", "both"),
    ],
)
def test_cuda_indexed_whole_leaf_participation_literal(frozen, outputs):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_leaf_participation

    check_leaf_participation(CudaAdaLNModulationOp(), "cuda", frozen, outputs)


@pytest.mark.cuda_only
def test_cuda_indexed_gate_copies_original_fp32_payload_literal():
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_gate_payload

    check_gate_payload(CudaAdaLNModulationOp(), "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("contribution", ["negative_zero", "subnormal"])
def test_cuda_indexed_signed_zero_and_subnormal_fold_literal(contribution):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_tiny_contributions

    check_tiny_contributions(CudaAdaLNModulationOp(), "cuda", contribution)


@pytest.mark.cuda_only
def test_cuda_indexed_all0_matches_shared_arithmetic_and_independent_gold():
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_shared_bank0

    op = CudaAdaLNModulationOp()
    check_shared_bank0(op, op, "cuda")
