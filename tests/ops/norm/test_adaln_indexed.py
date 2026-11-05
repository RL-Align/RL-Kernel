# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L2: indexed CUDA/Triton accuracy, bytes and public support boundaries."""

import pytest
import torch

from tests.helpers.adaln_indexed import check_small_case, require_cuda


@pytest.fixture(params=["cuda", "triton"])
def indexed_op(request):
    if request.param == "triton":
        from tests.helpers.adaln_indexed import require_triton_gpu

        require_triton_gpu()
        from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

        return TritonAdaLNModulationOp()
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    return CudaAdaLNModulationOp()


@pytest.mark.cuda_only
def test_indexed_small_fp32_direct_backend(indexed_op):
    check_small_case(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("route", ["all0", "all1", "alternating"])
def test_indexed_route_identity_literal(indexed_op, route):
    from tests.helpers.adaln_indexed import check_route_identity

    check_route_identity(indexed_op, "cuda", route)


@pytest.mark.cuda_only
@pytest.mark.parametrize("explicit_zero", [False, True])
def test_indexed_absent_vs_zero_dy_nonfinite_literal(indexed_op, explicit_zero):
    from tests.helpers.adaln_indexed import check_absent_vs_zero_dy

    check_absent_vs_zero_dy(indexed_op, "cuda", explicit_zero)


@pytest.mark.cuda_only
@pytest.mark.parametrize("order", ["lost_unit", "retained_unit"])
def test_indexed_selected_token_cancellation_literal(indexed_op, order):
    from tests.helpers.adaln_indexed import check_cancellation

    check_cancellation(indexed_op, "cuda", order)


@pytest.mark.cuda_only
def test_indexed_dscale_product_rounds_before_add_literal(indexed_op):
    from tests.helpers.adaln_indexed import check_product_rounding

    check_product_rounding(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("bank", [0, 1])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_indexed_nonfinite_selected_bank_isolation_literal(indexed_op, bank, nonfinite):
    from tests.helpers.adaln_indexed import check_nonfinite_isolation

    check_nonfinite_isolation(indexed_op, "cuda", bank, nonfinite)


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
def test_indexed_whole_leaf_participation_literal(indexed_op, frozen, outputs):
    from tests.helpers.adaln_indexed import check_leaf_participation

    check_leaf_participation(indexed_op, "cuda", frozen, outputs)


@pytest.mark.cuda_only
def test_indexed_gate_copies_original_fp32_payload_literal(indexed_op):
    from tests.helpers.adaln_indexed import check_gate_payload

    check_gate_payload(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("contribution", ["negative_zero", "subnormal"])
def test_indexed_signed_zero_and_subnormal_fold_literal(indexed_op, contribution):
    from tests.helpers.adaln_indexed import check_tiny_contributions

    check_tiny_contributions(indexed_op, "cuda", contribution)


@pytest.mark.cuda_only
def test_indexed_all0_matches_shared_arithmetic_and_independent_gold(indexed_op):
    from tests.helpers.adaln_indexed import check_shared_bank0

    check_shared_bank0(indexed_op, indexed_op, "cuda")


def test_indexed_two_bank_metadata_overflow_precedes_value_reads(metadata_op):
    import torch

    op = metadata_op
    # x fits int32, but 2*B*3*H does not. Meta .item()/copies must never run.
    x = torch.empty(65535, 1, 5462, device="meta")
    m = torch.empty(131070, 16386, device="meta")
    index = torch.empty(65535, 1, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="int32 indexing"):
        op(x, m, modulate_index=index)


@pytest.mark.cuda_only
def test_indexed_bf16_final_rounding(indexed_op):
    from tests.helpers.adaln_indexed import check_bf16_rounding

    check_bf16_rounding(indexed_op, "cuda")


@pytest.mark.cuda_only
def test_indexed_bf16_raw16_gate(indexed_op):
    from tests.helpers.adaln_indexed import check_bf16_gate_payload

    check_bf16_gate_payload(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "index_dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("broadcast", [False, True])
def test_indexed_full_route_domain(indexed_op, dtype, index_dtype, broadcast):
    from tests.helpers.adaln_indexed import check_domain_routes

    check_domain_routes(indexed_op, "cuda", dtype, index_dtype, broadcast)


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["strided", "expanded"])
def test_indexed_source_leaf_gradients(indexed_op, dtype, layout):
    from tests.helpers.adaln_indexed import check_view_edges

    check_view_edges(indexed_op, "cuda", dtype, layout)


@pytest.fixture(params=["cuda", "triton"])
def metadata_op(request):
    if request.param == "triton":
        pytest.importorskip("triton")
        from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

        return TritonAdaLNModulationOp()
    import os

    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    try:
        return CudaAdaLNModulationOp()
    except RuntimeError:
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")


@pytest.mark.parametrize(
    "shape,reason",
    [
        ((65536, 1, 2), "B <= 65535"),
        ((1, 1, 4097), "H <= 4096"),
        ((1, 2**31, 1), "int32 indexing"),
        ((1, 524289, 4096), "int32 indexing"),
    ],
)
def test_indexed_metadata_rejects_before_copy_or_allocation(metadata_op, shape, reason):
    b, s, h = shape
    x = torch.empty(shape, device="meta")
    m = torch.empty(2 * b, 3 * h, device="meta")
    index = torch.empty(1, s, dtype=torch.int32, device="meta")
    with pytest.raises((ValueError, RuntimeError), match=reason):
        metadata_op(x, m, modulate_index=index)


@pytest.mark.parametrize("which", ["x", "m", "index"])
def test_indexed_metadata_rejects_strided_access_span(metadata_op, which):
    x = torch.empty(2, 2, 2, device="meta")
    m = torch.empty(4, 6, device="meta")
    index = torch.empty(2, 2, dtype=torch.int64, device="meta")
    if which == "x":
        x = torch.empty_strided(x.shape, (2**31, 2, 1), device="meta")
    elif which == "m":
        m = torch.empty_strided(m.shape, (2**31, 1), device="meta")
    else:
        index = torch.empty_strided(index.shape, (2**31, 1), dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="int32 indexing"):
        metadata_op(x, m, modulate_index=index)


@pytest.mark.parametrize("shape", [(1, 2**31 - 1, 1), (65535, 1, 4096)])
def test_indexed_positive_metadata_bounds_reach_device_check(metadata_op, shape):
    b, s, h = shape
    with pytest.raises(RuntimeError, match="NVIDIA CUDA tensors"):
        metadata_op(
            torch.empty(shape, device="meta"),
            torch.empty(2 * b, 3 * h, device="meta"),
            modulate_index=torch.empty(1, s, dtype=torch.bool, device="meta"),
        )


@pytest.mark.cuda_only
@pytest.mark.parametrize("eps", [0.0, 2**-149, 2**-126])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexed_eps_arithmetic(indexed_op, eps, dtype):
    from tests.helpers.adaln_indexed import check_eps_arithmetic

    check_eps_arithmetic(indexed_op, "cuda", dtype, eps)


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("threads", [128, 256, 512])
def test_cuda_indexed_maximum_batch_both_banks_nonzero_dx(dtype, threads):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_maximum_batch

    check_maximum_batch(CudaAdaLNModulationOp(threads=threads), dtype)


@pytest.mark.cuda_only
@pytest.mark.parametrize("threads", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [1, 2, 3, 127, 128, 129, 255, 256, 257, 4095, 4096])
def test_cuda_indexed_threads_geometry_batch_layout_bytes(threads, dtype, hidden):
    require_cuda()
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_geometry_bytes

    check_geometry_bytes(CudaAdaLNModulationOp(threads=threads), "cuda", dtype, hidden)


@pytest.mark.cuda_only
def test_indexed_bf16_output_and_dx_ties(indexed_op):
    from tests.helpers.adaln_indexed import check_bf16_ties

    check_bf16_ties(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("invalid", ["value", "shape", "dtype", "eps"])
def test_indexed_direct_rejects_invalid_inputs(indexed_op, invalid):
    x = torch.zeros(1, 2, 3, device="cuda")
    m = torch.zeros(2, 9, device="cuda")
    index = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    eps = 7.0
    error, message = ValueError, "values must be 0 or 1"
    if invalid == "value":
        index.fill_(2)
    elif invalid == "shape":
        index = index[:, :1]
        message = "must have shape"
    elif invalid == "dtype":
        index = index.float()
        error, message = TypeError, "boolean or integer"
    else:
        eps, message = float("nan"), "finite and nonnegative"
    with pytest.raises(error, match=message):
        indexed_op(x, m, modulate_index=index, eps=eps)


def test_indexed_int32_maximum_access_offset_metadata(metadata_op):
    # The pointer is at the view start; the last accessed element offset is INT_MAX.
    x = torch.empty_strided((2, 1, 2), (2**31 - 2, 2, 1), device="meta")
    m = torch.empty(4, 6, device="meta")
    index = torch.empty(1, 1, dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA tensors"):
        metadata_op(x, m, modulate_index=index)


@pytest.mark.cuda_only
def test_indexed_bf16_parameter_fold_final_cast(indexed_op):
    from tests.helpers.adaln_indexed import check_bf16_parameter_fold_final_cast

    check_bf16_parameter_fold_final_cast(indexed_op, "cuda")


@pytest.mark.cuda_only
@pytest.mark.parametrize("output", ["y", "gate"])
def test_indexed_large_upstream_offset_rejected_before_copy(output, metadata_op):
    import os

    if os.environ.get("RLK_ADALN_LARGE_OFFSETS") != "1":
        pytest.skip(
            "opt-in: upstream storage requires over 4 GiB; metadata alone cannot test backward"
        )
    # The backend fixture already checks its own dependencies; CUDA must not
    # acquire a Triton requirement when sharing this public backward test.
    if torch.version.hip is not None or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required; not a large-offset GPU PASS")
    free, _ = torch.cuda.mem_get_info()
    assert free > 5 * 2**30, "large-offset acceptance needs >5 GiB free GPU memory"
    x = torch.tensor([[[-3.0, 3.0]]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    m = torch.zeros(2, 6, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    index = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    y, gate = metadata_op(x, m, modulate_index=index, eps=7.0)
    # Two logical elements, but final element address exceeds signed int32.
    # Initialize only accessed positions; never fill this entire 4+ GiB storage.
    upstream = torch.empty_strided((1, 1, 2), (0, 0, 2**31), device="cuda", dtype=x.dtype)
    upstream[0, 0, 0], upstream[0, 0, 1] = 1.0, 0.0
    with pytest.raises(ValueError, match="upstream requires int32 indexing"):
        torch.autograd.grad(y if output == "y" else gate, (x, m), upstream)


@pytest.mark.cuda_only
def test_indexed_bf16_all_parameter_ties(indexed_op):
    from tests.helpers.adaln_indexed import check_bf16_parameter_ties

    check_bf16_parameter_ties(indexed_op, "cuda")
