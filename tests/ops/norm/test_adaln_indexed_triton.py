# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L2: indexed NVIDIA Triton semantics, independent accuracy and fixed bytes."""

import pytest
import torch

from tests.helpers.adaln_indexed import require_triton_gpu, small_case


@pytest.fixture
def triton_op():
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    return TritonAdaLNModulationOp()


def test_triton_indexed_cpu_rejected_through_public_callable(triton_op):
    x, m, index, _, _ = small_case("cpu")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA tensors"):
        triton_op(x, m, modulate_index=index, eps=7.0)


@pytest.mark.cuda_only
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("reduction_tile", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [1, 2, 3, 127, 128, 129, 255, 256, 257, 3072, 4095, 4096])
def test_triton_indexed_launch_batch_layout_bytes(num_warps, reduction_tile, dtype, hidden):
    require_triton_gpu()
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_geometry_bytes

    check_geometry_bytes(
        TritonAdaLNModulationOp(num_warps=num_warps, reduction_tile=reduction_tile),
        "cuda",
        dtype,
        hidden,
    )


@pytest.mark.parametrize("num_warps,reduction_tile", [(1, 128), (4, 32), (8, 512)])
def test_triton_rejects_undeclared_launch_settings(num_warps, reduction_tile):
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    with pytest.raises(ValueError, match="num_warps"):
        TritonAdaLNModulationOp(num_warps=num_warps, reduction_tile=reduction_tile)


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("reduction_tile", [64, 128, 256])
def test_triton_indexed_maximum_batch_both_banks(num_warps, reduction_tile, dtype):
    require_triton_gpu()
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp
    from tests.helpers.adaln_indexed import check_maximum_batch

    check_maximum_batch(
        TritonAdaLNModulationOp(num_warps=num_warps, reduction_tile=reduction_tile), dtype
    )


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.parametrize("route", ["segmented", "dispersed"])
def test_triton_indexed_target_h3072_workload(triton_op, dtype, seq, route):
    import os

    if os.environ.get("RLK_ADALN_REAL_SHAPES") != "1":
        pytest.skip("opt-in target workload; requires >=8 GiB free GPU and >=16 GiB host memory")
    require_triton_gpu()
    from tests.helpers.adaln_indexed import check_geometry_bytes

    free, _ = torch.cuda.mem_get_info()
    assert free >= 8 * 2**30, "target comparison requires >=8 GiB free GPU memory"
    check_geometry_bytes(triton_op, "cuda", dtype, 3072, seq=seq, route=route)


@pytest.mark.parametrize("eps", [1e300, 2**-150])
def test_triton_indexed_direct_rejects_legacy_valid_eps_before_copy(triton_op, eps):
    x, m, index, _, _ = small_case("cpu")
    with pytest.raises(RuntimeError, match="finite FP32 eps"):
        triton_op(x, m, modulate_index=index, eps=eps)


@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "dtype",
        "device",
        "index_shape",
        "index_dtype",
        "negative_eps",
        "inf_eps",
        "nan_eps",
    ],
)
def test_triton_indexed_metadata_validation_before_device_or_values(triton_op, invalid):
    x, m, index, _, _ = small_case("cpu")
    eps = 7.0
    error, message = ValueError, "shape"
    if invalid == "shape":
        m = m[:, :-1]
    elif invalid == "dtype":
        m = m.double()
        error, message = TypeError, "share FP32 or BF16"
    elif invalid == "device":
        index = index.to("meta")
        message = "share device"
    elif invalid == "index_shape":
        index = index[:, :-1]
    elif invalid == "index_dtype":
        index = index.float()
        error, message = TypeError, "boolean or integer"
    else:
        eps = {"negative_eps": -1.0, "inf_eps": float("inf"), "nan_eps": float("nan")}[invalid]
        message = "finite and nonnegative"
    with pytest.raises(error, match=message):
        triton_op(x, m, modulate_index=index, eps=eps)


@pytest.mark.cuda_only
@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("outputs", ["gate", "both"])
def test_triton_indexed_bf16_subnormal_parameter_fold(triton_op, seq, outputs):
    require_triton_gpu()
    from tests.helpers.adaln_indexed import check_bf16_subnormal_parameter_fold

    check_bf16_subnormal_parameter_fold(triton_op, "cuda", seq, outputs)
