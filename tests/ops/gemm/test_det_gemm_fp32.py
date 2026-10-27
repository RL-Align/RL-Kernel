# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exercise the shared FP32 GEMM default index path on a real CUDA device."""

import inspect
import os

import pytest
import torch

from rl_engine.backends.shared.triton.gemm import det_gemm
from rl_engine.contracts.numerical import load_contract


@pytest.fixture(scope="module", autouse=True)
def require_fp32_gpu():
    reason = None
    if torch.version.hip is not None or not torch.cuda.is_available():
        reason = "Shared FP32 regression requires an NVIDIA CUDA device"
    elif torch.cuda.get_device_capability()[0] < 9:
        reason = "Shared FP32 regression requires CUDA SM90+"
    elif not det_gemm._TRITON_AVAILABLE:
        reason = "Shared FP32 regression requires Triton"
    if reason:
        if os.environ.get("RL_KERNEL_REQUIRE_FP32_GPU") == "1":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def default_index_launches(monkeypatch):
    """Observe real launches without substituting another implementation."""
    kernel = det_gemm._det_gemm_fp32_kernel
    default_index = inspect.signature(kernel.fn).parameters["INDEX_64"].default
    assert default_index is False
    launches = []

    class ObservedKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                # The general caller must exercise the default specialization.
                assert "INDEX_64" not in kwargs
                assert kwargs["PROMOTE_INPUTS"] is True
                assert kwargs["COMPENSATED"] is True
                result = kernel[grid](*args, **kwargs)
                launches.append(default_index)
                return result

            return launch

    monkeypatch.setattr(det_gemm, "_det_gemm_fp32_kernel", ObservedKernel())
    return launches


def assert_same_bytes(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("shape", [(1, 65, 73), (33, 129, 67), (129, 5376, 73)])
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
    ids=["bf16", "fp32", "bf16-fp32", "fp32-bf16"],
)
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_shared_fp32_default_index_accuracy_and_invariance(
    shape, a_dtype, b_dtype, strided, default_index_launches
):
    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(420)
    a = torch.randn((m, k), generator=generator, device="cuda").to(a_dtype)
    b = torch.randn((k, n), generator=generator, device="cuda").to(b_dtype)
    if strided:
        a = a.t().contiguous().t()
        b = b.t().contiguous().t()

    actual = det_gemm._triton_gemm_fp32(a, b)
    reference = a.float() @ b.float()
    tolerance = load_contract()["accuracy"]["default"]["reduction"]["float32"]
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, **tolerance)
    assert_same_bytes(actual, det_gemm._triton_gemm_fp32(a, b))
    assert_same_bytes(actual, det_gemm._triton_gemm_fp32(a.flip(0), b).flip(0))
    chunks = torch.cat([det_gemm._triton_gemm_fp32(part, b) for part in a.split(31)])
    assert_same_bytes(actual, chunks)
    # Cover the public FP32-output operator entry point as well as its helper.
    assert_same_bytes(actual, det_gemm.TritonDetGemmOp().forward_fp32(a, b))
    assert len(default_index_launches) == 4 + len(a.split(31))
    assert all(index_64 is False for index_64 in default_index_launches)


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
    ids=["bf16", "fp32", "bf16-fp32", "fp32-bf16"],
)
def test_shared_fp32_retains_small_terms_between_large_tiles(
    a_dtype, b_dtype, default_index_launches
):
    # Each tile contributes one exactly representable value. An ordinary FP32
    # left fold loses both unit terms; compensated accumulation retains them.
    a = torch.ones((1, 128), device="cuda", dtype=a_dtype)
    b = torch.zeros((128, 1), device="cuda", dtype=b_dtype)
    b[0, 0], b[32, 0], b[64, 0], b[96, 0] = 2**24, 1, 1, -(2**24)
    expected = (a.double() @ b.double()).float()
    assert expected.item() == 2.0
    actual = det_gemm.TritonDetGemmOp().forward_fp32(a, b)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert default_index_launches == [False]
