# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in tests for Triton addressing beyond the signed 32-bit element range."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("RLK_ATTN_OUT_LARGE_OFFSETS") != "1", reason="requires large GPU allocations"
)


@pytest.fixture
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    from rl_engine.backends.shared.triton.gemm import attn_out_bias_gemm as module

    torch.cuda.empty_cache()
    if not module._TRITON_AVAILABLE or torch.cuda.mem_get_info()[0] < 20 * 2**30:
        pytest.skip("requires Triton and 20 GiB of available device memory")
    return module


def test_tree_stores_beyond_int32(kernels):
    # The final row starts beyond INT_MAX. Launch just the first column tile.
    p, q = 32769, 65537
    a = torch.ones((p, 1), device="cuda", dtype=torch.float32)
    b = torch.ones((q, 1), device="cuda", dtype=torch.float32)
    out = torch.empty((p, q), device="cuda", dtype=torch.float32)
    tile = out[-1, :64]
    tile.fill_(-1)
    kernels._tree_gemm_kernel[((p + 31) // 32, 1)](
        a, b, out, p, q, 1, 1, DEPTH=0, BLOCK_M=32, BLOCK_N=64
    )
    assert torch.equal(tile, torch.ones_like(tile))


@pytest.mark.parametrize("wide_input", ["grad", "x"])
def test_dw_reads_beyond_int32(kernels, wide_input):
    # The third row starts past INT_MAX in one input. Only the first
    # output tile is launched; other columns do not need initialization.
    wide = 2**30 + 16
    n, k = (wide, 1) if wide_input == "grad" else (1, wide)
    grad = torch.empty((3, n), device="cuda", dtype=torch.float32)
    x = torch.empty((3, k), device="cuda", dtype=torch.float32)
    for row in range(3):
        grad[row, :64].fill_(row + 1)
        x[row, :64].fill_(2)
    out = torch.empty((n, k), device="cuda", dtype=torch.float32)
    tile = out[:64, :64]
    tile.fill_(-1)
    kernels._dw_left_fold_kernel[(1, 1)](grad, x, out, 3, n, k, BLOCK_N=64, BLOCK_K=64)
    assert torch.equal(tile, torch.full_like(tile, 12))


def test_dw_stores_beyond_int32(kernels):
    # Cover the first column tile of every row; the last row starts past INT_MAX.
    n, k = 32769, 65537
    grad = torch.empty((0, n), device="cuda", dtype=torch.float32)
    x = torch.empty((0, k), device="cuda", dtype=torch.float32)
    out = torch.empty((n, k), device="cuda", dtype=torch.float32)
    last_tile = out[-1, :64]
    last_tile.fill_(-1)
    kernels._dw_left_fold_kernel[((n + 63) // 64, 1)](grad, x, out, 0, n, k, BLOCK_N=64, BLOCK_K=64)
    assert torch.equal(last_tile, torch.zeros_like(last_tile))
