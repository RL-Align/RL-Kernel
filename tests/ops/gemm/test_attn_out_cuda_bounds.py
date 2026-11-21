# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise CUDA dimension/grid guards using zero-storage inputs."""

import pytest
import torch


@pytest.fixture
def extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import _C, _cuda_backend_available

    if not _cuda_backend_available():
        pytest.skip("CUDA extension unavailable")
    return _C


def _invoke(extension, entry, first_shape, second_shape):
    first = torch.empty(first_shape, device="cuda", dtype=torch.float32)
    second = torch.empty(second_shape, device="cuda", dtype=torch.float32)
    assert first.numel() == second.numel() == 0
    if entry == "forward":
        return extension.attn_out_bias_gemm_cuda_forward(first, second, None, False)
    if entry == "tree":
        return extension.attn_out_tree_gemm_cuda(first, second)
    return extension.attn_out_dw_left_fold_cuda(first, second)


@pytest.mark.parametrize("entry", ["forward", "tree", "dw"])
@pytest.mark.parametrize("dimension", ["rows", "output", "reduction"])
def test_cuda_rejects_dimensions_before_int32_narrowing(extension, entry, dimension):
    huge = 2**31
    if entry == "dw":
        shapes = {
            "rows": ((huge, 0), (huge, 0)),
            "output": ((0, huge), (0, 0)),
            "reduction": ((0, 0), (0, huge)),
        }
    else:
        shapes = {
            "rows": ((huge, 0), (0, 0)),
            "output": ((0, 0), (huge, 0)),
            "reduction": ((0, huge), (0, huge)),
        }
    with pytest.raises(RuntimeError, match="int32 dimension range"):
        _invoke(extension, entry, *shapes[dimension])


@pytest.mark.parametrize("entry", ["forward", "tree", "dw"])
def test_cuda_rejects_grid_overflow_before_output_allocation(extension, entry):
    # Individual dimensions fit int32; ceil(total / 256) does not.
    rows, cols = 2**31 - 1, 257
    shapes = ((0, rows), (0, cols)) if entry == "dw" else ((rows, 0), (cols, 0))
    with pytest.raises(RuntimeError, match=r"CUDA grid.x limit"):
        _invoke(extension, entry, *shapes)


@pytest.mark.parametrize("entry", ["forward", "tree", "dw"])
def test_cuda_allows_int32_max_dimension_with_empty_output(extension, entry):
    maximum = 2**31 - 1
    shapes = ((0, maximum), (0, 0)) if entry == "dw" else ((0, maximum), (0, maximum))
    result = _invoke(extension, entry, *shapes)
    assert result.numel() == 0
