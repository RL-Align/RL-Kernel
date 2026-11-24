# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent dyadic goldens for the declared CUDA/Triton K-tree profiles."""

import pytest
import torch

from tests.ops.gemm import test_projection_gemm as shared

backend = shared.backend
DEVICE = shared.DEVICE

pytestmark = pytest.mark.skipif(
    DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA/Triton K-tree profiles"
)


def _rounded_tree_reference(a, b, tile_midpoints):
    """CPU FP64 leaves followed by the profile's explicit BF16 rounding points.

    The dyadic operands below make leaf sums exactly representable in FP32.
    This avoids depending on torch FP32 GEMM or device contraction order in
    a golden intended to detect changed K partitions/node rounding.
    """
    a, b = a.cpu().double(), b.cpu().double()
    extent = a.shape[1] // 32 if tile_midpoints else a.shape[1]
    leaf_size = 1 if tile_midpoints else 32

    def visit(lo, hi):
        if hi - lo <= leaf_size:
            start, end = (lo * 32, hi * 32) if tile_midpoints else (lo, hi)
            return (a[:, start:end] @ b[start:end]).bfloat16()
        middle = (lo + hi) // 2
        left, right = visit(lo, middle), visit(middle, hi)
        return (left.double() + right.double()).bfloat16()

    return visit(0, extent).to(DEVICE)


@pytest.mark.parametrize("k", (64, 96, 160, 192, 2048))
@pytest.mark.parametrize("n", (64, 65))
@torch.no_grad()
def test_canonical_midpoint_partitions_and_rounding(backend, k, n):
    torch.manual_seed(647)
    a = torch.randint(-7, 8, (3, k), dtype=torch.int32).to(DEVICE, torch.bfloat16) / 8
    b = torch.randint(-7, 8, (k, n), dtype=torch.int32).to(DEVICE, torch.bfloat16) / 8
    tile_midpoints = backend[0] == "native" and n % 64 == 0
    expected = _rounded_tree_reference(a, b, tile_midpoints)
    shared._raw_equal(backend[1](a, b), expected)


def test_non_power_of_two_profiles_have_distinct_goldens():
    # A fixed negative control: a single universal tree oracle would certify
    # one of the two profiles incorrectly for these non-power-of-two K values.
    generator = torch.Generator().manual_seed(648)
    a = torch.randint(-7, 8, (3, 96), generator=generator).bfloat16() / 8
    b = torch.randint(-7, 8, (96, 64), generator=generator).bfloat16() / 8
    tile = _rounded_tree_reference(a, b, True)
    element = _rounded_tree_reference(a, b, False)
    assert not torch.equal(tile.view(torch.int16), element.view(torch.int16))
