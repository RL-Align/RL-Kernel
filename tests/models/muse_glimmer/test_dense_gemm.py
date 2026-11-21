# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #501 single-GPU qualification of the BF16 projection operators."""

import math

import pytest
import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _build_tree_plan
from rl_engine.models.muse_glimmer.dense_gemm import dense_gemm
from rl_engine.reference.gemm.det_gemm import DetGemmTreeReferenceOp

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA/Triton parity requires an NVIDIA SM80+ GPU",
)


def _assert_bits(actual, expected, label):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    different = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    count = different.sum().item()
    if count:
        index = different.flatten().int().argmax().item()
        pytest.fail(f"{label}: {count}/{actual.numel()} BF16 words differ, first index {index}")


def _evaluate(backend, x, weight, dy):
    x = x.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    y = dense_gemm(x, weight, backend=backend)
    return (y.detach(), *torch.autograd.grad(y, (x, weight), dy))


@requires_cuda
def test_kv_projection_parity():
    generator = torch.Generator(device="cuda").manual_seed(501)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(96, 6656, **options)
    weight = torch.randn(256, 6656, **options) / math.sqrt(6656)
    dy = torch.randn(96, 256, **options)

    row = slice(17, 18)
    for inputs, upstream in ((x, dy), (x[row], dy[row])):
        cuda = _evaluate("cuda", inputs, weight, upstream)
        triton = _evaluate("triton", inputs, weight, upstream)
        for name, left, right in zip(("Y", "dX", "dW"), cuda, triton, strict=True):
            _assert_bits(left, right, f"M={len(inputs)} {name}")
        if len(inputs) == len(x):
            batch_y, batch_dx = cuda[0][row], cuda[1][row]
        else:
            _assert_bits(cuda[0], batch_y, "single vs mixed Y")
            _assert_bits(cuda[1], batch_dx, "single vs mixed dX")


@pytest.mark.parametrize("k", (1, 65, 96, 256, 4096, 6656, 19968))
def test_fixed_leaf_host_plan(k):
    plan = _build_tree_plan(k, fixed_leaf_tree=True)
    starts = tuple(range(0, k, 32))
    assert plan.leaf_starts == starts
    assert plan.leaf_lengths == tuple(min(32, k - start) for start in starts)
    # Track intervals of leaf indices. Each midpoint splits the leaf count
    # with the smaller half on the left, even when K ends in a partial leaf.
    spans = {node: (i, i + 1) for i, node in enumerate(plan.leaf_nodes)}
    for level in plan.reduction_levels:
        for left, right, parent in level:
            lo, mid = spans.pop(left)
            right_lo, hi = spans.pop(right)
            assert mid == right_lo == lo + (hi - lo) // 2
            spans[parent] = (lo, hi)
    assert spans == {plan.root: (0, len(starts))}
    assert plan.node_count == 2 * len(starts) - 1


@requires_cuda
def test_kv_tree_rounding_witness():
    x = torch.ones((1, 6656), dtype=torch.bfloat16)
    w = torch.zeros((256, 6656), dtype=torch.bfloat16)
    w[0, 0], w[0, 25], w[0, 26] = 256, 1, -256
    w[1, 0], w[1, 31], w[1, 32] = 256, 1, -256
    # The 26-wide first leaf rounds away the unit in column 0. The 32-wide
    # first leaf rounds it away in column 1. A single FP64 sum retains both.
    gold = DetGemmTreeReferenceOp()(x, w[:2].T)
    assert gold.tolist() == [[1, 0]]
    assert (x.double() @ w[:2].double().T).tolist() == [[1, 1]]
    expected = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    expected[:, :2] = gold
    for backend in ("cuda", "triton"):
        _assert_bits(dense_gemm(x.cuda(), w.cuda(), backend=backend), expected, backend)
