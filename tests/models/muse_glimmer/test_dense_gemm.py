# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #501 single-GPU qualification of the BF16 projection operators."""

import math

import pytest
import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _build_tree_plan
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.models.muse_glimmer.dense_gemm import dense_gemm
from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec
from rl_engine.reference.gemm.det_gemm import DetGemmTreeReferenceOp

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA/Triton parity requires an NVIDIA SM80+ GPU",
)
_PROJECTIONS = MuseGlimmerSpec.from_manifest().gemm_shape_groups()


@pytest.mark.parametrize("n", (4096 + 256 + 256, 4096 + 256 + 256 + 4096, 19968 * 2))
def test_rejects_fused_projection_weights(n):
    x = torch.empty((1, 6656), device="meta", dtype=torch.bfloat16)
    weight = x.new_empty((n, 6656))
    with pytest.raises(ValueError, match="single projection"):
        dense_gemm(x, weight, backend="cuda")


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


def _qualify_batch(x, weight, dy, record_property):
    cuda = _evaluate("cuda", x, weight, dy)
    triton = _evaluate("triton", x, weight, dy)
    m, k = x.shape
    n = weight.size(0)
    for name, shape, left, right in zip(
        ("Y", "dX", "dW"), ((m, n), (m, k), (n, k)), cuda, triton, strict=True
    ):
        assert left.shape == right.shape == shape
        assert left.is_contiguous() and right.is_contiguous()
        _assert_bits(left, right, f"M={m} {name}")
        record_property(f"M{m}_{name}_bit_differences", 0)
    assert cuda[2].stride() == triton[2].stride() == (k, 1)

    # Evaluate the PyTorch tree on a few outputs with the entire reduction
    # axis intact. The independent FP64 checks below cover every output word.
    gold = DetGemmTreeReferenceOp()
    for name, lhs, rhs, actual in (
        ("Y", x, weight.T, cuda[0]),
        ("dX", dy, weight, cuda[1]),
        ("dW", dy.T, x, cuda[2]),
    ):
        expected = gold(lhs[:2].cpu(), rhs[:, :4].cpu())
        _assert_bits(actual[:2, :4].cpu(), expected, f"M={m} {name} PyTorch tree")

    contract = load_contract()
    cell = dict(op_class="reduction", dtype=torch.bfloat16)
    forward = resolve_tolerance(contract, judgment="forward_accuracy", **cell)
    gradient = resolve_tolerance(contract, judgment="gradient_accuracy", **cell)
    x64, w64, dy64 = x.double(), weight.double(), dy.double()
    y64 = x64 @ w64.T
    torch.testing.assert_close(cuda[0].double(), y64, atol=forward.atol, rtol=forward.rtol)
    record_property(f"M{m}_Y_max_abs", (cuda[0].double() - y64).abs().max().item())
    for name, actual, exact in (("dX", cuda[1], dy64 @ w64), ("dW", cuda[2], dy64.T @ x64)):
        error = torch.linalg.vector_norm(actual.double() - exact).item()
        norm = torch.linalg.vector_norm(exact).item()
        assert error <= gradient.rtol * norm, f"M={m} {name}: ||error||={error}, ||ref||={norm}"
        record_property(f"M{m}_{name}_relative_L2", error / norm if norm else 0.0)
    return cuda[:2]


@pytest.mark.ws1
@requires_cuda
@pytest.mark.parametrize(("n", "k"), _PROJECTIONS, ids=["+".join(v) for v in _PROJECTIONS.values()])
def test_projection_parity(n, k, record_property):
    generator = torch.Generator(device="cuda").manual_seed(501)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(96, k, **options)
    weight = torch.randn(n, k, **options) / math.sqrt(k)
    dy = torch.randn(96, n, **options)

    y, dx = _qualify_batch(x, weight, dy, record_property)
    row = slice(17, 18)
    single_y, single_dx = _qualify_batch(x[row], weight, dy[row], record_property)
    _assert_bits(single_y, y[row], "decode vs mixed Y")
    _assert_bits(single_dx, dx[row], "decode vs mixed dX")

    # Chunk 24 valid rows and shift their batch positions by five, with padding
    # on both sides. Keep their upstream gradients fixed for the dX comparison.
    padded_x = torch.nn.functional.pad(x[:24], (0, 0, 5, 7)).requires_grad_()
    padded_dy = torch.nn.functional.pad(dy[:24], (0, 0, 5, 7))
    valid = slice(5, 29)
    for backend in ("cuda", "triton"):
        with torch.inference_mode():
            prefill = dense_gemm(padded_x, weight, backend=backend)
        _assert_bits(prefill[valid], y[:24], f"{backend} padded prefill Y")
        train_y = dense_gemm(padded_x, weight, backend=backend)
        (padded_dx,) = torch.autograd.grad(train_y, padded_x, padded_dy)
        _assert_bits(padded_dx[valid], dx[:24], f"{backend} padded dX")


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
    w[2, 0], w[2, 32], w[2, 64] = 256, 1, -256
    # The 26-wide first leaf rounds away the unit in column 0. The 32-wide
    # first leaf rounds it away in column 1. A single FP64 sum retains both.
    # The first three leaves form L0 + (L1 + L2). Pairing L0 with L1 first
    # would instead return zero in column 2.
    gold = DetGemmTreeReferenceOp()(x, w[:3].T)
    assert gold.tolist() == [[1, 0, 1]]
    assert (x.double() @ w[:3].double().T).tolist() == [[1, 1, 1]]
    expected = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    expected[:, :3] = gold
    for backend in ("cuda", "triton"):
        _assert_bits(dense_gemm(x.cuda(), w.cuda(), backend=backend), expected, backend)

    # The 96-token dW kernel has exactly three leaves. These contributions
    # give 1 with L0 + (L1 + L2) and 0 with (L0 + L1) + L2.
    batch = torch.zeros(96, 6656, device="cuda", dtype=torch.bfloat16)
    batch[0, 0], batch[32, 0], batch[64, 0] = 256, 1, -256
    dy = torch.zeros(96, 256, device="cuda", dtype=torch.bfloat16)
    dy[:, 0] = 1
    weight = w.cuda().requires_grad_()
    for backend in ("cuda", "triton"):
        (dw,) = torch.autograd.grad(dense_gemm(batch, weight, backend=backend), weight, dy)
        assert dw[0, 0].item() == 1, backend


@requires_cuda
def test_projection_partial_tile_and_offset_views():
    # Both aligned and misaligned contiguous views can carry the same input.
    # Five rows leave a partially populated output tile.
    generator = torch.Generator(device="cuda").manual_seed(501)
    k, n = 6656, 256
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(5 * k + 1, **options)[1:].view(5, k)
    w = torch.randn(n * k + 1, **options)[1:].view(n, k)
    expected = dense_gemm(x, w, backend="cuda")
    for backend in ("cuda", "triton"):
        _assert_bits(dense_gemm(x.clone(), w.clone(), backend=backend), expected, backend)
        _assert_bits(dense_gemm(x, w, backend=backend), expected, backend)
