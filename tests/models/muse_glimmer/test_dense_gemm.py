# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #501 single-GPU qualification of the BF16 projection operators."""

import math

import pytest
import torch

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


def test_rejects_empty_batch():
    x = torch.empty((0, 6656), device="meta", dtype=torch.bfloat16)
    weight = x.new_empty((256, 6656))
    with pytest.raises(ValueError, match="positive M"):
        dense_gemm(x, weight, backend="cuda")


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


_LEAF_WITNESSES = (
    # Keep the unit until cancellation within one FP32 leaf. Early BF16 rounding loses it.
    ((11, 25, 26), (256, 1, -256), 1),
    # A single FP32 chain loses the unit, while separate eight-term chunks can keep it.
    ((11, 19, 27), (2**24, 1, -(2**24)), 0),
    # The first leaf rounds 257 to 256. Moving the boundary past index 32 yields 1.
    ((11, 31, 32), (256, 1, -256), 0),
)


def _projection_witnesses(k):
    midpoint = (k // 32 // 2) * 32
    return (
        *_LEAF_WITNESSES,
        # Only 256 is in the left root child. A later split (e.g. at 128 of 208
        # leaves or 512 of 624) pairs it with 1 before cancellation and yields 0.
        ((midpoint - 32, midpoint, k - 1), (256, 1, -256), 1),
        # The first two contributions meet at an inner merge and round to 256.
        # Keeping FP32 until the root would produce 1 instead of 0.
        ((11, 32, k - 1), (256, 1, -256), 0),
    )


@pytest.mark.ws1
@requires_cuda
@pytest.mark.parametrize(("n", "k"), _PROJECTIONS, ids=["+".join(v) for v in _PROJECTIONS.values()])
@pytest.mark.parametrize("m", (1, 79, 96))
def test_projection_tree_rounding_witness(n, k, m):
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.zeros((m, k), **options)
    w = torch.zeros((n, k), **options)
    dy = torch.zeros((m, n), **options)

    # The nonzeros in W's first rows drive Y, and those in its first columns
    # drive dX. The two blocks of W are disjoint, as are the inputs that reach them.
    y_cases, dx_cases = _projection_witnesses(k), _projection_witnesses(n)
    for row, (indices, values, _) in enumerate(y_cases):
        x[:, indices] = 1
        w[row, indices] = w.new_tensor(values)
    for col, (indices, values, _) in enumerate(dx_cases):
        dy[:, indices] = 1
        w[indices, col] = w.new_tensor(values)

    # Use unused feature indices to exercise the token reduction in dW.
    dw_row = len(y_cases)
    if m == 1:
        dy[:, dw_row] = 7
        dw_cases = (((0,), (37,), 260),)  # 259 rounds to 260 (even).
    else:
        dy[:, dw_row] = 1
        dw_cases = (
            *_LEAF_WITNESSES,
            # 256 + 1 rounds to 256 at the inner merge. Delaying that cast yields -1.
            ((0, 32, m - 1), (-258, 256, 1), -2),
            # L0 + (L1 + L2) gives 1. (L0 + L1) + L2 gives 0, as does
            # splitting at the element midpoint for the 79-token reduction.
            ((0, 32, m - 1), (256, 1, -256), 1),
        )
    dw_cols = slice(len(dx_cases), len(dx_cases) + len(dw_cases))
    for col, (indices, values, _) in enumerate(dw_cases, start=dw_cols.start):
        x[indices, col] = x.new_tensor(values)

    gold = DetGemmTreeReferenceOp()
    expected_y = x.new_zeros((1, n))
    expected_dx = x.new_zeros((1, k))
    y_slice = slice(0, len(y_cases))
    dx_slice = slice(0, len(dx_cases))
    dw_rows = slice(dw_row, dw_row + 1)
    expected_y[0, y_slice] = x.new_tensor([result for _, _, result in y_cases])
    expected_dx[0, dx_slice] = x.new_tensor([result for _, _, result in dx_cases])
    expected_dw = x.new_tensor([[result for _, _, result in dw_cases]])
    for name, lhs, rhs, expected in (
        ("Y", x[:1], w[y_slice].T, expected_y[:, y_slice]),
        ("dX", dy[:1], w[:, dx_slice], expected_dx[:, dx_slice]),
        ("dW", dy[:, dw_rows].T, x[:, dw_cols], expected_dw),
    ):
        _assert_bits(gold(lhs.cpu(), rhs.cpu()), expected.cpu(), f"{name} reference")

    for backend in ("cuda", "triton"):
        y, dx, dw = _evaluate(backend, x, w, dy)
        _assert_bits(y, expected_y.expand_as(y), f"{backend} Y")
        _assert_bits(dx, expected_dx.expand_as(dx), f"{backend} dX")
        _assert_bits(dw[dw_rows, dw_cols], expected_dw, f"{backend} dW")


@requires_cuda
@pytest.mark.parametrize("m", [1, 5, 35])
def test_projection_partial_tile_and_offset_views(m):
    # Both aligned and misaligned contiguous views can carry the same input.
    # Exercise decode, a thin batch, and the partial 32-row tile in prefill.
    generator = torch.Generator(device="cuda").manual_seed(501)
    k, n = 6656, 256
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(m * k + 1, **options)[1:].view(m, k)
    w = torch.randn(n * k + 1, **options)[1:].view(n, k)
    expected = DetGemmTreeReferenceOp()(x.cpu(), w.T.cpu())
    for backend in ("cuda", "triton"):
        _assert_bits(dense_gemm(x, w, backend=backend).cpu(), expected, f"{backend} offset views")
        aligned = dense_gemm(x.clone(), w.clone(), backend=backend)
        _assert_bits(aligned.cpu(), expected, f"{backend} contiguous clones")


@requires_cuda
@pytest.mark.parametrize("m", [5, 128, 130, 512])
def test_projection_arbitrary_batch(m):
    n, k = (256, 6656) if m == 5 else (19968, 6656)
    generator = torch.Generator(device="cuda").manual_seed(501)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=generator)

    def inputs(rows, cols):
        if m == 130:
            return torch.randn(rows * cols + 1, **options)[1:].view(rows, cols)
        return torch.randn(rows, cols, **options)

    x, weight, dy = inputs(m, k), inputs(n, k), inputs(m, n)
    cuda, triton = (_evaluate(backend, x, weight, dy) for backend in ("cuda", "triton"))
    gold = DetGemmTreeReferenceOp()
    # Row 96 is the first row of the second CUDA forward chunk on the gate
    # projection. The last row also covers the short final chunk at M=130.
    rows = slice(96, 98) if m >= 128 else slice(3, 5)
    for name, left, right, lhs, rhs, sample in (
        ("Y", cuda[0], triton[0], x, weight.T, rows),
        ("dX", cuda[1], triton[1], dy, weight, rows),
        ("dW", cuda[2], triton[2], dy.T, x, slice(96, 98)),
    ):
        assert left.is_contiguous() and right.is_contiguous()
        _assert_bits(left, right, f"M={m} {name}")
        expected = gold(lhs[sample].cpu(), rhs[:, :4].cpu())
        _assert_bits(left[sample, :4].cpu(), expected, f"M={m} {name} PyTorch tree")

    if m >= 128:
        for backend, (y, dx, _) in zip(("cuda", "triton"), (cuda, triton), strict=True):
            for row in (slice(96, 97), slice(m - 1, m)):
                token = x[row].detach().requires_grad_()
                single_y = dense_gemm(token, weight, backend=backend)
                (single_dx,) = torch.autograd.grad(single_y, token, dy[row])
                _assert_bits(single_y, y[row], f"{backend} M={m} single-row Y")
                _assert_bits(single_dx, dx[row], f"{backend} M={m} single-row dX")
