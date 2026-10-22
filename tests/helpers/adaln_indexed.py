# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Test-only indexed AdaLN fixtures and comparisons shared by L1/L2/L3."""

import os

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


def small_case(device):
    generator = torch.Generator().manual_seed(386)
    # Large positive eps keeps H2 dx sensitive; samples and banks have distinct tags.
    x = (torch.randn(2, 8, 2, generator=generator) * 2 + 0.3).to(device).requires_grad_()
    m = (torch.randn(4, 6, generator=generator) * 0.5).to(device).requires_grad_()
    index = torch.tensor(
        [[0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1]], dtype=torch.int32, device=device
    )
    dy = torch.randn(2, 8, 2, generator=generator).to(device)
    dg = torch.randn(2, 8, 2, generator=generator).to(device)
    return x, m, index, dy, dg


def check_small_case(op, device, trace=None):
    x, m, index, dy, dg = small_case(device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    if trace is not None:
        # Check routing before numerical comparisons: a correct fallback is still red.
        assert trace["actual_backend"] == "cuda"
        assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
        assert trace["kernel_fingerprint"] == "adaln_indexed_cuda_v1"
        assert not trace["fallback"]
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    for reference in (IndexedAdaLNFixedReference(), GtestAdaLNReference().forward_fp32):
        rx = x.detach().cpu().requires_grad_()
        rm = m.detach().cpu().requires_grad_()
        ry, rg = reference(rx, rm, modulate_index=index.cpu(), eps=7.0)
        rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy.cpu(), dg.cpu()))
        assert torch.isfinite(rdx).all() and torch.count_nonzero(rdx) == rdx.numel()
        for actual, expected, judgment in (
            (y, ry, "forward_accuracy"),
            (gate, rg, "forward_accuracy"),
            (dx, rdx, "gradient_accuracy"),
            (dm, rdm, "gradient_accuracy"),
        ):
            assert actual.dtype == expected.dtype == torch.float32
            if isinstance(reference, IndexedAdaLNFixedReference):
                assert_bytes(actual, expected)
            else:
                tolerance = resolve_tolerance(
                    load_contract(), judgment=judgment, op_class="reduction", dtype=actual.dtype
                )
                torch.testing.assert_close(
                    actual.cpu(), expected, atol=tolerance.atol, rtol=tolerance.rtol
                )


def require_cuda():
    if torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA-only test; ROCm is outside this profile")
    if not torch.cuda.is_available():
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            pytest.fail("NVIDIA CUDA is required for indexed acceptance")
        pytest.skip("NVIDIA CUDA required; not an indexed GPU PASS")
    from rl_engine.backends.extension import _C

    # Once CUDA is available, missing/old symbols must fail rather than hide a red test.
    assert hasattr(_C, "adaln_indexed_forward"), "fresh indexed CUDA extension required"
    assert hasattr(_C, "adaln_indexed_backward"), "fresh indexed CUDA extension required"


def check_route_identity(op, device, route):
    """Hand case: norm=(-3/4,3/4), rstd=1/4; distinct sample/bank tags."""
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1).requires_grad_()
    m = torch.tensor(
        [
            [10, 11, 1, 0, 20, 21],
            [30, 31, 2, 0, 40, 41],
            [50, 51, 3, 0, 60, 61],
            [70, 71, 4, 0, 80, 81],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    routes = {"all0": [0] * 8, "all1": [1] * 8, "alternating": [0, 1] * 4}
    index = torch.tensor([routes[route]] * 2, dtype=torch.int32, device=device)
    dy = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]], device=device).expand_as(x)
    dg = torch.tensor([[[3.0, 4.0]], [[6.0, 8.0]]], device=device).expand_as(x)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    # These four rows are the manually evaluated bank/sample results.
    y_rows = torch.tensor([[8.5, 11.75], [27.75, 31.75], [47.0, 51.75], [66.25, 71.75]])
    g_rows = torch.tensor([[20.0, 21.0], [40.0, 41.0], [60.0, 61.0], [80.0, 81.0]])
    dx_rows = torch.tensor(
        [[7 / 64, -7 / 64], [21 / 64, -21 / 64], [7 / 32, -7 / 32], [35 / 64, -35 / 64]]
    )
    rows = torch.arange(2)[:, None] + index.cpu() * 2
    for actual, expected in ((y, y_rows[rows]), (gate, g_rows[rows]), (dx, dx_rows[rows])):
        assert_bytes(actual, expected)
    counts = {"all0": [8, 8, 0, 0], "all1": [0, 0, 8, 8], "alternating": [4, 4, 4, 4]}[route]
    per_token = torch.tensor(
        [[1.0, 0.0, -0.75, 0.0, 3.0, 4.0], [2.0, 0.0, -1.5, 0.0, 6.0, 8.0]] * 2
    )
    expected_dm = torch.zeros(4, 6)
    for row, count in enumerate(counts):
        if count:
            expected_dm[row] = per_token[row] * count
    assert_bytes(dm, expected_dm)


def check_absent_vs_zero_dy(op, device, explicit_zero):
    x = torch.full((2, 8, 2), float("nan"), device=device, requires_grad=True)
    m = torch.zeros(4, 6, device=device, requires_grad=True)
    index = torch.zeros(2, 8, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    if explicit_zero:
        torch.autograd.backward((y, gate), (torch.zeros_like(y), torch.ones_like(gate)))
        assert torch.isnan(x.grad).all()
        assert torch.isnan(m.grad[:2, 2:4]).all()
        assert_bytes(m.grad[:2, :2], torch.zeros(2, 2))
    else:
        gate.backward(torch.ones_like(gate))
        assert x.grad is None
        assert_bytes(m.grad[:2, :4], torch.zeros(2, 4))
    assert_bytes(m.grad[:2, 4:], torch.full((2, 2), 8.0))
    assert_bytes(m.grad[2:], torch.zeros(2, 6))


def check_cancellation(op, device, order):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1).requires_grad_()
    m = torch.zeros(4, 6, device=device, requires_grad=True)
    index = torch.tensor([[0, 1] * 4] * 2, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    # Selected bank0: FP32 fold (2**26 + 1) - 2**26 = 0.
    # Reordering to cancel the large terms first leaves 1 (scale: +/-3/4).
    middle, last = (1.0, -(2**26)) if order == "lost_unit" else (-(2**26), 1.0)
    u = torch.tensor([2**26, 4.0, middle, 5.0, last, 6.0, 0.0, 7.0], device=device)
    u = u[:, None].expand(8, 2)
    _, dm = torch.autograd.grad(
        (y, gate), (x, m), (torch.stack((u, 2 * u)), torch.stack((u, 3 * u)))
    )
    bank0 = (
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2
        if order == "lost_unit"
        else [[1.0, 1.0, -0.75, 0.75, 1.0, 1.0], [2.0, 2.0, -1.5, 1.5, 3.0, 3.0]]
    )
    assert_bytes(
        dm,
        torch.tensor(
            [*bank0, [22.0, 22.0, -16.5, 16.5, 22.0, 22.0], [44.0, 44.0, -33.0, 33.0, 66.0, 66.0]]
        ),
    )


def check_product_rounding(op, device):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1).requires_grad_()
    m = torch.zeros(4, 6, device=device, requires_grad=True)
    index = torch.zeros(2, 8, dtype=torch.int32, device=device)
    y, _ = op(x, m, modulate_index=index, eps=7.0)
    dy = torch.zeros_like(x)
    dy[:, 0] = -1.0
    dy[:, 1] = 1 + 2**-23
    _, dm = torch.autograd.grad(y, (x, m), dy)
    # norm=+/-3/4 exactly. Product (1+2^-23)*3/4 is a tie:
    # .75 + 3*2^-25 rounds to .75 + 2^-23 (even mantissa).
    # Adding -.75 afterwards gives 2^-23; FMA instead gives 3*2^-25.
    assert_bytes(dm[:2], torch.tensor([[2**-23, 2**-23, -(2**-23), 2**-23, 0.0, 0.0]] * 2))
    assert_bytes(dm[2:], torch.zeros(2, 6))


def check_nonfinite_isolation(op, device, bank, nonfinite):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1)
    x[0, :, 0] = nonfinite
    x.requires_grad_()
    m = torch.zeros(4, 6, device=device, requires_grad=True)
    index = torch.full((2, 8), bank, dtype=torch.int32, device=device)
    dy = torch.tensor([[[nonfinite, 1.0]], [[1.0, 0.0]]], device=device).expand_as(x)
    dg = torch.tensor([[[nonfinite, 1.0]], [[1.0, 2.0]]], device=device).expand_as(x)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    assert torch.isnan(y[0]).all() and torch.isnan(dx[0]).all()
    assert_bytes(y[1], torch.tensor([[-0.75, 0.75]] * 8))
    assert_bytes(dx[1], torch.tensor([[7 / 128, -7 / 128]] * 8))
    bad = dm[bank * 2]
    classify = torch.isnan if nonfinite != nonfinite else torch.isposinf
    assert classify(bad[[0, 4]]).all()
    assert torch.isnan(bad[2:4]).all()
    assert_bytes(bad[[1, 5]], torch.tensor([8.0, 8.0]))
    assert_bytes(dm[bank * 2 + 1], torch.tensor([8.0, 0.0, -6.0, 0.0, 8.0, 16.0]))
    assert_bytes(dm[(1 - bank) * 2 : (1 - bank) * 2 + 2], torch.zeros(2, 6))


def check_leaf_participation(op, device, frozen, outputs):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1)
    x.requires_grad_(frozen != "x")
    m = torch.zeros(4, 6, device=device, requires_grad=frozen != "modulation")
    index = torch.tensor([[0, 1] * 4] * 2, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dy = torch.tensor([1.0, 0.0], device=device).expand_as(x)
    dg = torch.tensor([1.0, 2.0], device=device).expand_as(x)
    if outputs == "y":
        y.backward(dy)
    elif outputs == "gate":
        gate.backward(dg)
    else:
        torch.autograd.backward((y, gate), (dy, dg))
    if frozen == "x" or outputs == "gate":
        assert x.grad is None
    else:
        assert_bytes(x.grad, torch.tensor([7 / 128, -7 / 128]).repeat(2, 8, 1))
    if frozen == "modulation":
        assert m.grad is None
    else:
        expected = {
            "y": [4.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            "gate": [0.0, 0.0, 0.0, 0.0, 4.0, 8.0],
            "both": [4.0, 0.0, -3.0, 0.0, 4.0, 8.0],
        }[outputs]
        assert_bytes(m.grad, torch.tensor([expected] * 4))


def check_gate_payload(op, device):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1).requires_grad_()
    words = torch.tensor(
        [[0, -(2**31)], [1, 1 - 2**31], [0x7FC01234, 0x7F800000], [0x7FA01234, -0x800000]],
        dtype=torch.int32,
    )
    m = torch.zeros(4, 6)
    m[:, 4:] = words.view(torch.float32)
    m = m.to(device).requires_grad_()
    index = torch.tensor([[0, 1] * 4, [1, 0] * 4], dtype=torch.int32, device=device)
    _, gate = op(x, m, modulate_index=index, eps=7.0)
    rows = torch.arange(2)[:, None] + index.cpu() * 2
    assert_bytes(gate, words.view(torch.float32)[rows])
    gate.backward(torch.ones_like(gate))
    assert x.grad is None
    # Arithmetic dgate does not inherit payloads from forward gate values.
    assert_bytes(m.grad[:, :4], torch.zeros(4, 4))
    assert_bytes(m.grad[:, 4:], torch.full((4, 2), 4.0))


def check_tiny_contributions(op, device, contribution):
    x = torch.tensor([-3.0, 3.0], device=device).repeat(2, 8, 1).requires_grad_()
    m = torch.zeros(4, 6, device=device, requires_grad=True)
    index = torch.tensor([[0, 1] * 4] * 2, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    value = -0.0 if contribution == "negative_zero" else 2**-149
    u = torch.full_like(x, value)
    _, dm = torch.autograd.grad((y, gate), (x, m), (u, u))
    # Nonempty banks also start at +0: +0 + -0 = +0, without payload identity.
    words = [0] * 6 if contribution == "negative_zero" else [4, 4, 4 - 2**31, 4, 4, 4]
    assert_bytes(dm, torch.tensor([words] * 4, dtype=torch.int32).view(torch.float32))


def check_shared_bank0(indexed_op, shared_op, device):
    x, m, index, dy, dg = small_case(device)
    index.zero_()
    y, gate = indexed_op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    sx = x.detach().clone().requires_grad_()
    sm = m.detach()[:2].clone().requires_grad_()
    sy, sg = shared_op(sx, sm, eps=7.0)
    sdx, sdm = torch.autograd.grad((sy, sg), (sx, sm), (dy, dg[:, :1]))
    for actual, expected in ((y, sy), (dx, sdx), (dm[:2, :4], sdm[:, :4])):
        assert_bytes(actual, expected)
    assert_bytes(dm[2:], torch.zeros(2, 6))
    # Shared dgate is a passthrough; indexed dgate is a token sum, not compared.
    rx = x.detach().cpu().requires_grad_()
    rm = m.detach().cpu().requires_grad_()
    ry, rg = GtestAdaLNReference().forward_fp32(rx, rm, modulate_index=index.cpu(), eps=7.0)
    rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy.cpu(), dg.cpu()))
    for actual, expected in ((y, ry), (dx, rdx), (dm, rdm)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
