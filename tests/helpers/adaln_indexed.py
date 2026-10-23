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
    # Lost-unit bank0: (((2**26 + 1) - 2**26) + 1) = 1 in FP32.
    # Adjacent pairs instead give (2**26 + 1) + (-2**26 + 1) = 0.
    middle, last = (1.0, -(2**26)) if order == "lost_unit" else (-(2**26), 1.0)
    tail = 1.0 if order == "lost_unit" else 0.0
    u = torch.tensor([2**26, 4.0, middle, 5.0, last, 6.0, tail, 7.0], device=device)
    u = u[:, None].expand(8, 2)
    _, dm = torch.autograd.grad(
        (y, gate), (x, m), (torch.stack((u, 2 * u)), torch.stack((u, 3 * u)))
    )
    bank0 = [[1.0, 1.0, -0.75, 0.75, 1.0, 1.0], [2.0, 2.0, -1.5, 1.5, 3.0, 3.0]]
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
    for actual, expected, judgment in (
        (y, ry, "forward_accuracy"),
        (dx, rdx, "gradient_accuracy"),
        (dm, rdm, "gradient_accuracy"),
    ):
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="reduction", dtype=actual.dtype
        )
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance.atol, rtol=tolerance.rtol)


def check_bf16_rounding(op, device):
    # norm=(-.75,.75), shift=-.75, scale=2^-8: y[1]=3/1024.
    # Rounding the affine product early instead gives 1/256.
    x = torch.tensor([-3.0, 3.0], dtype=torch.bfloat16, device=device).repeat(1, 2, 1)
    x.requires_grad_()
    m = torch.zeros(2, 6, dtype=torch.bfloat16, device=device)
    m[:, :2] = -0.75
    m[:, 2:4] = 2**-8
    m.requires_grad_()
    index = torch.zeros(1, 2, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    # dy=(1,0) and (2^-8,0): sum is BF16 midpoint 1+2^-8 -> 1 (even).
    dy = torch.tensor([[[1.0, 0.0], [2**-8, 0.0]]], dtype=torch.bfloat16, device=device)
    dg = dy.clone()
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    assert_bytes(y, torch.tensor([[[-1.5, 3 / 1024]] * 2], dtype=torch.bfloat16))
    # dx0=(1+2^-8)*7/128 rounds to 225/4096; early dnorm cast gives 224/4096.
    assert_bytes(
        dx,
        torch.tensor(
            [[[225 / 4096, -225 / 4096], [225 / 1048576, -225 / 1048576]]], dtype=torch.bfloat16
        ),
    )
    assert_bytes(
        dm, torch.tensor([[1.0, 0.0, -0.75390625, 0.0, 1.0, 0.0], [0.0] * 6], dtype=torch.bfloat16)
    )


def check_bf16_gate_payload(op, device):
    # Includes signed zero, subnormals, infinities, quiet/signaling NaN payloads.
    words = torch.tensor(
        [[0, 0x8000], [1, 0x8001], [0x7F81, 0xFFC3], [0x7F80, 0xFF80]], dtype=torch.uint16
    )
    m = torch.zeros(4, 6, dtype=torch.bfloat16)
    m[:, 4:] = words.view(torch.bfloat16)
    m = m.to(device).requires_grad_()
    x = (
        torch.tensor([-3.0, 3.0], dtype=torch.bfloat16, device=device)
        .repeat(2, 2, 1)
        .requires_grad_()
    )
    index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device)
    _, gate = op(x, m, modulate_index=index, eps=7.0)
    expected = words.view(torch.bfloat16)[torch.tensor([[0, 2], [3, 1]])]
    assert_bytes(gate, expected)
    gate.backward(torch.ones_like(gate))
    assert x.grad is None
    assert_bytes(m.grad, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]] * 4, dtype=torch.bfloat16))


def check_domain_routes(op, device, dtype, index_dtype, broadcast):
    x = torch.tensor([-3.0, 3.0], dtype=dtype, device=device).repeat(2, 2, 1).requires_grad_()
    m = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 10.0, 11.0],
            [1.0, 1.0, 0.0, 0.0, 20.0, 21.0],
            [2.0, 2.0, 0.0, 0.0, 30.0, 31.0],
            [3.0, 3.0, 0.0, 0.0, 40.0, 41.0],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor(
        [[0, 1]] if broadcast else [[0, 1], [1, 0]], dtype=index_dtype, device=device
    )
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad(
        (y, gate),
        (x, m),
        (
            torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand_as(x),
            torch.tensor([2.0, 3.0], dtype=dtype, device=device).expand_as(x),
        ),
    )
    rows = torch.tensor([[0, 2], [1, 3]] if broadcast else [[0, 2], [3, 1]])
    assert_bytes(
        y,
        torch.tensor([[-0.75, 0.75], [0.25, 1.75], [1.25, 2.75], [2.25, 3.75]], dtype=dtype)[rows],
    )
    assert_bytes(
        gate,
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]], dtype=dtype)[rows],
    )
    assert_bytes(dx, torch.tensor([7 / 128, -7 / 128], dtype=dtype).expand_as(dx.cpu()))
    assert_bytes(dm, torch.tensor([[1.0, 0.0, -0.75, 0.0, 2.0, 3.0]] * 4, dtype=dtype))


def check_view_edges(op, device, dtype, layout):
    batch, seq = 2, 4
    if layout == "strided":
        source_x = (
            torch.tensor([-3.0, 99.0, 3.0, 99.0], dtype=dtype, device=device)
            .repeat(batch, seq, 1)
            .requires_grad_()
        )
        source_m = torch.zeros(8, 12, dtype=dtype, device=device, requires_grad=True)
        x, m = source_x[..., ::2], source_m[::2, ::2]
        route_storage = torch.tensor(
            [[0, 9, 1, 9, 0, 9, 1, 9]] * batch, dtype=torch.int16, device=device
        )
        index = route_storage[:, ::2]
        dy = torch.tensor([1.0, 99.0, 0.0, 99.0], dtype=dtype, device=device).repeat(batch, seq, 1)[
            ..., ::2
        ]
        dg = torch.tensor([2.0, 99.0, 3.0, 99.0], dtype=dtype, device=device).repeat(batch, seq, 1)[
            ..., ::2
        ]
    else:
        source_x = (
            torch.tensor([-3.0, 3.0], dtype=dtype, device=device).repeat(1, seq, 1).requires_grad_()
        )
        source_m = torch.zeros(1, 6, dtype=dtype, device=device, requires_grad=True)
        x, m = source_x.expand(batch, -1, -1), source_m.expand(2 * batch, -1)
        index = torch.tensor([[0, 1, 0, 1]], dtype=torch.int16, device=device).expand(batch, -1)
        dy = torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand_as(x)
        dg = torch.tensor([2.0, 3.0], dtype=dtype, device=device).expand_as(x)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    torch.autograd.backward((y, gate), (dy, dg))
    assert_bytes(y, torch.tensor([-0.75, 0.75], dtype=dtype).expand(batch, seq, 2))
    assert_bytes(gate, torch.zeros(batch, seq, 2, dtype=dtype))
    if layout == "strided":
        expected_x = torch.tensor([7 / 128, 0.0, -7 / 128, 0.0], dtype=dtype).repeat(batch, seq, 1)
        expected_m = torch.zeros(8, 12, dtype=dtype)
        expected_m[::2, ::2] = torch.tensor([2.0, 0.0, -1.5, 0.0, 4.0, 6.0], dtype=dtype)
    else:
        expected_x = torch.tensor([7 / 64, -7 / 64], dtype=dtype).repeat(1, seq, 1)
        expected_m = torch.tensor([[8.0, 0.0, -6.0, 0.0, 16.0, 24.0]], dtype=dtype)
    assert source_x.is_leaf and source_m.is_leaf
    assert_bytes(source_x.grad, expected_x)
    assert_bytes(source_m.grad, expected_m)


def check_eps_arithmetic(op, device, dtype, eps):
    x = torch.zeros(1, 1, 2, dtype=dtype, device=device, requires_grad=True)
    m = torch.zeros(2, 6, dtype=dtype, device=device, requires_grad=True)
    index = torch.zeros(1, 1, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=eps)
    dx, dm = torch.autograd.grad(
        (y, gate),
        (x, m),
        (
            torch.tensor([[[1.0, 0.0]]], dtype=dtype, device=device),
            torch.tensor([[[1.0, 2.0]]], dtype=dtype, device=device),
        ),
    )
    if eps == 0:
        assert torch.isnan(y).all() and torch.isnan(dx).all()
        assert torch.isnan(dm[0, 2:4]).all()
    else:
        # Decimal sqrt -> FP32 sqrt -> reciprocal independently pins 2^-149.
        value = (
            torch.tensor([0x643504F3], dtype=torch.int32).view(torch.float32)[0]
            if eps == 2**-149
            else 2**62
        )
        expected_dx = torch.stack((torch.as_tensor(value), -torch.as_tensor(value))).to(dtype)
        assert_bytes(dx, expected_dx.reshape(1, 1, 2))
        assert_bytes(y, torch.zeros(1, 1, 2, dtype=dtype))
        assert_bytes(dm[0, 2:4], torch.zeros(2, dtype=dtype))
    assert_bytes(dm[0, [0, 1, 4, 5]], torch.tensor([1.0, 0.0, 1.0, 2.0], dtype=dtype))
    assert_bytes(dm[1], torch.zeros(6, dtype=dtype))


def check_geometry_bytes(op, device, dtype, hidden, *, seq=3, route="mixed"):
    """Compare fixed/gold and equivalent batch layouts for identical logical tokens."""
    generator = torch.Generator().manual_seed(303)
    x = torch.randn(3, seq, hidden, generator=generator).to(dtype)
    m = (torch.randn(6, 3 * hidden, generator=generator) / 4).to(dtype)
    if seq == 3 and route == "mixed":
        index = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.int32)
    else:
        token = torch.arange(seq)
        first = (token >= seq // 2) if route == "segmented" else (token % 3 == 1)
        index = torch.stack((first, ~first, first)).to(torch.int32)
    dy = torch.randn(x.shape, generator=generator).to(dtype)
    dg = torch.randn(x.shape, generator=generator).to(dtype)

    def execute(callable_op, target, xx, mm, ii, yy, gg):
        xx = xx.to(target).detach().requires_grad_()
        mm = mm.to(target).detach().requires_grad_()
        y, gate = callable_op(xx, mm, modulate_index=ii.to(target), eps=7.0)
        dx, dm = torch.autograd.grad((y, gate), (xx, mm), (yy.to(target), gg.to(target)))
        return tuple(t.detach().cpu() for t in (y, gate, dx, dm))

    expected = execute(IndexedAdaLNFixedReference(), "cpu", x, m, index, dy, dg)
    # Gold uses FP32 leaves, so its gradient tolerance check never compares two BF16 leaves.
    gold = execute(
        GtestAdaLNReference().forward_fp32,
        "cpu",
        x.float(),
        m.float(),
        index,
        dy.float(),
        dg.float(),
    )
    actual = execute(op, device, x, m, index, dy, dg)
    for i, (a, e, g) in enumerate(zip(actual, expected, gold, strict=True)):
        assert_bytes(a, e)
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="forward_accuracy" if i < 2 else "gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(a.float(), g, atol=tolerance.atol, rtol=tolerance.rtol)
    # Every logical sample moves to a different batch position; both banks move with it.
    permutation = torch.tensor([2, 0, 1])
    bank_rows = torch.cat((permutation, permutation + 3))
    inverse = torch.argsort(permutation)
    inverse_banks = torch.cat((inverse, inverse + 3))

    def strided(t):
        return torch.stack((t, torch.full_like(t, 99)), dim=-1).to(device)[..., 0]

    moved = execute(
        op,
        device,
        strided(x[permutation]),
        strided(m[bank_rows]),
        strided(index[permutation]),
        strided(dy[permutation]),
        strided(dg[permutation]),
    )
    for i, (a, e) in enumerate(zip(moved, actual, strict=True)):
        assert_bytes(a[inverse_banks if i == 3 else inverse], e)


def check_bf16_ties(op, device):
    x = (
        torch.tensor([-3.0, 3.0], dtype=torch.bfloat16, device=device)
        .repeat(1, 2, 1)
        .requires_grad_()
    )
    # y: .75 + {65/256,67/256} are opposite-parity midpoints.
    m = torch.tensor(
        [[65 / 256, 65 / 256, 0.0, 0.0, 0.0, 0.0], [67 / 256, 67 / 256, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    m.requires_grad_()
    index = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    y, _ = op(x, m, modulate_index=index, eps=7.0)
    assert_bytes(y[..., 1], torch.tensor([[1.0, 65 / 64]], dtype=torch.bfloat16))
    # dx = (1+scale)*7/128: 455/8192 -> 228/4096 (up),
    # 469/8192 -> 234/4096 (down), both ties-to-even.
    with torch.no_grad():
        m[0, 2] = 1 / 64
        m[1, 2] = 3 / 64
    y, _ = op(x, m, modulate_index=index, eps=7.0)
    dx, _ = torch.autograd.grad(
        y, (x, m), torch.tensor([1.0, 0.0], dtype=torch.bfloat16, device=device).expand_as(x)
    )
    assert_bytes(
        dx,
        torch.tensor(
            [[[228 / 4096, -228 / 4096], [234 / 4096, -234 / 4096]]], dtype=torch.bfloat16
        ),
    )


def check_bf16_parameter_fold_final_cast(op, device):
    x = (
        torch.tensor([-3.0, 3.0], dtype=torch.bfloat16, device=device)
        .repeat(1, 3, 1)
        .requires_grad_()
    )
    m = torch.zeros(2, 6, dtype=torch.bfloat16, device=device, requires_grad=True)
    index = torch.zeros(1, 3, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dy = torch.tensor(
        [[[1.0, 1.0], [1 / 512, 1 / 256], [1 / 512, 1 / 256]]], dtype=torch.bfloat16, device=device
    )
    dg = torch.tensor(
        [[[1.0, 1.0], [1 / 256, 1 / 256], [1 / 256, 1 / 256]]], dtype=torch.bfloat16, device=device
    )
    _, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    # dshift[1]/dgate: 1+1/256+1/256 ->129/128 (per-add BF16 gives 1).
    # dscale[0]: -.75-3/2048-3/2048 ->-193/256 (per-add gives -.75).
    assert_bytes(
        dm,
        torch.tensor(
            [[1.0, 129 / 128, -193 / 256, 194 / 256, 129 / 128, 129 / 128], [0.0] * 6],
            dtype=torch.bfloat16,
        ),
    )


def require_triton_gpu():
    if torch.version.hip is not None or not torch.cuda.is_available():
        pytest.skip("NVIDIA Triton GPU required; not an indexed PASS")
    pytest.importorskip("triton")


def check_maximum_batch(op, dtype):
    b = 65535
    x = torch.tensor([-3.0, 3.0], dtype=dtype, device="cuda").repeat(b, 2, 1).requires_grad_()
    m = torch.zeros(2 * b, 6, dtype=dtype, device="cuda")
    tags = (torch.arange(b, device="cuda") % 17).to(dtype)
    m[:b, 4], m[b:, 4] = tags, tags + 32
    m.requires_grad_()
    index = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad(
        (y, gate),
        (x, m),
        (
            torch.tensor([1.0, 0.0], dtype=dtype, device="cuda").expand_as(x),
            torch.tensor([2.0, 3.0], dtype=dtype, device="cuda").expand_as(x),
        ),
    )
    assert_bytes(y, torch.tensor([-0.75, 0.75], dtype=dtype).expand(b, 2, 2))
    assert_bytes(gate[:, 0, 0], tags.cpu())
    assert_bytes(gate[:, 1, 0], (tags + 32).cpu())
    assert_bytes(dx, torch.tensor([7 / 128, -7 / 128], dtype=dtype).expand(b, 2, 2))
    assert_bytes(dm, torch.tensor([1.0, 0.0, -0.75, 0.0, 2.0, 3.0], dtype=dtype).expand(2 * b, 6))


def check_bf16_parameter_ties(op, device):
    # eps=0 gives exact norm=(-1,1), so all dmod slices see opposite-parity ties.
    x = torch.tensor([-1.0, 1.0], dtype=torch.bfloat16, device=device).repeat(1, 4, 1)
    x.requires_grad_()
    m = torch.zeros(2, 6, dtype=x.dtype, device=device, requires_grad=True)
    index = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=0.0)
    upstream = torch.tensor([1.0, 1.0, 1 / 256, 3 / 256], dtype=x.dtype, device=device)
    upstream = upstream[None, :, None].expand_as(x)
    _, dm = torch.autograd.grad((y, gate), (x, m), (upstream, upstream))
    assert_bytes(
        dm,
        torch.tensor(
            [
                [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
                [65 / 64, 65 / 64, -65 / 64, 65 / 64, 65 / 64, 65 / 64],
            ],
            dtype=torch.bfloat16,
        ),
    )


def check_bf16_subnormal_parameter_fold(op, device, seq, outputs):
    x = torch.tensor([-3.0, 3.0], dtype=torch.bfloat16, device=device).repeat(1, seq, 1)
    x.requires_grad_()
    m = torch.zeros(2, 6, dtype=x.dtype, device=device, requires_grad=True)
    index = torch.zeros(1, seq, dtype=torch.int32, device=device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    upstream = torch.tensor([1, 0x8001], dtype=torch.uint16).view(torch.bfloat16)
    upstream = upstream.to(device).reshape(1, 1, 2).expand_as(x)
    if outputs == "gate":
        gate.backward(upstream)
        assert x.grad is None
        words = [0, 0, 0, 0, seq, 0x8000 + seq]
    else:
        torch.autograd.backward((y, gate), (upstream, upstream))
        # -.75 * {1,3} BF16 minimum-subnormal units rounds to {-1,-2} units.
        scale_units = 1 if seq == 1 else 2
        words = [seq, 0x8000 + seq, 0x8000 + scale_units, 0x8000 + scale_units, seq, 0x8000 + seq]
    expected = torch.tensor([words, [0] * 6], dtype=torch.uint16).view(torch.bfloat16)
    assert_bytes(m.grad, expected)
