# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L1: independent indexed accuracy gold and fixed-byte reference checks."""

import pytest
import torch

from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference
from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference
from tests.helpers.adaln_indexed import assert_bytes, check_small_case


@pytest.fixture(
    params=[IndexedAdaLNFixedReference(), GtestAdaLNReference().forward_fp32],
    ids=["fixed", "gold"],
)
def reference(request):
    return request.param


def test_indexed_references_match_hand_calculated_nonzero_vjp():
    # Mean=1, variance+eps=16, norm=[-3/4, 3/4]. No H2 zero-dx degeneration.
    for op in (IndexedAdaLNFixedReference(), GtestAdaLNReference().forward_fp32):
        x = torch.tensor([[[-2.0, 4.0], [-2.0, 4.0]]], requires_grad=True)
        m = torch.tensor(
            [[0.5, -0.5, 1.0, 0.0, 2.0, 3.0], [-1.0, 1.0, 0.0, 1.0, 4.0, 5.0]], requires_grad=True
        )
        index = torch.tensor([[0, 1]], dtype=torch.int32)
        y, gate = op(x, m, modulate_index=index, eps=7.0)
        dx, dm = torch.autograd.grad(
            (y, gate),
            (x, m),
            (torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]), torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])),
        )
        assert_bytes(y, torch.tensor([[[-1.0, 0.25], [-1.75, 2.5]]]))
        assert_bytes(gate, torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))
        assert_bytes(dx, torch.tensor([[[7 / 64, -7 / 64], [-7 / 32, 7 / 32]]]))
        assert_bytes(
            dm, torch.tensor([[1.0, 0.0, -0.75, 0.0, 2.0, 3.0], [0.0, 2.0, 0.0, 1.5, 4.0, 5.0]])
        )
        assert torch.count_nonzero(dx) == dx.numel()


def test_indexed_byte_reference_matches_independent_small_fp32_case():
    check_small_case(IndexedAdaLNFixedReference(), "cpu")


@pytest.mark.parametrize(
    "eps,supported", [(0.0, True), (2**-149, True), (2**-150, False), (1e300, False)]
)
def test_shared_fp32_eps_boundaries_preserve_wider_indexed_fallback(eps, supported):
    op = NativeAdaLNModulationOp()
    x = torch.tensor([[[-2.0, 4.0]]])
    if supported:
        y, _ = op(x, torch.zeros(1, 6), eps=eps)
        assert torch.isfinite(y).all()
    else:
        with pytest.raises(ValueError, match="finite FP32 without positive underflow"):
            op(x, torch.zeros(1, 6), eps=eps)
    y, _ = op(x, torch.zeros(2, 6), modulate_index=torch.zeros(1, 1, dtype=torch.int32), eps=eps)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("broadcast", [False, True])
def test_indexed_byte_reference_preserves_selected_token_fold_and_sample_routing(broadcast):
    x = torch.tensor([-3.0, 3.0]).repeat(2, 6, 1).requires_grad_()
    m = torch.zeros(4, 6, requires_grad=True)
    index = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.int32)
    if not broadcast:
        index = index.expand(2, -1).clone()
    y, gate = IndexedAdaLNFixedReference()(x, m, modulate_index=index, eps=7.0)
    # In FP32, 2**26 + 1 - 2**26 is 0 in the declared left-fold order.
    upstream = torch.tensor([2**26, 4.0, 1.0, 5.0, -(2**26), 6.0])[:, None].expand(-1, 2)
    _, dm = torch.autograd.grad(
        (y, gate),
        (x, m),
        (torch.stack((upstream, 2 * upstream)), torch.stack((upstream, 3 * upstream))),
    )
    assert_bytes(
        dm,
        torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [15.0, 15.0, -11.25, 11.25, 15.0, 15.0],
            [30.0, 30.0, -22.5, 22.5, 45.0, 45.0],
        ]),
    )

@pytest.mark.parametrize("route", ["all0", "all1", "alternating"])
def test_indexed_route_identity_literal(reference, route):
    from tests.helpers.adaln_indexed import check_route_identity

    check_route_identity(reference, "cpu", route)


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_indexed_absent_vs_zero_dy_nonfinite_literal(reference, explicit_zero):
    from tests.helpers.adaln_indexed import check_absent_vs_zero_dy

    check_absent_vs_zero_dy(reference, "cpu", explicit_zero)


@pytest.mark.parametrize("order", ["lost_unit", "retained_unit"])
def test_indexed_selected_token_cancellation_literal(reference, order):
    from tests.helpers.adaln_indexed import check_cancellation

    check_cancellation(reference, "cpu", order)


def test_indexed_dscale_product_rounds_before_add_literal(reference):
    from tests.helpers.adaln_indexed import check_product_rounding

    check_product_rounding(reference, "cpu")


@pytest.mark.parametrize("bank", [0, 1])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_indexed_nonfinite_selected_bank_isolation_literal(reference, bank, nonfinite):
    from tests.helpers.adaln_indexed import check_nonfinite_isolation

    check_nonfinite_isolation(reference, "cpu", bank, nonfinite)


@pytest.mark.parametrize(
    "frozen,outputs",
    [
        ("none", "y"),
        ("none", "gate"),
        ("none", "both"),
        ("x", "y"),
        ("x", "gate"),
        ("x", "both"),
        ("modulation", "y"),
        ("modulation", "both"),
    ],
)
def test_indexed_whole_leaf_participation_literal(reference, frozen, outputs):
    from tests.helpers.adaln_indexed import check_leaf_participation

    check_leaf_participation(reference, "cpu", frozen, outputs)


def test_indexed_gate_copies_original_fp32_payload_literal(reference):
    from tests.helpers.adaln_indexed import check_gate_payload

    check_gate_payload(reference, "cpu")


@pytest.mark.parametrize("contribution", ["negative_zero", "subnormal"])
def test_indexed_signed_zero_and_subnormal_fold_literal(reference, contribution):
    from tests.helpers.adaln_indexed import check_tiny_contributions

    check_tiny_contributions(reference, "cpu", contribution)


def test_indexed_all0_matches_shared_arithmetic_and_independent_gold():
    from tests.helpers.adaln_indexed import check_shared_bank0

    check_shared_bank0(IndexedAdaLNFixedReference(), NativeAdaLNModulationOp(), "cpu")
