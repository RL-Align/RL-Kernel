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
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [15.0, 15.0, -11.25, 11.25, 15.0, 15.0],
                [30.0, 30.0, -22.5, 22.5, 45.0, 45.0],
            ]
        ),
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


def test_indexed_bf16_reference_casts_once_at_outputs_and_gradients():
    from tests.helpers.adaln_indexed import check_bf16_rounding

    check_bf16_rounding(IndexedAdaLNFixedReference(), "cpu")


def test_bf16_rounding_literals_reject_specific_early_casts():
    from fractions import Fraction as F

    # Exact rational arithmetic independently checks the discriminating literals.
    product = F(3, 4) * (1 + F(1, 256))
    assert product - F(3, 4) == F(3, 1024)
    assert F(193, 256) - F(3, 4) == F(1, 256)  # early BF16 product
    exact_dx = F(257, 256) * F(7, 128)
    assert F(449, 8192) < exact_dx < F(451, 8192)  # rounds to 225/4096
    assert F(7, 128) != F(225, 4096)  # early BF16 dnorm loses scale
    assert F(1) + F(1, 256) == (F(1) + F(129, 128)) / 2  # dshift/dgate tie -> even 1
    assert -F(3, 4) * (1 + F(1, 256)) == -F(771, 1024)  # final dscale -> -193/256


def test_indexed_bf16_reference_preserves_raw16_gate_payload():
    from tests.helpers.adaln_indexed import check_bf16_gate_payload

    check_bf16_gate_payload(IndexedAdaLNFixedReference(), "cpu")
    # A numeric equality test cannot reject canonicalized NaNs; raw16 can.
    payload = torch.tensor([0x7F81], dtype=torch.uint16).view(torch.bfloat16)
    assert not torch.equal(
        payload.view(torch.uint16), payload.float().bfloat16().view(torch.uint16)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "index_dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("broadcast", [False, True])
def test_indexed_reference_full_route_domain(dtype, index_dtype, broadcast):
    from tests.helpers.adaln_indexed import check_domain_routes

    check_domain_routes(IndexedAdaLNFixedReference(), "cpu", dtype, index_dtype, broadcast)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["strided", "expanded"])
def test_indexed_reference_source_leaf_gradients(dtype, layout):
    from tests.helpers.adaln_indexed import check_view_edges

    check_view_edges(IndexedAdaLNFixedReference(), "cpu", dtype, layout)


@pytest.mark.parametrize("eps", [0.0, 2**-149, 2**-126])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexed_reference_eps_arithmetic(eps, dtype):
    from tests.helpers.adaln_indexed import check_eps_arithmetic

    check_eps_arithmetic(IndexedAdaLNFixedReference(), "cpu", dtype, eps)


def test_subnormal_eps_dx_literal_independent_decimal_sqrt():
    import struct
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 100
        root = struct.unpack("f", struct.pack("f", float((Decimal(2) ** -149).sqrt())))[0]
        reciprocal = struct.unpack("f", struct.pack("f", 1 / root))[0]
    assert struct.unpack("I", struct.pack("f", reciprocal / 2))[0] == 0x643504F3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [1, 3, 127, 128, 129, 4095, 4096])
def test_indexed_reference_geometry_and_batch_layout(dtype, hidden):
    from tests.helpers.adaln_indexed import check_geometry_bytes

    check_geometry_bytes(IndexedAdaLNFixedReference(), "cpu", dtype, hidden)


def test_indexed_reference_bf16_output_and_dx_ties():
    from fractions import Fraction as F
    from tests.helpers.adaln_indexed import check_bf16_ties

    assert F(3, 4) + F(65, 256) == (F(1) + F(129, 128)) / 2
    assert F(3, 4) + F(67, 256) == (F(129, 128) + F(65, 64)) / 2
    assert F(65, 64) * F(7, 128) == (F(227, 4096) + F(228, 4096)) / 2
    assert F(67, 64) * F(7, 128) == (F(234, 4096) + F(235, 4096)) / 2
    check_bf16_ties(IndexedAdaLNFixedReference(), "cpu")


def test_indexed_reference_bf16_parameter_fold_cast_counterexample():
    from fractions import Fraction as F
    from tests.helpers.adaln_indexed import check_bf16_parameter_fold_final_cast

    assert F(1) + F(1, 256) + F(1, 256) == F(129, 128)
    # First per-add BF16 tie rounds to 1, and the second repeats that tie.
    assert F(1) + F(1, 256) == (F(1) + F(129, 128)) / 2
    exact_scale = -F(3, 4) * (1 + F(1, 512) + F(1, 512))
    assert -F(387, 512) < exact_scale < -F(385, 512)  # -> -193/256
    # Each individual tiny contribution is < half an ULP of -.75:
    assert F(3, 2048) < F(1, 512)  # wrong per-add cast remains -.75
    check_bf16_parameter_fold_final_cast(IndexedAdaLNFixedReference(), "cpu")


def test_indexed_reference_bf16_all_parameter_ties():
    from fractions import Fraction as F
    from tests.helpers.adaln_indexed import check_bf16_parameter_ties

    assert F(1) + F(1, 256) == (F(1) + F(129, 128)) / 2
    assert F(1) + F(3, 256) == (F(129, 128) + F(65, 64)) / 2
    check_bf16_parameter_ties(IndexedAdaLNFixedReference(), "cpu")


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("outputs", ["gate", "both"])
def test_indexed_reference_bf16_subnormal_parameter_fold(seq, outputs):
    from tests.helpers.adaln_indexed import check_bf16_subnormal_parameter_fold

    check_bf16_subnormal_parameter_fold(IndexedAdaLNFixedReference(), "cpu", seq, outputs)
