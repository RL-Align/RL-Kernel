# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L1: independent indexed accuracy gold and fixed-byte reference checks."""

import pytest
import torch

from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference
from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference
from tests.helpers.adaln_indexed import assert_bytes, check_small_case


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
