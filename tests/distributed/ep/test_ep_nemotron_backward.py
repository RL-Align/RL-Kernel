# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Nemotron EP backward on the threaded transport. See check_ep_nemotron.py for real groups."""

import pytest
import torch
import torch.utils.checkpoint as checkpoint

from rl_engine.distributed.ep import EpExchangeError, EpPlanError
from rl_engine.distributed.ep.backward import accumulate_slots
from rl_engine.distributed.ep.nemotron import (
    NUM_EXPERTS,
    TOP_K,
    EpInvocation,
    ep_backward,
    ep_backward_preflight,
)
from rl_engine.reference.moe.nemotron_router import dispatch_tokens, route_scores
from tests.distributed.ep.nemotron_support import (
    H,
    bits,
    contiguous_placement,
    expert_scales,
    fixed_order_combine,
    global_inputs,
    make_exchange,
    rank_slice,
    reference_dx_fp64,
    row_experts,
    run_layer,
    shuffled_placement,
    split_sizes,
    stand_in_expert,
    train_rank,
)
from tests.distributed.ep.test_ep_exchange import run_ranks

pytestmark = pytest.mark.unit

TOTAL = 48
SIZES = (1, 2, 4, 8)


def train_all(ep, *, sizes=None, placement=None, perm=None, weights_fn=None):
    x, ids, weights, target = global_inputs(TOTAL)
    if perm is not None:
        x, ids, weights, target = x[perm], ids[perm], weights[perm], target[perm]
    if weights_fn is not None:
        weights = weights_fn(weights)
    sizes = split_sizes(x.shape[0], ep) if sizes is None else sizes
    placement = contiguous_placement(ep) if placement is None else placement
    scales = expert_scales()

    def body(rank, transport):
        return train_rank(
            make_exchange(placement, transport),
            rank_slice(x, sizes, rank),
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
        )

    results = run_ranks(ep, body)
    for result in results:
        assert isinstance(result, tuple), result
    out = torch.cat([r[0] for r in results])
    dx = torch.cat([r[1] for r in results])
    return out, dx, (x, ids, weights, target, scales)


def test_b1_gradient_bits_do_not_depend_on_ep_size():
    _, baseline, _ = train_all(1)
    assert baseline.dtype == torch.bfloat16
    assert bool((baseline != 0).any())
    for ep in SIZES[1:]:
        _, dx, _ = train_all(ep)
        assert torch.equal(bits(dx), bits(baseline)), ep


@pytest.mark.parametrize("ep", (2, 4, 8))
def test_b1_shuffled_placement_gives_the_same_gradient_bits(ep):
    _, baseline, _ = train_all(1)
    _, dx, _ = train_all(ep, placement=shuffled_placement(ep))
    assert torch.equal(bits(dx), bits(baseline))


@pytest.mark.parametrize("ep", SIZES)
def test_b2_gradient_matches_independent_fp64_closed_form(ep):
    _, dx, (x, ids, weights, target, scales) = train_all(ep)
    expected = reference_dx_fp64(x, ids, weights, target, scales)
    # BF16 stages round the combine, expert and accumulation; the bound reflects that only.
    assert torch.allclose(dx.to(torch.float64), expected, rtol=0.05, atol=0.05)


def test_b2_closed_form_passes_a_finite_difference_check():
    x, ids, weights, target = (
        t.to(torch.float64) if t.is_floating_point() else t for t in global_inputs(1)
    )
    scales = expert_scales().to(torch.float64)

    def loss(value):
        rows = value[:, None] * scales[ids][:, :, None] + value[:, None] ** 2 * 0.25
        return ((weights[:, :, None] * rows).sum(dim=1) * target).sum()

    point = x[:, :4].clone()
    target = target[:, :4]
    direction = torch.ones_like(point)
    step = 1e-6
    numeric = (loss(point + step * direction) - loss(point - step * direction)) / (2 * step)
    analytic = (reference_dx_fp64(point, ids, weights, target, scales) * direction).sum()
    assert numeric.item() == pytest.approx(analytic.item(), rel=1e-6)


def test_b3a_payload_branch_matches_router_style_unpack():
    """Same cotangent into ``x_rows`` and router-packed rows, with no routing-weight branch."""
    x, ids, _, _ = global_inputs(TOTAL)
    grad = torch.randn(TOTAL * TOP_K, H, generator=torch.Generator().manual_seed(5))
    grad = grad.to(torch.bfloat16)
    permutation, _, _ = dispatch_tokens(x, ids, NUM_EXPERTS)

    # The router's _unpack_grad: FP32 from zero, slots 0..5, cast once.
    inverse = torch.argsort(permutation)
    expected = torch.zeros(TOTAL, H)
    for slot in range(TOP_K):
        expected = expected + grad[inverse[torch.arange(TOTAL) * TOP_K + slot]].float()
    expected = expected.to(torch.bfloat16)

    def body(_, transport):
        leaf = x.clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(1), transport)
        call = ep_exchange.dispatch(leaf, ids, EpInvocation(0, 0, 0))
        y = call.x_rows * 1
        rows = call.return_outputs(y)
        # ``grad`` is in packed order; ``rows`` is in (token, slot) order.
        rows.backward(grad[inverse])
        return leaf.grad

    (dx,) = run_ranks(1, body)
    assert isinstance(dx, torch.Tensor), dx
    assert torch.equal(bits(dx), bits(expected))


def test_b3a_signed_zero_accumulation_matches_router_not_pr633():
    rows = torch.full((TOP_K, 2), -0.0, dtype=torch.bfloat16)
    total = accumulate_slots(rows, TOP_K, torch.bfloat16)
    assert not bool(torch.signbit(total.float()).any())


def test_accumulation_order_is_ascending_slot():
    # Exact in FP32 for ordinary rows, so only cancellation exposes the order: ascending
    # loses the 1 against 2^24 and ends at 0; descending would end at 1.
    big = 2.0**24
    rows = torch.tensor([[big], [1.0], [-big], [0.0], [0.0], [0.0]]).to(torch.bfloat16)
    assert accumulate_slots(rows, TOP_K, torch.float32).item() == 0.0


def test_accumulation_keeps_fp32_until_the_single_cast():
    # 1 + 6 * 2^-9 rounds to 1 if each BF16 addition rounds, but not in FP32.
    rows = torch.tensor([[1.0], [2.0**-9], [2.0**-9], [2.0**-9], [2.0**-9], [2.0**-9]])
    total = accumulate_slots(rows.to(torch.bfloat16), TOP_K, torch.bfloat16)
    step = total.float() - 1
    assert step.item() > 0


ROUTER_WEIGHT = torch.randn(NUM_EXPERTS, H, generator=torch.Generator().manual_seed(13)) / 128
ROUTER_BIAS = torch.rand(NUM_EXPERTS, generator=torch.Generator().manual_seed(17)) * 0.05


def route(x):
    """The reference router on ``x``, with weights that carry a gradient back to ``x``.

    Logits are taken one token at a time so that no GEMM blocking depends on the batch,
    which would change bits for reasons that have nothing to do with the exchange.
    """
    rows = [x[t : t + 1].to(torch.float32) @ ROUTER_WEIGHT.T for t in range(x.shape[0])]
    logits = torch.cat(rows) if rows else x.to(torch.float32) @ ROUTER_WEIGHT.T
    return route_scores(logits.sigmoid(), ROUTER_BIAS, TOP_K)


def full_chain_ep(ep):
    """Router, exchange, stand-in expert and combine; per-slot ``dy``, ``dw`` and total ``dx``."""
    x, _, _, target = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, ep)
    scales = expert_scales()

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ids, weights = route(leaf)
        weights.retain_grad()
        ep_exchange = make_exchange(contiguous_placement(ep), transport)
        call, _, loss = run_layer(
            ep_exchange, leaf, ids, weights, rank_slice(target, sizes, rank), scales
        )
        call.rows.retain_grad()
        ep_backward(ep_exchange, loss, [call])
        return call.rows.grad, weights.grad, leaf.grad

    results = run_ranks(ep, body)
    for result in results:
        assert isinstance(result, tuple), result
    return [torch.cat(parts) for parts in zip(*results)]


def full_chain_without_exchange():
    """The router's own packing on one device: autograd sums the payload branch itself."""
    x, _, _, target = global_inputs(TOTAL)
    scales = expert_scales()
    leaf = x.clone().requires_grad_(True)
    ids, weights = route(leaf)
    weights.retain_grad()
    permutation, _, packed = dispatch_tokens(leaf, ids, NUM_EXPERTS)
    y = stand_in_expert(packed, ids.reshape(-1)[permutation], scales)
    rows = y.index_select(0, torch.argsort(permutation))
    rows.retain_grad()
    out = fixed_order_combine(rows, weights)
    (out.to(torch.float32) * target.to(torch.float32)).sum().backward()
    return rows.grad, weights.grad, leaf.grad, ids


def full_chain_fp64(ids, *, routing_branch=True):
    """The same loss in FP64 with the selection held fixed, sharing no code with the above."""
    x, _, _, target = global_inputs(TOTAL)
    f = x.to(torch.float64).requires_grad_(True)
    selected = (f @ ROUTER_WEIGHT.to(torch.float64).T).sigmoid().gather(1, ids)
    w = selected / selected.sum(dim=1, keepdim=True) * 2.5
    if not routing_branch:
        w = w.detach()
    a = expert_scales().to(torch.float64)[ids][:, :, None]
    rows = f[:, None] * a + f[:, None] * f[:, None] * 0.25
    ((w[:, :, None] * rows).sum(dim=1) * target.to(torch.float64)).sum().backward()
    return f.grad


@pytest.mark.parametrize("ep", SIZES)
def test_b3b_full_chain_with_router_weights_that_depend_on_x(ep):
    """Total ``x.grad`` is the payload branch through the exchange plus the routing branch."""
    dy, dw, dx = full_chain_ep(ep)
    ref_dy, ref_dw, ref_dx, ids = full_chain_without_exchange()
    # Same rows and the same combine, so the cotangents into the exchange are the same bits.
    assert torch.equal(bits(dy), bits(ref_dy))
    assert torch.equal(dw, ref_dw)
    if ep > 1:
        assert all(torch.equal(bits(a), bits(b)) for a, b in zip((dy, dw, dx), full_chain_ep(1)))

    # The reference sums the payload branch inside index_select's backward, whose rounding
    # points are not specified, so total dx is compared within BF16 rounding (on CPU the two
    # currently agree bit for bit; that is not asserted). Against FP64 the same tolerance
    # must hold, and must fail for a gradient without the routing branch, so the check
    # would notice that branch going missing.
    torch.testing.assert_close(dx.float(), ref_dx.float(), rtol=2**-7, atol=2**-6)
    exact = full_chain_fp64(ids)
    payload_only = full_chain_fp64(ids, routing_branch=False)
    tolerance = 2**-6 + 2**-7 * exact.abs()
    assert bool(((dx.to(torch.float64) - exact).abs() <= tolerance).all())
    assert ((payload_only - exact).abs() / tolerance).max().item() > 8


def routing_weight_grads(ep):
    x, ids, weights, target = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, ep)
    scales = expert_scales()

    def body(rank, transport):
        leaf_w = rank_slice(weights, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(ep), transport)
        call, _, loss = run_layer(
            ep_exchange,
            rank_slice(x, sizes, rank).clone().requires_grad_(True),
            rank_slice(ids, sizes, rank),
            leaf_w,
            rank_slice(target, sizes, rank),
            scales,
        )
        ep_backward(ep_exchange, loss, [call])
        return leaf_w.grad

    return torch.cat(run_ranks(ep, body)), (x, ids, target, scales)


@pytest.mark.parametrize("ep", SIZES[1:])
def test_b3c_leaf_routing_weight_gradient_bits_do_not_depend_on_ep_size(ep):
    """Each slot's weight gradient comes from its own returned row, so EP must not move it."""
    grad_w, (x, ids, target, scales) = routing_weight_grads(ep)
    reference, _ = routing_weight_grads(1)
    assert torch.equal(bits(grad_w), bits(reference))

    rows = stand_in_expert(x.repeat_interleave(TOP_K, 0), ids.reshape(-1), scales)
    rows = rows.view(-1, TOP_K, H).to(torch.float64)
    expected = (rows * target.to(torch.float64)[:, None]).sum(dim=2)
    torch.testing.assert_close(grad_w.to(torch.float64), expected, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("ep", (2, 8))
def test_b4_gradient_bits_do_not_depend_on_batch_or_position(ep):
    _, baseline, _ = train_all(ep)
    perm = torch.randperm(TOTAL, generator=torch.Generator().manual_seed(3))
    _, moved, _ = train_all(ep, perm=perm)
    assert torch.equal(bits(moved), bits(baseline)[perm])
    subset = torch.arange(TOTAL // 2)
    _, small, _ = train_all(ep, perm=subset)
    assert torch.equal(bits(small), bits(baseline)[subset])


@pytest.mark.parametrize("sizes", ([48, 0, 0, 0], [0, 0, 0, 48], [0, 48, 0, 0], [47, 1, 0, 0]))
def test_b5_empty_ranks_complete(sizes):
    _, baseline, _ = train_all(1)
    _, dx, _ = train_all(4, sizes=sizes)
    assert torch.equal(bits(dx), bits(baseline))


def test_b5_rank_receiving_no_rows_completes():
    placement = type(contiguous_placement(4)).from_owners([0] * NUM_EXPERTS, 4)
    _, baseline, _ = train_all(1)
    _, dx, _ = train_all(4, placement=placement)
    assert torch.equal(bits(dx), bits(baseline))


def test_b5_frozen_input_rank_still_serves_peers():
    """Rank 0 trains on frozen ``x``; its expert must still return ``dX`` to rank 1."""
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        local = rank_slice(x, sizes, rank).clone().requires_grad_(rank == 1)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call, _, loss = run_layer(
            ep_exchange,
            local,
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
        )
        ep_backward(ep_exchange, loss, [call])
        return local.grad

    frozen, live = run_ranks(2, body)
    assert frozen is None
    assert isinstance(live, torch.Tensor), live
    _, baseline, _ = train_all(2, sizes=sizes)
    assert torch.equal(bits(live), bits(rank_slice(baseline, sizes, 1)))


def test_b5_frozen_expert_output_still_returns_zero_gradients():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call = ep_exchange.dispatch(leaf, rank_slice(ids, sizes, rank), EpInvocation(0, 0, 0))
        rows = call.return_outputs(call.x_rows.detach())
        ep_backward(ep_exchange, rows.float().sum(), [call])
        return leaf.grad

    results = run_ranks(2, body)
    for result in results:
        assert isinstance(result, torch.Tensor), result
        assert not bool(result.any())


def test_b5_output_left_out_of_the_loss_is_rejected_on_every_rank():
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    sizes = split_sizes(TOTAL, 4, empty_rank=None)
    payload_moves = []

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(4), transport)
        call, _, loss = run_layer(
            ep_exchange,
            leaf,
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
        )
        if rank == 2:
            loss = (torch.ones((), requires_grad=True) * 2).sum()
        moves = transport.fabric.moves
        try:
            ep_backward(ep_exchange, loss, [call])
        finally:
            payload_moves.append(transport.fabric.moves - moves)
        return leaf.grad

    results = run_ranks(4, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "does not reach the loss" in str(results[2])
    assert "ranks [2] failed" in str(results[0])
    assert set(payload_moves) == {0}


def test_b5_loss_without_graph_is_rejected_on_every_rank():
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call, _, loss = run_layer(
            ep_exchange,
            leaf,
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
        )
        ep_backward(ep_exchange, torch.zeros(()) if rank == 0 else loss, [call])

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results


def test_b5_call_without_returned_outputs_is_rejected():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call = ep_exchange.dispatch(leaf, rank_slice(ids, sizes, rank), EpInvocation(0, 0, 0))
        ep_backward_preflight(ep_exchange, leaf.float().sum(), [call])

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results


def _two_layers(rank, transport, ep, sizes):
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    ep_exchange = make_exchange(contiguous_placement(ep), transport)
    made = []
    for layer in range(2):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        call, _, loss = run_layer(
            ep_exchange,
            leaf,
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
            EpInvocation(0, 0, layer),
        )
        made.append((call, loss, leaf))
    return ep_exchange, made


def test_b6_replayed_backward_is_rejected_on_every_rank():
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call, _, loss = run_layer(
            ep_exchange,
            leaf,
            rank_slice(ids, sizes, rank),
            rank_slice(weights, sizes, rank),
            rank_slice(target, sizes, rank),
            scales,
        )
        loss.backward(retain_graph=True)
        loss.backward()

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "out of order" in str(results[0])


def test_b6_gradient_with_wrong_shape_fails_every_rank():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        ep_exchange = make_exchange(contiguous_placement(2), transport)
        call = ep_exchange.dispatch(leaf, rank_slice(ids, sizes, rank), EpInvocation(0, 0, 0))
        rows = call.return_outputs(call.x_rows)
        saved = call.holder["saved"]
        dy = torch.zeros_like(rows)
        if rank == 1:
            dy = dy[:-1]
        return ep_exchange.exchange.dispatch_dy(saved, dy)

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results


def test_b6_non_strict_exchange_cannot_train():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        ep_exchange = make_exchange(contiguous_placement(2), transport, strict=False)
        return ep_exchange.dispatch(
            rank_slice(x, sizes, rank),
            rank_slice(ids, sizes, rank),
            EpInvocation(0, 0, 0),
        )

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "strict" in str(results[0])


def test_b7_reversed_layer_order_is_rejected_by_preflight_before_payload():
    sizes = split_sizes(TOTAL, 2, empty_rank=None)
    payload_moves = []

    def body(rank, transport):
        ep_exchange, made = _two_layers(rank, transport, 2, sizes)
        first, second = (made[0], made[1]) if rank == 0 else (made[1], made[0])
        moves = transport.fabric.moves
        try:
            ep_backward(ep_exchange, first[1], [first[0]])
            ep_backward(ep_exchange, second[1], [second[0]])
        finally:
            payload_moves.append(transport.fabric.moves - moves)

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "disagree" in str(results[0])
    assert set(payload_moves) == {0}


def test_b7_reversed_layer_order_without_preflight_is_rejected_before_payload():
    """Even a caller that skips ``ep_backward`` cannot exchange the wrong layer's rows."""
    sizes = split_sizes(TOTAL, 2, empty_rank=None)
    payload_moves = []

    def body(rank, transport):
        _, made = _two_layers(rank, transport, 2, sizes)
        first, second = (made[0], made[1]) if rank == 0 else (made[1], made[0])
        moves = transport.fabric.moves
        try:
            first[1].backward()
            second[1].backward()
        finally:
            payload_moves.append(transport.fabric.moves - moves)

    results = run_ranks(2, body)
    assert all(isinstance(result, Exception) for result in results), results
    assert "disagree" in str(results[0]) and "disagree" in str(results[1])
    assert set(payload_moves) == {0}


@pytest.mark.parametrize("ep", (1, 2, 4))
def test_b7_two_layers_in_the_same_order_complete_with_the_same_bits(ep):
    sizes = split_sizes(TOTAL, ep, empty_rank=None)
    _, baseline, _ = train_all(ep, sizes=sizes)

    def body(rank, transport):
        ep_exchange, made = _two_layers(rank, transport, ep, sizes)
        for call, loss, _ in made:
            ep_backward(ep_exchange, loss, [call])
        return torch.stack([made[0][2].grad, made[1][2].grad])

    results = run_ranks(ep, body)
    for rank, result in enumerate(results):
        assert isinstance(result, torch.Tensor), result
        expected = rank_slice(baseline, sizes, rank)
        assert torch.equal(bits(result[0]), bits(expected))
        assert torch.equal(bits(result[1]), bits(expected))


def test_stacked_layers_in_one_graph_run_nodes_in_a_matching_order():
    """Layer 1 consumes layer 0's output; one backward serves both on every rank."""
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()
    sizes = split_sizes(TOTAL, 4)

    def body(rank, transport):
        leaf = rank_slice(x, sizes, rank).clone().requires_grad_(True)
        local_ids = rank_slice(ids, sizes, rank)
        local_w = rank_slice(weights, sizes, rank)
        ep_exchange = make_exchange(contiguous_placement(4), transport)
        hidden, calls = leaf, []
        for layer in range(3):
            call = ep_exchange.dispatch(hidden, local_ids, EpInvocation(0, 0, layer))
            y_rows = stand_in_expert(call.x_rows, row_experts(call), scales)
            hidden = fixed_order_combine(call.return_outputs(y_rows), local_w)
            calls.append(call)
        ep_backward(
            ep_exchange,
            (hidden.float() * rank_slice(target, sizes, rank).float()).sum(),
            calls,
        )
        return leaf.grad

    results = run_ranks(4, body)
    for result in results:
        assert isinstance(result, torch.Tensor), result


def test_reentrant_checkpoint_recompute_is_rejected():
    x, ids, weights, target = global_inputs(TOTAL)
    scales = expert_scales()

    def body(_, transport):
        ep_exchange = make_exchange(contiguous_placement(1), transport)
        leaf = x.clone().requires_grad_(True)

        def segment(value):
            _, out, _ = run_layer(ep_exchange, value, ids, weights, target, scales)
            return out

        out = checkpoint.checkpoint(segment, leaf, use_reentrant=True)
        (out.float() * target.float()).sum().backward()

    (result,) = run_ranks(1, body)
    assert isinstance(result, Exception)
    assert "inside backward" in str(result)


def test_save_route_refuses_a_route_that_has_not_returned():
    x, ids, *_ = global_inputs(TOTAL)

    def body(_, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(1), transport)
            call = ep_exchange.dispatch(x, ids, EpInvocation(0, 0, 0))
        with pytest.raises(EpPlanError, match="returned route"):
            ep_exchange.exchange.save_route(call.route)
        return True

    assert run_ranks(1, body) == [True]
