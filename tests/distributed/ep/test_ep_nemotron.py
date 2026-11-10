# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Nemotron EP forward on the threaded transport. See check_ep_nemotron.py for real groups."""

import pytest
import torch

from rl_engine.distributed.ep import EpExchangeError, EpPlanError
from rl_engine.distributed.ep.cuda_transport import CudaNcclAllToAll
from rl_engine.distributed.ep.nemotron import (
    MAX_TOKENS,
    NUM_EXPERTS,
    TOP_K,
    EpInvocation,
    NemotronEpExchange,
)
from rl_engine.reference.moe.nemotron_router import dispatch_tokens
from tests.distributed.ep.nemotron_support import (
    H,
    bits,
    contiguous_placement,
    expert_scales,
    fixed_order_combine,
    global_inputs,
    make_exchange,
    rank_slice,
    reference_forward,
    row_experts,
    run_layer,
    shuffled_placement,
    split_sizes,
    stand_in_expert,
)
from tests.distributed.ep.test_ep_exchange import run_ranks

pytestmark = pytest.mark.unit

TOTAL = 48
SIZES = (1, 2, 4, 8)


def forward_all(ep, total=TOTAL, sizes=None, placement=None, seed=0, perm=None):
    x, ids, weights, target = global_inputs(total, seed)
    if perm is not None:
        x, ids, weights, target = x[perm], ids[perm], weights[perm], target[perm]
    sizes = split_sizes(x.shape[0], ep) if sizes is None else sizes
    placement = contiguous_placement(ep) if placement is None else placement
    scales = expert_scales()

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(placement, transport)
            _, out, _ = run_layer(
                ep_exchange,
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                rank_slice(weights, sizes, rank),
                rank_slice(target, sizes, rank),
                scales,
            )
        return out

    results = run_ranks(ep, body)
    for result in results:
        assert isinstance(result, torch.Tensor), result
    return torch.cat(results), (x, ids, weights, scales)


@pytest.mark.parametrize("ep", SIZES)
def test_f1_output_matches_reference_without_exchange(ep):
    out, (x, ids, weights, scales) = forward_all(ep)
    assert torch.equal(bits(out), bits(reference_forward(x, ids, weights, scales)))


def test_f1_output_bits_do_not_depend_on_ep_size():
    baseline, _ = forward_all(1)
    for ep in SIZES[1:]:
        assert torch.equal(bits(forward_all(ep)[0]), bits(baseline))


@pytest.mark.parametrize("ep", (2, 8))
def test_f2_row_bits_do_not_depend_on_batch_or_position(ep):
    baseline, _ = forward_all(ep)
    perm = torch.randperm(TOTAL, generator=torch.Generator().manual_seed(3))
    moved, _ = forward_all(ep, perm=perm)
    assert torch.equal(bits(moved), bits(baseline)[perm])
    subset = torch.arange(TOTAL // 2)
    small, _ = forward_all(ep, perm=subset)
    assert torch.equal(bits(small), bits(baseline)[subset])


@pytest.mark.parametrize("ep", SIZES)
def test_f3_round_trip_preserves_every_bit_pattern(ep):
    patterns = torch.tensor(
        [0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x8000, 0x0000, 0x0001, 0x8001],
        dtype=torch.uint16,
    ).view(torch.int16)
    x_bits = patterns[torch.arange(16 * H) % patterns.numel()].reshape(16, H)
    x = x_bits.view(torch.bfloat16)
    ids = global_inputs(16)[1]
    sizes = split_sizes(16, ep)

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(ep), transport)
            call = ep_exchange.dispatch(
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )
            return call.return_outputs(call.x_rows)

    returned = torch.cat(run_ranks(ep, body)).view(-1, TOP_K, H)
    expected = x_bits[:, None].expand(16, TOP_K, H)
    assert torch.equal(returned.view(torch.int16), expected)


@pytest.mark.parametrize(
    "sizes",
    ([48, 0, 0, 0], [0, 0, 0, 48], [0, 48, 0, 0], [12, 12, 12, 12], [47, 1, 0, 0]),
)
def test_f4_empty_and_skewed_ranks(sizes):
    out, (x, ids, weights, scales) = forward_all(4, sizes=sizes)
    assert torch.equal(bits(out), bits(reference_forward(x, ids, weights, scales)))


@pytest.mark.parametrize("ep", (2, 4, 8))
def test_f4_shuffled_placement(ep):
    out, (x, ids, weights, scales) = forward_all(ep, placement=shuffled_placement(ep))
    assert torch.equal(bits(out), bits(reference_forward(x, ids, weights, scales)))


def test_f4_all_rows_to_one_owner():
    owners = [0] * NUM_EXPERTS
    placement = type(contiguous_placement(4)).from_owners(owners, 4)
    out, (x, ids, weights, scales) = forward_all(4, placement=placement)
    assert torch.equal(bits(out), bits(reference_forward(x, ids, weights, scales)))


def test_f5_ep1_rows_equal_router_reference_packing():
    x, ids, _, _ = global_inputs(TOTAL)

    def body(_, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(1), transport)
            return ep_exchange.dispatch(x, ids, EpInvocation(0, 0, 0)).x_rows

    (rows,) = run_ranks(1, body)
    _, _, packed = dispatch_tokens(x, ids, NUM_EXPERTS)
    assert torch.equal(bits(rows), bits(packed))


def _single_rank_fault(ep, fault, bad_rank=1):
    """``fault`` corrupts the inputs of one rank; every rank must raise and name the cause."""
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, ep, empty_rank=None)

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(ep), transport)
            args = [
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            ]
            if rank == bad_rank:
                args = fault(*args)
            return ep_exchange.dispatch(*args)

    return run_ranks(ep, body)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (lambda x, i, n: (x.float(), i, n), "BF16"),
        (lambda x, i, n: (x[:, :-1], i, n), "shape"),
        (lambda x, i, n: (x, i.to(torch.int32), n), "int64"),
        (lambda x, i, n: (x, i[:, :-1], n), "shape"),
        (lambda x, i, n: (x, i + NUM_EXPERTS, n), "outside"),
        (lambda x, i, n: (x, i.flip(1), n), "ascending"),
        (lambda x, i, n: (x, torch.zeros_like(i), n), "ascending"),
    ],
)
def test_f7_one_rank_with_bad_input_fails_every_rank(fault, message):
    results = _single_rank_fault(4, fault)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert message in str(results[1])
    assert "ranks [1] failed" in str(results[0])


def test_f7_token_limit_is_enforced_before_any_payload():
    def fault(x, i, n):
        return (
            torch.empty(MAX_TOKENS + 1, H, dtype=torch.bfloat16),
            torch.zeros(MAX_TOKENS + 1, TOP_K, dtype=torch.int64),
            n,
        )

    results = _single_rank_fault(2, fault)
    assert all(isinstance(result, EpExchangeError) for result in results)
    assert "at most" in str(results[1])


def test_f7_disagreeing_invocation_is_rejected():
    results = _single_rank_fault(4, lambda x, i, n: (x, i, EpInvocation(0, 0, 1)))
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "disagree" in str(results[0])


def test_f7_disagreeing_placement_is_rejected():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 4, empty_rank=None)

    def body(rank, transport):
        placement = shuffled_placement(4) if rank == 2 else contiguous_placement(4)
        with torch.no_grad():
            return make_exchange(placement, transport).dispatch(
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )

    results = run_ranks(4, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results


def test_f7_one_rank_training_while_peers_do_not_is_rejected():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        with torch.set_grad_enabled(rank == 0):
            return make_exchange(contiguous_placement(2), transport).dispatch(
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results


@pytest.mark.parametrize("strict_rank", (0, 1))
def test_f7_mixed_strict_settings_are_rejected_before_any_payload(strict_rank):
    """Strict adds an echo collective, so ranks that disagree on it must fail together."""
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(
                contiguous_placement(2), transport, strict=rank == strict_rank
            )
            try:
                call = ep_exchange.dispatch(
                    rank_slice(x, sizes, rank),
                    rank_slice(ids, sizes, rank),
                    EpInvocation(0, 0, 0),
                )
                return call.return_outputs(call.x_rows)
            except EpExchangeError as exc:
                return exc, transport.fabric.moves

    results = run_ranks(2, body, timeout=2)
    for result in results:
        assert isinstance(result, tuple), result
        error, moves = result
        assert "disagree" in str(error) and moves == 0, result


def test_f7_reused_invocation_is_rejected_on_every_rank():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(2), transport)
            args = (
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )
            call = ep_exchange.dispatch(*args)
            call.return_outputs(call.x_rows)
            return ep_exchange.dispatch(*args)

    results = run_ranks(2, body)
    assert all(isinstance(result, EpExchangeError) for result in results), results
    assert "already used" in str(results[0])


def test_f7_wrong_expert_output_fails_every_rank():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 2, empty_rank=None)

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(2), transport)
            call = ep_exchange.dispatch(
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )
            y = call.x_rows.float() if rank == 1 else call.x_rows
            return call.return_outputs(y)

    results = run_ranks(2, body)
    assert isinstance(results[0], EpExchangeError), results
    assert "ranks [1] failed" in str(results[0])
    # The failing rank keeps its own error type and message.
    assert isinstance(results[1], EpPlanError), results
    assert "BF16" in str(results[1])


@pytest.mark.parametrize("ep", (2, 4, 8))
def test_f8_weights_are_applied_once_in_slot_order(ep):
    """Six distinct rows per token, so a repeated or dropped weight changes the bits."""
    x, ids, weights, target = global_inputs(TOTAL)
    weights = (weights * 4 + 1).to(torch.float32)
    sizes = split_sizes(TOTAL, ep)
    scales = expert_scales()

    def body(rank, transport):
        with torch.no_grad():
            ep_exchange = make_exchange(contiguous_placement(ep), transport)
            _, out, _ = run_layer(
                ep_exchange,
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                rank_slice(weights, sizes, rank),
                rank_slice(target, sizes, rank),
                scales,
            )
        return out

    out = torch.cat(run_ranks(ep, body))
    f = x.to(torch.float32)
    rows = f[:, None] * scales[ids][:, :, None] + f[:, None] * f[:, None] * 0.25
    rows = rows.to(torch.bfloat16).to(torch.float32)
    total = torch.zeros_like(f)
    for slot in range(TOP_K):
        total = total + weights[:, slot : slot + 1] * rows[:, slot]
    assert torch.equal(bits(out), bits(total.to(torch.bfloat16)))


def test_row_experts_recover_each_rows_owner_and_id():
    x, ids, *_ = global_inputs(TOTAL)
    sizes = split_sizes(TOTAL, 4)
    placement = shuffled_placement(4)

    def body(rank, transport):
        with torch.no_grad():
            call = make_exchange(placement, transport).dispatch(
                rank_slice(x, sizes, rank),
                rank_slice(ids, sizes, rank),
                EpInvocation(0, 0, 0),
            )
            experts = row_experts(call)
            recv = call.route.recv
            assert torch.equal(experts, recv.experts[recv.to_expert])
            assert bool((placement.expert_owner[experts] == rank).all())
            return call.provenance()

    results = run_ranks(4, body)
    assert all(isinstance(result, dict) for result in results), results
    assert results[2]["rank"] == 2 and results[2]["ep_size"] == 4
    assert results[2]["fallback"] is False and results[2]["payload_dtype"] == "bf16"


def test_construction_rejects_unsupported_geometry():
    with pytest.raises(EpPlanError, match="128"):
        NemotronEpExchange(
            type(contiguous_placement(1)).from_owners([0] * 256, 1),
            model_version="m",
            placement_version="p",
            transport=object(),
        )
    with pytest.raises(EpPlanError, match="EP size"):
        NemotronEpExchange(
            type(contiguous_placement(1)).from_owners([e % 3 for e in range(128)], 3),
            model_version="m",
            placement_version="p",
            transport=object(),
        )


def test_invocation_rejects_non_integers():
    with pytest.raises(EpPlanError, match="non-negative"):
        EpInvocation(0, 0, -1)
    with pytest.raises(EpPlanError, match="non-negative"):
        EpInvocation(0, 0, True)


def test_cuda_transport_refuses_without_a_process_group():
    with pytest.raises(EpExchangeError, match="process group"):
        CudaNcclAllToAll(None)


def test_stand_ins_do_not_depend_on_batching():
    x, ids, weights, _ = global_inputs(8)
    scales = expert_scales()
    row = x.repeat_interleave(TOP_K, dim=0)
    experts = ids.reshape(-1)
    together = stand_in_expert(row, experts, scales)
    alone = torch.cat(
        [stand_in_expert(row[i : i + 1], experts[i : i + 1], scales) for i in range(row.shape[0])]
    )
    assert torch.equal(bits(together), bits(alone))
    combined = fixed_order_combine(together, weights)
    assert combined.shape == (8, H)
