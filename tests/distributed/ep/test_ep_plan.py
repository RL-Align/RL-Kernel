# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Task 35 acceptance on a simulated all-to-all; see check_ep_plan_rccl.py for real RCCL."""

import pytest
import torch

from rl_engine.distributed.ep import (
    WIRE_COLUMNS,
    EpIdentity,
    EpPlacement,
    EpPlanError,
    build_recv_plan,
    build_send_plan,
)
from rl_engine.distributed.ep.plan import WIRE_RUN_TAG, WIRE_SLOT

pytestmark = pytest.mark.unit

E = 256
K = 6
H = 8
IDENTITY = EpIdentity("run", "model-v1", "place-v1")


def owners(ep):
    return [e * ep // E for e in range(E)]


def topk(num_tokens, seed):
    g = torch.Generator().manual_seed(seed)
    if num_tokens == 0:
        return torch.empty((0, K), dtype=torch.int64)
    return torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(num_tokens)])


def all_to_all(sends, send_counts):
    """Deliver rows the way all_to_all_single does: by source, then send position."""
    p = len(sends)
    offsets = [torch.cat((c.new_zeros(1), c.cumsum(0))) for c in send_counts]
    return [
        torch.cat([sends[s][offsets[s][d] : offsets[s][d + 1]] for s in range(p)])
        for d in range(p)
    ]


def split(total, ep):
    """Uneven token split with rank 1 empty when there are several ranks."""
    if ep == 1:
        return [total]
    parts = [0] * ep
    parts[0] = total // 2
    rest = total - parts[0]
    for i in range(2, ep):
        parts[i] = rest // (ep - 2) if ep > 2 else 0
    parts[-1 if ep > 2 else 0] += total - sum(parts)
    return parts


def fake_kernel(rows, experts):
    """Per-row, byte-level stand-in for a batch-invariant expert kernel."""
    return (rows.view(torch.int16) ^ (experts[:, None].to(torch.int16) + 1)).view(rows.dtype)


def run_ep(ep, x, expert_ids):
    """Full round trip on simulated transport; returns rows in global (token, slot) order."""
    placement = EpPlacement.from_owners(owners(ep), ep)
    sizes = split(x.shape[0], ep)
    xs, ids = x.split(sizes), expert_ids.split(sizes)
    sends = [build_send_plan(ids[r], placement, IDENTITY, r) for r in range(ep)]
    counts = [s.counts for s in sends]

    metas = all_to_all([s.wire_metadata() for s in sends], counts)
    inbound = all_to_all([s.pack_token_rows(xs[r]) for r, s in enumerate(sends)], counts)
    recv_counts = torch.stack(counts).T
    recvs = [
        build_recv_plan(metas[r], recv_counts[r], placement, IDENTITY, r, K) for r in range(ep)
    ]

    returned = []
    for r, recv in enumerate(recvs):
        sorted_rows = recv.to_expert_order(inbound[r])
        sorted_experts = recv.to_expert_order(recv.experts)
        returned.append(recv.to_arrival_order(fake_kernel(sorted_rows, sorted_experts)))

    homes = all_to_all(returned, [recv.counts for recv in recvs])
    echoed = all_to_all(metas, [recv.counts for recv in recvs])
    outputs = []
    for r, send in enumerate(sends):
        send.verify_echo(echoed[r])
        outputs.append(send.unpack_return(homes[r]))
    return sends, recvs, outputs


def recvs_local_count(ep, rank):
    return EpPlacement.from_owners(owners(ep), ep).local_counts[rank]


def test_placement_local_ids_are_dense_and_ascending():
    placement = EpPlacement.from_owners([1, 0, 1, 2, 0], 4)
    assert placement.expert_local.tolist() == [0, 0, 1, 0, 1]
    assert placement.local_counts.tolist() == [2, 2, 1, 0]


@pytest.mark.parametrize("bad", [[0, 4], [-1, 0]])
def test_placement_rejects_owner_outside_group(bad):
    with pytest.raises(EpPlanError, match="owner outside"):
        EpPlacement.from_owners(bad, 4)


def test_send_order_is_destination_token_slot():
    placement = EpPlacement.from_owners(owners(4), 4)
    plan = build_send_plan(topk(50, 1), placement, IDENTITY, 0)
    keys = list(zip(plan.destinations.tolist(), plan.tokens.tolist(), plan.slots.tolist()))
    assert keys == sorted(keys)
    assert plan.counts.dtype == plan.offsets.dtype == torch.int64
    assert plan.offsets.tolist() == [0, *plan.counts.cumsum(0).tolist()]


def test_token_expert_slot_stay_paired():
    ids = topk(50, 2)
    plan = build_send_plan(ids, EpPlacement.from_owners(owners(4), 4), IDENTITY, 0)
    assert torch.equal(plan.experts, ids[plan.tokens, plan.slots])


def test_send_order_matches_expert_backward_receipt_rule():
    plan = build_send_plan(topk(20, 3), EpPlacement.from_owners(owners(8), 8), IDENTITY, 0)
    route = plan.saved_route_slots()
    assert [(d, t, s) for t, s, _, d in route] == sorted((d, t, s) for t, s, _, d in route)


def test_pack_routes_token_and_slot_values():
    ids = topk(10, 4)
    plan = build_send_plan(ids, EpPlacement.from_owners(owners(4), 4), IDENTITY, 0)
    x = torch.arange(10, dtype=torch.float32)[:, None]
    weights = torch.arange(60, dtype=torch.float32).reshape(10, K)
    assert torch.equal(plan.pack_token_rows(x)[:, 0], plan.tokens.float())
    assert torch.equal(plan.pack_slot_values(weights), (plan.tokens * K + plan.slots).float())


def test_repeated_expert_in_one_token_keeps_distinct_rows():
    ids = torch.full((2, K), 7, dtype=torch.int64)
    plan = build_send_plan(ids, EpPlacement.from_owners(owners(4), 4), IDENTITY, 0)
    assert plan.num_rows == 2 * K
    assert plan.slots.tolist() == list(range(K)) * 2


def test_invalid_slots_stay_absent_and_p6_rejects_incomplete_top_k():
    ids = topk(4, 5)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, 2] = False
    plan = build_send_plan(ids, EpPlacement.from_owners(owners(4), 4), IDENTITY, 0, valid=valid)
    assert plan.num_rows == 4 * K - 1
    assert (1, 2) not in set(zip(plan.tokens.tolist(), plan.slots.tolist()))

    moe = pytest.importorskip("rl_engine.contracts.operators.moe")
    MoEOutputCombineInputs = moe.MoEOutputCombineInputs
    MoEOutputCombinePlan = moe.MoEOutputCombinePlan
    mapping = plan.unpack_return(torch.zeros(plan.num_rows, H, dtype=torch.bfloat16))
    p6 = MoEOutputCombinePlan(top_k=K, hidden_size=H)
    with pytest.raises(ValueError, match="num_tokens \\* top_k"):
        MoEOutputCombineInputs(
            mapping.rows,
            mapping.token_indices,
            mapping.expert_indices,
            mapping.slot_indices,
            routing_weights=torch.ones(plan.num_rows),
            mapping_run_id=mapping.mapping_run_id,
            num_tokens=4,
        ).validate(p6)


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_inverse_permutations_are_exact(ep):
    x = torch.randint(-(2**15), 2**15, (64, H), dtype=torch.int16).view(torch.bfloat16)
    _, recvs, _ = run_ep(ep, x, topk(64, 6))
    for recv in recvs:
        rows = torch.arange(recv.num_rows)
        assert torch.equal(recv.to_arrival_order(recv.to_expert_order(rows)), rows)
        assert torch.equal(recv.to_expert_order(recv.to_arrival_order(rows)), rows)


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_expert_order_has_a_unique_total_key_and_valid_offsets(ep):
    _, recvs, _ = run_ep(ep, torch.zeros((64, H), dtype=torch.bfloat16), topk(64, 7))
    for r, recv in enumerate(recvs):
        order = recv.to_expert
        keys = list(
            zip(
                recv.local_experts[order].tolist(),
                recv.sources[order].tolist(),
                recv.tokens[order].tolist(),
                recv.slots[order].tolist(),
            )
        )
        assert keys == sorted(keys) and len(set(keys)) == len(keys)
        assert recv.expert_offsets.dtype == torch.int64
        assert recv.expert_offsets[0] == 0 and recv.expert_offsets[-1] == recv.num_rows
        assert recv.expert_offsets.numel() == int(recvs_local_count(ep, r)) + 1
        sorted_local = recv.local_experts[order]
        for e in range(recv.expert_offsets.numel() - 1):
            rows = sorted_local[recv.expert_offsets[e] : recv.expert_offsets[e + 1]]
            assert bool((rows == e).all())


def test_output_is_byte_identical_across_ep_sizes_with_nan_patterns():
    x = torch.randint(-(2**15), 2**15, (97, H), dtype=torch.int16).view(torch.bfloat16)
    ids = topk(97, 8)
    expected = fake_kernel(x.repeat_interleave(K, 0), ids.reshape(-1))
    for ep in (1, 2, 4, 8):
        _, _, outputs = run_ep(ep, x, ids)
        got = torch.cat([o.rows for o in outputs]).view(torch.int16)
        assert torch.equal(got, expected.view(torch.int16)), f"ep={ep}"


def test_output_feeds_the_p6_boundary():
    moe = pytest.importorskip("rl_engine.contracts.operators.moe")
    MoEOutputCombineInputs = moe.MoEOutputCombineInputs
    MoEOutputCombinePlan = moe.MoEOutputCombinePlan
    ids = topk(8, 9)
    x = torch.zeros((8, H), dtype=torch.bfloat16)
    _, _, outputs = run_ep(1, x, ids)
    m = outputs[0]
    MoEOutputCombineInputs(
        m.rows,
        m.token_indices,
        m.expert_indices,
        m.slot_indices,
        routing_weights=torch.ones(8 * K),
        mapping_schema_version=m.mapping_schema_version,
        mapping_run_id=m.mapping_run_id,
        num_tokens=8,
    ).validate(MoEOutputCombinePlan(top_k=K, hidden_size=H))
    assert torch.equal(m.expert_indices, ids.reshape(-1))


def test_all_ranks_empty_is_a_valid_plan():
    sends, recvs, outputs = run_ep(4, torch.zeros((0, H), dtype=torch.bfloat16), topk(0, 0))
    assert all(s.num_rows == 0 for s in sends)
    assert all(r.num_rows == 0 and r.expert_offsets.numel() == 65 for r in recvs)
    assert all(o.rows.shape == (0, H) for o in outputs)


def test_plan_is_reproducible_and_int64_on_large_indices():
    placement = EpPlacement.from_owners(owners(2), 2)
    ids = topk(30, 10)
    a, b = (build_send_plan(ids, placement, IDENTITY, 1) for _ in range(2))
    assert torch.equal(a.wire_metadata(), b.wire_metadata())

    token = 2**40
    meta = torch.tensor([[a.run_tag, 0, token, 5, 0]], dtype=torch.int64)
    recv = build_recv_plan(meta, torch.tensor([1, 0]), placement, IDENTITY, 0, K)
    assert recv.tokens.tolist() == [token]


@pytest.fixture
def received():
    placement = EpPlacement.from_owners(owners(2), 2)
    sends = [build_send_plan(topk(20, 11 + r), placement, IDENTITY, r) for r in range(2)]
    meta = all_to_all([s.wire_metadata() for s in sends], [s.counts for s in sends])[0]
    counts = torch.stack([s.counts for s in sends]).T[0]
    return placement, meta, counts


def test_recv_accepts_clean_metadata(received):
    placement, meta, counts = received
    build_recv_plan(meta, counts, placement, IDENTITY, 0, K)


def test_recv_detects_missing_row(received):
    placement, meta, counts = received
    with pytest.raises(EpPlanError, match="missing or extra"):
        build_recv_plan(meta[:-1], counts, placement, IDENTITY, 0, K)


def test_recv_detects_duplicate_row(received):
    placement, meta, counts = received
    meta = meta.clone()
    meta[1] = meta[0]
    with pytest.raises(EpPlanError, match="duplicate"):
        build_recv_plan(meta, counts, placement, IDENTITY, 0, K)


def test_recv_detects_foreign_row(received):
    placement, meta, counts = received
    meta = meta.clone()
    meta[3, WIRE_RUN_TAG] ^= 1
    with pytest.raises(EpPlanError, match="another run"):
        build_recv_plan(meta, counts, placement, IDENTITY, 0, K)


@pytest.mark.parametrize(
    "other",
    [
        EpIdentity("other", "model-v1", "place-v1"),
        EpIdentity("run", "model-v2", "place-v1"),
        EpIdentity("run", "model-v1", "place-v2"),
    ],
)
def test_recv_detects_identity_mismatch(received, other):
    placement, meta, counts = received
    with pytest.raises(EpPlanError, match="another run"):
        build_recv_plan(meta, counts, placement, other, 0, K)


def test_recv_detects_placement_table_mismatch(received):
    _, meta, counts = received
    shifted = EpPlacement.from_owners([(o + 1) % 2 for o in owners(2)], 2)
    with pytest.raises(EpPlanError, match="another run"):
        build_recv_plan(meta, counts, shifted, IDENTITY, 0, K)


def test_recv_detects_misrouted_row(received):
    placement, meta, counts = received
    with pytest.raises(EpPlanError, match="does not own"):
        build_recv_plan(meta, counts, placement, IDENTITY, 1, K)


def test_recv_detects_wrong_source_position(received):
    placement, meta, counts = received
    with pytest.raises(EpPlanError, match="source disagrees"):
        build_recv_plan(meta, counts.flip(0), placement, IDENTITY, 0, K)


def test_recv_detects_out_of_order_rows(received):
    placement, meta, counts = received
    meta = meta.clone()
    meta[[0, 1]] = meta[[1, 0]]
    with pytest.raises(EpPlanError, match="out of token/slot order"):
        build_recv_plan(meta, counts, placement, IDENTITY, 0, K)


def test_recv_rejects_slot_outside_top_k(received):
    placement, meta, counts = received
    meta = meta.clone()
    meta[0, WIRE_SLOT] = K
    with pytest.raises(EpPlanError, match="outside top_k"):
        build_recv_plan(meta, counts, placement, IDENTITY, 0, K)


def test_recv_rejects_wrong_shape_and_dtype(received):
    placement, meta, counts = received
    with pytest.raises(EpPlanError, match="metadata must be"):
        build_recv_plan(meta.to(torch.int32), counts, placement, IDENTITY, 0, K)
    with pytest.raises(EpPlanError, match="metadata must be"):
        build_recv_plan(meta[:, :WIRE_COLUMNS - 1], counts, placement, IDENTITY, 0, K)


@pytest.fixture
def sent():
    return build_send_plan(topk(20, 13), EpPlacement.from_owners(owners(2), 2), IDENTITY, 0)


def test_echo_accepts_clean(sent):
    sent.verify_echo(sent.wire_metadata())


def test_echo_detects_missing_row_even_when_counts_agree(sent):
    with pytest.raises(EpPlanError, match="missing or extra"):
        sent.verify_echo(sent.wire_metadata()[:-1])


def test_echo_detects_foreign_row(sent):
    echoed = sent.wire_metadata()
    echoed[0, WIRE_RUN_TAG] += 1
    with pytest.raises(EpPlanError, match="another run"):
        sent.verify_echo(echoed)


def test_echo_detects_duplicate_row(sent):
    echoed = sent.wire_metadata()
    echoed[1] = echoed[0]
    with pytest.raises(EpPlanError, match="duplicate"):
        sent.verify_echo(echoed)


def test_echo_detects_reordered_rows(sent):
    echoed = sent.wire_metadata()
    echoed[[0, 1]] = echoed[[1, 0]]
    with pytest.raises(EpPlanError, match="differs"):
        sent.verify_echo(echoed)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"topk_experts": torch.zeros((2, K))}, "integer"),
        ({"topk_experts": torch.zeros(2, dtype=torch.int64)}, "integer"),
        ({"topk_experts": torch.full((2, K), E, dtype=torch.int64)}, "outside the placement"),
        ({"topk_experts": torch.full((2, K), -1, dtype=torch.int64)}, "outside the placement"),
        ({"source_rank": 2}, "source_rank"),
        ({"valid": torch.ones((2, K))}, "bool"),
    ],
)
def test_send_rejects_bad_input(kwargs, match):
    args = {
        "topk_experts": torch.zeros((2, K), dtype=torch.int64),
        "placement": EpPlacement.from_owners(owners(2), 2),
        "identity": IDENTITY,
        "source_rank": 0,
    } | kwargs
    with pytest.raises(EpPlanError, match=match):
        build_send_plan(**args)


def test_identity_rejects_empty_field():
    with pytest.raises(EpPlanError, match="non-empty"):
        EpIdentity("", "m", "p")


def test_pack_and_unpack_reject_wrong_row_count(sent):
    with pytest.raises(EpPlanError, match="one row per token"):
        sent.pack_token_rows(torch.zeros(19, H))
    with pytest.raises(EpPlanError, match="num_tokens, top_k"):
        sent.pack_slot_values(torch.zeros(20, K - 1))
    with pytest.raises(EpPlanError, match="returned rows"):
        sent.unpack_return(torch.zeros(sent.num_rows - 1, H))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_plan_is_identical_on_cpu_and_gpu():
    placement = EpPlacement.from_owners(owners(8), 8)
    ids = topk(512, 14)
    cpu = build_send_plan(ids, placement, IDENTITY, 3)
    gpu = build_send_plan(ids.cuda(), placement, IDENTITY, 3)
    assert torch.equal(cpu.wire_metadata(), gpu.wire_metadata().cpu())
    assert torch.equal(cpu.to_canonical, gpu.to_canonical.cpu())
    meta = cpu.wire_metadata()[: int(cpu.counts[0])]
    counts = torch.zeros(8, dtype=torch.int64)
    counts[3] = meta.shape[0]
    a = build_recv_plan(meta, counts, placement, IDENTITY, 0, K)
    b = build_recv_plan(meta.cuda(), counts.cuda(), placement, IDENTITY, 0, K)
    assert torch.equal(a.to_expert, b.to_expert.cpu())
    assert torch.equal(a.expert_offsets, b.expert_offsets.cpu())
