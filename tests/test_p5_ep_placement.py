# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-7 routed-expert EP placement: etp=1 gate and single-process EP byte-equality."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.contract import tensor_bytes
from rl_engine.moe.parallel.ep_placement import (
    ExpertParallelConfigError,
    ExpertPlacement,
    MockExchangePlan,
    check_expert_parallel_config,
    resolve_expert_tensor_parallel_size,
)

EP_SIZES = (1, 2, 4, 8)


# etp gate


@pytest.mark.parametrize(
    "config",
    [
        {"expert_tensor_parallel_size": 1, "tensor_model_parallel_size": 4},
        {"expert_tensor_parallel_size": None, "tensor_model_parallel_size": 1},
        SimpleNamespace(expert_tensor_parallel_size=1, tensor_model_parallel_size=8),
    ],
)
def test_etp_one_passes(config) -> None:
    check_expert_parallel_config(config)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size=2"),
        # Megatron resolves an unset etp to TP: unset under TP=4 is etp=4.
        ({"expert_tensor_parallel_size": None, "tensor_model_parallel_size": 4}, "=4"),
        ({}, "cannot resolve"),
        ({"expert_tensor_parallel_size": 0}, "positive int"),
        ({"expert_tensor_parallel_size": "1"}, "positive int"),
    ],
)
def test_etp_not_one_fails_closed(config, match) -> None:
    with pytest.raises(ExpertParallelConfigError, match=match):
        check_expert_parallel_config(config)


def test_etp_follows_megatron_default() -> None:
    assert resolve_expert_tensor_parallel_size({"tensor_model_parallel_size": 2}) == 2


def test_placement_rejects_etp_injection() -> None:
    with pytest.raises(ExpertParallelConfigError):
        ExpertPlacement(n_experts=256, ep_size=8, etp_size=2)


# placement


def test_placement_dsv4_shape() -> None:
    p = ExpertPlacement(n_experts=256, ep_size=8)
    assert p.experts_per_rank == 32
    assert p.rank_of(37) == 1 and p.local_index(37) == 5
    assert p.experts_of(7) == range(224, 256)
    owners = [p.rank_of(e) for e in range(256)]
    assert all(owners.count(r) == 32 for r in range(8))


def test_placement_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ExpertPlacement(n_experts=16, ep_size=3)
    with pytest.raises(ValueError):
        ExpertPlacement(n_experts=16, ep_size=4).rank_of(16)


def test_placement_fingerprint_tracks_geometry() -> None:
    a = ExpertPlacement(16, 4).fingerprint
    assert a == ExpertPlacement(16, 4).fingerprint
    assert a != ExpertPlacement(16, 8).fingerprint


# EP=1/2/4/8 byte-equality against the P5-S0 oracle


def _run_ep(batch, ep: int, dy: torch.Tensor):
    plan = MockExchangePlan.from_batch(batch, ExpertPlacement(batch.expert_offsets.numel() - 1, ep))
    shards = plan.dispatch(batch)
    ys, dxs, dps = [], [], []
    for shard, dy_local in zip(shards, plan.dispatch_rows(dy), strict=True):
        shard.validate()
        y, saved = oracle.routed_expert_forward(shard)
        grads = oracle.routed_expert_backward(shard, saved, dy_local)
        ys.append(y)
        dxs.append(grads["dx"])
        dps.append(grads["dp_s"])
    return (
        plan,
        shards,
        plan.return_rows(shards, ys),
        plan.return_rows(shards, dxs),
        plan.return_rows(shards, dps),
    )


@pytest.mark.parametrize("name", list(fixtures.EP_CASES))
@pytest.mark.parametrize("ep", EP_SIZES)
def test_ep_routed_output_byte_equal_to_oracle(name: str, ep: int) -> None:
    batch = fixtures.make_expert_batch(name)
    dy = fixtures.make_grad_output(name, (batch.rows, batch.hidden))
    y_ref, saved = oracle.routed_expert_forward(batch)
    ref = oracle.routed_expert_backward(batch, saved, dy)

    _, _, y, dx, dp_s = _run_ep(batch, ep, dy)

    assert tensor_bytes(y) == tensor_bytes(y_ref)
    assert tensor_bytes(dx) == tensor_bytes(ref["dx"])
    # dp_s: FP32 [rows] back on the token host, in token-slot order.
    assert dp_s.dtype == torch.float32 and dp_s.shape == (batch.rows,)
    assert tensor_bytes(dp_s) == tensor_bytes(ref["dp_s"])


@pytest.mark.parametrize("ep", EP_SIZES)
def test_lora_shares_the_base_plan_unsplit(ep: int) -> None:
    batch = fixtures.make_expert_batch("ep16_base_plus_lora")
    plan = MockExchangePlan.from_batch(batch, ExpertPlacement(16, ep))
    shards = plan.dispatch(batch)
    # One plan carries base and LoRA rows; A/B reach every rank whole, not EP-sharded.
    assert all(s.lora is batch.lora for s in shards)
    assert {s.lora.fingerprint() for s in shards} == {batch.lora.fingerprint()}
    assert all(s.w1.shape[0] == 16 // ep for s in shards)


def test_empty_rank_still_participates() -> None:
    batch = fixtures.make_expert_batch("ep16_uneven_empty_rank")
    plan = MockExchangePlan.from_batch(batch, ExpertPlacement(16, 8))
    assert plan.recv_counts[1] == 0 and plan.recv_counts[5] == 0
    assert sum(plan.recv_counts) == batch.rows


def test_return_rejects_missing_and_duplicate_slots() -> None:
    batch = fixtures.make_expert_batch("ep16_base_plus_lora")
    plan = MockExchangePlan.from_batch(batch, ExpertPlacement(16, 2))
    shards = plan.dispatch(batch)
    outs = [s.p_s.clone() for s in shards]
    # A row lost in transit: rank 1 returns one row short.
    short = dataclasses.replace(
        shards[1], x=shards[1].x[:-1], output_slot=shards[1].output_slot[:-1]
    )
    with pytest.raises(ValueError, match="missing"):
        plan.return_rows([shards[0], short], [outs[0], outs[1][:-1]])
    dup = [shards[0], shards[0]]
    with pytest.raises(ValueError, match="duplicate"):
        plan.return_rows(dup, [outs[0], outs[0]])
