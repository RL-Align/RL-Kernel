# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2): cross-rank gradient evidence on a 2-process gloo double of Megatron DDP.

The double mirrors ``param_and_grad_buffer.py`` (Megatron-LM 1dcf0da) as far as
the observer depends on it: contiguous buckets with ``offset``/``params_list``,
gradient scaling before the collective, one reduce-scatter per bucket through
the module-level ``dist_reduce_scatter_func`` with the distributed optimizer or
``torch.distributed.all_reduce`` otherwise, and the distributed optimizer's
``gbuf_ranges`` (``distrib_optimizer.py``).
"""

import copy
import json
import os
import socket
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from p7_doubles.megatron_ddp import FakeMegatronDDP, TinyModel

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
from rl_engine.alignment.cross_config.adapters.megatron_training_gradient_ranks import (
    GradientRankReadback,
    gather_gradient_readbacks,
    validate_gradient_rank_set,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_ranks import (
    ExpectedRankTopology,
    observe_rank_coordinates,
)
from rl_engine.integrations.megatron_training_gradients import (
    GradientStepObserver,
    observe_optimizer_ownership,
)

pytestmark = pytest.mark.unit

BUFFER_MODULE = "megatron.core.distributed.param_and_grad_buffer"
WORLD = 2


def _emulated_reduce_scatter(output, input, op=None, group=None, async_op=False):
    """gloo has no reduce-scatter on tensors; same signature, same result."""

    reduced = input.clone()
    # The unwrapped function: this is part of the emulated collective, not another sync call.
    dist.distributed_c10d.all_reduce(reduced, group=group)
    size = dist.get_world_size(group=group)
    shard = input.numel() // size
    rank = dist.get_rank(group=group)
    output.copy_(reduced[rank * shard : (rank + 1) * shard])
    return None


class DistributedFakeDDP(FakeMegatronDDP):
    """FakeMegatronDDP plus Megatron's bucket view and gradient sync on a DP group."""

    def __init__(self, module, *, distributed_optimizer, group, bucket_numel=24):
        super().__init__(module)
        self.ddp_config = SimpleNamespace(
            use_distributed_optimizer=distributed_optimizer, average_in_collective=False
        )
        self.group = group
        size = dist.get_world_size(group=group)
        buffer = self.buffers[0]
        buckets, start, params = [], 0, []
        ordered = sorted(buffer.param_index_map.items(), key=lambda item: item[1][0])
        for index, (param, (_low, high, _)) in enumerate(ordered):
            params.append(param)
            last = index == len(ordered) - 1
            if high - start >= bucket_numel or last:
                padded = -(-(high - start) // size) * size
                buckets.append((start, high, padded, params))
                start, params = high, []
        total = sum(padded for _, _, padded, _ in buckets)
        grad_data = torch.zeros(total)
        remap, objects, offset = {}, [], 0
        for bucket_id, (low, high, padded, members) in enumerate(buckets):
            for param in members:
                p_low, p_high, _ = buffer.param_index_map[param]
                remap[param] = (offset + p_low - low, offset + p_high - low, bucket_id)
            objects.append(
                SimpleNamespace(
                    bucket_id=bucket_id,
                    offset=offset,
                    params_list=list(members),
                    grad_data=grad_data[offset : offset + padded],
                    numel_unpadded=high - low,
                    gradient_scaling_factor=1.0 / size,
                )
            )
            offset += padded
        buffer.param_index_map = remap
        buffer.grad_data = grad_data
        buffer.buckets = objects
        buffer.bucket_indices = [(b.offset, b.offset + b.grad_data.numel()) for b in objects]

    def finish_grad_sync(self):
        self.sync_calls += 1
        buffer = self.buffers[0]
        buffer.grad_data.zero_()
        for param, (start, end, _) in buffer.param_index_map.items():
            buffer.grad_data[start:end] = param.main_grad.reshape(-1)
        size, rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        reduce_scatter = sys.modules[BUFFER_MODULE].dist_reduce_scatter_func
        for bucket in buffer.buckets:
            bucket.grad_data *= bucket.gradient_scaling_factor
            if self.ddp_config.use_distributed_optimizer:
                shard = bucket.grad_data.numel() // size
                local = bucket.grad_data[rank * shard : (rank + 1) * shard]
                reduce_scatter(local, bucket.grad_data, op=dist.ReduceOp.SUM, group=self.group)
            else:
                dist.all_reduce(bucket.grad_data, op=dist.ReduceOp.SUM, group=self.group)


class FakeDistributedOptimizer:
    """``gbuf_ranges`` built the way DistributedOptimizer._build_model_gbuf_range does."""

    def __init__(self, ddp, group):
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        buffer = ddp.buffers[0]
        self.buffers = [buffer]
        bucket_ranges = []
        for bucket in buffer.buckets:
            shard = bucket.grad_data.numel() // size
            low, high = bucket.offset + rank * shard, bucket.offset + (rank + 1) * shard
            param_map = {}
            for param, (start, end, bucket_id) in buffer.param_index_map.items():
                if bucket_id != bucket.bucket_id:
                    continue
                own_low, own_high = max(start, low), min(end, high)
                if own_high > own_low:
                    param_map[param] = {
                        "gbuf_world": SimpleNamespace(start=own_low, end=own_high),
                        "param": SimpleNamespace(start=own_low - start, end=own_high - start),
                    }
            bucket_ranges.append({"param_map": param_map})
        self.gbuf_ranges = [{(torch.float32, torch.float32): bucket_ranges}]


class WideModel(TinyModel):
    """TinyModel plus a second trainable matrix so the buffer spans several buckets."""

    def __init__(self):
        super().__init__()
        self.extra = torch.nn.Parameter(torch.full((5, 4), 0.125))

    def forward(self, x):
        return super().forward(x) + (x @ self.extra.t()).sum()


def _worker(rank, port, distributed_optimizer, out_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    sys.modules[BUFFER_MODULE] = types.SimpleNamespace(
        dist_reduce_scatter_func=_emulated_reduce_scatter
    )
    try:
        group = dist.new_group(list(range(WORLD)), backend="gloo")
        torch.manual_seed(0)
        ddp = DistributedFakeDDP(
            WideModel(), distributed_optimizer=distributed_optimizer, group=group
        )
        observer = GradientStepObserver(ddp, observe_sync=True)
        observer.install()
        ddp.zero_grad_buffer()
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / (8 + rank)
        dist.all_reduce(torch.ones(1), group=group)  # unrelated collective in the window
        ddp.module(x).sum().backward()
        ddp.finish_grad_sync()
        evidence = observer.end_step()
        observer.uninstall()
        ownership = (
            observe_optimizer_ownership(FakeDistributedOptimizer(ddp, group), ddp)
            if distributed_optimizer
            else None
        )
        coords = observe_rank_coordinates(tp_group=None, dp_group=group)
        readbacks = gather_gradient_readbacks(
            GradientRankReadback.from_evidence(evidence, coords, ownership), group
        )
        if rank == 0:
            with open(out_path, "w") as handle:
                json.dump([item.to_dict() for item in readbacks], handle)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module", params=[True, False], ids=["distopt", "allreduce"])
def gathered(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("grad_ranks") / "readbacks.json"
    mp.spawn(_worker, args=(_free_port(), request.param, str(out)), nprocs=WORLD, join=True)
    payload = json.loads(out.read_text())
    return request.param, payload


def _readbacks(payload):
    return [GradientRankReadback.from_dict(copy.deepcopy(item)) for item in payload]


TOPOLOGY = ExpectedRankTopology(tp=1, dp=WORLD)


def test_real_gloo_step_passes_and_is_not_certified(gathered):
    distributed, payload = gathered
    report = validate_gradient_rank_set(_readbacks(payload), TOPOLOGY)
    assert report.complete and not report.certified
    assert report.collective == ("reduce_scatter" if distributed else "all_reduce")
    assert [item["field"] for item in report.missing_evidence] == ["gradient.producer_emitted"]
    (group,) = report.groups
    assert group["dp_ranks"] == [0, 1] and group["buckets"] >= 2
    assert set(group["emitted"]) == {"adapter", "head", "extra"}
    for item in payload:
        sync = item["evidence"]["sync"]
        assert sync["average_in_collective"] == {"source": "configured", "value": False}
        assert sync["unmatched_calls"] == 1, "the unrelated all-reduce is not a gradient sync"
        assert len(sync["calls"]) == len(sync["buckets"])
    if distributed:
        assert group["owned_ranges"] >= 3


def _expect(payload, status, mutate=None, *, topology=TOPOLOGY, drop=None):
    items = copy.deepcopy(payload)
    if mutate is not None:
        mutate(items)
    if drop is not None:
        items = [item for index, item in enumerate(items) if index != drop]
    readbacks = [GradientRankReadback.from_dict(item) for item in items]
    with pytest.raises(T05ContractError) as info:
        validate_gradient_rank_set(readbacks, topology)
    assert info.value.status == status, info.value
    return info.value


def test_missing_and_duplicate_ranks(gathered):
    _, payload = gathered
    _expect(payload, "MISSING_RANK", drop=1)
    _expect(payload, "IDENTITY_DRIFT", lambda items: items.append(items[0]))
    _expect(payload, "IDENTITY_DRIFT", topology=ExpectedRankTopology(tp=1, dp=4))


def test_unobserved_sync_is_missing_provenance(gathered):
    _, payload = gathered
    _expect(payload, "MISSING_PROVENANCE", lambda items: items[1]["evidence"].pop("sync"))
    _expect(
        payload, "MISSING_PROVENANCE", lambda items: items[0]["evidence"]["sync"]["calls"].pop()
    )


def test_bucket_synchronised_twice_is_drift(gathered):
    _, payload = gathered

    def twice(items):
        calls = items[0]["evidence"]["sync"]["calls"]
        calls.append(dict(calls[0]))

    _expect(payload, "IDENTITY_DRIFT", twice)


def test_wrong_collective_is_silent_fallback(gathered):
    distributed, payload = gathered

    def swap(items):
        for call in items[1]["evidence"]["sync"]["calls"]:
            call["kind"] = "all_reduce" if distributed else "reduce_scatter"

    _expect(payload, "SILENT_FALLBACK", swap)


@pytest.mark.parametrize("average", [False, True], ids=["sum", "avg"])
@pytest.mark.parametrize("op", ["RedOpType.MAX", "RedOpType.MIN", "RedOpType.PRODUCT"])
def test_wrong_reduction_op_is_silent_fallback(gathered, average, op):
    """Both DDP collectives must reject non-gradient reductions, including MAX."""

    _, payload = gathered

    def wrong_op(items):
        for item in items:
            sync = item["evidence"]["sync"]
            sync["average_in_collective"]["value"] = average
            for call in sync["calls"]:
                call["op"] = "RedOpType.AVG" if average else "RedOpType.SUM"
        items[1]["evidence"]["sync"]["calls"][0]["op"] = op

    assert _expect(payload, "SILENT_FALLBACK", wrong_op).field_name == "sync.calls.op"


@pytest.mark.parametrize("average", [False, True], ids=["sum", "avg"])
def test_sum_and_avg_must_match_ddp_configuration(gathered, average):
    """Accept each configured op and reject swapping SUM and AVG."""

    _, original = gathered
    payload = copy.deepcopy(original)
    for item in payload:
        sync = item["evidence"]["sync"]
        sync["average_in_collective"]["value"] = average
        for call in sync["calls"]:
            call["op"] = "RedOpType.AVG" if average else "RedOpType.SUM"
    assert validate_gradient_rank_set(_readbacks(payload), TOPOLOGY).complete

    def swap(items):
        items[0]["evidence"]["sync"]["calls"][0]["op"] = (
            "RedOpType.SUM" if average else "RedOpType.AVG"
        )

    _expect(payload, "SILENT_FALLBACK", swap)


def test_missing_reduction_configuration_and_op_fail_closed(gathered):
    _, payload = gathered

    def no_config(items):
        for item in items:
            item["evidence"]["sync"].pop("average_in_collective")

    _expect(payload, "MISSING_PROVENANCE", no_config)
    _expect(
        payload,
        "MISSING_PROVENANCE",
        lambda items: items[0]["evidence"]["sync"]["calls"][0].pop("op"),
    )
    _expect(
        payload,
        "IDENTITY_DRIFT",
        lambda items: items[1]["evidence"]["sync"]["average_in_collective"].update(value=True),
    )


def test_wrong_group_and_replica_drift(gathered):
    _, payload = gathered

    def group(items):
        items[1]["evidence"]["sync"]["calls"][0]["group_ranks"] = [1]

    _expect(payload, "IDENTITY_DRIFT", group)

    def layout(items):
        items[1]["evidence"]["sync"]["buckets"][0]["offset"] += 1

    _expect(payload, "IDENTITY_DRIFT", layout)

    def decision(items):
        record = next(r for r in items[1]["evidence"]["records"] if r["name"] == "extra")
        record["wrote"], record["write_path"] = False, "none"

    _expect(payload, "IDENTITY_DRIFT", decision)

    def unknown(items):
        items[0]["evidence"]["unknown"] = ["head"]

    _expect(payload, "MISSING_PROVENANCE", unknown)


def test_shard_offsets_are_checked(gathered):
    distributed, payload = gathered
    if not distributed:
        _expect(
            payload,
            "IDENTITY_DRIFT",
            lambda items: items[0]["evidence"]["sync"]["calls"][0].update(output_numel=1),
        )
        return

    def swapped(items):
        call = items[1]["evidence"]["sync"]["calls"][0]
        call["output_offset"] = 0

    _expect(payload, "IDENTITY_DRIFT", swapped)


def test_ownership_coverage(gathered):
    distributed, payload = gathered
    if not distributed:
        pytest.skip("ownership applies to the distributed optimizer")
    _expect(payload, "MISSING_PROVENANCE", lambda items: items[0].update(ownership=None))
    _expect(payload, "INCOMPLETE_ARTIFACT", lambda items: items[1]["ownership"]["ranges"].pop())

    def overlap(items):
        stolen = dict(items[1]["ownership"]["ranges"][0])
        items[0]["ownership"]["ranges"].append(stolen)

    _expect(payload, "IDENTITY_DRIFT", overlap)  # outside rank 0's shard

    def misaligned(items):
        items[0]["ownership"]["ranges"][0]["param_start"] += 1

    _expect(payload, "IDENTITY_DRIFT", misaligned)


def test_gather_requires_gloo(monkeypatch):
    monkeypatch.setattr(dist, "get_backend", lambda group=None: "nccl")
    with pytest.raises(T05ContractError) as info:
        gather_gradient_readbacks(GradientRankReadback({}, {}), None)
    assert info.value.status == "UNSUPPORTED_CAPABILITY"


def test_schema_version_is_checked():
    with pytest.raises(T05ContractError) as info:
        GradientRankReadback.from_dict({"schema_version": "other"})
    assert info.value.status == "SCHEMA_MISMATCH"


def test_sync_spy_is_removed_on_uninstall():
    ddp = FakeMegatronDDP(TinyModel())
    original = dist.all_reduce
    observer = GradientStepObserver(ddp, observe_sync=True)
    observer.install()
    assert dist.all_reduce is not original
    observer.uninstall()
    assert dist.all_reduce is original


def test_sync_off_keeps_the_evidence_shape():
    ddp = FakeMegatronDDP(TinyModel())
    observer = GradientStepObserver(ddp)
    observer.install()
    ddp.zero_grad_buffer()
    ddp.finish_grad_sync()
    payload = observer.end_step().to_dict()
    observer.uninstall()
    assert "sync" not in payload
