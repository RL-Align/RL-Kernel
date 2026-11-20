# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.recent128 import (
    RECENT128_BACKEND_ID,
    CacheRequest,
    CacheVersion,
    Recent128Contract,
    Recent128ContractError,
)
from rl_engine.integrations.engines.rollout.vllm.recent128 import bind_recent128_metadata
from rl_engine.models.deepseek_v4.recent128 import Recent128Cache, Recent128Workspace
from tests.models.deepseek_v4.recent128_helpers import metadata

pytestmark = pytest.mark.unit
VERSION = CacheVersion("checkpoint-0731", "lora-0", "scales-0")
REQUEST = CacheRequest("request-A")


class RecordingBackend:
    """No numerical claims: CPU tests cover ownership/state, GPU tests use vLLM."""

    backend_id = RECENT128_BACKEND_ID

    def __init__(self):
        self.calls = []
        self.fail = False

    def write(self, *args):
        self.calls.append("write")
        if self.fail:
            raise RuntimeError("injected write failure")

    def gather(self, *args):
        self.calls.append("gather")

    def indices(self, *args):
        self.calls.append("indices")


def bind(meta, *, requests=(REQUEST,), version=VERSION, contract=None):
    return bind_recent128_metadata(
        meta,
        requests=requests,
        version=version,
        contract=contract or Recent128Contract(512),
    )


def session(contract=None, *, pages=32):
    contract = contract or Recent128Contract(512)
    backend = RecordingBackend()
    cache = Recent128Cache(
        torch.zeros((pages, contract.page_bytes), dtype=torch.uint8),
        contract,
        layer_id="layer-0",
        backend=backend,
    )
    cache.register(REQUEST, VERSION)
    return cache, backend


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_boundaries(length):
    batch = bind(metadata([length], [length]))
    assert batch.previous_lengths == (0,)
    assert batch.final_lengths == (length,)
    assert batch.gather_lens.tolist() == [length]
    assert batch.gather_starts == (0,)


@pytest.mark.parametrize("prefix,count", [(128, 2), (127, 129), (512, 257), (3, 4)])
def test_chunk_retains_prefix_and_entire_query(prefix, count):
    batch = bind(metadata([prefix + count], [count]))
    assert batch.gather_lens.tolist() == [min(prefix, 127) + count]
    assert batch.gather_starts == (max(0, prefix - 127),)


def test_128_ring_is_rejected_before_write():
    # Queries 128/129 need positions 1..129. Recycling page 0 for page 2
    # would overwrite some of those positions before query 128 consumes them.
    with pytest.raises(Recent128ContractError, match="live page alias"):
        bind(metadata([130], [2], tables=[[0, 1, 0]]))


def test_old_pages_can_be_recycled_outside_live_union():
    batch = bind(metadata([321], [1], tables=[[3, 4, 5, 0, 1, 2]]))
    assert batch.gather_starts == (193,)


def test_no_shared_writable_pages_without_cow():
    with pytest.raises(Recent128ContractError, match="shared page"):
        bind(
            metadata([1, 1], [1, 1], tables=[[0], [0]]),
            requests=(REQUEST, CacheRequest("B")),
        )


def test_bad_slots_and_holes_rejected():
    meta = metadata([129], [2])
    meta.slot_mapping[0] += 1
    with pytest.raises(Recent128ContractError, match="logical position"):
        bind(meta)
    meta.slot_mapping[0] = -1
    with pytest.raises(Recent128ContractError, match="logical position"):
        bind(meta)


def test_suffix_padding_and_empty_request():
    batch = bind(
        metadata([2, 0], [2, 0], padding=3),
        requests=(REQUEST, CacheRequest("empty")),
    )
    assert batch.num_tokens == 5 and batch.num_actual_tokens == 2
    assert batch.gather_lens.tolist() == [2, 0]


def test_snapshot_detaches_scheduler_mutation_and_handles_inference_tensors():
    with torch.inference_mode():
        meta = metadata([2], [2])
        batch = bind(meta)
        meta.slot_mapping.fill_(-1)
    batch.validate_snapshot()
    assert batch.slot_mapping.tolist() == [0, 1]
    batch.slot_mapping[0] = -1
    with pytest.raises(Recent128ContractError, match="mutated"):
        batch.validate_snapshot()


def test_update_replay_duplicate_and_versions():
    cache, backend = session()
    batch = bind(metadata([4], [4]))
    kv = torch.zeros((4, 512), dtype=torch.bfloat16)
    cache.update(kv, batch)
    cache.replay(batch)
    assert backend.calls == ["write", "gather", "indices", "gather", "indices"]
    with pytest.raises(Recent128ContractError, match="duplicate"):
        cache.update(kv, batch)
    wrong = bind(metadata([5], [1]), version=CacheVersion("checkpoint-0731", "lora-1", "scales-0"))
    with pytest.raises(Recent128ContractError, match="version"):
        cache.update(kv[:1], wrong)
    assert backend.calls.count("write") == 1


def test_generation_and_adoption_are_explicit():
    cache, _ = session()
    cache.release(REQUEST)
    with pytest.raises(Recent128ContractError, match="newer generation"):
        cache.register(REQUEST, VERSION)
    key = CacheRequest("request-A", 1)
    cache.register(key, VERSION, cached_length=128)
    batch = bind(metadata([130], [2]), requests=(key,))
    cache.update(torch.zeros((2, 512), dtype=torch.bfloat16), batch)


def test_backend_failure_poisoning():
    cache, backend = session()
    backend.fail = True
    batch = bind(metadata([1], [1]))
    with pytest.raises(RuntimeError, match="injected"):
        cache.update(torch.zeros((1, 512), dtype=torch.bfloat16), batch)
    backend.fail = False
    with pytest.raises(Recent128ContractError, match="failed"):
        cache.update(torch.zeros((1, 512), dtype=torch.bfloat16), batch)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_query_tokens": 0},
        {"max_query_tokens": True},
        {"window_size": 127},
        {"block_size": 16},
        {"head_dim": 576},
        {"encoding": "fp8"},
        {"rope_state": "pre_rope"},
        {"cp_size": 2},
        {"graph_capture": True},
        {"backend_id": "fallback"},
        {"layout_id": "bf16"},
        {"max_query_tokens": 2**31},
    ],
)
def test_unsupported_contracts(changes):
    with pytest.raises(Recent128ContractError):
        replace(Recent128Contract(8), **changes)


def test_invalid_storage_capacity_and_autograd_do_not_write():
    cache, backend = session(pages=1)
    batch = bind(metadata([65], [65]))
    with pytest.raises(Recent128ContractError, match="out of range"):
        cache.update(torch.zeros((65, 512), dtype=torch.bfloat16), batch)
    batch = bind(metadata([1], [1]))
    with pytest.raises(Recent128ContractError, match="without grad"):
        cache.update(torch.zeros((1, 512), dtype=torch.bfloat16, requires_grad=True), batch)
    assert backend.calls == []


def test_metadata_bounds_and_fixed_capacity():
    with pytest.raises(Recent128ContractError, match="capacity"):
        bind(metadata([3], [3]), contract=Recent128Contract(2))
    bad = metadata([1], [1])
    bad.block_table[0, 0] = -1
    with pytest.raises(Recent128ContractError, match="missing live page"):
        bind(bad)
    with pytest.raises(Recent128ContractError, match="row count"):
        bind(metadata([1, 1], [1, 1]))


def test_megatron_contract_rejects_incompatible_encoding_before_import():
    from rl_engine.integrations.engines.train.megatron.recent128 import recent128_visible_k

    with pytest.raises(Recent128ContractError, match="OCP/64"):
        recent128_visible_k(
            torch.zeros((1, 512), dtype=torch.bfloat16),
            contract=Recent128Contract(1, encoding="e4m3fnuz"),
        )


def test_workspace_cannot_alias_metadata():
    cache, backend = session()
    batch = bind(metadata([1], [1]))
    workspace = Recent128Workspace.allocate(batch)
    workspace.lengths = batch.seq_lens
    with pytest.raises(Recent128ContractError, match="alias"):
        cache.update(
            torch.zeros((1, 512), dtype=torch.bfloat16),
            batch,
            workspace=workspace,
        )
    assert backend.calls == []


def test_zero_request_batch():
    cache, backend = session()
    batch = bind(metadata([], [], padding=0), requests=())
    view = cache.update(torch.empty((0, 512), dtype=torch.bfloat16), batch)
    assert view.kv.shape == (0, cache.contract.workspace_width, 512)
    assert view.indices.shape == (0, 128)


def test_fixed_workspace_int32_bounds():
    with pytest.raises(Recent128ContractError, match="int32"):
        bind(
            metadata([1, 1], [1, 1]),
            requests=(REQUEST, CacheRequest("B")),
            contract=Recent128Contract(2**30),
        )
