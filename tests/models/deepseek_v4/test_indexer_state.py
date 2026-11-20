# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.compressor_state import (
    CompressorStateContract,
    CompressorStateError,
)
from rl_engine.contracts.operators.indexer_state import IndexerStateContract
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache
from rl_engine.models.deepseek_v4.indexer_state import IndexerStateCache
from tests.models.deepseek_v4.indexer_state_helpers import make, LayoutOnlyBackend
from tests.models.deepseek_v4.compressor_state_helpers import byte_equal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 7, 8, 17, 65])
def test_complete_mxfp4_rows_and_invalid_defaults(count):
    cache, x, _, _ = make(LayoutOnlyBackend(), tokens=128)
    h = cache.create_request(7, version="v1", writer_rank=0)
    lease = cache.prepare(h, x[:count], start_position=0)
    cache.commit(lease)
    snap = cache.snapshot(h)
    assert snap["packed"].shape == (count // 4, 64)
    assert snap["scales"].shape == (count // 4, 4)
    assert snap["packed"].dtype == snap["scales"].dtype == torch.uint8
    assert bool((snap["packed"] == 17).all()) and bool((snap["scales"] == 129).all())
    assert bool((cache._cache.flatten(0, 1)[count // 4 :] == 0).all())
    assert bool((cache._scale_cache.flatten(0, 1)[count // 4 :] == 127).all())
    with pytest.raises(CompressorStateError, match="consumed"):
        cache.commit(lease)
    byte_equal(cache.snapshot(h)["packed"], snap["packed"])


def test_subclass_reuses_lifecycle_and_separates_main_handles():
    assert IndexerStateCache.prepare is CompressorStateCache.prepare
    assert IndexerStateCache.commit is CompressorStateCache.commit
    assert IndexerStateCache.release is CompressorStateCache.release
    assert IndexerStateCache.replace_model is CompressorStateCache.replace_model
    cache, x, _, params = make(LayoutOnlyBackend(), tokens=128)
    main_contract = CompressorStateContract(max_sequence_tokens=128)
    with pytest.raises(CompressorStateError, match="typed"):
        IndexerStateCache(
            main_contract,
            expected_arithmetic=main_contract.arithmetic_id,
            ape=params[0],
            norm=params[1],
            cos=params[2],
            sin=params[3],
            version="v1",
            writer_rank=0,
            backend=LayoutOnlyBackend(),
        )
    with pytest.raises(CompressorStateError, match="typed"):
        CompressorStateCache(
            cache.contract,
            expected_arithmetic=cache.contract.arithmetic_id,
            ape=params[0],
            norm=params[1],
            cos=params[2],
            sin=params[3],
            version="v1",
            writer_rank=0,
            backend=LayoutOnlyBackend(),
        )
    other, _, _, _ = make(LayoutOnlyBackend(), tokens=128)
    h = cache.create_request(7, version="v1", writer_rank=0)
    with pytest.raises(CompressorStateError, match="foreign"):
        other.prepare(h, x[:4], start_position=0)


@pytest.mark.parametrize(
    "change",
    [
        {"ratio": 128},
        {"ratio": 8},
        {"max_requests": 0},
        {"max_arena_bytes": 1},
        {"graph": 1},
        {"max_sequence_tokens": 2**31},
    ],
)
def test_invalid_indexer_state_contract(change):
    with pytest.raises(CompressorStateError):
        replace(IndexerStateContract(), **change)


def test_release_reinitializes_scales_and_old_snapshots_are_private():
    cache, x, _, _ = make(LayoutOnlyBackend(), tokens=128, max_requests=1)
    h = cache.create_request(7, version="v1", writer_rank=0)
    cache.append(h, x[:16], start_position=0)
    snap = cache.snapshot(h)
    snap["packed"].fill_(99)
    snap["scales"].fill_(255)
    assert bool((cache.snapshot(h)["scales"] == 129).all())
    cache.release(h)
    assert bool((cache._scale_cache == 127).all())
    new = cache.create_request(7, version="v1", writer_rank=0)
    assert new.generation != h.generation
    assert cache.snapshot(new)["packed"].numel() == 0
    with pytest.raises(CompressorStateError, match="stale"):
        cache.snapshot(h)


def test_corrupt_input_and_stale_version_do_not_change_payload_or_scales():
    cache, x, _, _ = make(LayoutOnlyBackend(), tokens=128)
    h = cache.create_request(7, version="v1", writer_rank=0)
    cache.append(h, x[:8], start_position=0)
    snap = cache.snapshot(h)
    x[8, 0] = torch.nan
    with pytest.raises(CompressorStateError, match="finite"):
        cache.append(h, x[8:12], start_position=8)
    with pytest.raises(CompressorStateError, match="version"):
        cache.create_request(8, version="old", writer_rank=0)
    byte_equal(cache.snapshot(h)["packed"], snap["packed"])
    byte_equal(cache.snapshot(h)["scales"], snap["scales"])


def test_shared_framework_factory_does_not_duplicate_lifecycle():
    from rl_engine.integrations.engines.rollout.vllm.indexer_state import (
        bind_indexer_state as rollout,
    )
    from rl_engine.integrations.engines.train.megatron.indexer_state import (
        bind_indexer_state as train,
    )

    assert rollout is train
    cache, _, _, (a, n, co, si) = make(LayoutOnlyBackend(), tokens=128)
    mirror = train(
        cache.contract,
        expected_arithmetic=cache.contract.arithmetic_id,
        ape=a,
        norm=n,
        cos=co,
        sin=si,
        version="v1",
        writer_rank=0,
        backend=LayoutOnlyBackend(),
        expected_implementation=cache.implementation_fingerprint,
    )
    assert isinstance(mirror, IndexerStateCache)
