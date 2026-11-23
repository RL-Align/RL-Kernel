# SPDX-License-Identifier: Apache-2.0
"""INK-22 per-visit cache contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.visit_cache import VisitKey, VisitStateCache


def test_cache_isolates_physical_layer_pass_and_logical_visit():
    cache = VisitStateCache()
    first = VisitKey(6, 0, 6)
    second = VisitKey(6, 1, 18)
    cache.put(first, kv=torch.ones(2, 3), conv=torch.zeros(2, 3))
    cache.put(second, kv=torch.full((2, 3), 2.0), conv=torch.full((2, 3), 3.0))
    assert torch.equal(cache.require(first).kv, torch.ones(2, 3))
    assert torch.equal(cache.require(second).kv, torch.full((2, 3), 2.0))
    assert cache.keys() == (first, second)


def test_cache_returns_clones_and_prevents_state_aliasing():
    cache = VisitStateCache()
    key = VisitKey(7, 2, 19)
    source = torch.ones(2, 2)
    cache.put(key, kv=source)
    source.fill_(4)
    fetched = cache.require(key)
    fetched.kv.fill_(9)
    assert torch.equal(cache.require(key).kv, torch.ones(2, 2))


def test_clear_pass_removes_only_matching_pass():
    cache = VisitStateCache()
    cache.put(VisitKey(1, 0, 1), kv=torch.ones(1))
    cache.put(VisitKey(1, 1, 13), kv=torch.ones(1))
    cache.put(VisitKey(2, 1, 14), conv=torch.ones(1))
    cache.clear_pass(1)
    assert cache.get(VisitKey(1, 0, 1)) is not None
    assert cache.get(VisitKey(1, 1, 13)) is None
    assert cache.get(VisitKey(2, 1, 14)) is None


def test_missing_state_fails_closed():
    with pytest.raises(KeyError, match="no cached state"):
        VisitStateCache().require(VisitKey(0, 0, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_visit_state_clone_matches_source():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    from rl_engine.models.inkling2.visit_cache import clone_visit_state_cuda

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_visit_state_clone"):
        pytest.skip("Inkling 2 visit-state CUDA extension is unavailable")
    source = torch.randn(3, 4, device="cuda", dtype=torch.bfloat16)
    clone = clone_visit_state_cuda(source)
    assert torch.equal(clone, source)
    clone.add_(1)
    assert not torch.equal(clone, source)
