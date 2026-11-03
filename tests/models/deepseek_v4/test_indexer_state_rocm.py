# SPDX-License-Identifier: Apache-2.0
"""Real MI300X FP32 history, native compression/Hadamard/MXFP4 and cache writes."""

from concurrent.futures import ThreadPoolExecutor
import gc
import json

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_state import AiterIndexerStateBackend
from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.compressor_state import CompressorStateError
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from tests.models.deepseek_v4.indexer_state_helpers import make
from tests.models.deepseek_v4.compressor_state_helpers import byte_equal

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterIndexerStateBackend()


def standalone(backend, x, ids, parameters):
    op = IndexerC4Compressor(IndexerC4Contract(), backend=backend.postprocess)
    bound = op.bind(x, ids, version="v1")
    return op(bound, *parameters, version="v1")


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 4, 17, 129, 271])
def test_bytes_scales_and_state_match_task20_across_chunking_and_graph(backend, graph, chunk):
    cache, x, ids, parameters = make(backend, graph=graph, device="cuda")
    expected = standalone(backend, x, ids, parameters)
    h = cache.create_request(7, version="v1", writer_rank=0)
    written = set()
    for start in range(0, len(x), chunk):
        receipt = cache.append(h, x[start : start + chunk], start_position=start)
        for group in range(receipt.first_group, receipt.end_group):
            assert group not in written
            written.add(group)
    actual = cache.snapshot(h)
    byte_equal(actual["packed"], expected.packed)
    byte_equal(actual["scales"], expected.scales)
    assert written == set(range(len(x) // 4))
    ring = torch.zeros((8, 512), dtype=torch.float32, device="cuda")
    for pos in range(len(x) - 8, len(x)):
        ring[pos % 8, :256] = x[pos, :256].float()
        ring[pos % 8, 256:] = x[pos, 256:].float() + parameters[0][pos % 4]
    byte_equal(actual["partial_ring"], ring)
    rows = actual["packed"].shape[0]
    assert bool((cache._cache.flatten(0, 1)[rows:] == 0).all())
    assert bool((cache._scale_cache.flatten(0, 1)[rows:] == 127).all())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("graph", [False, True])
def test_two_requests_variable_chunks_disjoint_cache_and_page_reuse(backend, seed, graph):
    cache, x, ids, parameters = make(backend, graph=graph, seed=seed, device="cuda", max_requests=2)
    expected = [standalone(backend, values, ids, parameters) for values in (x, -x)]
    handles = [cache.create_request(rid, version="v1", writer_rank=0) for rid in (7, 8)]
    starts = [0, 0]
    step = seed
    while min(starts) < len(x):
        i = step % 2
        first = starts[i]
        count = min([1, 5, 17, 31][step % 4], len(x) - first)
        if count:
            values = x if i == 0 else -x
            cache.append(handles[i], values[first : first + count], start_position=first)
            starts[i] += count
        step += 1
    for h, ref in zip(handles, expected, strict=True):
        out = cache.snapshot(h)
        byte_equal(out["packed"], ref.packed)
        byte_equal(out["scales"], ref.scales)
    saved = cache.snapshot(handles[1])
    cache.release(handles[0])
    new = cache.create_request(7, version="v1", writer_rank=0)
    assert new.generation > handles[0].generation and new.slot == handles[0].slot
    cache.append(new, x, start_position=0)
    byte_equal(cache.snapshot(handles[1])["packed"], saved["packed"])
    byte_equal(cache.snapshot(handles[1])["scales"], saved["scales"])
    byte_equal(cache.snapshot(new)["packed"], expected[0].packed)
    with pytest.raises(CompressorStateError, match="stale"):
        cache.prepare(handles[0], x[:4], start_position=0)


def test_incomplete_groups_never_pollute_cache_and_capture_has_no_writes(backend):
    cache, x, _, _ = make(backend, graph=True, step=1, device="cuda")
    assert bool((cache._cache == 0).all()) and bool((cache._scale_cache == 127).all())
    h = cache.create_request(7, version="v1", writer_rank=0)
    for pos in range(3):
        cache.append(h, x[pos : pos + 1], start_position=pos)
        assert cache.snapshot(h)["packed"].numel() == 0
        assert bool((cache._cache == 0).all()) and bool((cache._scale_cache == 127).all())
    cache.append(h, x[3:4], start_position=3)
    assert cache.snapshot(h)["packed"].shape == (1, 64)
    assert bool((cache._cache.flatten(0, 1)[1:] == 0).all())
    assert bool((cache._scale_cache.flatten(0, 1)[1:] == 127).all())


@pytest.mark.parametrize("graph", [False, True])
def test_duplicate_thread_commit_alias_reuse_and_cross_stream(backend, graph):
    cache, x, _, _ = make(backend, graph=graph, device="cuda")
    h = cache.create_request(7, version="v1", writer_rank=0)
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    a.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(a):
        lease = cache.prepare(h, x[:8], start_position=0)
    x[:8].data.fill_(torch.nan)
    with torch.cuda.stream(b):
        cache.commit(lease)
    original = cache.snapshot(h)
    lease = cache.prepare(h, x[8:16], start_position=8)

    def run(_):
        with torch.cuda.device(cache.device):
            try:
                cache.commit(lease)
                return 1
            except CompressorStateError:
                return 0

    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(run, range(2))) == 1
    out = cache.snapshot(h)
    byte_equal(out["packed"][:2], original["packed"])
    byte_equal(out["scales"][:2], original["scales"])
    assert out["commits"] == 2


@pytest.mark.parametrize("graph", [False, True])
def test_model_replacement_zero_values_and_snapshot_independence(backend, graph):
    cache, x, ids, (a, n, co, si) = make(backend, graph=graph, device="cuda")
    h = cache.create_request(7, version="v1", writer_rank=0)
    cache.append(h, x, start_position=0)
    old = cache.snapshot(h)
    cache.release(h)
    assert bool((cache._scale_cache == 127).all())
    new_a, new_n = a * 2, n * 0.5
    cache.replace_model(version="v2", ape=new_a, norm=new_n, cos=co, sin=si)
    h = cache.create_request(7, version="v2", writer_rank=0)
    cache.append(h, x, start_position=0)
    expected = standalone(backend, x, ids, (new_a, new_n, co, si))
    actual = cache.snapshot(h)
    byte_equal(actual["packed"], expected.packed)
    byte_equal(actual["scales"], expected.scales)
    assert not torch.equal(actual["scales"], old["scales"])
    actual["packed"].zero_()
    actual["scales"].fill_(255)
    byte_equal(cache.snapshot(h)["scales"], expected.scales)
    cache.release(h)
    h = cache.create_request(7, version="v2", writer_rank=0)
    x.zero_()
    cache.append(h, x, start_position=0)
    expected = standalone(backend, x, ids, (new_a, new_n, co, si))
    byte_equal(cache.snapshot(h)["packed"], expected.packed)
    byte_equal(cache.snapshot(h)["scales"], expected.scales)


def test_native_postprocessing_and_mask_scatter_are_reused(backend, monkeypatch):
    cache, x, _, _ = make(backend, device="cuda")
    h = cache.create_request(7, version="v1", writer_rank=0)
    order = []
    compress, rotate, quant, scatter, state = (
        backend.compress,
        backend.postprocess._rotate,
        backend.postprocess.quantize,
        backend._scatter,
        backend.save_partial,
    )

    def wrap(name, fn):
        def call(*args, **kwargs):
            order.append(name)
            return fn(*args, **kwargs)

        return call

    with monkeypatch.context() as patch:
        patch.setattr(backend, "compress", wrap("compress", compress))
        patch.setattr(backend.postprocess, "_rotate", wrap("rotate", rotate))
        patch.setattr(backend.postprocess, "quantize", wrap("quant", quant))
        patch.setattr(backend, "_scatter", wrap("scatter", scatter))
        patch.setattr(backend, "save_partial", wrap("state", state))
        cache.append(h, x[:17], start_position=0)
    assert order == ["compress", "rotate", "quant", "scatter", "state"]


def test_fixed_graph_addresses_and_release_reuse_allocation(backend, tmp_path):
    cache, x, _, _ = make(backend, graph=True, device="cuda")
    names = ("_cache", "_scale_cache", "_bf16_stage", "_gather_rows", "_commit_mask", "_state")
    pointers = {name: getattr(cache, name).data_ptr() for name in names}
    h = cache.create_request(7, version="v1", writer_rank=0)
    cache.append(h, x[:17], start_position=0)
    cache.release(h)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(40):
        h = cache.create_request(7, version="v1", writer_rank=0)
        cache.append(h, x[:17], start_position=0)
        cache.release(h)
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    # Delayed allocator cleanup from prior Graph instances may reduce live
    # bytes. The invariant is no growth, not exact allocator bookkeeping.
    assert after <= before
    assert pointers == {name: getattr(cache, name).data_ptr() for name in names}
    assert bool((cache._cache == 0).all()) and bool((cache._scale_cache == 127).all())
    (tmp_path / "allocation.json").write_text(
        json.dumps({"before": before, "after": after, "cycles": 40})
    )


def test_payload_scale_partial_write_failure_poisons_without_publishing_cursor(
    backend, monkeypatch
):
    cache, x, _, _ = make(backend, device="cuda")
    h = cache.create_request(7, version="v1", writer_rank=0)
    lease = cache.prepare(h, x[:4], start_position=0)

    def fail_after_payload(arena, payload, scale):
        rows = arena._cache.flatten(0, 1)
        rows.masked_scatter_(arena._commit_mask[:, None].expand_as(rows), payload.flatten())
        torch.cuda.synchronize()
        raise RuntimeError("simulated scale write failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "_scatter", fail_after_payload)
        with pytest.raises(RuntimeError, match="scale write"):
            cache.commit(lease)
    assert cache._requests[7].tokens == 0
    assert bool((cache._scale_cache == 127).all())
    for method, args in [(cache.commit, (lease,)), (cache.snapshot, (h,)), (cache.release, (h,))]:
        with pytest.raises(CompressorStateError, match="poisoned"):
            method(*args)
