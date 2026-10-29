# SPDX-License-Identifier: Apache-2.0
"""Native aiter/vLLM lifecycle, partial-state, single-commit and Graph tests."""

from concurrent.futures import ThreadPoolExecutor
import gc
import json

import pytest
import torch

from rl_engine.backends.rocm.attention.compressor_state import AiterCompressorStateBackend
from rl_engine.contracts.operators.compressor_state import (
    CompressorStateContract,
    CompressorStateError,
)
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache
from tests.models.deepseek_v4.compressor_state_helpers import (
    tensors,
    byte_equal,
    ReferenceStateBackend,
)

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterCompressorStateBackend()


def make(backend, ratio, *, graph=False, seed=21, max_tokens=271, **kwargs):
    c = CompressorStateContract(
        ratio=ratio,
        graph=graph,
        max_sequence_tokens=max_tokens,
        max_tokens_per_step=max_tokens,
        **kwargs,
    )
    x, a, n, co, si = tensors(c, seed=seed, device="cuda")
    engine = CompressorStateCache(
        c,
        ape=a,
        norm=n,
        cos=co,
        sin=si,
        version="v1",
        writer_rank=0,
        expected_arithmetic=c.arithmetic_id,
        backend=backend,
    )
    return engine, x, (a, n, co, si)


def run(engine, x, chunk):
    h = engine.create_request(7, version="v1", writer_rank=0)
    rows_written = set()
    for first in range(0, len(x), chunk):
        receipt = engine.append(h, x[first : first + chunk], start_position=first)
        for g in range(receipt.first_group, receipt.end_group):
            assert g not in rows_written
            rows_written.add(g)
    assert rows_written == set(range(len(x) // engine.contract.ratio))
    return engine.snapshot(h), h


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 17, 129, 271])
def test_complete_state_and_cache_bytes_across_prefill_decode_chunks_graph(
    backend, ratio, graph, chunk
):
    baseline, x, _ = make(backend, ratio)
    expected, _ = run(baseline, x, len(x))
    engine, x, _ = make(backend, ratio, graph=graph)
    actual, _ = run(engine, x, chunk)
    byte_equal(actual["rows"], expected["rows"])
    byte_equal(actual["partial_ring"], expected["partial_ring"])
    assert actual["rows"].shape == (271 // ratio, 512)
    assert actual["incomplete_tokens"] == 271 % ratio
    if graph:
        assert engine._graph is not None


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("seed", [0, 5, 21, 73])
def test_numerical_cache_vs_independent_fp64_and_exact_ring_state(backend, ratio, seed):
    native, x, (a, n, co, si) = make(backend, ratio, seed=seed, max_tokens=265)
    actual, _ = run(native, x, 7)
    reference = CompressorStateCache(
        native.contract,
        ape=a.cpu(),
        norm=n.cpu(),
        cos=co.cpu(),
        sin=si.cpu(),
        version="v1",
        writer_rank=0,
        expected_arithmetic=native.contract.arithmetic_id,
        backend=ReferenceStateBackend(),
    )
    expected, _ = run(reference, x.cpu(), 265)
    byte_equal(actual["partial_ring"].cpu(), expected["partial_ring"])
    # The reference rounds its FP64 result to the declared BF16 output. A
    # separate byte-exact test above covers BI rather than numerical tolerance.
    torch.testing.assert_close(
        actual["rows"].float().cpu(), expected["rows"].float(), rtol=0.009, atol=1e-4
    )


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("graph", [False, True])
def test_interleaved_requests_duplicate_lease_release_page_reuse_and_new_version(
    backend, ratio, graph
):
    engine, x, parameters = make(backend, ratio, graph=graph, max_requests=2)
    h1 = engine.create_request(10, version="v1", writer_rank=0)
    h2 = engine.create_request(11, version="v1", writer_rank=0)
    for first in range(0, len(x), 17):
        for handle, values in [(h1, x), (h2, -x)]:
            lease = engine.prepare(handle, values[first : first + 17], start_position=first)
            engine.commit(lease)
            previous = engine.snapshot(handle)
            with pytest.raises(CompressorStateError, match="consumed"):
                engine.commit(lease)
            byte_equal(engine.snapshot(handle)["rows"], previous["rows"])
    saved = engine.snapshot(h2)
    engine.release(h1)
    new = engine.create_request(10, version="v1", writer_rank=0)
    assert new.slot == h1.slot and new.generation > h1.generation
    assert engine.snapshot(new)["rows"].shape == (0, 512)
    assert bool((engine.snapshot(new)["partial_ring"] == 0).all())
    with pytest.raises(CompressorStateError, match="stale"):
        engine.prepare(h1, x[:1], start_position=0)
    engine.append(new, x, start_position=0)
    byte_equal(engine.snapshot(h2)["rows"], saved["rows"])
    byte_equal(engine.snapshot(h2)["partial_ring"], saved["partial_ring"])
    engine.release(new)
    engine.release(h2)
    engine.replace_model(
        version="v2", **dict(zip(("ape", "norm", "cos", "sin"), parameters, strict=True))
    )
    with pytest.raises(CompressorStateError, match="version"):
        engine.create_request(10, version="v1", writer_rank=0)
    h3 = engine.create_request(10, version="v2", writer_rank=0)
    engine.append(h3, x, start_position=0)
    assert engine.snapshot(h3)["version"] == "v2"


@pytest.mark.parametrize("ratio", [4, 128])
def test_capture_warmup_does_not_commit_or_write_partial_state(backend, ratio):
    engine, x, _ = make(backend, ratio, graph=True)
    assert bool((engine._cache == 0).all()) and bool((engine._state == 0).all())
    h = engine.create_request(7, version="v1", writer_rank=0)
    engine.append(h, x[: ratio - 1], start_position=0)
    assert engine.snapshot(h)["rows"].shape == (0, 512)
    assert bool((engine._cache == 0).all())
    engine.append(h, x[ratio - 1 : ratio], start_position=ratio - 1)
    snapshot = engine.snapshot(h)
    assert snapshot["rows"].shape == (1, 512)
    # Every unused cache row must retain its zero initializer.
    assert bool((engine._cache.flatten(0, 1)[1:] == 0).all())


@pytest.mark.parametrize("ratio", [4, 128])
def test_upstream_order_unique_ring_writers_and_no_custom_math(backend, ratio, monkeypatch):
    engine, x, _ = make(backend, ratio)
    h = engine.create_request(7, version="v1", writer_rank=0)
    observed = []
    compress, writer = backend.compress, backend.save_partial

    def compress_spy(**kwargs):
        observed.append("compress")
        assert kwargs["head_dim"] == 512
        assert kwargs.get("k_split_num_waves", 1) == 1
        return compress(**kwargs)

    def writer_spy(**kwargs):
        observed.append("state")
        slots = kwargs["slot_mapping"]
        active = slots[slots >= 0].cpu().tolist()
        assert len(active) == len(set(active)) == engine.contract.state_rows
        return writer(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "compress", compress_spy)
        patch.setattr(backend, "save_partial", writer_spy)
        engine.append(h, x, start_position=0)
    assert observed == ["compress", "state"]


def test_staging_owned_aliases_input_validation_and_writer_identity(backend):
    engine, x, _ = make(backend, 4, graph=True, world_size=2)
    h = engine.create_request(7, version="v1", writer_rank=0)
    expected = x[:4].float().clone()
    lease = engine.prepare(h, x[:4], start_position=0)
    x[:4].data.zero_()
    engine.commit(lease)
    byte_equal(engine.snapshot(h)["partial_ring"][:4, :1024], expected[:, :1024])
    other = engine.create_request(8, version="v1", writer_rank=1)
    with pytest.raises(CompressorStateError, match="writer"):
        engine.prepare(other, x[:4], start_position=0)
    # Rejection precedes any staging/cache/state write.
    before = engine.snapshot(h)
    x[4, 0] = torch.inf
    with pytest.raises(CompressorStateError, match="finite"):
        engine.prepare(h, x[4:8], start_position=4)
    byte_equal(engine.snapshot(h)["partial_ring"], before["partial_ring"])
    byte_equal(engine.snapshot(h)["rows"], before["rows"])


@pytest.mark.parametrize("graph", [False, True])
def test_cross_stream_handoff_host_race_and_no_early_buffer_reuse(backend, graph):
    engine, x, _ = make(backend, 4, graph=graph)
    h = engine.create_request(7, version="v1", writer_rank=0)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        lease = engine.prepare(h, x[:8], start_position=0)
    with torch.cuda.stream(consumer):
        engine.commit(lease)
    assert engine.snapshot(h)["tokens"] == 8
    lease = engine.prepare(h, x[8:16], start_position=8)

    def commit():
        with torch.cuda.device(engine.device):
            try:
                engine.commit(lease)
                return 1
            except CompressorStateError:
                return 0

    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(lambda _: commit(), range(2))) == 1
    assert engine.snapshot(h)["commits"] == 2


def test_static_graph_buffers_stay_stable_across_requests_and_updates(backend, tmp_path):
    engine, x, parameters = make(backend, 4, graph=True)
    buffers = (
        "_state",
        "_cache",
        "_input",
        "_plan",
        "_positions",
        "_write_slots",
        "_read_kv",
        "_read_score",
        "_ape",
        "_norm",
        "_cos",
        "_sin",
    )
    pointers = {name: getattr(engine, name).data_ptr() for name in buffers}
    h = engine.create_request(7, version="v1", writer_rank=0)
    engine.append(h, x[:4], start_position=0)
    engine.release(h)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(40):
        h = engine.create_request(7, version="v1", writer_rank=0)
        engine.append(h, x[:4], start_position=0)
        engine.release(h)
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert before == after
    engine.replace_model(
        version="v2", **dict(zip(("ape", "norm", "cos", "sin"), parameters, strict=True))
    )
    assert pointers == {name: getattr(engine, name).data_ptr() for name in buffers}
    (tmp_path / "allocation.json").write_text(
        json.dumps({"before": before, "after": after, "cycles": 40})
    )


def test_model_weights_are_snapshotted_and_c128_profile_mismatch_is_rejected(backend):
    engine, x, (a, n, co, si) = make(backend, 128)
    reference, _, _ = make(backend, 128)
    a.data.zero_()
    n.data.zero_()
    co.data.zero_()
    si.data.zero_()
    actual, _ = run(engine, x, 17)
    expected, _ = run(reference, x, len(x))
    byte_equal(actual["rows"], expected["rows"])
    with pytest.raises(CompressorStateError, match="arithmetic mismatch"):
        CompressorStateCache(
            engine.contract,
            ape=a,
            norm=n,
            cos=co,
            sin=si,
            version="v1",
            writer_rank=0,
            expected_arithmetic="aiter.hca.eight_wave.fp32",
            backend=backend,
        )


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("seed", [0, 9, 73])
def test_seeded_uneven_graph_updates_equal_full_native_execution(backend, ratio, seed):
    baseline, x, _ = make(backend, ratio, seed=seed)
    expected, _ = run(baseline, x, len(x))
    engine, x, _ = make(backend, ratio, seed=seed, graph=True)
    h = engine.create_request(7, version="v1", writer_rank=0)
    sizes = [1, 3, 4, 7, 31, 127]
    start, step = 0, seed
    while start < len(x):
        count = min(sizes[step % len(sizes)], len(x) - start)
        engine.append(h, x[start : start + count], start_position=start)
        step += 1
        start += count
    actual = engine.snapshot(h)
    byte_equal(actual["rows"], expected["rows"])
    byte_equal(actual["partial_ring"], expected["partial_ring"])


@pytest.mark.parametrize("ratio", [4, 128])
def test_graph_model_replacement_uses_new_parameters_and_no_stale_request_data(backend, ratio):
    engine, x, (a, n, co, si) = make(backend, ratio, graph=True)
    old, h = run(engine, x, 17)
    engine.release(h)
    new_a, new_n = a * 1.5, n * 0.5
    engine.replace_model(version="v2", ape=new_a, norm=new_n, cos=co, sin=si)
    fresh = CompressorStateCache(
        engine.contract,
        ape=new_a,
        norm=new_n,
        cos=co,
        sin=si,
        version="v2",
        writer_rank=0,
        expected_arithmetic=engine.contract.arithmetic_id,
        backend=backend,
    )
    snapshots = []
    for cache in (engine, fresh):
        handle = cache.create_request(7, version="v2", writer_rank=0)
        for start in range(0, len(x), 17):
            cache.append(handle, x[start : start + 17], start_position=start)
        snapshots.append(cache.snapshot(handle))
    byte_equal(snapshots[0]["rows"], snapshots[1]["rows"])
    byte_equal(snapshots[0]["partial_ring"], snapshots[1]["partial_ring"])
    assert not torch.equal(old["rows"], snapshots[0]["rows"])


def test_zero_length_no_native_launch_and_external_graph_lifecycle_rejection(backend, monkeypatch):
    engine, x, _ = make(backend, 4)
    h = engine.create_request(7, version="v1", writer_rank=0)

    def forbidden(_):
        raise AssertionError("zero-length commit must not launch")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "execute", forbidden)
        receipt = engine.append(h, x[:0], start_position=0)
        assert receipt.first_group == receipt.end_group == 0
    assert engine.snapshot(h)["tokens"] == 0
    with pytest.raises(CompressorStateError, match="external Graph"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            engine.prepare(h, x[:1], start_position=0)
