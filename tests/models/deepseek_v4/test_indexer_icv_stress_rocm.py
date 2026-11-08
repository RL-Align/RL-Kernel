# SPDX-License-Identifier: Apache-2.0
"""Seeded property, stream, memory and isolation tests on actual library kernels."""

import gc
import json
import os
import subprocess
import sys

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend
from rl_engine.contracts.operators.indexer_icv import IndexerICVContract, IndexerICVError
from rl_engine.integrations.common.indexer_icv import bind_icv_queries, bind_global_icv_keys
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV
from tests.models.deepseek_v4.icv_helpers import byte_equal, decode, ids, payload

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterIndexerICVBackend()


def make_contract(precision, **kwargs):
    values = dict(num_heads=4, head_dim=128, query_tile_rows=7, key_tile_rows=31)
    values.update(kwargs)
    return IndexerICVContract(
        precision="mxfp4" if precision == "mxfp4" else "fp8",
        fp8_encoding=None if precision == "mxfp4" else precision,
        **values,
    )


def pair(op, m=17, n=65):
    c = op.contract
    q = bind_icv_queries(
        op,
        *payload(c, m, q=True, device="cuda"),
        ids(7, m, device="cuda"),
        layout=c.layout,
        version="v1",
    )
    k = bind_global_icv_keys(
        op,
        *payload(c, n, seed=1, device="cuda"),
        ids(7, n, device="cuda"),
        layout=c.layout,
        version="v1",
        request_key_counts={7: n},
    )
    return q, k


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
@pytest.mark.parametrize("alias_kind", ["data", "set_storage"])
def test_snapshot_protects_gpu_queries_and_prepared_keys_from_caller_aliases(
    backend, precision, alias_kind
):
    c = make_contract(precision)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 3, q=True, device="cuda")
    kv, ks = payload(c, 5, seed=42, device="cuda")
    q = bind_icv_queries(
        op,
        qv,
        qs,
        ids(7, 3, device="cuda"),
        layout=c.layout,
        version="v1",
        ownership="snapshot",
    )
    k = bind_global_icv_keys(
        op,
        kv,
        ks,
        ids(7, 5, device="cuda"),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 5},
        ownership="snapshot",
    )
    before = op.materialize(q, k)[0].values.clone()
    tensors = [qv, qs, kv, ks]
    versions = [t._version for t in tensors]
    for t in tensors:
        if alias_kind == "data":
            t.data.view(torch.uint8).zero_()
        else:
            torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                t.untyped_storage(), 0, (t.numel() * t.element_size(),), (1,)
            ).zero_()
    assert [t._version for t in tensors] == versions
    byte_equal(op.materialize(q, k)[0].values, before)


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
@pytest.mark.parametrize("seed", range(10))
def test_seeded_ragged_multi_request_geometry_and_cp_properties(backend, precision, seed):
    c = make_contract(
        precision,
        num_heads=(1, 3, 8, 64, 128)[seed % 5],
        head_dim=(1, 31, 64, 127, 128, 129, 255, 256)[seed % 8],
        query_tile_rows=(1, 3, 7, 32)[seed % 4],
        key_tile_rows=(7, 31, 64)[seed % 3],
    )
    op = IndexerICV(c, backend=backend)
    counts, q_arrays, q_scales, k_arrays, k_scales, q_ids, k_ids = {}, [], [], [], [], [], []
    baselines, inputs = {}, {}
    for r in range(3):
        request = 100 + 7 * r
        m, n = (0, 1, 3, 9)[(seed + r) % 4], (0, 1, 17, 65)[(seed * 3 + r) % 4]
        qv, qs = payload(c, m, q=True, seed=seed * 100 + r, device="cuda")
        kv, ks = payload(c, n, seed=seed * 100 + r + 40, device="cuda")
        qi, ki = ids(request, m, device="cuda"), ids(request, n, device="cuda")
        qb = bind_icv_queries(op, qv, qs, qi, layout=c.layout, version="v1")
        kb = bind_global_icv_keys(
            op, kv, ks, ki, layout=c.layout, version="v1", request_key_counts={request: n}
        )
        result = op.materialize(qb, kb)
        if m:
            baselines[request] = result[0].values
            inputs[request] = (qv, qs, kv, ks)
        counts[request] = n
        q_arrays.append(qv)
        q_scales.append(qs)
        q_ids.append(qi)
        k_arrays.append(kv)
        k_scales.append(ks)
        k_ids.append(ki)
    # Valid payload with invalid metadata must contribute neither rows nor columns.
    pq, pqs = payload(c, 3, q=True, seed=999, device="cuda")
    pk, pks = payload(c, 2, seed=998, device="cuda")
    q_arrays.append(pq)
    q_scales.append(pqs)
    k_arrays.append(pk)
    k_scales.append(pks)
    q_ids.append(torch.full((3, 2), -1, dtype=torch.int64, device="cuda"))
    k_ids.append(torch.full((2, 2), -1, dtype=torch.int64, device="cuda"))
    qv, qs, qi = torch.cat(q_arrays), torch.cat(q_scales), torch.cat(q_ids)
    kv, ks, ki = torch.cat(k_arrays), torch.cat(k_scales), torch.cat(k_ids)
    generator = torch.Generator().manual_seed(seed)
    qp = torch.randperm(len(qv), generator=generator).cuda()
    kp = torch.randperm(len(kv), generator=generator).cuda()
    cp_size = (1, 2, 4)[seed % 3]
    qb = bind_icv_queries(op, qv[qp].contiguous(), qs[qp], qi[qp], layout=c.layout, version="v1")
    kb = bind_global_icv_keys(
        op,
        kv[kp].contiguous(),
        ks[kp],
        ki[kp],
        layout=c.layout,
        version="v1",
        request_key_counts=counts,
        key_owners=kp % cp_size,
        cp_size=cp_size,
    )
    blocks = op.materialize(qb, kb)
    assert {b.request_id for b in blocks} == set(baselines)
    for block in blocks:
        selected = list(block.query_positions)
        byte_equal(block.values, baselines[block.request_id][selected])
        q0, s0, k0, sk0 = inputs[block.request_id]
        if len(k0):
            oracle = (
                torch.einsum("mhd,nd->mhn", decode(q0, s0, c), decode(k0, sk0, c)).float().cuda()
            )
            torch.testing.assert_close(block.values, oracle[selected], rtol=4e-6, atol=3e-4)
    for tile in op.iter_tiles(qb, kb):
        byte_equal(
            tile.values,
            baselines[tile.request_id][
                list(tile.query_positions), :, tile.key_start : tile.key_end
            ],
        )


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_two_streams_reuse_prepared_keys_without_races(backend, precision):
    op = IndexerICV(make_contract(precision), backend=backend)
    queries, keys = pair(op)
    expected = op.materialize(queries, keys)[0].values
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(current)
    outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            for _ in range(8):
                result = op.materialize(queries, keys)[0].values
            outputs.append(result)
    for stream in streams:
        current.wait_stream(stream)
    for result in outputs:
        byte_equal(result, expected)
    keys.validate()
    queries.validate()


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_repeated_execution_no_live_tensor_memory_growth(backend, precision):
    op = IndexerICV(make_contract(precision), backend=backend)
    queries, keys = pair(op, 33, 129)
    for _ in range(5):
        result = op.materialize(queries, keys)
        del result
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(50):
        result = op.materialize(queries, keys)
        del result
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after <= before + 4096, (before, after)


def test_streaming_peak_memory_stays_below_full_icv(backend, tmp_path):
    c = make_contract(
        "mxfp4",
        num_heads=64,
        query_tile_rows=32,
        key_tile_rows=256,
        max_output_bytes=32 * 64 * 256 * 4,
    )
    op = IndexerICV(c, backend=backend)
    queries, keys = pair(op, 64, 4096)
    with pytest.raises(IndexerICVError, match="use iter_tiles"):
        op.materialize(queries, keys)
    for tile in op.iter_tiles(queries, keys):
        del tile
    gc.collect()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    count = 0
    for tile in op.iter_tiles(queries, keys):
        assert tile.values.numel() * 4 <= c.max_output_bytes
        count += 1
        del tile
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - start
    assert count == 32
    full_bytes = 64 * 64 * 4096 * 4
    assert peak < full_bytes // 2, (peak, full_bytes)
    (tmp_path / "memory-evidence.json").write_text(
        json.dumps(
            {
                "stream_peak_extra_bytes": peak,
                "full_icv_bytes": full_bytes,
                "tile_count": count,
                "max_tile_output_bytes": c.max_output_bytes,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "target", ["q_data", "q_scales", "k_data", "k_scales", "prepared", "selector"]
)
def test_gpu_mutation_is_detected_before_next_numerical_use(backend, target):
    op = IndexerICV(make_contract("mxfp4"), backend=backend)
    queries, keys = pair(op)
    source = {
        "q_data": queries.operand.data,
        "q_scales": queries.operand.scales,
        "k_data": keys.operand.data,
        "k_scales": keys.operand.scales,
        "prepared": keys.requests[7].values,
        "selector": queries.requests[0].selector,
    }[target]
    source.view(torch.uint8).fill_(0)
    with pytest.raises(IndexerICVError, match="changed|mutated"):
        op.materialize(queries, keys)


def test_graph_capture_is_explicitly_rejected(backend):
    op = IndexerICV(make_contract("mxfp4"), backend=backend)
    queries, keys = pair(op)
    with pytest.raises(IndexerICVError, match="Graph"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op.materialize(queries, keys)


def test_fresh_processes_with_distinct_jit_caches_produce_identical_bytes(tmp_path):
    outputs = []
    for run in range(2):
        output = tmp_path / f"run{run}.json"
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton{run}"))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.models.deepseek_v4.icv_process_probe",
                "--mode",
                "replay",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1]


def test_real_two_rank_gloo_transport_then_gpu_icv(tmp_path):
    output = tmp_path / "gloo-result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.models.deepseek_v4.icv_process_probe",
            "--mode",
            "gloo",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(output.read_text())
    assert evidence["backend"] == "gloo_cpu_transport_then_one_gpu_icv"
    assert evidence["world_size"] == 2
    assert evidence["byte_equal"] is True


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
def test_long_batch_at_full_64_head_geometry(backend, precision):
    c = make_contract(precision, num_heads=64, head_dim=128, query_tile_rows=32, key_tile_rows=128)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 257, q=True, seed=2301, device="cuda")
    kv, ks = payload(c, 257, seed=2302, device="cuda")
    keys = bind_global_icv_keys(
        op,
        kv,
        ks,
        ids(7, 257, device="cuda"),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 257},
    )
    full_q = bind_icv_queries(op, qv, qs, ids(7, 257, device="cuda"), layout=c.layout, version="v1")
    expected = op.materialize(full_q, keys)[0].values
    for count in (1, 31, 32, 33, 127, 128, 129, 256, 257):
        query = bind_icv_queries(
            op, qv[:count], qs[:count], ids(7, count, device="cuda"), layout=c.layout, version="v1"
        )
        byte_equal(op.materialize(query, keys)[0].values, expected[:count])
    selection = torch.tensor([256, 0, 128, 31, 129], dtype=torch.int64, device="cuda")
    query = bind_icv_queries(
        op,
        qv[selection].contiguous(),
        qs[selection],
        ids(7, 257, device="cuda")[selection],
        layout=c.layout,
        version="v1",
    )
    byte_equal(op.materialize(query, keys)[0].values, expected[selection])


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
def test_equal_key_values_keep_distinct_global_columns(backend, precision):
    c = make_contract(precision)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 5, q=True, seed=2401, device="cuda")
    k_one, s_one = payload(c, 1, seed=2402, device="cuda")
    kv, ks = k_one.repeat(17, 1), s_one.repeat(17, 1)
    keys = bind_global_icv_keys(
        op,
        kv,
        ks,
        ids(7, 17, device="cuda"),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 17},
        cp_size=2,
        key_owners=torch.arange(17, dtype=torch.int64, device="cuda") % 2,
    )
    query = bind_icv_queries(op, qv, qs, ids(7, 5, device="cuda"), layout=c.layout, version="v1")
    output = op.materialize(query, keys)[0]
    assert output.key_end == 17
    for column in range(1, 17):
        byte_equal(output.values[..., :1], output.values[..., column : column + 1])


def test_wide_scale_spectrum_against_forward_error_bound_and_bi(backend):
    c = make_contract("mxfp4", num_heads=4, head_dim=128)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 9, q=True, seed=2501, device="cuda")
    kv, ks = payload(c, 17, seed=2502, device="cuda")
    exponents = torch.tensor([87, 107, 147, 167], dtype=torch.uint8, device="cuda")
    qs.copy_(exponents.expand_as(qs))
    ks.copy_(torch.tensor([127, 147, 87, 167], dtype=torch.uint8, device="cuda").expand_as(ks))
    query = bind_icv_queries(op, qv, qs, ids(7, 9, device="cuda"), layout=c.layout, version="v1")
    keys = bind_global_icv_keys(
        op,
        kv,
        ks,
        ids(7, 17, device="cuda"),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 17},
    )
    actual = op.materialize(query, keys)[0].values
    q64, k64 = decode(qv, qs, c), decode(kv, ks, c)
    expected = torch.einsum("mhd,nd->mhn", q64, k64)
    # Bound rounding error by the sum of absolute products, not relative error
    # to a cancellation-prone near-zero result.
    magnitude = torch.einsum("mhd,nd->mhn", q64.abs(), k64.abs())
    bound = 4 * c.head_dim * torch.finfo(torch.float32).eps * magnitude
    assert bool(torch.isfinite(actual).all())
    assert bool(((actual.double().cpu() - expected).abs() <= bound).all())
    for count in (1, 3, 9):
        subset = bind_icv_queries(
            op, qv[:count], qs[:count], ids(7, count, device="cuda"), layout=c.layout, version="v1"
        )
        byte_equal(op.materialize(subset, keys)[0].values, actual[:count])


@pytest.mark.parametrize("failure", ["not_rocm", "wrong_arch", "missing_library"])
def test_unvalidated_backend_environment_fails_without_fallback(monkeypatch, failure):
    import importlib
    from types import SimpleNamespace

    if failure == "not_rocm":
        monkeypatch.setattr(torch.version, "hip", None)
    elif failure == "wrong_arch":
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(gcnArchName="gfx950")
        )
    else:
        native_import = importlib.import_module

        def missing(name, *args, **kwargs):
            if name == "aiter.ops.triton.moe.quant_moe":
                raise ImportError("injected missing ABI")
            return native_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(IndexerICVError, match="ROCm|architecture|ABI"):
        AiterIndexerICVBackend()


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_default_public_constructor_selects_real_backend_and_declares_arithmetic(precision):
    c = make_contract(precision)
    # Public callers normally omit the injected backend used by most fixtures.
    op = IndexerICV(c)
    assert isinstance(op.backend, AiterIndexerICVBackend)
    assert op.backend.provenance["sources"]
    assert "dot_fp32" in c.arithmetic and "split1" in c.arithmetic
    if precision == "mxfp4":
        assert "exact_e2m1" in c.arithmetic
    else:
        assert "mul(q_scale*k_scale)_once" in c.arithmetic
    queries, keys = pair(op, 3, 5)
    output = op.materialize(queries, keys)[0]
    assert output.values.shape == (3, c.num_heads, 5)
    assert output.contract_fingerprint == c.fingerprint
    assert output.implementation_fingerprint == op.implementation_fingerprint


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
@pytest.mark.parametrize("m,n", [(0, 5), (3, 0)])
def test_empty_direct_library_boundary_never_launches_gemm(backend, monkeypatch, precision, m, n):
    c = make_contract(precision)
    qv, qs = payload(c, m, q=True, device="cuda")
    kv, ks = payload(c, n, device="cuda")
    q, qscale = backend.prepare(qv, qs, c)
    k, kscale = backend.prepare(kv, ks, c)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty numerical boundary launched GEMM")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "gemm16", forbidden)
        patch.setattr(backend, "gemm8", forbidden)
        actual = backend.dot(q, k, qscale, kscale, c)
    assert actual.shape == (m, c.num_heads, n)
    assert actual.dtype == torch.float32 and actual.numel() == 0
