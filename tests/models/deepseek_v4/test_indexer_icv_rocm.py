# SPDX-License-Identifier: Apache-2.0
"""Real-kernel ICV acceptance; quantized-byte BI and numerical accuracy are separate."""

import ast
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend
from rl_engine.contracts.operators.indexer_icv import IndexerICVContract, IndexerICVError
from rl_engine.integrations.engines.rollout.vllm.indexer_icv import (
    bind_global_icv_keys,
    bind_icv_queries,
)
from rl_engine.integrations.engines.train.megatron import indexer_icv as megatron_icv
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV
from tests.models.deepseek_v4.icv_helpers import byte_equal, decode, decode_fp4, ids, payload

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    # No skip for a missing ABI on a live ROCm validation environment.
    return AiterIndexerICVBackend()


def contract(precision, **kwargs):
    params = dict(num_heads=3, query_tile_rows=7, key_tile_rows=31)
    params.update(kwargs)
    return IndexerICVContract(
        precision="mxfp4" if precision == "mxfp4" else "fp8",
        fp8_encoding=None if precision == "mxfp4" else precision,
        **params,
    )


def bind(op, qdata, qscale, kdata, kscale, *, qids=None, kids=None, cp_size=1, owners=None):
    c = op.contract
    q = bind_icv_queries(
        op,
        qdata,
        qscale,
        ids(7, len(qdata), device="cuda") if qids is None else qids,
        layout=c.layout,
        version="model-scale-v1",
    )
    k = bind_global_icv_keys(
        op,
        kdata,
        kscale,
        ids(7, len(kdata), device="cuda") if kids is None else kids,
        layout=c.layout,
        version="model-scale-v1",
        request_key_counts={7: len(kdata)},
        cp_size=cp_size,
        key_owners=owners,
    )
    return q, k


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
def test_full_64_head_geometry_raw_icv_vs_float64(backend, precision):
    c = contract(precision, num_heads=64, head_dim=128)
    qv, qs = payload(c, 33, q=True, seed=16, device="cuda")
    kv, ks = payload(c, 137, seed=17, device="cuda")
    op = IndexerICV(c, backend=backend)
    q, k = bind(op, qv, qs, kv, ks)
    got = op.materialize(q, k)[0].values
    expected = torch.einsum("mhd,nd->mhn", decode(qv, qs, c), decode(kv, ks, c)).float().cuda()
    torch.testing.assert_close(got, expected, rtol=3e-6, atol=2e-4)
    assert got.dtype == torch.float32 and got.shape == (33, 64, 137)
    assert bool((got < 0).any()) and bool((got > 0).any())  # No hidden ReLU.
    assert not torch.equal(got, got.bfloat16().float())  # No BF16-rounded output.
    for m, n in ((1, 7), (2, 65), (7, 137), (8, 65), (17, 137), (33, 137)):
        qm, km = bind(op, qv[:m], qs[:m], kv[:n], ks[:n])
        byte_equal(op.materialize(qm, km)[0].values, got[:m, :, :n])
    # Scale application occurs exactly once. No head weights or global scale input.
    changed = qs + 1 if precision == "mxfp4" else qs * 2
    q2 = bind_icv_queries(
        op, qv, changed, ids(7, 33, device="cuda"), layout=c.layout, version="model-scale-v1"
    )
    doubled = op.materialize(q2, k)[0].values
    byte_equal(doubled, got * 2)


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
@pytest.mark.parametrize("dim", [1, 31, 32, 33, 63, 65, 127, 129, 256])
def test_reduction_tails_and_batch_key_width_changes(backend, precision, dim):
    c = contract(precision, head_dim=dim, query_tile_rows=32, key_tile_rows=256)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 65, q=True, seed=31, device="cuda")
    kv, ks = payload(c, 67, seed=32, device="cuda")
    q, k = bind(op, qv, qs, kv, ks)
    reference = op.materialize(q, k)[0].values
    for m, n in ((1, 1), (2, 7), (7, 31), (33, 65), (65, 67)):
        qt, kt = bind(op, qv[:m], qs[:m], kv[:n], ks[:n])
        byte_equal(op.materialize(qt, kt)[0].values, reference[:m, :, :n])
    expected = (
        torch.einsum(
            "mhd,nd->mhn",
            decode(qv[:1], qs[:1], c),
            decode(kv[:7], ks[:7], c),
        )
        .float()
        .cuda()
    )
    torch.testing.assert_close(reference[:1, :, :7], expected, rtol=4e-6, atol=2e-4)


@pytest.mark.parametrize("encoded_scale", [2, 3, 100, 126, 127, 128, 200, 251, 252])
def test_all_e2m1_values_decode_exactly_in_admitted_scale_range(backend, encoded_scale):
    c = contract("mxfp4", num_heads=1, head_dim=32)
    # Each byte enumerates two adjacent nibbles, including positive/negative zero.
    values = torch.tensor(
        [i | ((i + 1) << 4) for i in range(0, 16, 2)] * 2, dtype=torch.uint8, device="cuda"
    ).reshape(1, 16)
    scales = torch.full((1, 1), encoded_scale, dtype=torch.uint8, device="cuda")
    actual, _ = backend.prepare(values, scales, c)
    expected = decode_fp4(values, scales, 32).bfloat16().cuda()
    byte_equal(actual, expected)
    assert bool(torch.isfinite(actual).all())


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_global_k_cp_partition_reorder_and_missing_shard(backend, precision, cp_size):
    c = contract(precision)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 17, q=True, seed=42, device="cuda")
    kv, ks = payload(c, 71, seed=43, device="cuda")
    q, k = bind(op, qv, qs, kv, ks)
    expected = op.materialize(q, k)[0].values
    all_rows = torch.arange(71, device="cuda")
    # Materialized native-transport result: deliberately reverse rank arrival.
    permutation = torch.cat(
        [all_rows[all_rows % cp_size == rank] for rank in reversed(range(cp_size))]
    )
    rank_major_ids = ids(7, 71, device="cuda")[permutation]
    qt, kt = bind(
        op,
        qv,
        qs,
        kv[permutation].contiguous(),
        ks[permutation],
        kids=rank_major_ids,
        cp_size=cp_size,
        owners=permutation % cp_size,
    )
    byte_equal(op.materialize(qt, kt)[0].values, expected)
    with pytest.raises(IndexerICVError, match="global K"):
        bind_global_icv_keys(
            op,
            kv[permutation[:-1]].contiguous(),
            ks[permutation[:-1]],
            rank_major_ids[:-1],
            cp_size=cp_size,
            key_owners=permutation[:-1] % cp_size,
            request_key_counts={7: 71},
            layout=c.layout,
            version="model-scale-v1",
        )


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_mixed_requests_padding_reordering_and_query_chunks(backend, precision):
    c = contract(precision)
    op = IndexerICV(c, backend=backend)
    av, sa = payload(c, 33, q=True, seed=51, device="cuda")
    ak, ka = payload(c, 65, seed=52, device="cuda")
    qa, keys_a = bind(op, av, sa, ak, ka)
    expected = op.materialize(qa, keys_a)[0].values
    bv, sb = payload(c, 7, q=True, seed=53, device="cuda")
    bk, kb = payload(c, 17, seed=54, device="cuda")
    pq, ps = payload(c, 3, q=True, seed=55, device="cuda")
    pk, pks = payload(c, 3, seed=56, device="cuda")
    key_payload, key_scales = torch.cat((bk, pk, ak)), torch.cat((kb, pks, ka))
    key_ids = torch.cat(
        (
            ids(9, 17, device="cuda"),
            torch.full((3, 2), -1, dtype=torch.int64, device="cuda"),
            ids(7, 65, device="cuda"),
        )
    )
    q_payload, q_scales = torch.cat((bv, pq, av.flip(0))), torch.cat((sb, ps, sa.flip(0)))
    q_ids = torch.cat(
        (
            ids(9, 7, device="cuda"),
            torch.full((3, 2), -1, dtype=torch.int64, device="cuda"),
            ids(7, 33, device="cuda").flip(0),
        )
    )
    keys = bind_global_icv_keys(
        op,
        key_payload,
        key_scales,
        key_ids,
        layout=c.layout,
        version="model-scale-v1",
        request_key_counts={7: 65, 9: 17},
        cp_size=2,
        key_owners=torch.arange(len(key_ids), device="cuda") % 2,
    )
    queries = bind_icv_queries(
        op, q_payload, q_scales, q_ids, layout=c.layout, version="model-scale-v1"
    )
    blocks = {b.request_id: b for b in op.materialize(queries, keys)}
    byte_equal(blocks[7].values, expected.flip(0))
    assert blocks[9].values.shape == (7, 3, 17)
    for begin, end in ((0, 1), (1, 5), (5, 17), (17, 33)):
        query = bind_icv_queries(
            op,
            av[begin:end],
            sa[begin:end],
            ids(7, end - begin, start=begin, device="cuda"),
            layout=c.layout,
            version="model-scale-v1",
        )
        byte_equal(op.materialize(query, keys)[0].values, expected[begin:end])


def test_streaming_budget_and_plan_do_not_change_arithmetic(backend):
    c = contract(
        "mxfp4", num_heads=64, query_tile_rows=3, key_tile_rows=7, max_output_bytes=3 * 64 * 7 * 4
    )
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 9, q=True, seed=61, device="cuda")
    kv, ks = payload(c, 17, seed=62, device="cuda")
    q, k = bind(op, qv, qs, kv, ks)
    with pytest.raises(IndexerICVError, match="use iter_tiles"):
        op.materialize(q, k)
    expected = torch.einsum("mhd,nd->mhn", decode(qv, qs, c), decode(kv, ks, c)).float().cuda()
    count = 0
    for tile in op.iter_tiles(q, k):
        count += 1
        torch.testing.assert_close(
            tile.values,
            expected[list(tile.query_physical_rows), :, tile.key_start : tile.key_end],
            rtol=3e-6,
            atol=2e-4,
        )
        assert tile.values.numel() * 4 <= c.max_output_bytes
    assert count == 9
    wider = replace(c, query_tile_rows=9, key_tile_rows=17, max_output_bytes=1 << 20)
    op2 = IndexerICV(wider, backend=backend)
    q2, k2 = bind(op2, qv, qs, kv, ks)
    full = op2.materialize(q2, k2)[0].values
    for tile in op.iter_tiles(q, k):
        byte_equal(
            tile.values, full[list(tile.query_physical_rows), :, tile.key_start : tile.key_end]
        )


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
@pytest.mark.parametrize("m,n", [(0, 0), (0, 5), (3, 0)])
def test_empty_inputs(backend, precision, m, n):
    c = contract(precision)
    op = IndexerICV(c, backend=backend)
    q, k = bind(op, *payload(c, m, q=True, device="cuda"), *payload(c, n, device="cuda"))
    blocks = op.materialize(q, k)
    assert list(op.iter_tiles(q, k)) == []
    if m:
        assert blocks[0].values.shape == (m, c.num_heads, n)
    else:
        assert blocks == ()


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_vllm_megatron_forward_adapters_match_without_hidden_postprocess(backend, precision):
    c = contract(precision)
    op = IndexerICV(c, backend=backend)
    mirror = IndexerICV(c, backend=backend, expected_implementation=op.implementation_fingerprint)
    qv, qs = payload(c, 9, q=True, seed=71, device="cuda")
    kv, ks = payload(c, 11, seed=72, device="cuda")
    q, k = bind(op, qv, qs, kv, ks)
    qm = megatron_icv.bind_icv_queries(
        mirror,
        qv,
        qs,
        ids(7, 9, device="cuda"),
        layout=c.layout,
        version="model-scale-v1",
    )
    km = megatron_icv.bind_global_icv_keys(
        mirror,
        kv,
        ks,
        ids(7, 11, device="cuda"),
        layout=c.layout,
        version="model-scale-v1",
        request_key_counts={7: 11},
    )
    byte_equal(mirror.materialize(qm, km)[0].values, op.materialize(q, k)[0].values)
    with pytest.raises(IndexerICVError, match="independent Q scales"):
        megatron_icv.bind_icv_queries(
            mirror,
            qv,
            None,
            ids(7, 9, device="cuda"),
            layout=c.layout,
            version="v1",
        )


def test_against_pinned_aiter_float_scoring_reference(backend):
    path = os.environ.get("RLK_AITER_INDEXER_REFERENCE")
    if not path:
        pytest.skip("set RLK_AITER_INDEXER_REFERENCE for native floating reference")
    tree = ast.parse(Path(path).read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_causal_mask", "indexer_fwd")
    ]
    assert len(functions) == 2
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    c = contract("mxfp4", num_heads=4)
    op = IndexerICV(c, backend=backend)
    qv, qs = payload(c, 33, q=True, seed=81, device="cuda")
    kv, ks = payload(c, 9, seed=82, device="cuda")
    q, k = bind(op, qv, qs, kv, ks)
    raw = op.materialize(q, k)[0].values
    qf, kf = decode(qv, qs, c).bfloat16().cuda(), decode(kv, ks, c).bfloat16().cuda()
    weights = torch.randn((33, 4), dtype=torch.float32, device="cuda")
    native = namespace["indexer_fwd"](qf, kf, weights, 4)
    # Test-only postprocess. Production item 16 never applies these operations.
    composed = (raw.relu() * weights[..., None]).sum(dim=1)
    mask = namespace["_causal_mask"](33, 9, 4, qv.device)
    composed = torch.where(mask, composed, -torch.inf)
    torch.testing.assert_close(composed, native, rtol=3e-6, atol=0.002)
    assert raw.shape == (33, 4, 9)


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fn", "e4m3fnuz"])
def test_no_host_readback_in_compute_and_reuses_actual_library_functions(
    backend, monkeypatch, precision
):
    from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale

    assert backend.upcast is upcast_from_mxfp
    assert backend.gemm16 is gemm_a16w16
    assert backend.gemm8 is gemm_a8w8_per_token_scale
    c = contract(precision)
    op = IndexerICV(c, backend=backend)
    q, k = bind(op, *payload(c, 3, q=True, device="cuda"), *payload(c, 5, device="cuda"))
    expected = op.materialize(q, k)[0].values

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU readback in numerical ICV")

    with monkeypatch.context() as patch:
        for method in ("cpu", "item", "tolist", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        actual = op.materialize(q, k)[0].values
    byte_equal(actual, expected)


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_accepts_outputs_of_existing_library_quantizers(backend, precision):
    c = contract(precision, num_heads=4, head_dim=128)
    torch.manual_seed(1616)
    q_source = torch.randn((9, 4, 128), dtype=torch.bfloat16, device="cuda")
    k_source = torch.randn((25, 128), dtype=torch.bfloat16, device="cuda")
    if precision == "mxfp4":
        from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

        qv, qs = downcast_to_mxfp(q_source, torch.uint8, axis=-1)
        kv, ks = downcast_to_mxfp(k_source, torch.uint8, axis=-1)
    else:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        qv, qs = per_token_group_quant_fp8(
            q_source.flatten(0, 1),
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
            column_major_scales=False,
        )
        qv, qs = qv.view(9, 4, 128), qs.view(9, 4, 1)
        kv, ks = per_token_group_quant_fp8(
            k_source,
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
            column_major_scales=False,
        )
    op = IndexerICV(c, backend=backend)
    q, k = bind(op, qv, qs, kv, ks)
    actual = op.materialize(q, k)[0].values
    expected = torch.einsum("mhd,nd->mhn", decode(qv, qs, c), decode(kv, ks, c)).float().cuda()
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-4)
    for m in (1, 3, 9):
        small, _ = bind(op, qv[:m], qs[:m], kv, ks)
        byte_equal(op.materialize(small, k)[0].values, actual[:m])


@pytest.mark.parametrize("precision", ["mxfp4", "e4m3fnuz"])
def test_native_quantized_zero_rows_are_not_rejected_as_underflow_scales(backend, precision):
    c = contract(precision, num_heads=4)
    q_source = torch.zeros((3, 4, 128), dtype=torch.bfloat16, device="cuda")
    k_source = torch.zeros((5, 128), dtype=torch.bfloat16, device="cuda")
    if precision == "mxfp4":
        from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

        qv, qs = downcast_to_mxfp(q_source, torch.uint8, axis=-1)
        kv, ks = downcast_to_mxfp(k_source, torch.uint8, axis=-1)
    else:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        qv, qs = per_token_group_quant_fp8(
            q_source.flatten(0, 1),
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
        )
        qv, qs = qv.view(3, 4, 128), qs.view(3, 4, 1)
        kv, ks = per_token_group_quant_fp8(
            k_source,
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
        )
    op = IndexerICV(c, backend=backend)
    q, k = bind(op, qv, qs, kv, ks)
    actual = op.materialize(q, k)[0].values
    byte_equal(actual, torch.zeros_like(actual))
