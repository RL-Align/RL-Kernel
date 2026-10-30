# SPDX-License-Identifier: Apache-2.0

"""MI300X acceptance: real upstream kernels, byte equality, not tolerance checks."""

import ast
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from rl_engine.backends.rocm.attention.recent128_cache import VllmRecent128Backend
from rl_engine.contracts.operators.recent128 import (
    CacheRequest,
    CacheVersion,
    Recent128Contract,
)
from rl_engine.integrations.engines.rollout.vllm.recent128 import (
    bind_recent128_metadata,
    recent128_cache_update,
)
from rl_engine.models.deepseek_v4.recent128 import Recent128Cache, Recent128Workspace
from tests.models.deepseek_v4.recent128_helpers import (
    assert_bytes_equal,
    canonical_cache_bytes,
    metadata,
)

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(),
        reason="ROCm GPU required",
    ),
]
VERSION = CacheVersion("flash-0731", "adapter-0", "scale-0")


@pytest.fixture(scope="module")
def native():
    # A missing installed ABI is a FAILURE on a ROCm acceptance run, not a skip.
    from vllm.models.deepseek_v4.common.ops import cache_utils

    return cache_utils


@pytest.fixture(scope="module")
def backend():
    return VllmRecent128Backend()


def rows(n, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, 512), generator=generator, dtype=torch.bfloat16).cuda()


def bind(meta, requests, contract):
    return bind_recent128_metadata(
        meta,
        requests=tuple(requests),
        version=VERSION,
        contract=contract,
    )


def full_native_cache(native, contract, sequences, tables, pages):
    cache = torch.full((pages, contract.page_bytes), 0xA5, dtype=torch.uint8, device="cuda")
    lengths = [x.shape[0] for x in sequences]
    meta = metadata(lengths, lengths, tables=tables, block_size=contract.block_size, device="cuda")
    native.quantize_and_insert_k_cache(
        torch.cat(sequences),
        cache,
        meta.slot_mapping,
        block_size=contract.block_size,
        use_fnuz=contract.use_fnuz,
    )
    visible = torch.empty((len(lengths), max(lengths), 512), dtype=torch.bfloat16, device="cuda")
    native.dequantize_and_gather_k_cache_triton(
        visible,
        cache,
        meta.seq_lens,
        None,
        meta.block_table,
        contract.block_size,
        0,
        use_fnuz=contract.use_fnuz,
    )
    return cache, visible


def check_view(view, full_visible, prefixes, query_counts):
    batch, contract = view.batch, view.batch.contract
    actual_indices = view.indices.cpu()
    actual_lengths = view.lengths.cpu()
    cursor = 0
    for row, (prefix, count) in enumerate(zip(prefixes, query_counts, strict=False)):
        for p in range(prefix, prefix + count):
            start = max(0, p - 127)
            count_visible = p - start + 1
            first = row * contract.workspace_width + start - batch.gather_starts[row]
            expected = torch.arange(first, first + count_visible, dtype=torch.int32)
            assert torch.equal(actual_indices[cursor, :count_visible], expected)
            assert (actual_indices[cursor, count_visible:] == -1).all()
            assert actual_lengths[cursor].item() == count_visible
            gathered = view.kv.reshape(-1, 512).index_select(
                0,
                view.indices[cursor, :count_visible].long(),
            )
            assert_bytes_equal(gathered, full_visible[row, start : p + 1])
            cursor += 1
    assert (actual_indices[cursor:] == -1).all()
    assert (actual_lengths[cursor:] == 0).all()


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("encoding", ["e4m3fn", "e4m3fnuz"])
def test_real_writer_gather_indices_chunked_prefill_and_decode(
    native, backend, page_size, encoding
):
    contract = Recent128Contract(260, block_size=page_size, encoding=encoding)
    lengths = [260, 137]
    sequences = [rows(length, seed) for length, seed in zip(lengths, (17, 21), strict=False)]
    counts = [(n + page_size - 1) // page_size for n in lengths]
    # Deliberately scrambled, disjoint physical pages, plus untouched canary pages.
    permutation = list(reversed(range(sum(counts))))
    tables = [permutation[: counts[0]], permutation[counts[0] :]]
    pages = sum(counts) + 2
    expected_cache, visible = full_native_cache(native, contract, sequences, tables, pages)
    cache = torch.full_like(expected_cache, 0xA5)
    session = Recent128Cache(cache, contract, layer_id="layer-12", backend=backend)
    requests = [CacheRequest("A"), CacheRequest("B")]
    for request in requests:
        session.register(request, VERSION)
    prefixes = [0, 0]
    for chunk in (127, 1, 2, 129, 1):
        query = [min(chunk, total - p) for total, p in zip(lengths, prefixes, strict=False)]
        ends = [p + q for p, q in zip(prefixes, query, strict=False)]
        meta = metadata(
            ends,
            query,
            tables=tables,
            block_size=page_size,
            padding=3,
            device="cuda",
        )
        # Exercise the actual vLLM dataclass, not only a duck-typed fixture.
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

        swa = DeepseekSparseSWAMetadata(**vars(meta))
        batch = bind(swa, requests, contract)
        inputs = torch.cat(
            [
                sequence[p : p + q]
                for sequence, p, q in zip(sequences, prefixes, query, strict=False)
            ]
            + [torch.full((3, 512), float("nan"), dtype=torch.bfloat16, device="cuda")]
        )
        workspace = Recent128Workspace.allocate(batch)
        view = recent128_cache_update(session, inputs, batch, workspace=workspace)
        check_view(view, visible, prefixes, query)
        snapshot = cache.clone()
        replay = session.replay(batch, workspace=workspace)
        check_view(replay, visible, prefixes, query)
        assert_bytes_equal(cache, snapshot)
        prefixes = ends
    assert prefixes == lengths
    # Covers token data, all scales, scale padding and untouched page bytes.
    assert_bytes_equal(cache, expected_cache)


def _run_schedule(backend, native, *, with_neighbor, reverse, chunks):
    contract = Recent128Contract(257)
    a, neighbor = rows(257, 1234), rows(513, 42)
    sequences = [a, neighbor] if with_neighbor else [a]
    if reverse:
        sequences.reverse()
    request_names = ["A", "B"] if with_neighbor else ["A"]
    if reverse:
        request_names.reverse()
    requests = [CacheRequest(name) for name in request_names]
    lengths = [len(sequence) for sequence in sequences]
    n_pages = sum((n + 63) // 64 for n in lengths)
    # Physical allocation changes in every scenario independently of logical order.
    available = list(reversed(range(n_pages)))
    tables, cursor = [], 0
    for length in lengths:
        n = (length + 63) // 64
        tables.append(available[cursor : cursor + n])
        cursor += n
    expected, visible = full_native_cache(native, contract, sequences, tables, n_pages)
    cache = torch.full_like(expected, 0xA5)
    session = Recent128Cache(cache, contract, layer_id="layer-0", backend=backend)
    for request in requests:
        session.register(request, VERSION)
    prefixes = [0] * len(requests)
    last_a = None
    schedule = list(chunks)
    while any(p < length for p, length in zip(prefixes, lengths, strict=False)):
        chunk = schedule.pop(0) if schedule else 257
        query = [min(chunk, length - p) for length, p in zip(lengths, prefixes, strict=False)]
        ends = [p + q for p, q in zip(prefixes, query, strict=False)]
        batch = bind(
            metadata(ends, query, tables=tables, padding=2, device="cuda"),
            requests,
            contract,
        )
        inputs = torch.cat(
            [
                sequence[p : p + q]
                for sequence, p, q in zip(sequences, prefixes, query, strict=False)
            ]
            + [torch.zeros((2, 512), dtype=torch.bfloat16, device="cuda")]
        )
        view = session.update(inputs, batch)
        check_view(view, visible, prefixes, query)
        a_row = request_names.index("A")
        if query[a_row]:
            a_token = sum(query[: a_row + 1]) - 1
            valid = int(view.lengths[a_token].item())
            last_a = (
                view.kv.reshape(-1, 512)
                .index_select(
                    0,
                    view.indices[a_token, :valid].long(),
                )
                .clone()
            )
        prefixes = ends
    a_row = request_names.index("A")
    return canonical_cache_bytes(cache, tables[a_row], 257, 64), last_a


@pytest.mark.parametrize(
    "neighbor,reverse,chunks",
    [
        (False, False, [1] * 257),
        (True, False, [3, 4, 63, 64, 1, 128]),
        (True, True, [127, 2, 128]),
        (True, True, [257]),
    ],
)
def test_batch_reorder_padding_chunk_and_page_invariance(
    backend, native, neighbor, reverse, chunks
):
    baseline = _run_schedule(backend, native, with_neighbor=False, reverse=False, chunks=[257])
    actual = _run_schedule(
        backend,
        native,
        with_neighbor=neighbor,
        reverse=reverse,
        chunks=chunks,
    )
    for got, expected in zip(actual, baseline, strict=False):
        assert_bytes_equal(got, expected)


def test_empty_update_and_page_reuse(native, backend):
    contract = Recent128Contract(129)
    cache = torch.full((4, contract.page_bytes), 0xA5, dtype=torch.uint8, device="cuda")
    session = Recent128Cache(cache, contract, layer_id="layer-0", backend=backend)
    request = CacheRequest("reuse")
    session.register(request, VERSION)
    empty = bind(metadata([0], [0], padding=2, device="cuda"), [request], contract)
    before = cache.clone()
    view = session.update(torch.empty((2, 512), dtype=torch.bfloat16, device="cuda"), empty)
    assert (view.lengths == 0).all()
    assert_bytes_equal(cache, before)
    old = rows(129, 12)
    batch = bind(metadata([129], [129], device="cuda"), [request], contract)
    session.update(old, batch)
    session.release(request)
    new_request = CacheRequest("reuse", 1)
    session.register(new_request, VERSION)
    fresh = rows(3, 55)
    batch = bind(metadata([3], [3], device="cuda"), [new_request], contract)
    view = session.update(fresh, batch)
    _, visible = full_native_cache(native, contract, [fresh], [[0]], 4)
    check_view(view, visible, [0], [3])


def test_hot_path_has_no_cpu_readback(backend, monkeypatch):
    contract = Recent128Contract(4)
    request = CacheRequest("hot")
    batch = bind(metadata([4], [4], device="cuda"), [request], contract)
    cache = torch.empty((1, contract.page_bytes), dtype=torch.uint8, device="cuda")
    session = Recent128Cache(cache, contract, layer_id="layer-0", backend=backend)
    session.register(request, VERSION)
    workspace = Recent128Workspace.allocate(batch)
    inputs = rows(4, 88)

    def forbidden(*args, **kwargs):
        raise AssertionError("host readback in hot path")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        session.update(inputs, batch, workspace=workspace)
        session.replay(batch, workspace=workspace)
    torch.cuda.synchronize()


def _load_native_visible_source():
    """Execute the 3 unmodified upstream helpers without importing NVIDIA-only CSA.

    This tests the actual pinned function bodies, not full Megatron integration.
    CI may supply the pinned source explicitly; a missing source is a visible skip.
    """
    path = os.environ.get("RLK_MEGATRON_DSV4_RUNTIME_SOURCE")
    if not path:
        pytest.skip("set RLK_MEGATRON_DSV4_RUNTIME_SOURCE for pinned native-helper parity")
    source = ast.parse(Path(path).read_text())
    names = {"_packed_cache", "_dequantize_packed_cache", "quantized_main_k_visible"}
    functions = [
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = types.ModuleType("megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime")
    module.__dict__["torch"] = torch
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("length", [0, 129, 257])
def test_megatron_native_visible_k_and_gradient_parity(native, backend, monkeypatch, length):
    from rl_engine.integrations.engines.train.megatron.recent128 import recent128_visible_k

    module = _load_native_visible_source()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    contract = Recent128Contract(max(length, 1))
    source = rows(length, 901).requires_grad_(True)
    direct = module.quantized_main_k_visible(source)
    got = recent128_visible_k(source, contract=contract)
    assert_bytes_equal(got, direct)
    gradient = torch.randn_like(source)
    expected_grad = torch.autograd.grad(direct, source, gradient, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(got, source, gradient)[0]
    assert_bytes_equal(actual_grad, expected_grad)
    if length:
        # The common raw cache representation is shared with rollout; native
        # helper owns its visible-value/VJP expression (including signed zeros).
        cache, visible = full_native_cache(
            native,
            contract,
            [source.detach()],
            [list(range((length + 63) // 64))],
            (length + 63) // 64,
        )
        assert_bytes_equal(got, visible[0] + (source - source.detach()))
        assert cache.dtype == torch.uint8
