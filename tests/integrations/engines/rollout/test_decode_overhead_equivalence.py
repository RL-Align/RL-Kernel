# SPDX-License-Identifier: Apache-2.0
"""Bitwise equivalence of the host-overhead-free decode/MoE paths with their references.

CPU tests stub only the cache dequant kernel (same stub for both builders); the GPU tests
use the real vLLM dequant on random cache bytes and the ROCm combine kernel.
"""

import pytest
import torch

native_decode = pytest.importorskip("rl_engine.integrations.engines.rollout.vllm.native_decode")
routed_combine = pytest.importorskip("rl_engine.integrations.engines.rollout.vllm.routed_combine")

gpu = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)

SWA_BLOCK, SWA_BLOCKS = 8, 64
TOP_BLOCK, TOP_TABLE, TOP_BLOCKS = 16, 32, 256  # 512 logical compressed positions per request


def _cache(num_blocks, block, g, device):
    page_bytes = ((block * 584 + 575) // 576) * 576
    storage = torch.randint(0, 256, (num_blocks, page_bytes), generator=g, dtype=torch.uint8)
    return storage.to(device).as_strided((num_blocks, block, 584), (page_bytes, 584, 1))


def _fake_dequant(out, flat, lengths, table, block, fnuz):
    pages = table[:, 0].long()
    j = torch.arange(block, device=out.device)
    d = torch.arange(512, device=out.device)
    vals = pages[:, None, None] * 131 + j[None, :, None] * 7 + d[None, None, :] + int(fnuz) * 3
    out.copy_((vals % 251).to(out.dtype))


def _ragged(rows, lens, capacity_extra, g, device):
    flat = [r for r in rows]
    ptr = torch.zeros(len(rows) + 1, dtype=torch.int32)
    ptr[1:] = torch.cumsum(torch.tensor([len(r) for r in rows], dtype=torch.int64), 0)
    parts = [torch.tensor(r, dtype=torch.int32) for r in flat]
    parts = parts or [torch.empty(0, dtype=torch.int32)]
    body = torch.cat(parts)
    junk = torch.randint(0, 1000, (capacity_extra,), generator=g, dtype=torch.int32)
    return torch.cat((body, junk)).to(device), ptr.to(device)


def _dense(rows, width, g, device):
    out = torch.full((len(rows), width), -1, dtype=torch.int32)
    for i, r in enumerate(rows):
        # Valid entries at random columns (padding may be interleaved) keep original order.
        cols = torch.randperm(width, generator=g)[: len(r)].sort().values
        if len(r):
            out[i, cols] = torch.tensor(r, dtype=torch.int32)
    return out.to(device)


def _metadata(batch, seed, device, swa_only=False, swa_ragged=True, top_ragged=True, empty=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    swa_rows, top_rows, tables = [], [], []
    for _ in range(batch):
        n_swa = 0 if empty else int(torch.randint(0, 129, (1,), generator=g))
        swa_rows.append(torch.randint(0, SWA_BLOCKS * SWA_BLOCK, (n_swa,), generator=g).tolist())
        table = torch.randperm(TOP_BLOCKS, generator=g)[:TOP_TABLE]
        tables.append(table)
        n_top = 0 if (empty or swa_only) else int(torch.randint(0, 481, (1,), generator=g))
        logical = torch.randperm(TOP_TABLE * TOP_BLOCK, generator=g)[:n_top]
        ids = table[logical // TOP_BLOCK] * TOP_BLOCK + logical % TOP_BLOCK
        top_rows.append(ids.tolist())  # unsorted: the builder must reorder by logical position
    kwargs = dict(
        swa_only=swa_only,
        swa_k_cache=_cache(SWA_BLOCKS, SWA_BLOCK, g, device),
        kv_cache=None if swa_only else _cache(TOP_BLOCKS, TOP_BLOCK, g, device),
        swa_lens=torch.tensor([len(r) for r in swa_rows], dtype=torch.int32, device=device),
        topk_lens=torch.tensor([len(r) for r in top_rows], dtype=torch.int32, device=device),
        q=torch.empty(batch, 1, device=device),
    )
    if swa_ragged:
        kwargs["swa_ragged_indices"], kwargs["swa_ragged_indptr"] = _ragged(
            swa_rows, None, 37, g, device
        )
    else:
        kwargs["swa_indices"] = _dense(swa_rows, 160, g, device)
    if top_ragged:
        kwargs["topk_ragged_indices"], kwargs["topk_ragged_indptr"] = _ragged(
            top_rows, None, 11, g, device
        )
    else:
        kwargs["topk_indices"] = _dense(top_rows, 512, g, device)
    block_table = torch.stack(tables).to(torch.int32).to(device)
    return kwargs, block_table


def _compare(kwargs, block_table, device, batch, fnuz):
    ref_kv, ref_idx = native_decode._build_operands_reference(
        kwargs, block_table, device, batch, fnuz
    )
    new_kv, new_idx = native_decode._build_operands(kwargs, block_table, device, batch, fnuz)
    assert new_kv.shape == ref_kv.shape and new_kv.dtype == ref_kv.dtype
    assert new_idx.shape == (batch, 640) and new_idx.dtype == ref_idx.dtype == torch.int32
    assert torch.equal(new_idx, ref_idx)
    # Bit-level compare (real dequant of random bytes may contain NaN).
    assert torch.equal(new_kv.view(torch.int16), ref_kv.view(torch.int16))
    assert new_kv.is_contiguous()


@pytest.mark.parametrize("batch", [1, 7, 128])
@pytest.mark.parametrize("swa_ragged", [True, False])
@pytest.mark.parametrize("top_ragged", [True, False])
@pytest.mark.parametrize("fnuz", [False, True])
def test_operands_match_reference_cpu(monkeypatch, batch, swa_ragged, top_ragged, fnuz):
    monkeypatch.setattr(native_decode, "_dequant", _fake_dequant)
    kwargs, table = _metadata(batch, 1234 + batch, "cpu", False, swa_ragged, top_ragged)
    _compare(kwargs, table, "cpu", batch, fnuz)


@pytest.mark.parametrize("batch", [1, 7, 128])
def test_operands_swa_only_and_sink_only_cpu(monkeypatch, batch):
    monkeypatch.setattr(native_decode, "_dequant", _fake_dequant)
    kwargs, table = _metadata(batch, 99, "cpu", swa_only=True)
    _compare(kwargs, table, "cpu", batch, False)
    kwargs, table = _metadata(batch, 100, "cpu", empty=True)
    _compare(kwargs, table, "cpu", batch, False)
    kv, idx = native_decode._build_operands(kwargs, table, "cpu", batch, False)
    assert kv.shape == (1, 512) and not kv.any() and bool((idx == -1).all())


@pytest.mark.parametrize("batch", [1, 7])
def test_warmup_without_metadata_cpu(monkeypatch, batch):
    # vLLM warm-up/profile runs pass attn_metadata=None; the original loop only touched
    # block_table for real top-k rows, so no-top-k batches must build without it.
    monkeypatch.setattr(native_decode, "_dequant", _fake_dequant)
    for kwargs, _ in (_metadata(batch, 101, "cpu", swa_only=True), _metadata(batch, 102, "cpu", empty=True)):
        _compare(kwargs, None, "cpu", batch, False)
    kwargs, _ = _metadata(batch, 103, "cpu")
    with pytest.raises(ValueError, match="block table"):
        native_decode._build_operands(kwargs, None, "cpu", batch, False)


def test_operand_errors_match_reference_cpu(monkeypatch):
    monkeypatch.setattr(native_decode, "_dequant", _fake_dequant)
    kwargs, table = _metadata(7, 5, "cpu")
    kwargs["swa_lens"] = kwargs["swa_lens"] + 1  # recorded lengths disagree with indptr
    for build in (native_decode._build_operands_reference, native_decode._build_operands):
        with pytest.raises(ValueError, match="disagrees"):
            build(kwargs, table, "cpu", 7, False)
    kwargs, table = _metadata(3, 6, "cpu", swa_ragged=False)
    kwargs["swa_indices"] = torch.arange(3 * 700, dtype=torch.int32).reshape(3, 700) % 512
    kwargs["swa_lens"] = torch.full((3,), 700, dtype=torch.int32)
    for build in (native_decode._build_operands_reference, native_decode._build_operands):
        with pytest.raises(ValueError, match="exceeds"):
            build(kwargs, table, "cpu", 3, False)


@gpu
@pytest.mark.parametrize("batch", [1, 7, 128])
@pytest.mark.parametrize("fnuz", [False, True])
def test_operands_match_reference_gpu_real_dequant(batch, fnuz):
    pytest.importorskip("vllm.models.deepseek_v4.common.ops.cache_utils")
    kwargs, table = _metadata(batch, 4321 + batch, "cuda")
    _compare(kwargs, table, torch.device("cuda"), batch, fnuz)


# ---- MoE fixed combine fast path -------------------------------------------------------


def _routing(tokens, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(256, generator=g)[:6] for _ in range(tokens)])
    return ids.to(torch.int32)


@pytest.mark.parametrize("tokens", [1, 7, 128, 2048])
def test_fast_rows_match_prepare_fixed_output_combine_cpu(tokens):
    from rl_engine.contracts.operators.fixed_output_combine import prepare_fixed_output_combine
    from rl_engine.contracts.operators.moe import MoEOutputCombineInputs, MoEOutputCombinePlan

    ids = _routing(tokens, tokens)
    toks, experts, valid = routed_combine.routing_row_metadata(ids)
    assert bool(valid.all())
    hidden = 64
    y = torch.randn(tokens * 6, hidden).to(torch.bfloat16)
    order = (toks * 256 + experts).argsort(stable=True)
    slots = torch.empty_like(toks)
    slots[order] = torch.arange(toks.numel(), dtype=torch.int64) % 6
    plan = MoEOutputCombinePlan(6, hidden, expert_weight_applied=True)
    inputs = MoEOutputCombineInputs(
        y, toks, experts, slots, mapping_run_id="ref", num_tokens=tokens
    )
    ref = prepare_fixed_output_combine(plan, inputs, expected_mapping_run_id="ref", num_experts=256)

    routed_combine._FAST["count"].clear()
    first = routed_combine._fast_prepared(y, toks, experts, tokens, "run-1")  # full validation
    fast = routed_combine._fast_prepared(y, toks, experts, tokens, "run-2")  # trusted path
    assert fast.provenance.get("fast_path") is True
    for prepared in (first, fast):
        rows = prepared.canonical_rows()
        assert rows.dtype == torch.int64 and rows.shape == (tokens, 6)
        assert torch.equal(rows, ref.canonical_rows())
        assert torch.equal(prepared._tokens, ref._tokens)
    assert fast.num_tokens == tokens and fast.mapping_run_id == "run-2"
    # Execution-side metadata checks accept the trusted object.
    fast.validate_execution(y, None, "run-2")


def test_fast_rows_sampling_cadence_cpu():
    tokens, hidden = 7, 64
    ids = _routing(tokens, 3)
    toks, experts, _ = routed_combine.routing_row_metadata(ids)
    y = torch.randn(tokens * 6, hidden).to(torch.bfloat16)
    routed_combine._FAST["count"].clear()
    kinds = [
        routed_combine._fast_prepared(y, toks, experts, tokens, f"r{i}").provenance.get("fast_path")
        for i in range(2 * routed_combine._VALIDATE_EVERY + 1)
    ]
    full = [i for i, k in enumerate(kinds) if k is None]
    assert full == [0, 256, 512]


@gpu
@pytest.mark.parametrize("tokens", [1, 7, 128, 2048])
def test_fast_combine_output_matches_reference_gpu(tokens):
    from rl_engine.integrations.engines.train.megatron.native_combine import combine_rows
    from rl_engine.ops.moe.output_combine import output_combine

    ids = _routing(tokens, 10 + tokens).cuda()
    toks, experts, valid = routed_combine.routing_row_metadata(ids)
    y = torch.randn(tokens * 6, 256, device="cuda").to(torch.bfloat16)
    ref = combine_rows(y, toks, experts, tokens, "ref-run")
    routed_combine._FAST["count"].clear()
    routed_combine._fast_prepared(y, toks, experts, tokens, "warm")  # consume validated call
    out = torch.empty_like(ref)
    routed_combine.fixed_combine_rows(y, toks, experts, valid, tokens, "fast-run", out, True)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    prepared = routed_combine._fast_prepared(y, toks, experts, tokens, "fast-run-2")
    direct = output_combine(prepared, y, mapping_run_id="fast-run-2")
    assert torch.equal(direct.view(torch.int16), ref.view(torch.int16))
