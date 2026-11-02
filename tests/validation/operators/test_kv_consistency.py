# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe tests for WS1 C6/C7 decode-prefill and stateful KV."""

from __future__ import annotations

import pytest
import torch

from rl_engine.config.workload import load_manifest
from rl_engine.contracts.numerical import load_contract
from rl_engine.reference.attention.kv_cache import NativeKVCacheAttnOp
from rl_engine.reference.attention.stateful_kv import StatefulKVCache
from rl_engine.validation.operators.kv_consistency import (
    B2_PRODUCTION_KV_STATUS,
    DecodePrefillCase,
    assert_decode_prefill_consistent,
    assert_stateful_kv_consistent,
    build_decode_prefill_cases,
    resolve_attention_candidate,
)


@pytest.fixture()
def contract():
    return load_contract()


@pytest.fixture()
def manifest():
    return load_manifest()


def test_c6_cases_all_include_direct_decode(manifest):
    cases = build_decode_prefill_cases(manifest)
    assert len(cases) >= 6
    assert all(case.include_direct_decode for case in cases)
    ids = {case.case_id for case in cases}
    assert "decode-b1-short" in ids
    assert "decode-b1-long" in ids
    assert "decode-bn-varlen" in ids
    assert "decode-bn-padded-right" in ids
    assert "decode-bn-padded-left" in ids


def test_c6_gold_decode_matches_prefill_on_cpu(contract, manifest):
    cases = (
        DecodePrefillCase(
            case_id="cpu-b1-short",
            batch=1,
            seq_lens=(8,),
            pad_side=None,
            fixture_id="short_full_model_seq8",
        ),
    )
    report = assert_decode_prefill_consistent(
        backend_profile="cuda_bf16",
        candidate="pytorch",
        contract=contract,
        manifest=manifest,
        device="cpu",
        cases=cases,
        require_declared_candidate=False,
    )
    assert report.passed
    assert report.fallback_reason is None
    cell = report.cells[0]
    assert cell.attention_compare.passed
    assert cell.logprob_verdict.passed
    assert cell.stored_kv_layout == "[B, Hkv, S, D]"
    names = {item.metric for item in cell.logprob_verdict.metrics}
    assert names == {"max_abs_dlogp", "approx_kl0", "clipfrac0"}


def test_c6_rejects_missing_direct_decode_flag(contract, manifest):
    cases = (
        DecodePrefillCase(
            case_id="bad",
            batch=1,
            seq_lens=(8,),
            pad_side=None,
            fixture_id="short",
            include_direct_decode=False,
        ),
    )
    with pytest.raises(RuntimeError, match="direct decode"):
        assert_decode_prefill_consistent(
            backend_profile="cuda_bf16",
            candidate="pytorch",
            contract=contract,
            manifest=manifest,
            device="cpu",
            cases=cases,
            require_declared_candidate=False,
        )


def test_c6_profile_candidate_family(manifest):
    cuda = resolve_attention_candidate("cuda_bf16", manifest=manifest)
    triton = resolve_attention_candidate("triton_cuda_bf16", manifest=manifest)
    assert cuda["candidate"] == "cuda"
    assert triton["candidate"] == "triton"
    with pytest.raises(RuntimeError, match="requires 'cuda'"):
        resolve_attention_candidate("cuda_bf16", candidate="triton", manifest=manifest)


def test_c7_stateful_cache_is_not_concat():
    cache = StatefulKVCache.allocate(
        n_layers=1,
        batch=2,
        n_kv_heads=8,
        max_seq_len=8,
        head_dim=128,
        dtype=torch.float32,
        device="cpu",
    )
    k = torch.randn(2, 8, 3, 128)
    v = torch.randn(2, 8, 3, 128)
    cache.write(k, v, layer=0)
    k_read, v_read, length = cache.read(layer=0)
    assert length == 3
    assert torch.equal(k_read, k)
    assert torch.equal(v_read, v)
    assert cache.identity()["kind"] == "stateful_kv_buffer"
    assert "NativeKVCacheAttnOp" not in cache.identity()["writer"]


def test_c7_stateful_cache_chunked_backward_survives_later_writes():
    cache = StatefulKVCache.allocate(
        n_layers=1,
        batch=1,
        n_kv_heads=1,
        max_seq_len=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    k1 = torch.randn(1, 1, 2, 2, requires_grad=True)
    v1 = torch.randn(1, 1, 2, 2, requires_grad=True)
    cache.write(k1, v1)
    k_prefix, v_prefix, _ = cache.read()
    loss = k_prefix.square().sum() + v_prefix.square().sum()

    k2 = torch.randn(1, 1, 1, 2, requires_grad=True)
    v2 = torch.randn(1, 1, 1, 2, requires_grad=True)
    cache.write(k2, v2)
    k_all, v_all, _ = cache.read()
    loss = loss + k_all.square().sum() + v_all.square().sum()
    loss.backward()

    for tensor in (k1, v1, k2, v2):
        assert tensor.grad is not None


def test_c7_stateful_cache_preserves_prefix_validity():
    cache = StatefulKVCache.allocate(
        n_layers=1,
        batch=2,
        n_kv_heads=1,
        max_seq_len=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    values = torch.ones(2, 1, 2, 2)
    valid = torch.tensor([[True, True], [True, False]])
    cache.write(values, values, valid_mask=valid)
    assert torch.equal(cache.read_valid_mask(), valid)


def test_c7_gold_b1_and_generate_rescore(contract, manifest):
    report = assert_stateful_kv_consistent(
        backend_profile="cuda_bf16",
        candidate="pytorch",
        contract=contract,
        manifest=manifest,
        device="cpu",
        require_declared_candidate=False,
    )
    assert report.b1_passed
    assert report.generate_rescore.passed
    assert report.b2_status == B2_PRODUCTION_KV_STATUS
    assert report.passed
    names = {item.metric for item in report.generate_rescore.metrics}
    assert names == {"max_abs_dlogp", "approx_kl0", "clipfrac0"}


def test_c7_rejects_concat_reference_as_b1(contract, manifest):
    with pytest.raises(RuntimeError, match="does not satisfy C7 B1"):
        assert_stateful_kv_consistent(
            backend_profile="cuda_bf16",
            candidate="pytorch",
            contract=contract,
            manifest=manifest,
            device="cpu",
            attn_op=NativeKVCacheAttnOp(),
            require_declared_candidate=False,
        )


def test_c6_c7_use_c1_not_private_thresholds():
    source = open("rl_engine/validation/operators/kv_consistency.py", encoding="utf-8").read()
    assert "_DECODE_ATOL" not in source
    assert "_PADDING_ATOL" not in source
