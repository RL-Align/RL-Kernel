# SPDX-License-Identifier: Apache-2.0
"""C128 compressor batch invariance: patch logic (CPU) and pinned schedule (ROCm + vLLM)."""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "rl_engine/backends/rocm/attention/compressor_schedule.py"
)
# Load by path: the attention package __init__ pulls in ROCm-only imports.
_spec = importlib.util.spec_from_file_location("compressor_schedule_under_test", _PATH)
sched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sched)


def _fake_module(monkeypatch):
    calls = []
    mod = types.ModuleType(sched._MODULE)

    def _pick_compress_num_splits(num_actual, compress_ratio, head_dim):
        return 2

    def _launch_two_stage_sparse_attn_compressor(**kw):
        # Like vLLM: resolve the module global at call time (``_pick_compress_num_splits(``).
        return mod._pick_compress_num_splits(1, 128, 512)

    def compress_norm_rope_store_two_stage_triton(**kw):
        calls.append(kw)

    mod._pick_compress_num_splits = _pick_compress_num_splits
    mod._launch_two_stage_sparse_attn_compressor = _launch_two_stage_sparse_attn_compressor
    mod.compress_norm_rope_store_two_stage_triton = compress_norm_rope_store_two_stage_triton
    other = types.ModuleType("fake_importer")
    other.compress_norm_rope_store_two_stage_triton = compress_norm_rope_store_two_stage_triton
    monkeypatch.setitem(sys.modules, sched._MODULE, mod)
    monkeypatch.setitem(sys.modules, "fake_importer", other)
    monkeypatch.setitem(sched._STATE, "bound", False)
    monkeypatch.setitem(sched._STATE, "executed", False)
    return mod, other, calls


def test_patch_logic_cpu(monkeypatch):
    mod, other, calls = _fake_module(monkeypatch)
    sched.pin_c128_compressor_schedule()
    assert mod._pick_compress_num_splits(99999, 128, 512) == 16
    assert mod._launch_two_stage_sparse_attn_compressor() == 16
    with pytest.raises(RuntimeError):
        mod._pick_compress_num_splits(10, 4, 512)
    assert (
        other.compress_norm_rope_store_two_stage_triton
        is mod.compress_norm_rope_store_two_stage_triton
    )
    scratch = torch.empty(8, 512)
    mod.compress_norm_rope_store_two_stage_triton(
        num_actual=8, head_dim=512, num_decode_tokens=5, compress_scratch=scratch
    )
    assert calls[0]["num_decode_tokens"] == 0
    with pytest.raises(RuntimeError):
        mod.compress_norm_rope_store_two_stage_triton(
            num_actual=9, head_dim=512, num_decode_tokens=0, compress_scratch=scratch
        )
    patched = mod.compress_norm_rope_store_two_stage_triton
    sched.pin_c128_compressor_schedule()  # idempotent
    assert mod.compress_norm_rope_store_two_stage_triton is patched


def test_patch_fails_closed(monkeypatch):
    mod, _, _ = _fake_module(monkeypatch)
    mod._launch_two_stage_sparse_attn_compressor = lambda **kw: 1  # no picker call
    with pytest.raises(RuntimeError):
        sched.pin_c128_compressor_schedule()


# ---------------------------------------------------------------- GPU tests
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm GPU")


def _real_vllm():
    pytest.importorskip("vllm")
    try:
        return importlib.import_module(sched._MODULE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vllm deepseek_v4 ops unavailable: {exc}")


@pytest.fixture
def pinned():
    _real_vllm()
    from rl_engine.backends.rocm.attention.compressor_schedule import (
        _STATE,
        pin_c128_compressor_schedule,
    )

    pin_c128_compressor_schedule()
    return _STATE


def _inputs(t, b, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(t, b, 512, device="cuda", dtype=torch.float32, generator=g)
    score = torch.randn(t, b, 512, device="cuda", dtype=torch.float32, generator=g)
    ape = torch.randn(128, 512, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.rand(512, device="cuda", dtype=torch.float32, generator=g) + 0.5
    cache = torch.randn(t + 128, 64, device="cuda", dtype=torch.float32, generator=g)
    return kv, score, ape, weight, cache


def _actor(kv, score, ape, weight, cache):
    from rl_engine.backends.rocm.attention.native_compressor import PINNED_C128_PROFILE, _forward

    return _forward(kv, score, ape, weight, cache, 128, 1e-6, PINNED_C128_PROFILE)


@gpu
@pytest.mark.parametrize("t", [128, 2180, 4096])
@pytest.mark.parametrize("b", [1, 2, 3])
def test_actor_batch_invariant(pinned, t, b):
    kv, score, ape, weight, cache = _inputs(t, b)
    full = _actor(kv, score, ape, weight, cache)
    for i in range(b):
        single = _actor(kv[:, i : i + 1], score[:, i : i + 1], ape, weight, cache)
        assert torch.equal(full[:, i], single[:, 0]), (t, b, i)


@gpu
def test_pinned_equals_unpinned_where_matched(pinned):
    m = _real_vllm()
    original_pick, _ = pinned["originals"]
    inputs = _inputs(2180, 1)
    pinned_out = _actor(*inputs)
    patched = m._pick_compress_num_splits
    m._pick_compress_num_splits = original_pick
    try:
        assert original_pick(2180, 128, 512) == 16  # precondition on this device
        unpinned_out = _actor(*inputs)
    finally:
        m._pick_compress_num_splits = patched
    assert torch.equal(pinned_out, unpinned_out)


def _rollout(seqs, ape, weight, cache):
    """Packed vLLM-side call over several sequences; returns per-sequence [n, 512] rows."""
    from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_two_stage_triton,
    )
    from vllm.models.deepseek_v4.common.ops.save_partial_states import _SAVE_PARTIAL_STATES_KERNEL
    from vllm.models.deepseek_v4.compressor import CompressorMetadata

    ratio, width, sblock, kblock = 128, 512, 64, 64
    lens = [s[0].shape[0] for s in seqs]
    n_seq, total = len(seqs), sum(lens)
    spages = max((n + sblock - 1) // sblock for n in lens)
    kpages = max((n // ratio + kblock - 1) // kblock for n in lens)
    dev = "cuda"
    state = torch.zeros((n_seq * spages, sblock, 2 * width), device=dev, dtype=torch.float32)
    positions = torch.cat([torch.arange(n, device=dev, dtype=torch.int64) for n in lens])
    owners = torch.cat(
        [torch.full((n,), i, device=dev, dtype=torch.int32) for i, n in enumerate(lens)]
    )
    slots = positions + owners.long() * spages * sblock
    table = torch.arange(n_seq * spages, device=dev, dtype=torch.int32).reshape(n_seq, spages)
    page_bytes = ((kblock * 584 + 575) // 576) * 576
    packed = torch.zeros((n_seq * kpages, page_bytes), device=dev, dtype=torch.uint8)
    kcache = packed.as_strided((n_seq * kpages, kblock, 576), (page_bytes, 576, 1))
    ktable = torch.arange(n_seq * kpages, device=dev, dtype=torch.int32).reshape(n_seq, kpages)
    kslots = positions // ratio + owners.long() * kpages * kblock
    metadata = CompressorMetadata(block_table=ktable, slot_mapping=kslots, block_size=kblock)
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=torch.cat([s[0] for s in seqs]),
        score=torch.cat([s[1] for s in seqs]),
        ape=ape,
        positions=positions,
        state_cache=state,
        slot_mapping=slots,
        block_size=sblock,
        state_width=width,
        compress_ratio=ratio,
        pdl_kwargs={},
    )
    compress_norm_rope_store_two_stage_triton(
        state_cache=state,
        num_actual=total,
        token_to_req_indices=owners,
        positions=positions,
        slot_mapping=slots,
        block_table=table,
        block_size=sblock,
        state_width=width,
        cos_sin_cache=cache,
        kv_cache=kcache,
        k_cache_metadata=metadata,
        pdl_kwargs={},
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=ratio,
        overlap=False,
        use_fp4_cache=False,
        rms_norm_weight=weight,
        rms_norm_eps=1e-6,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        num_decode_tokens=0,
        compress_scratch=torch.empty((total, 512), device=dev, dtype=torch.float32),
    )
    outs = []
    for i, n in enumerate(lens):
        out = torch.empty((1, n // ratio, 512), device=dev, dtype=torch.bfloat16)
        lengths = torch.full((1,), n // ratio, device=dev, dtype=torch.int32)
        dequantize_and_gather_k_cache(
            out, packed, lengths, None, ktable[i : i + 1], kblock, 0, use_fnuz=False
        )
        outs.append(out[0])
    return outs


@gpu
def test_rollout_packed_batch_invariant(pinned):
    ka, sa, ape, weight, cache = _inputs(130, 1, seed=1)
    kb, sb, _, _, _ = _inputs(2180, 1, seed=2)
    cache = torch.randn(2400, 64, device="cuda", dtype=torch.float32)
    a, b = (ka[:, 0], sa[:, 0]), (kb[:, 0], sb[:, 0])
    packed = _rollout([a, b], ape, weight, cache)
    alone = _rollout([b], ape, weight, cache)
    assert torch.equal(packed[1], alone[0])
