# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rl_engine.contracts.operators.compressor_state import (
    CompressorStateContract,
    CompressorStateError,
)
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache
from tests.models.deepseek_v4.compressor_state_helpers import (
    make_cache,
    tensors,
    ReferenceStateBackend,
    byte_equal,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("ratio", [4, 128])
def test_request_cursor_incomplete_groups_and_duplicate_commit(ratio):
    arena, x, _ = make_cache(ratio, max_sequence_tokens=512)
    h = arena.create_request(7, version="v1", writer_rank=0)
    lease = arena.prepare(h, x[: ratio - 1], start_position=0)
    assert arena.snapshot(h)["tokens"] == 0
    arena.commit(lease)
    snapshot = arena.snapshot(h)
    assert snapshot["rows"].shape == (0, 512)
    assert snapshot["incomplete_tokens"] == ratio - 1
    with pytest.raises(CompressorStateError, match="consumed"):
        arena.commit(lease)
    byte_equal(arena.snapshot(h)["partial_ring"], snapshot["partial_ring"])
    receipt = arena.append(h, x[ratio - 1 : ratio], start_position=ratio - 1)
    assert (receipt.first_group, receipt.end_group) == (0, 1)
    assert arena.snapshot(h)["rows"].shape == (1, 512)
    with pytest.raises(CompressorStateError, match="committed"):
        arena.append(h, x[:1], start_position=0)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("chunk", [1, 3, 7, 127, 256])
def test_chunked_and_full_state_and_rows_match(ratio, chunk):
    n = 269
    baseline, x, _ = make_cache(ratio, max_tokens_per_step=n, max_sequence_tokens=n)
    h0 = baseline.create_request(1, version="v1", writer_rank=0)
    baseline.append(h0, x, start_position=0)
    expected = baseline.snapshot(h0)
    arena, x, _ = make_cache(ratio, max_tokens_per_step=n, max_sequence_tokens=n)
    h = arena.create_request(1, version="v1", writer_rank=0)
    for start in range(0, n, chunk):
        arena.append(h, x[start : start + chunk], start_position=start)
    out = arena.snapshot(h)
    byte_equal(out["rows"], expected["rows"])
    byte_equal(out["partial_ring"], expected["partial_ring"])
    assert out["tokens"] == n


def test_cancel_and_prepared_original_alias_reuse():
    arena, x, _ = make_cache(max_sequence_tokens=256)
    h = arena.create_request(7, version="v1", writer_rank=0)
    lease = arena.prepare(h, x[:4], start_position=0)
    with pytest.raises(CompressorStateError, match="outstanding"):
        arena.prepare(h, x[:4], start_position=0)
    with pytest.raises(CompressorStateError, match="outstanding"):
        arena.release(h)
    arena.cancel(lease)
    assert arena.snapshot(h)["tokens"] == 0
    with pytest.raises(CompressorStateError, match="stale"):
        arena.commit(lease)
    lease = arena.prepare(h, x[:4], start_position=0)
    expected = x[:4].float().clone()
    x[:4].data.fill_(torch.nan)
    arena.commit(lease)
    state = arena.snapshot(h)["partial_ring"]
    byte_equal(state[:4, :1024], expected[:, :1024])
    for method in (arena.cancel, arena.commit):
        with pytest.raises(CompressorStateError, match="stale"):
            method(None)


def test_release_reuse_stale_foreign_and_model_barriers():
    arena, x, parameters = make_cache(max_requests=1, max_sequence_tokens=256)
    h = arena.create_request(7, version="v1", writer_rank=0)
    arena.append(h, x[:8], start_position=0)
    with pytest.raises(CompressorStateError, match="release all"):
        arena.replace_model(
            version="v2", **dict(zip(("ape", "norm", "cos", "sin"), parameters, strict=True))
        )
    with pytest.raises(CompressorStateError, match="capacity"):
        arena.create_request(8, version="v1", writer_rank=0)
    arena.release(h)
    with pytest.raises(CompressorStateError, match="stale"):
        arena.release(h)
    h2 = arena.create_request(7, version="v1", writer_rank=0)
    assert h2.slot == h.slot and h2.generation > h.generation
    assert bool((arena.snapshot(h2)["partial_ring"] == 0).all())
    with pytest.raises(CompressorStateError, match="stale"):
        arena.snapshot(h)
    other, _, _ = make_cache(max_sequence_tokens=256)
    with pytest.raises(CompressorStateError, match="foreign"):
        other.snapshot(h2)
    arena.release(h2)
    arena.replace_model(
        version="v2", **dict(zip(("ape", "norm", "cos", "sin"), parameters, strict=True))
    )
    with pytest.raises(CompressorStateError, match="version"):
        arena.create_request(7, version="v1", writer_rank=0)
    new = arena.create_request(7, version="v2", writer_rank=0)
    arena.append(new, x[:8], start_position=0)
    assert arena.snapshot(new)["version"] == "v2"


def test_unique_writer_and_request_validation():
    arena, x, _ = make_cache(world_size=2, max_sequence_tokens=256)
    other = arena.create_request(9, version="v1", writer_rank=1)
    with pytest.raises(CompressorStateError, match="writer"):
        arena.prepare(other, x[:4], start_position=0)
    for rid, version, rank in [(True, "v1", 0), (-1, "v1", 0), (10, "v2", 0), (10, "v1", 2)]:
        with pytest.raises(CompressorStateError):
            arena.create_request(rid, version=version, writer_rank=rank)
    with pytest.raises(CompressorStateError, match="already active"):
        arena.create_request(9, version="v1", writer_rank=1)


@pytest.mark.parametrize(
    "change",
    [
        {"ratio": 8},
        {"ratio": True},
        {"max_requests": 0},
        {"world_size": 0},
        {"max_sequence_tokens": 1},
        {"max_sequence_tokens": 2**31},
        {"max_arena_bytes": 1},
        {"graph": 1},
        {"max_requests": 2**31, "max_arena_bytes": 2**100},
    ],
)
def test_invalid_capacity_and_contract(change):
    with pytest.raises(CompressorStateError):
        replace(CompressorStateContract(), **change)


@pytest.mark.parametrize(
    "bad", ["dtype", "width", "stride", "nan", "inf", "grad", "capacity", "start"]
)
def test_bad_input_before_state_mutation(bad):
    arena, x, _ = make_cache(max_sequence_tokens=256, max_tokens_per_step=8)
    h = arena.create_request(1, version="v1", writer_rank=0)
    source, start = x[:4].clone(), 0
    if bad == "dtype":
        source = source.float()
    elif bad == "width":
        source = source[:, :512].contiguous()
    elif bad == "stride":
        source = source.T.contiguous().T
    elif bad == "nan":
        source[0, 0] = torch.nan
    elif bad == "inf":
        source[0, 0] = torch.inf
    elif bad == "grad":
        source.requires_grad_()
    elif bad == "capacity":
        source = x[:9]
    else:
        start = 1
    with pytest.raises(CompressorStateError):
        arena.prepare(h, source, start_position=start)
    assert arena.backend.calls == 0
    assert arena.snapshot(h)["tokens"] == 0


def test_sequence_capacity_and_poisoned_arena_never_retries_partial_write():
    arena, x, _ = make_cache(max_sequence_tokens=8)
    h = arena.create_request(1, version="v1", writer_rank=0)
    arena.append(h, x, start_position=0)
    with pytest.raises(CompressorStateError, match="sequence capacity"):
        arena.prepare(h, x[:1], start_position=8)

    class Fault(ReferenceStateBackend):
        def execute(self, arena):
            super().execute(arena)
            raise RuntimeError("simulated failure after native writes")

    arena, x, _ = make_cache(backend=Fault(), max_sequence_tokens=128)
    h = arena.create_request(1, version="v1", writer_rank=0)
    lease = arena.prepare(h, x[:4], start_position=0)
    with pytest.raises(RuntimeError, match="simulated"):
        arena.commit(lease)
    for method, args in [(arena.commit, (lease,)), (arena.snapshot, (h,)), (arena.release, (h,))]:
        with pytest.raises(CompressorStateError, match="poisoned"):
            method(*args)


def test_concurrent_commits_consume_one_lease_once():
    arena, x, _ = make_cache(max_sequence_tokens=128)
    h = arena.create_request(1, version="v1", writer_rank=0)
    lease = arena.prepare(h, x[:8], start_position=0)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(arena.commit, lease) for _ in range(2)]
        success = 0
        for f in futures:
            try:
                f.result()
                success += 1
            except CompressorStateError:
                pass
    assert success == 1 and arena.backend.calls == 1
    assert arena.snapshot(h)["commits"] == 1


def test_explicit_profile_handshake_and_bad_model_inputs():
    c = CompressorStateContract(ratio=128, max_sequence_tokens=128)
    _, a, n, co, si = tensors(c)
    common = dict(
        ape=a, norm=n, cos=co, sin=si, version="v1", writer_rank=0, backend=ReferenceStateBackend()
    )
    with pytest.raises(CompressorStateError, match="arithmetic mismatch"):
        CompressorStateCache(c, expected_arithmetic="aiter.hca.eight_waves", **common)
    for bad in [
        {"ape": a.bfloat16()},
        {"norm": n[:127]},
        {"cos": co.float()},
        {"sin": torch.full_like(si, torch.nan)},
        {"rms_eps": 0},
    ]:
        with pytest.raises(CompressorStateError):
            CompressorStateCache(c, expected_arithmetic=c.arithmetic_id, **(common | bad))
    with pytest.raises(CompressorStateError, match="fingerprint"):
        CompressorStateCache(
            c, expected_arithmetic=c.arithmetic_id, expected_implementation="wrong", **common
        )
    with pytest.raises(CompressorStateError, match="Graph"):
        CompressorStateCache(replace(c, graph=True), expected_arithmetic=c.arithmetic_id, **common)


@pytest.mark.parametrize("operation", ["create", "release", "model"])
def test_lifecycle_device_failure_also_poisons_arena(monkeypatch, operation):
    arena, x, parameters = make_cache(max_sequence_tokens=128)
    if operation == "release":
        h = arena.create_request(7, version="v1", writer_rank=0)
        arena.append(h, x[:4], start_position=0)

    def fault():
        raise RuntimeError("completion failed")

    monkeypatch.setattr(arena, "_complete_device", fault)
    with pytest.raises(RuntimeError, match="completion"):
        if operation == "create":
            arena.create_request(7, version="v1", writer_rank=0)
        elif operation == "release":
            arena.release(h)
        else:
            arena.replace_model(
                version="v2", **dict(zip(("ape", "norm", "cos", "sin"), parameters, strict=True))
            )
    with pytest.raises(CompressorStateError, match="poisoned"):
        arena.create_request(8, version="v1", writer_rank=0)


def test_framework_factory_reuse_and_immutable_snapshot_outputs():
    from rl_engine.integrations.engines.rollout.vllm.compressor_state import (
        bind_compressor_state as rollout,
    )
    from rl_engine.integrations.engines.train.megatron.compressor_state import (
        bind_compressor_state as train,
    )

    assert rollout is train
    c = CompressorStateContract(max_sequence_tokens=128)
    x, a, n, co, si = tensors(c)
    arena = rollout(
        c,
        expected_arithmetic=c.arithmetic_id,
        ape=a,
        norm=n,
        cos=co,
        sin=si,
        version="v1",
        writer_rank=0,
        backend=ReferenceStateBackend(),
    )
    h = arena.create_request(7, version="v1", writer_rank=0)
    arena.append(h, x[:8], start_position=0)
    snapshot = arena.snapshot(h)
    snapshot["rows"].fill_(torch.nan)
    snapshot["partial_ring"].fill_(torch.nan)
    actual = arena.snapshot(h)
    assert bool(torch.isfinite(actual["rows"]).all())
    assert bool(torch.isfinite(actual["partial_ring"]).all())
