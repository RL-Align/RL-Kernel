# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P0: failure-batch selection, pre-update weight lock and synthetic fixtures."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from p7_doubles.fixtures import build_fixtures, fixture_digest, vime_request
from p7_doubles.t01_abi import TraceEnvelopeDouble, envelope_missing_fields
from p7_doubles.t01_status import CONTRACT_V3_STATUSES

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    T05_EMITTED_STATUSES,
    FailureBatchEntry,
    FailureBatchFilter,
    FailureBatchManifest,
    PreUpdateWeightLock,
    ShardDigest,
    T05ContractError,
    select_failure_batch,
)

pytestmark = pytest.mark.unit

FIXTURES = build_fixtures()


def _lock(fixture=FIXTURES["normal"], *, checkpoint="ckpt-0", weight="w-0"):
    shards = tuple(
        ShardDigest.from_tensor(
            fixture.lm_head_weight[start:end], tp_rank=rank, vocab_start=start, vocab_end=end
        )
        for rank, (start, end) in enumerate(fixture.vocab_shards)
    )
    return PreUpdateWeightLock(checkpoint_id=checkpoint, weight_id=weight, lm_head_shards=shards)


def _entry(case, *, sample="s0", rank="tp0", module="M10", run="run-a", checkpoint="ckpt-0"):
    return FailureBatchEntry(
        stable_case_id=case,
        run_id=run,
        checkpoint_id=checkpoint,
        sample_id=sample,
        rank=rank,
        module=module,
        global_token_ids=(0, 1, 2),
    )


def _manifest(entries, lock=None):
    return FailureBatchManifest(
        manifest_id="fb-1", entries=tuple(entries), weight_lock=lock or _lock()
    )


def test_emitted_statuses_are_contract_names():
    assert T05_EMITTED_STATUSES <= set(CONTRACT_V3_STATUSES)
    assert len(CONTRACT_V3_STATUSES) == 28 == len(set(CONTRACT_V3_STATUSES))
    with pytest.raises(ValueError):
        T05ContractError("NOT_A_STATUS", "f", "d")


def test_fixture_checksums_are_pinned():
    manifest = json.loads(
        (Path(__file__).parent / "p7_doubles" / "fixtures" / "manifest.json").read_text()
    )
    assert manifest["digests"] == {name: fixture_digest(fx) for name, fx in FIXTURES.items()}


def test_zigzag_rows_match_existing_megatron_helpers():
    from rl_engine.integrations.canonical_cp import packed_gather_indices
    from rl_engine.integrations.framework_operators import _megatron_zigzag_layout

    fixture = FIXTURES["packed_varlen_zigzag_cp2"]
    world = fixture.cp_world
    per_rank = [fixture.cp_local_rows(rank) for rank in range(world)]
    local_rows = len(per_rank[0])
    rank_major = [row for rows in per_rank for row in rows]
    gather = packed_gather_indices(list(fixture.sample_lengths), local_rows, world)
    assert [rank_major[index] for index in gather] == list(range(fixture.num_tokens))

    # The first packed sample (length 8, 4 local rows per rank) must follow the
    # single-sequence zigzag layout used by the strict Megatron attention path.
    for rank in range(world):
        positions, *_ = _megatron_zigzag_layout(4, cp_rank=rank, cp_world_size=world)
        assert tuple(positions) == fixture.cp_local_rows(rank)[:4]


def test_vime_request_double_matches_vime_dataclass():
    vime = pytest.importorskip("vime.backends.megatron_utils.linear_logp_provider")
    double = vime_request(FIXTURES["padded_vocab_tp2"], tp_rank=1)
    real = vime.LinearLogpRequest(
        logits=double.logits,
        target_ids=double.target_ids,
        tensor_parallel_group=None,
        token_layout=vime.TokenLayout(**vars(double.token_layout)),
        with_entropy=double.with_entropy,
        with_entropy_grad=double.with_entropy_grad,
        chunk_size=double.chunk_size,
        context=vime.LinearLogpContext(
            hidden=double.context.hidden,
            projection=vime.LinearProjection(weight=double.context.projection.weight),
            vocab_partition=vime.VocabPartition(**vars(double.context.vocab_partition)),
        ),
        temperature=double.temperature,
        metadata=double.metadata,
    )
    assert set(vars(double)) == {field.name for field in dataclasses.fields(real)}


def test_selection_is_deterministic_and_bound_to_lock():
    entries = [_entry("c2", sample="s1"), _entry("c1", sample="s0"), _entry("c3", rank="tp1")]
    manifest = _manifest(entries)
    reordered = _manifest(list(reversed(entries)))
    assert manifest.fingerprint == reordered.fingerprint
    picked = select_failure_batch(manifest, FailureBatchFilter.from_mapping({"rank": ["tp0"]}))
    assert [e.stable_case_id for e in picked.entries] == ["c1", "c2"]
    again = select_failure_batch(reordered, FailureBatchFilter.from_mapping({"rank": ["tp0"]}))
    assert picked.selection_fingerprint == again.selection_fingerprint
    other_filter = select_failure_batch(manifest, FailureBatchFilter())
    assert other_filter.selection_fingerprint != picked.selection_fingerprint


def test_empty_selection_fails_closed():
    manifest = _manifest([_entry("c1")])
    with pytest.raises(T05ContractError) as info:
        select_failure_batch(manifest, FailureBatchFilter.from_mapping({"module": ["M3"]}))
    assert info.value.status == "IDENTITY_DRIFT"


def test_unknown_filter_dimension_is_rejected():
    with pytest.raises(ValueError):
        FailureBatchFilter.from_mapping({"layer": ["7"]})


def test_entry_checkpoint_outside_lock_is_weight_drift():
    with pytest.raises(T05ContractError) as info:
        _manifest([_entry("c1", checkpoint="ckpt-after-update")])
    assert info.value.status == "PRE_UPDATE_WEIGHT_DRIFT"


def test_duplicate_global_token_is_ambiguous():
    with pytest.raises(T05ContractError) as info:
        FailureBatchEntry("c", "r", "k", "s", "tp0", "M10", global_token_ids=(1, 1))
    assert info.value.status == "AMBIGUOUS_GLOBAL_TOKEN_MAPPING"


def test_weight_lock_accepts_identical_shard_and_rejects_one_bit_drift():
    lock = _lock()
    weight = FIXTURES["normal"].lm_head_weight
    same = ShardDigest.from_tensor(weight, tp_rank=0, vocab_start=0, vocab_end=64)
    lock.verify_shard(same, checkpoint_id="ckpt-0", weight_id="w-0")
    drifted = ShardDigest.from_tensor(
        FIXTURES["weight_drift"].lm_head_weight, tp_rank=0, vocab_start=0, vocab_end=64
    )
    with pytest.raises(T05ContractError) as info:
        lock.verify_shard(drifted, checkpoint_id="ckpt-0", weight_id="w-0")
    assert info.value.status == "PRE_UPDATE_WEIGHT_DRIFT"
    with pytest.raises(T05ContractError) as info:
        lock.verify_shard(same, checkpoint_id="ckpt-1", weight_id="w-0")
    assert info.value.status == "PRE_UPDATE_WEIGHT_DRIFT"


def test_weight_lock_missing_rank_is_missing_provenance():
    lock = _lock()
    other = ShardDigest.from_tensor(
        FIXTURES["normal"].lm_head_weight[:32], tp_rank=1, vocab_start=32, vocab_end=64
    )
    with pytest.raises(T05ContractError) as info:
        lock.verify_shard(other, checkpoint_id="ckpt-0", weight_id="w-0")
    assert info.value.status == "MISSING_PROVENANCE"


def test_schema_version_mismatch_is_rejected():
    with pytest.raises(T05ContractError) as info:
        PreUpdateWeightLock("k", "w", _lock().lm_head_shards, schema_version="v0")
    assert info.value.status == "SCHEMA_MISMATCH"


def test_shard_digest_distinguishes_signed_zero():
    plus = FIXTURES["normal"].lm_head_weight.clone()
    minus = plus.clone()
    plus[0, 0] = 0.0
    minus[0, 0] = -0.0
    assert torch.equal(plus, minus), "value comparison cannot see the difference"
    a = ShardDigest.from_tensor(plus, tp_rank=0, vocab_start=0, vocab_end=64)
    b = ShardDigest.from_tensor(minus, tp_rank=0, vocab_start=0, vocab_end=64)
    assert a.sha256 != b.sha256, "+0 and -0 must hash differently"
    assert a.dtype == "bfloat16"


def test_trace_envelope_double_reports_missing_fields():
    envelope = TraceEnvelopeDouble(
        producer="P3",
        upstream_contract_id="P3.route.v1",
        producer_schema_version="1",
        run_id="run-a",
        case_id="c1",
        absolute_layer=3,
        pass_name="forward",
        global_token_ids=(0, 1),
        requested_backend="ref",
        actual_backend="ref",
        kernel_build="b",
        comparison_mode="byte_exact",
        producer_verdict="",
        fixture_checksum="x",
    )
    assert envelope_missing_fields(envelope.to_dict()) == ("producer_verdict",)
