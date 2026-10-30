# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P1: fail-closed gate over real operator/provider evidence (CPU, strict-kernel double).

The real ``MegatronLogpOperator`` and the real VIME ``_provider_impl`` run;
only the CUDA-only device check is bypassed and the strict kernel is replaced
by ``CpuStrictLinearLogpDouble``. Every fail-closed branch has a negative test.
"""

from types import SimpleNamespace

import pytest
import torch
from p7_doubles import t02_sampling
from p7_doubles.fixtures import vime_request
from p7_doubles.harness import (
    FIXTURES,
    TP_GROUP,
    expect,
    make_context,
    patch_tp2,
    run,
    strict_observer,
)
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

import rl_engine.integrations.megatron_training_readback as readback
from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.integrations import framework_operators
from rl_engine.integrations.engines.train.megatron import operators as megatron_operators
from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver

pytestmark = pytest.mark.unit


@pytest.fixture
def observer(monkeypatch):
    with strict_observer(monkeypatch) as instance:
        yield instance


@pytest.fixture
def tp2(monkeypatch):
    return patch_tp2(monkeypatch)


# --------------------------------------------------------------------------- positive paths


def test_recompute_path_produces_fp32_evidence(observer):
    fixture = FIXTURES["normal"]
    snapshot = run(observer, vime_request(fixture))
    evidence = gate.validate_training_call(snapshot, make_context(fixture))
    assert evidence.path == "recompute"
    assert evidence.entrypoint == "sm90_deterministic_linear_logp"
    assert evidence.selected_logprob.dtype == torch.float32
    assert evidence.selected_logprob.shape == (8,)
    active = evidence.selected_logprob[fixture.active_mask]
    assert evidence.active_raw_bytes == active.view(torch.int32).numpy().tobytes()
    assert evidence.row_keys == tuple(("s0", position) for position in range(8))
    assert (evidence.tp_rank, evidence.tp_world_size) == (0, 1)
    assert (
        evidence.fingerprint
        == gate.validate_training_call(snapshot, make_context(fixture)).fingerprint
    )


def test_reuse_path_is_identified(observer):
    fixture = FIXTURES["normal"]
    request = vime_request(fixture, metadata_overrides={"complete_sampling_support": True})
    evidence = gate.validate_training_call(run(observer, request), make_context(fixture))
    assert evidence.path == "reuse_local_logits"
    assert evidence.entrypoint == "sm90_deterministic_logp_from_local_logits_tp"


def test_tp2_rank1_ownership(observer, tp2):
    fixture = FIXTURES["padded_vocab_tp2"]
    tp2.rank = 1
    request = vime_request(fixture, tp_rank=1)
    request.tensor_parallel_group = TP_GROUP
    evidence = gate.validate_training_call(run(observer, request), make_context(fixture))
    assert (evidence.tp_rank, evidence.tp_world_size) == (1, 2)
    assert (evidence.vocab_start, evidence.vocab_end) == (32, 64)
    assert (evidence.real_vocab_size, evidence.padded_vocab_size) == (60, 64)
    assert evidence.entrypoint == "sm90_deterministic_linear_logp_tp"


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_zigzag_cp2_row_keys(observer, cp_rank):
    fixture = FIXTURES["packed_varlen_zigzag_cp2"]
    snapshot = run(observer, vime_request(fixture, cp_rank=cp_rank))
    evidence = gate.validate_training_call(snapshot, make_context(fixture, cp_rank=cp_rank))
    identity = fixture.row_identity()
    expected = tuple(
        (f"s{identity[row][0]}", identity[row][1]) for row in fixture.cp_local_rows(cp_rank)
    )
    assert evidence.row_keys == expected


def test_configured_and_declared_fields_never_certify(observer):
    fixture = FIXTURES["normal"]
    evidence = gate.validate_training_call(
        run(observer, vime_request(fixture)), make_context(fixture)
    )
    assert "context.hidden" in evidence.certifying_sources
    for name in ("metadata", "result.provenance", "operator.provenance", "linear_logp.provenance"):
        assert name not in evidence.certifying_sources


def test_sampling_sentinels_normalize_to_identity(observer):
    fixture = FIXTURES["normal"]
    raw = {"temperature": None, "top_k": 0, "top_p": None, "min_p": None}
    context = make_context(fixture, sampling_record=t02_sampling.record(raw))
    gate.validate_training_call(run(observer, vime_request(fixture)), context)


# --------------------------------------------------------------------------- snapshot / schema


def test_capture_error_is_missing_provenance():
    snapshot = readback.TrainingCallSnapshot(call_index=0, fields=(), capture_error="boom")
    expect("MISSING_PROVENANCE", snapshot, make_context(FIXTURES["normal"]), "snapshot")


def test_operator_error_is_missing_provenance(observer):
    request = vime_request(FIXTURES["normal"])
    request.context = SimpleNamespace(
        hidden=request.context.hidden, projection=None, vocab_partition=None
    )
    with pytest.raises(RuntimeError):
        run(observer, request)
    expect(
        "MISSING_PROVENANCE", observer.snapshots[-1], make_context(FIXTURES["normal"]), "snapshot"
    )


def test_fp32_hidden_is_schema_mismatch(observer):
    request = vime_request(FIXTURES["normal"])
    request.context.hidden = request.context.hidden.float()
    request.context.projection.weight = request.context.projection.weight.float()
    expect(
        "SCHEMA_MISMATCH",
        run(observer, request),
        make_context(FIXTURES["normal"]),
        "context.hidden",
    )


def test_bf16_output_is_schema_mismatch(observer):
    wrapper = CpuStrictLinearLogpDouble(out_dtype=torch.bfloat16)
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)
    expect("SCHEMA_MISMATCH", snapshot, make_context(FIXTURES["normal"]), "result.logp")


def test_weight_without_digest_is_missing_provenance(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL_STRICT", "1")
    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda tensor, module: None)
    monkeypatch.setattr(megatron_operators, "_require_nvidia_cuda", lambda tensor, module: None)
    observer = TrainingLogpObserver(digest_tensors=False)
    observer.install()
    try:
        snapshot = run(observer, vime_request(FIXTURES["normal"]))
    finally:
        observer.uninstall()
    expect(
        "MISSING_PROVENANCE",
        snapshot,
        make_context(FIXTURES["normal"]),
        "context.projection.weight",
    )


# --------------------------------------------------------------------------- identity / layout


def test_metadata_tp_rank_disagreeing_with_group_is_identity_drift(observer):
    request = vime_request(FIXTURES["normal"], metadata_overrides={"tp_rank": 1})
    expect(
        "IDENTITY_DRIFT",
        run(observer, request),
        make_context(FIXTURES["normal"]),
        "metadata.tp_rank",
    )


def test_metadata_tp_world_without_group_is_identity_drift(observer):
    fixture = FIXTURES["padded_vocab_tp2"]
    request = vime_request(fixture, tp_rank=0)  # declares tp_world_size=2 but passes no group
    expect(
        "IDENTITY_DRIFT", run(observer, request), make_context(fixture), "metadata.tp_world_size"
    )


def test_wrong_vocab_offset_is_identity_drift(observer, tp2):
    fixture = FIXTURES["padded_vocab_tp2"]
    tp2.rank = 0
    request = vime_request(fixture, tp_rank=1)  # rank-1 shard sent from a rank-0 process
    request.tensor_parallel_group = TP_GROUP
    request.metadata["tp_rank"] = 0
    expect(
        "IDENTITY_DRIFT", run(observer, request), make_context(fixture), "context.vocab_partition"
    )


def test_row_layout_disagreeing_with_call_is_identity_drift(observer):
    fixture = FIXTURES["normal"]
    context = make_context(fixture, row_layout=gate.RowLayout(("s0",), (6,)))
    expect("IDENTITY_DRIFT", run(observer, vime_request(fixture)), context, "row_layout")


def test_unknown_layout_is_unsupported(observer):
    request = vime_request(FIXTURES["normal"])
    request.token_layout = SimpleNamespace(world_size=2, rank=0, layout="ring")
    expect(
        "UNSUPPORTED_CAPABILITY",
        run(observer, request),
        make_context(FIXTURES["normal"]),
        "token_layout.layout",
    )


def _allgather_request(rank, rows):
    base = vime_request(FIXTURES["normal"])
    request = vime_request(FIXTURES["normal"])
    request.logits, request.target_ids = base.logits[rows], base.target_ids[rows]
    request.context.hidden = base.context.hidden[rows]
    request.token_layout = SimpleNamespace(world_size=2, rank=rank, layout="allgather")
    return request


@pytest.mark.parametrize("rank", [0, 1])
def test_allgather_layout_follows_vime_contiguous_split(observer, rank):
    rows = slice(4 * rank, 4 * rank + 4)  # one sample of 8 tokens over CP=2 -> 4 rows each
    mask = FIXTURES["normal"].active_mask[rows].clone()
    context = make_context(FIXTURES["normal"], active_token_mask=mask)
    evidence = gate.validate_training_call(run(observer, _allgather_request(rank, rows)), context)
    assert evidence.row_keys == tuple(
        ("s0", position) for position in range(4 * rank, 4 * rank + 4)
    )


def test_allgather_tail_rows_are_padding(observer):
    rows = slice(0, 4)  # rank 1 of 6 tokens over CP=2 with 4 local rows: tokens 4,5 + 2 padding
    layout = gate.RowLayout(("s0",), (6,))
    request = _allgather_request(1, rows)
    snapshot = run(observer, request)
    ok = make_context(
        FIXTURES["normal"],
        row_layout=layout,
        active_token_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
    )
    assert gate.validate_training_call(snapshot, ok).row_keys == (("s0", 4), ("s0", 5), None, None)
    bad = make_context(
        FIXTURES["normal"],
        row_layout=layout,
        active_token_mask=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    )
    expect("IDENTITY_DRIFT", snapshot, bad, "active_token_mask")


def test_active_padding_row_is_identity_drift(observer):
    base = vime_request(FIXTURES["normal"])
    rows = slice(0, 4)  # sample length 6 over zigzag CP=2 -> 4 local rows, rows 2-3 padding
    request = vime_request(FIXTURES["normal"])
    request.logits, request.target_ids = base.logits[rows], base.target_ids[rows]
    request.context.hidden = base.context.hidden[rows]
    request.token_layout = SimpleNamespace(world_size=2, rank=0, layout="zigzag")
    layout = gate.RowLayout(("s0",), (6,))
    snapshot = run(observer, request)
    ok = make_context(
        FIXTURES["normal"],
        row_layout=layout,
        active_token_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
    )
    evidence = gate.validate_training_call(snapshot, ok)
    assert evidence.row_keys == (("s0", 0), ("s0", 1), None, None)
    bad = make_context(
        FIXTURES["normal"],
        row_layout=layout,
        active_token_mask=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    )
    expect("IDENTITY_DRIFT", snapshot, bad, "active_token_mask")


# --------------------------------------------------------------------------- weights


def test_weight_drift_against_lock(observer):
    snapshot = run(observer, vime_request(FIXTURES["weight_drift"]))
    expect("PRE_UPDATE_WEIGHT_DRIFT", snapshot, make_context(FIXTURES["normal"]), "lm_head_shards")


def test_wrong_checkpoint_against_lock(observer):
    snapshot = run(observer, vime_request(FIXTURES["normal"]))
    expect(
        "PRE_UPDATE_WEIGHT_DRIFT",
        snapshot,
        make_context(FIXTURES["normal"], checkpoint_id="ckpt-1"),
        "checkpoint_id",
    )


def test_lm_head_bias_is_unsupported(observer):
    request = vime_request(FIXTURES["normal"])
    request.context.projection.bias = torch.zeros(64, dtype=torch.bfloat16)
    expect(
        "UNSUPPORTED_CAPABILITY",
        run(observer, request),
        make_context(FIXTURES["normal"]),
        "context.projection.bias",
    )


# ---------------------------------------------------------------- path / staleness / fallback


def test_stale_wrapper_provenance_is_detected(observer):
    wrapper = CpuStrictLinearLogpDouble(freeze_after_calls=1)
    run(observer, vime_request(FIXTURES["normal"]), wrapper)
    request = vime_request(FIXTURES["packed_varlen_zigzag_cp2"], cp_rank=0)  # 6 rows, not 8
    snapshot = run(observer, request, wrapper)
    expect(
        "MISSING_PROVENANCE",
        snapshot,
        make_context(FIXTURES["packed_varlen_zigzag_cp2"]),
        "linear_logp.",
    )


def test_unknown_entrypoint_is_missing_provenance(observer):
    wrapper = CpuStrictLinearLogpDouble(entrypoint="native_log_softmax")
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)
    expect(
        "MISSING_PROVENANCE",
        snapshot,
        make_context(FIXTURES["normal"]),
        "linear_logp.strict_entrypoint",
    )


def test_reuse_flag_contradiction_is_missing_provenance(observer):
    wrapper = CpuStrictLinearLogpDouble(entrypoint="sm90_deterministic_logp_from_local_logits_tp")
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)  # recompute call
    expect(
        "MISSING_PROVENANCE",
        snapshot,
        make_context(FIXTURES["normal"]),
        "provenance.execution.lm_head_result_reused",
    )


def test_fallback_flag_is_silent_fallback(observer):
    wrapper = CpuStrictLinearLogpDouble(extra_provenance={"fallback_used": True})
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)
    expect("SILENT_FALLBACK", snapshot, make_context(FIXTURES["normal"]), "provenance")


def test_non_strict_backend_is_silent_fallback(observer):
    wrapper = CpuStrictLinearLogpDouble(extra_provenance={"actual_backend": "native"})
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)
    expect("SILENT_FALLBACK", snapshot, make_context(FIXTURES["normal"]), "backend_id")


def test_non_deterministic_route_is_silent_fallback(observer):
    wrapper = CpuStrictLinearLogpDouble(extra_provenance={"deterministic_linear_logp": False})
    snapshot = run(observer, vime_request(FIXTURES["normal"]), wrapper)
    expect("SILENT_FALLBACK", snapshot, make_context(FIXTURES["normal"]), "provenance")


# --------------------------------------------------------------------------- sampling


def test_temperature_not_one_is_sampling_mismatch(observer):
    snapshot = run(observer, vime_request(FIXTURES["normal"], temperature=0.7))
    expect("SAMPLING_PARAMS_MISMATCH", snapshot, make_context(FIXTURES["normal"]), "temperature")


def test_prescaled_logits_still_check_requested_temperature(observer):
    request = vime_request(
        FIXTURES["normal"],
        temperature=0.7,
        metadata_overrides={"logits_are_temperature_scaled": True},
    )
    expect("SAMPLING_PARAMS_MISMATCH", run(observer, request), make_context(FIXTURES["normal"]))


def test_tensor_temperature_identity_passes_and_non_identity_fails(observer):
    fixture = FIXTURES["normal"]
    ok = run(observer, vime_request(fixture, temperature=torch.ones(8)))
    gate.validate_training_call(ok, make_context(fixture))
    bad = run(observer, vime_request(fixture, temperature=torch.full((8,), 0.5)))
    expect("SAMPLING_PARAMS_MISMATCH", bad, make_context(fixture))


def test_top_p_keep_mask_is_sampling_mismatch(observer):
    fixture = FIXTURES["normal"]
    keep = torch.ones(8, 64, dtype=torch.bool)
    keep[:, -5:] = False
    snapshot = run(observer, vime_request(fixture, keep_mask=keep))
    expect("SAMPLING_PARAMS_MISMATCH", snapshot, make_context(fixture))


def test_sparse_nucleus_metadata_is_sampling_mismatch(observer):
    fixture = FIXTURES["normal"]
    request = vime_request(fixture)
    snapshot = run(observer, request)
    request_with_ids = vime_request(
        fixture, metadata_overrides={"top_p_sparse_token_ids": torch.zeros(8, 4)}
    )
    fields = [
        (
            readback.EvidenceField(
                "metadata",
                readback.EvidenceSource.CONFIGURED,
                "request.metadata",
                readback.json_safe(request_with_ids.metadata),
            )
            if item.name == "metadata"
            else item
        )
        for item in snapshot.fields
    ]
    patched = readback.TrainingCallSnapshot(
        call_index=snapshot.call_index,
        fields=tuple(fields),
        selected_logprob=snapshot.selected_logprob,
    )
    expect(
        "SAMPLING_PARAMS_MISMATCH",
        patched,
        make_context(fixture),
        "metadata.top_p_sparse_token_ids",
    )


@pytest.mark.parametrize(
    "layer, key, value",
    [
        ("normalized", "top_k", 50),
        ("normalized", "min_p", 0.05),
        ("materialized", "top_p", 0.9),
        ("actual", "temperature", 0.8),
        ("actual", "top_k", 20),
    ],
)
def test_non_identity_sampling_layer_is_mismatch(observer, layer, key, value):
    fixture = FIXTURES["normal"]
    record = t02_sampling.record(t02_sampling.IDENTITY_REQUEST)
    record[layer][key] = value
    snapshot = run(observer, vime_request(fixture))
    expect("SAMPLING_PARAMS_MISMATCH", snapshot, make_context(fixture, sampling_record=record))


def test_requested_top_k_normalizes_to_non_identity(observer):
    fixture = FIXTURES["normal"]
    record = t02_sampling.record({**t02_sampling.IDENTITY_REQUEST, "top_k": 40})
    snapshot = run(observer, vime_request(fixture))
    expect("SAMPLING_PARAMS_MISMATCH", snapshot, make_context(fixture, sampling_record=record))


def test_missing_sampling_record_or_layer_is_missing_provenance(observer):
    fixture = FIXTURES["normal"]
    snapshot = run(observer, vime_request(fixture))
    expect("MISSING_PROVENANCE", snapshot, make_context(fixture, sampling_record=None))
    record = t02_sampling.record(t02_sampling.IDENTITY_REQUEST)
    del record["materialized"]
    expect("MISSING_PROVENANCE", snapshot, make_context(fixture, sampling_record=record))
    record = t02_sampling.record(t02_sampling.IDENTITY_REQUEST)
    del record["actual"]["min_p"]
    expect("MISSING_PROVENANCE", snapshot, make_context(fixture, sampling_record=record))


# --------------------------------------------------------------------------- capability / output


def test_lm_head_lora_states(observer):
    fixture = FIXTURES["normal"]
    snapshot = run(observer, vime_request(fixture))
    expect("MISSING_PROVENANCE", snapshot, make_context(fixture, lm_head_lora=None))
    expect("UNSUPPORTED_CAPABILITY", snapshot, make_context(fixture, lm_head_lora=True))


def test_zero_active_tokens(observer):
    fixture = FIXTURES["zero_active"]
    expect(
        "ZERO_ACTIVE_TOKENS",
        run(observer, vime_request(fixture)),
        make_context(fixture),
        "active_token_mask",
    )


def test_active_mask_must_be_bool_rows(observer):
    fixture = FIXTURES["normal"]
    snapshot = run(observer, vime_request(fixture))
    bad = make_context(fixture, active_token_mask=fixture.active_mask.to(torch.int64))
    expect("SCHEMA_MISMATCH", snapshot, bad)


def test_non_finite_active_logprob(observer):
    fixture = FIXTURES["normal"]
    active_row = int(fixture.active_mask.nonzero()[0])
    inactive_row = int((~fixture.active_mask).nonzero()[0])
    bad = run(observer, vime_request(fixture), CpuStrictLinearLogpDouble(poison_rows=(active_row,)))
    expect("NON_FINITE", bad, make_context(fixture), "result.logp")
    ok = run(
        observer, vime_request(fixture), CpuStrictLinearLogpDouble(poison_rows=(inactive_row,))
    )
    gate.validate_training_call(ok, make_context(fixture))


def test_duplicate_snapshot_field_names_are_rejected():
    item = readback.EvidenceField("x", readback.EvidenceSource.OBSERVED, "o", 1)
    with pytest.raises(ValueError):
        readback.TrainingCallSnapshot(call_index=0, fields=(item, item))
