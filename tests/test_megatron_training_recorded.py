# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P1: recorded training provider over the real operator/provider route (CPU double)."""

from types import SimpleNamespace

import pytest
import torch
from p7_doubles import t02_sampling
from p7_doubles.fixtures import build_fixtures, vime_request
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.alignment.cross_config.adapters.megatron_training import (
    PreUpdateWeightLock,
    ShardDigest,
    T05ContractError,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (
    M10ProviderInput,
    RecordedTrainingProvider,
)
from rl_engine.integrations import framework_operators
from rl_engine.integrations.engines.train.megatron import operators as megatron_operators
from rl_engine.integrations.framework_operators import MegatronLogpOperator
from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver, tensor_identity
from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

pytestmark = pytest.mark.unit

FIXTURES = build_fixtures()


@pytest.fixture(autouse=True)
def _cpu_strict_route(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL_STRICT", "1")
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "R/R")
    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda tensor, module: None)
    monkeypatch.setattr(megatron_operators, "_require_nvidia_cuda", lambda tensor, module: None)


def _operator():
    return MegatronLogpOperator(_provider_impl, linear_logp=CpuStrictLinearLogpDouble())


def _namespace_request(m10, logits):
    return SimpleNamespace(
        logits=logits,
        target_ids=m10.target_token_ids,
        tensor_parallel_group=m10.tensor_parallel_group,
        token_layout=SimpleNamespace(
            world_size=m10.token_layout_world_size,
            rank=m10.token_layout_rank,
            layout=m10.token_layout,
        ),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=m10.chunk_size,
        log_prob_keep_mask=None,
        context=SimpleNamespace(
            hidden=m10.final_decoder_rows,
            projection=SimpleNamespace(weight=m10.lm_head_weight_local, bias=None),
            vocab_partition=SimpleNamespace(
                local_start=m10.vocab_start,
                local_size=m10.vocab_local_size,
                real_size=m10.real_vocab_size,
                padded_size=m10.padded_vocab_size,
            ),
        ),
        temperature=m10.temperature,
        metadata=dict(m10.metadata),
    )


def _m10(fixture, *, cp_rank=0, **overrides):
    rows = list(fixture.cp_local_rows(cp_rank))
    start, end = fixture.vocab_shards[0]
    shard = ShardDigest.from_tensor(
        fixture.lm_head_weight[start:end], tp_rank=0, vocab_start=start, vocab_end=end
    )
    values = dict(
        final_decoder_rows=fixture.hidden[rows].clone(),
        lm_head_weight_local=fixture.lm_head_weight[start:end].clone(),
        target_token_ids=fixture.target_ids[rows].clone(),
        active_token_mask=fixture.active_mask[rows].clone(),
        vocab_start=start,
        real_vocab_size=fixture.real_vocab,
        padded_vocab_size=fixture.padded_vocab,
        token_layout_world_size=fixture.cp_world,
        token_layout_rank=cp_rank,
        token_layout="single" if fixture.cp_world == 1 else "zigzag",
        row_layout=gate.RowLayout(
            tuple(f"s{i}" for i in range(len(fixture.sample_lengths))), fixture.sample_lengths
        ),
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", (shard,)),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
        metadata={"tp_rank": 0, "tp_world_size": 1},
    )
    values.update(overrides)
    return M10ProviderInput(**values)


def _bytes(tensor):
    return tensor.contiguous().view(torch.int32).numpy().tobytes()


@pytest.mark.parametrize("fixture_name, cp_rank", [("normal", 0), ("packed_varlen_zigzag_cp2", 1)])
def test_recorded_matches_live_route_bitwise(fixture_name, cp_rank):
    fixture = FIXTURES[fixture_name]
    m10 = _m10(fixture, cp_rank=cp_rank)
    recorded = RecordedTrainingProvider(
        operator_factory=_operator, request_factory=_namespace_request
    ).score(m10)

    observer = TrainingLogpObserver()
    observer.install(MegatronLogpOperator)
    try:
        _operator()(vime_request(fixture, cp_rank=cp_rank))
    finally:
        observer.uninstall()
    context = gate.TrainingCallContext(
        checkpoint_id=m10.checkpoint_id,
        weight_id=m10.weight_id,
        weight_lock=m10.weight_lock,
        row_layout=m10.row_layout,
        active_token_mask=m10.active_token_mask,
        lm_head_lora=False,
        sampling_record=m10.sampling_record,
    )
    live = gate.validate_training_call(observer.snapshots[0], context)

    assert recorded.path == live.path == "recompute"
    assert recorded.row_keys == live.row_keys
    assert _bytes(recorded.selected_logprob) == _bytes(live.selected_logprob)
    assert recorded.active_bytes_sha256 == live.active_bytes_sha256


def test_recorded_with_real_vime_request_types():
    pytest.importorskip("vime.backends.megatron_utils.linear_logp_provider")
    evidence = RecordedTrainingProvider(operator_factory=_operator).score(_m10(FIXTURES["normal"]))
    assert evidence.path == "recompute" and evidence.selected_logprob.dtype == torch.float32


def test_recorded_logits_enable_reuse_path():
    fixture = FIXTURES["normal"]
    logits = vime_request(fixture).logits
    m10 = _m10(fixture, local_logits=logits, metadata={"complete_sampling_support": True})
    evidence = RecordedTrainingProvider(
        operator_factory=_operator, request_factory=_namespace_request
    ).score(m10)
    assert evidence.path == "reuse_local_logits"


def test_placeholder_logits_consumed_by_reuse_path_fails_closed():
    m10 = _m10(FIXTURES["normal"], metadata={"complete_sampling_support": True})
    with pytest.raises(T05ContractError) as info:
        RecordedTrainingProvider(
            operator_factory=_operator, request_factory=_namespace_request
        ).score(m10)
    assert (info.value.status, info.value.field_name) == ("MISSING_PROVENANCE", "logits")


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("final_decoder_rows", lambda m: m.final_decoder_rows.float()),
        ("lm_head_weight_local", lambda m: m.lm_head_weight_local.float()),
        ("target_token_ids", lambda m: m.target_token_ids.to(torch.int32)),
        ("active_token_mask", lambda m: m.active_token_mask.to(torch.uint8)),
        ("local_logits", lambda m: torch.zeros(3, 3, dtype=torch.bfloat16)),
    ],
)
def test_input_schema_violations(field_name, value):
    base = _m10(FIXTURES["normal"])
    with pytest.raises(T05ContractError) as info:
        _m10(FIXTURES["normal"], **{field_name: value(base)})
    assert info.value.status == "SCHEMA_MISMATCH"


def test_recorded_refuses_to_share_an_observed_class():
    other = TrainingLogpObserver()
    other.install(MegatronLogpOperator)
    try:
        with pytest.raises(RuntimeError):
            RecordedTrainingProvider(
                operator_factory=_operator, request_factory=_namespace_request
            ).score(_m10(FIXTURES["normal"]))
    finally:
        other.uninstall()


def test_zero_stride_placeholder_is_hashed_without_materializing():
    placeholder = torch.zeros((), dtype=torch.bfloat16).expand(4096, 32768)
    identity = tensor_identity(placeholder)
    assert identity["broadcast_dims"] == [0, 1]
    assert identity["sha256"] == tensor_identity(torch.zeros(1, 1, dtype=torch.bfloat16))["sha256"]
