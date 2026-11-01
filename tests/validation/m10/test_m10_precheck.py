# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T02 M10 pre-checks: positive stub path and fail-closed negative tests."""

import pytest
import torch

from rl_engine.validation.m10 import (
    M10Status,
    StubM10Provider,
    build_m10_stub,
    evaluate_m10_precheck,
)
from rl_engine.validation.sampling_params.schema import (
    CanonicalSamplingLayer,
    CanonicalSamplingRecord,
)


def _stub(**kwargs):
    return build_m10_stub(**kwargs)


def _non_identity_record(case_id="stub-case"):
    drifted = CanonicalSamplingLayer(top_k=50)
    identity = CanonicalSamplingLayer()
    return CanonicalSamplingRecord(
        case_id=case_id,
        engine="stub",
        requested=drifted,
        normalized=identity,
        materialized=identity,
        runtime_actual=identity,
    )


class TestM10PrecheckPassing:
    def test_stub_input_passes(self):
        result = evaluate_m10_precheck(_stub())
        assert result.status is M10Status.PASS
        assert result.passed
        assert result.failing_field is None

    def test_partial_active_ratio_still_passes(self):
        result = evaluate_m10_precheck(_stub(active_ratio=0.5))
        assert result.status is M10Status.PASS

    def test_stub_provider_exposes_the_read_abi(self):
        provider = StubM10Provider(tokens=4, hidden=32, local_vocab=64)
        inp = provider.read("case-42")
        assert inp.sampling_record.case_id == "case-42"
        assert evaluate_m10_precheck(inp).status is M10Status.PASS

    def test_matching_pre_update_weight_lock_passes(self):
        inp = _stub()
        result = evaluate_m10_precheck(
            inp, expected_pre_update_weight_lock=inp.pre_update_weight_lock
        )
        assert result.status is M10Status.PASS


class TestM10PrecheckFailClosed:
    def test_missing_sampling_record_fails(self):
        inp = _stub()
        inp.sampling_record = None
        result = evaluate_m10_precheck(inp)
        assert result.status is M10Status.MISSING_PROVENANCE
        assert result.failing_field == "sampling_record"

    def test_non_identity_sampling_fails(self):
        inp = _stub()
        inp.sampling_record = _non_identity_record()
        result = evaluate_m10_precheck(inp)
        assert result.status is M10Status.SAMPLING_PARAMS_MISMATCH
        assert result.failing_field == "sampling_record.requested"

    def test_lm_head_lora_is_unsupported(self):
        result = evaluate_m10_precheck(_stub(lm_head_lora=True))
        assert result.status is M10Status.UNSUPPORTED_CAPABILITY
        assert result.failing_field == "lm_head_lora"

    def test_missing_weight_lock_fails(self):
        inp = _stub()
        inp.pre_update_weight_lock = ""
        result = evaluate_m10_precheck(inp)
        assert result.status is M10Status.MISSING_PROVENANCE

    def test_weight_lock_drift_fails(self):
        inp = _stub()
        result = evaluate_m10_precheck(inp, expected_pre_update_weight_lock="other-lock")
        assert result.status is M10Status.PRE_UPDATE_WEIGHT_DRIFT

    def test_wrong_rows_dtype_fails(self):
        inp = _stub()
        inp.final_decoder_rows = inp.final_decoder_rows.float()
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_hidden_dim_mismatch_fails(self):
        inp = _stub()
        inp.lm_head_weight_local = torch.randn(
            inp.local_vocab_size, inp.final_decoder_rows.size(1) + 8, dtype=torch.bfloat16
        )
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_target_shape_mismatch_fails(self):
        inp = _stub()
        inp.target_token_ids = inp.target_token_ids[:1]
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_bad_vocab_range_fails(self):
        inp = _stub()
        inp.vocab_end = inp.vocab_start
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_vocab_width_mismatch_fails(self):
        inp = _stub()
        inp.vocab_start += 1
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_shard_beyond_real_vocab_fails(self):
        inp = _stub()
        inp.global_vocab_size = inp.vocab_end - 1
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_padded_target_fails(self):
        inp = _stub()
        inp.global_vocab_size = inp.vocab_end + 128
        inp.target_token_ids[0] = inp.global_vocab_size + 1
        assert evaluate_m10_precheck(inp).status is M10Status.INVALID_PROFILE

    def test_zero_active_tokens_fails(self):
        inp = _stub()
        inp.active_token_mask[:] = False
        assert evaluate_m10_precheck(inp).status is M10Status.ZERO_ACTIVE_TOKENS

    def test_non_finite_rows_fail(self):
        inp = _stub()
        inp.final_decoder_rows[0, 0] = float("inf")
        assert evaluate_m10_precheck(inp).status is M10Status.NON_FINITE


@pytest.mark.parametrize("kwargs", [{"tokens": 0}, {"hidden": -1}, {"active_ratio": 1.5}])
def test_invalid_stub_dimensions_raise(kwargs):
    with pytest.raises(ValueError):
        build_m10_stub(**kwargs)
