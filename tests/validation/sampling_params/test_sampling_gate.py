# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T02 sampling-parameters gate: positive and fail-closed negative tests.

Two orthogonal properties are exercised: framework-raw parameters normalize to
the single canonical sentinel, and any non-identity / unprovenanced record is
rejected before any scoring happens.
"""

import json

import pytest

from rl_engine.validation.sampling_params import (
    CanonicalSamplingLayer,
    CanonicalSamplingRecord,
    SamplingGateStatus,
    SamplingLayerName,
    SamplingParamsError,
    evaluate_sampling_gate,
    normalize_sampling_layer,
)

IDENTITY_LAYER = CanonicalSamplingLayer()


def _record(**overrides):
    layers = {
        "requested": IDENTITY_LAYER,
        "normalized": IDENTITY_LAYER,
        "materialized": IDENTITY_LAYER,
        "runtime_actual": IDENTITY_LAYER,
    }
    layers.update(overrides)
    return CanonicalSamplingRecord(case_id="case-0", engine="stub", **layers)


class TestSamplingNormalization:
    @pytest.mark.parametrize("top_k", [None, 0, -1])
    @pytest.mark.parametrize("top_p", [None, 1.0])
    @pytest.mark.parametrize("min_p", [None, 0.0])
    def test_raw_identity_forms_normalize_to_sentinel(self, top_k, top_p, min_p):
        layer = normalize_sampling_layer(temperature=1.0, top_k=top_k, top_p=top_p, min_p=min_p)
        assert layer.is_identity()

    def test_absent_parameters_normalize_to_identity(self):
        assert normalize_sampling_layer().is_identity()

    def test_enabled_values_are_preserved(self):
        layer = normalize_sampling_layer(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05)
        assert (layer.temperature, layer.top_k, layer.top_p, layer.min_p) == (
            0.7,
            50,
            0.9,
            0.05,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 0.0},
            {"temperature": -1.0},
            {"temperature": float("nan")},
            {"top_k": -2},
            {"top_k": 1.5},
            {"top_k": True},
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"top_p": -0.1},
            {"min_p": -0.1},
            {"min_p": 1.5},
        ],
    )
    def test_structurally_invalid_parameters_raise(self, kwargs):
        with pytest.raises(SamplingParamsError):
            normalize_sampling_layer(**kwargs)


class TestSamplingGate:
    def test_identity_consistency_record_passes(self):
        result = evaluate_sampling_gate(_record())
        assert result.status is SamplingGateStatus.PASS
        assert result.passed
        assert result.canonical_fingerprint is not None
        assert result.failing_layer is None
        assert result.failing_field is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("temperature", 0.7),
            ("top_k", 50),
            ("top_p", 0.9),
            ("min_p", 0.05),
        ],
    )
    def test_non_identity_layer_fails(self, field, value):
        result = evaluate_sampling_gate(_record(requested=CanonicalSamplingLayer(**{field: value})))
        assert result.status is SamplingGateStatus.SAMPLING_PARAMS_MISMATCH
        assert not result.passed
        assert result.failing_layer == SamplingLayerName.REQUESTED.value
        assert result.failing_field == field

    def test_runtime_drift_is_reported_on_the_actual_layer(self):
        result = evaluate_sampling_gate(_record(runtime_actual=CanonicalSamplingLayer(top_k=50)))
        assert result.status is SamplingGateStatus.SAMPLING_PARAMS_MISMATCH
        assert result.failing_layer == SamplingLayerName.RUNTIME_ACTUAL.value
        assert result.failing_field == "top_k"

    def test_missing_layer_fails_closed(self):
        record = CanonicalSamplingRecord(case_id="case-1", engine="stub", requested=IDENTITY_LAYER)
        result = evaluate_sampling_gate(record)
        assert result.status is SamplingGateStatus.MISSING_PROVENANCE
        assert not result.passed
        assert result.failing_layer == SamplingLayerName.NORMALIZED.value
        assert result.canonical_fingerprint is None

    def test_silent_fallback_fails_even_when_values_are_identity(self):
        record = _record()
        record.provenance = {"fallback_layers": (SamplingLayerName.RUNTIME_ACTUAL.value,)}
        result = evaluate_sampling_gate(record)
        assert result.status is SamplingGateStatus.SILENT_FALLBACK
        assert not result.passed
        assert result.failing_layer == SamplingLayerName.RUNTIME_ACTUAL.value

    def test_fallback_takes_precedence_over_mismatch(self):
        record = _record(requested=CanonicalSamplingLayer(temperature=0.7))
        record.provenance = {"fallback_layers": (SamplingLayerName.REQUESTED.value,)}
        assert evaluate_sampling_gate(record).status is SamplingGateStatus.SILENT_FALLBACK


class TestCanonicalRecord:
    def test_fingerprint_is_stable_and_value_sensitive(self):
        assert _record().canonical_fingerprint() == _record().canonical_fingerprint()
        drifted = _record(runtime_actual=CanonicalSamplingLayer(top_k=50))
        assert drifted.canonical_fingerprint() != _record().canonical_fingerprint()

    def test_fingerprint_is_none_when_incomplete(self):
        record = CanonicalSamplingRecord(case_id="case-2", engine="stub", requested=IDENTITY_LAYER)
        assert record.canonical_fingerprint() is None

    def test_record_serializes_with_version(self):
        payload = _record().to_dict()
        assert payload["record_version"] == "p7-sampling-record-v1"
        assert json.loads(_record().to_json())["engine"] == "stub"
