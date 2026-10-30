# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from rl_engine.p6.__main__ import conformance, main
from rl_engine.p6.artifact import build_payload, publish, read_artifact, resume, source_hash, verify
from rl_engine.p6.contract import ContractError, digest
from rl_engine.p6.provider import ProviderRegistry, RecordedProvider, compare_envelopes
from rl_engine.p6.recordings import golden_records


@pytest.fixture
def payload():
    records = golden_records()[:1]
    provenance, checks = conformance(records, "cpu")
    return build_payload(records, provenance, checks, {"device": "cpu"})


def test_publish_verify_provider_and_resume(tmp_path, payload):
    directory = tmp_path / "attempt"
    publish(directory, payload)
    assert verify(directory)["operator_recordings"] == 5
    assert verify(directory)["gpu_reexecuted"] is False
    assert read_artifact(directory)[0] == payload
    provider = RecordedProvider.from_artifact(directory)
    registry = ProviderRegistry()
    registry.register("recorded", provider)
    for r in payload["operator_recordings"]:
        result = registry.run("recorded", r["operator"], r["case_id"], r["inputs"])
        assert result["stages"] == r["expected"]
        assert compare_envelopes(result, copy.deepcopy(result))["production_certified"] is False
    assert (
        resume(directory, payload["cases"], payload["request"])["status"]
        == "COMPLETE_ATTEMPT_REUSED"
    )
    assert source_hash() == payload["source_sha256"]
    with pytest.raises(ContractError, match="OUTPUT_EXISTS"):
        publish(directory, payload)
    with pytest.raises(ContractError, match="IDENTITY_DRIFT"):
        resume(directory, payload["cases"], {"device": "cuda:0"})


@pytest.mark.parametrize("filename", ["manifest.json", "payload.json", "seal.sha256"])
def test_corruption_or_missing_component(tmp_path, payload, filename):
    directory = tmp_path / "attempt"
    publish(directory, payload)
    path = directory / filename
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ContractError, match="INCOMPLETE_ARTIFACT"):
        verify(directory)
    path.write_bytes(original + b"x")
    with pytest.raises((ContractError, ValueError)):
        verify(directory)


@pytest.mark.parametrize("field", ["operator_recordings", "cases", "catalog", "required_evidence"])
def test_resealed_incomplete_evidence_rejected(tmp_path, payload, field):
    changed = copy.deepcopy(payload)
    changed[field] = []
    with pytest.raises(ContractError):
        publish(tmp_path / "attempt", changed)


def test_recorded_input_binding_and_no_fallback(payload):
    provider = RecordedProvider(payload["operator_recordings"], digest(payload))
    r = payload["operator_recordings"][0]
    registry = ProviderRegistry()
    registry.register("recorded", provider)
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY"):
        registry.run("live", r["operator"], r["case_id"], r["inputs"])
    with pytest.raises(ContractError, match="SILENT_FALLBACK"):
        registry.register("live", provider)
    changed = copy.deepcopy(r["inputs"])
    changed["rows"][0][0] = -changed["rows"][0][0]
    with pytest.raises(ContractError, match="IDENTITY_DRIFT"):
        registry.run("recorded", r["operator"], r["case_id"], changed)
    changed = copy.deepcopy(r["inputs"])
    changed["context"]["forward_id"] = "other"
    with pytest.raises(ContractError, match="STALE_RUN_METADATA"):
        registry.run("recorded", r["operator"], r["case_id"], changed)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("order_hash", "IDENTITY_DRIFT"),
        ("case_id", "IDENTITY_DRIFT"),
        ("stages", "BYTE_MISMATCH"),
        ("provenance", "MISSING_PROVENANCE"),
        ("fallback", "SILENT_FALLBACK"),
    ],
)
def test_comparator_catches_false_candidate(payload, mutation, code):
    provider = RecordedProvider(payload["operator_recordings"], digest(payload))
    r = payload["operator_recordings"][0]
    ref = provider.run(r["operator"], r["case_id"], r["inputs"])
    candidate = copy.deepcopy(ref)
    if mutation == "stages":
        candidate["stages"]["canonical_fp32"] = "00000000"
        candidate["boundary_hashes"] = {k: digest(v) for k, v in candidate["stages"].items()}
    elif mutation == "provenance":
        candidate["provenance"] = {}
    elif mutation == "fallback":
        candidate["fallback"] = True
    else:
        candidate[mutation] = "0" * 64
    with pytest.raises(ContractError, match=code):
        compare_envelopes(ref, candidate)


def test_explicit_live_candidate_replacement(payload):
    # This is a fake provider to test routing, never GPU or production evidence.
    recorded = RecordedProvider(payload["operator_recordings"], digest(payload))

    class Candidate:
        def describe(self):
            result = recorded.describe()
            result["kind"] = "live"
            return result

        def run(self, operator, case_id, inputs):
            result = recorded.run(operator, case_id, inputs)
            result["kind"] = "live"
            result["provenance"] = {
                "readback_kind": "actual",
                "backend": "test-fake",
                "device": "cpu",
                "implementation_sha256": "a" * 64,
            }
            return result

    registry = ProviderRegistry()
    registry.register("live", Candidate())
    r = payload["operator_recordings"][0]
    assert registry.run("live", r["operator"], r["case_id"], r["inputs"])["kind"] == "live"


def test_identity_drift_precedes_numerical_attribution(payload):
    provider = RecordedProvider(payload["operator_recordings"], digest(payload))
    r = payload["operator_recordings"][0]
    ref = provider.run(r["operator"], r["case_id"], r["inputs"])
    bad = copy.deepcopy(ref)
    bad["context"]["checkpoint_id"] = "wrong-checkpoint"
    bad["stages"]["canonical_fp32"] = "bad-output"
    with pytest.raises(ContractError, match="IDENTITY_DRIFT"):
        compare_envelopes(ref, bad)


def test_cli_fails_instead_of_pretending_h100(tmp_path, capsys):
    assert (
        main(
            ["conformance", "--device", "cpu", "--require-h100", "--output", str(tmp_path / "gpu")]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["code"] == "UNSUPPORTED_CAPABILITY"
    assert not (tmp_path / "gpu").exists()
